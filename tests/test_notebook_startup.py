"""Execute embedded notebook programs with a controlled lifecycle owner; no GPU."""
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import socket
import sys
from types import ModuleType, SimpleNamespace
import textwrap
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime/minimax_h3"))
import startup
from test_notebook_layout import execute_program, program_tree


def execute_owner_guard(program, owner_name, state):
    """Run the owner guard embedded in the generated notebook."""
    tree = program_tree(program)
    for block in ast.walk(tree):
        body = getattr(block, "body", None)
        if not isinstance(body, list):
            continue
        for index, node in enumerate(body):
            if (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == owner_name):
                guard_tree = ast.Module(body=body[index:index + 2], type_ignores=[])
                scope = {"startup_host": SimpleNamespace(snapshot=lambda: dict(state))} if state is not None else {}
                exec(compile(ast.fix_missing_locations(guard_tree), "notebook-owner-guard", "exec"), scope)
                return
    raise AssertionError(f"Guard not found: {owner_name}")


@pytest.mark.parametrize("program, owner_name", [
    ("install", "existing_host"),
    ("start", "previous_host"),
])
@pytest.mark.parametrize("phase, cleanup, server_alive, tunnel_alive", [
    ("failed", False, True, True),
    ("failed", False, False, False),
    ("failed", True, True, False),
    ("failed", True, False, True),
    ("stopped", False, False, False),
    ("stopped", True, True, False),
    ("stopped", True, False, True),
    ("ready", True, False, False),
    ("new", False, True, False),
])
def test_owner_guard_rejects_uncleared_or_active_run(program, owner_name, phase, cleanup,
                                                    server_alive, tunnel_alive):
    state = dict(phase=phase, cleanup_complete=cleanup, server_alive=server_alive, tunnel_alive=tunnel_alive)
    with pytest.raises(RuntimeError, match="尚未完成清理"):
        execute_owner_guard(program, owner_name, state)


@pytest.mark.parametrize("program, owner_name", [
    ("install", "existing_host"),
    ("start", "previous_host"),
])
@pytest.mark.parametrize("phase, cleanup", [("failed", True), ("stopped", True), ("new", False)])
def test_owner_guard_allows_cleaned_terminal_or_unstarted_run(program, owner_name, phase, cleanup):
    execute_owner_guard(program, owner_name, dict(phase=phase, cleanup_complete=cleanup,
                                               server_alive=False, tunnel_alive=False))


@pytest.mark.parametrize("program, owner_name", [
    ("install", "existing_host"),
    ("start", "previous_host"),
])
def test_owner_guard_requires_cleanup_evidence_and_allows_absent_owner(program, owner_name):
    with pytest.raises(RuntimeError, match="尚未完成清理"):
        execute_owner_guard(program, owner_name, dict(phase="failed", server_alive=False, tunnel_alive=False))
    execute_owner_guard(program, owner_name, None)


@pytest.mark.parametrize("public", [False, True])
def test_start_returns_before_join_and_join_does_not_launch_again(tmp_path, monkeypatch, public):
    events = []
    state = dict(phase="preparing", ready=False, model_ready=False, tunnel_connected=False,
                 public_ready=False, public_requested=public, elapsed_seconds=0,
                 instance_id="current-instance", public_url="https://example.trycloudflare.com",
                 server_alive=True, tunnel_alive=public)

    class Owner:
        def __init__(self, directory, **options):
            assert options["public"] == public
            assert options["install_cloudflared"] == public
            self.instance_id = state["instance_id"]
            self.api_key = "test-only-key-" + "x" * 40
            self.state_path, self.server_log, self.tunnel_log = [tmp_path / p for p in ("state", "server", "tunnel")]
            events.append("construct")

        def start(self):
            events.append("start")
            return self

        def snapshot(self):
            return dict(state)

        def wait_ready(self, on_update):
            events.append("join")
            on_update(dict(state))
            state.update(phase="ready", ready=True, model_ready=True,
                         public_ready=public, tunnel_connected=public)
            return dict(state)

        def stop(self):
            events.append("stop")
            state.update(phase="stopped", ready=False, server_alive=False, tunnel_alive=False,
                         model_alive=False, manager_alive=False, cleanup_complete=True)
            return dict(state)

    monkeypatch.setattr(startup, "StartupSession", Owner)
    probe = Mock()
    probe.__enter__ = Mock(return_value=probe)
    probe.__exit__ = Mock(return_value=False)
    probe.connect_ex.return_value = 1
    monkeypatch.setattr(socket, "socket", lambda: probe)
    scope = dict(os=os, sys=sys, contextlib=contextlib, io=io,
                 PROJECT_DIR=tmp_path, INSTALL_OK=True, RESIDENCY="resident",
                 DEPENDENCY_CHECK="unused-test-command", json_probe=lambda *a, **k: {},
                 check_resources=lambda *a, **k: {}, StatusCard=Mock(), elapsed_text=str)
    monkeypatch.setenv("H3_MODE", "bf16")
    execute_program("start", scope, START_GPU_API=True, OPEN_PUBLIC_API=public)
    assert events == ["construct", "start"]
    assert scope["PUBLIC_API_URL"] is None
    execute_program("wait", scope)
    assert events == ["construct", "start", "join"]
    assert bool(scope["PUBLIC_API_URL"]) == public
    execute_program("wait", scope)
    assert events.count("start") == 1
    execute_program("stop", scope, STOP_SERVER=True)
    assert events[-1] == "stop" and scope["PUBLIC_API_URL"] is None


@pytest.mark.parametrize("program, owner_name", [("install", "existing_host"), ("start", "previous_host")])
@pytest.mark.parametrize("module", ["model_alive", "manager_alive"])
def test_guard_includes_new_independent_processes(program, owner_name, module):
    state = dict(phase="stopped",cleanup_complete=True,server_alive=False,tunnel_alive=False)
    state[module] = True
    with pytest.raises(RuntimeError):
        execute_owner_guard(program, owner_name, state)


def test_join_failure_does_not_destroy_control_plane(tmp_path):
    host = Mock()
    host.state_path = tmp_path / "status.json"
    host.wait_ready.side_effect = RuntimeError("model failed")
    scope = dict(startup_host=host, StatusCard=Mock(), elapsed_text=str, json=json)
    with pytest.raises(RuntimeError, match="model failed"):
        execute_program("wait", scope)
    host.stop.assert_not_called()


def test_readiness_shows_public_api_pass_before_model_join(tmp_path):
    """Public API reachability can pass while generation is still unavailable."""
    callback = next(node for node in ast.walk(program_tree("wait"))
                    if isinstance(node, ast.FunctionDef) and node.name == "show_readiness")
    card = Mock()
    scope = dict(join_card=card, elapsed_text=str,
                 startup_host=SimpleNamespace(state_path=tmp_path / "state",
                                              server_log=tmp_path / "model.log",
                                              tunnel_log=tmp_path / "tunnel.log"))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[callback], type_ignores=[])),
                 "notebook-readiness-display", "exec"), scope)
    state = dict(phase="preparing", elapsed_seconds=5, api_ready=True,
                 public_api_ready=True, model_ready=False, public_ready=False,
                 ready=False, public_requested=True, tunnel_connected=True)
    original = dict(state)

    scope["show_readiness"](state)

    fields = dict(card.update.call_args.kwargs["fields"])
    assert fields["API后台"] == "可用"
    assert fields["公网探测"] == "通过"
    assert fields["模型容器"] == "未就绪"
    assert state == original  # Displaying a ready branch must not open the join gate.


def execute_current_stop(scope, *, release=True):
    """Exercise the builder's stop program before publishing a new notebook."""
    builder = ast.parse((ROOT / 'tools/build_notebook.py').read_text(encoding='utf-8'))
    source, = (node.value for node in ast.walk(builder)
               if isinstance(node, ast.Constant) and isinstance(node.value, str)
               and node.value.lstrip().startswith('#@title 关闭选项'))
    tree = ast.parse(textwrap.dedent(source))
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id == 'STOP_SERVER':
                node.value = ast.Constant(True)
            elif node.targets[0].id == 'RELEASE_COLAB_RUNTIME':
                node.value = ast.Constant(release)
    exec(compile(ast.fix_missing_locations(tree), 'builder-stop-program', 'exec'), scope)


@pytest.mark.parametrize('changed, expected', [
    ({'model_alive': True}, '仍有进程存活：模型'),
    ({'model_alive': None}, '进程退出状态尚未确认：模型'),
    ({'process_observation_errors': {'model': 'AccessDenied'}}, '进程退出状态尚未确认'),
    ({}, '进程已退出，但最终清理审计尚未确认'),
    ({'cleanup_complete': True, 'phase': 'preparing'}, '最终清理审计尚未确认'),
    ({'cleanup_complete': 'true'}, '最终清理审计尚未确认'),
])
def test_stop_reports_distinct_unconfirmed_states_and_never_releases(changed, expected, monkeypatch, capsys):
    state = dict(phase='failed', cleanup_complete=False, manager_alive=False, server_alive=False,
                 model_alive=False, tunnel_alive=False, error='Manager final audit missing',
                 process_observation_errors={}, cleanup_errors=[])
    state.update(changed)
    runtime = SimpleNamespace(unassign=Mock())
    google, colab = ModuleType('google'), ModuleType('google.colab')
    google.colab, colab.runtime = colab, runtime
    monkeypatch.setitem(sys.modules, 'google', google)
    monkeypatch.setitem(sys.modules, 'google.colab', colab)
    host = SimpleNamespace(stop=Mock(return_value=dict(state)))
    scope = dict(startup_host=host, StatusCard=Mock(), json=json, PUBLIC_API_URL='old-public-address')
    with pytest.raises(RuntimeError, match=expected) as raised:
        execute_current_stop(scope)
    assert state['phase'] in str(raised.value) and state['error'] in str(raised.value)
    assert scope['STOP_RECEIPT']['phase'] == state['phase']
    assert scope['STOP_RECEIPT']['error'] == state['error']
    assert '停止回执：' in capsys.readouterr().out
    assert scope['PUBLIC_API_URL'] == 'old-public-address'
    runtime.unassign.assert_not_called()
    scope['StatusCard'].assert_not_called()


def test_stop_redacts_failure_details_without_rewriting_actual_receipt(capsys):
    public_key, private_key = 'public-stop-secret-0123456789', 'private-stop-secret-0123456789'
    state = dict(phase='failed', cleanup_complete=False, manager_alive=False, server_alive=False,
                 model_alive=False, tunnel_alive=False, error='audit ' + public_key + ' ' + private_key)
    host = SimpleNamespace(stop=Mock(return_value=state), api_key=public_key, _control_key=private_key)
    scope = dict(startup_host=host, StatusCard=Mock(), json=json)
    with pytest.raises(RuntimeError) as raised:
        execute_current_stop(scope, release=False)
    printed = capsys.readouterr().out
    for secret in (public_key, private_key):
        assert secret not in printed and secret not in str(raised.value)
    assert scope['STOP_RECEIPT']['error'] == state['error']


def test_release_only_after_explicit_final_cleanup_receipt(monkeypatch, capsys):
    state = dict(phase='stopped', cleanup_complete=True, manager_alive=False, server_alive=False,
                 model_alive=False, tunnel_alive=False, error=None, process_observation_errors={})
    calls = []
    runtime = SimpleNamespace(unassign=lambda: calls.append('release'))
    google, colab = ModuleType('google'), ModuleType('google.colab')
    google.colab, colab.runtime = colab, runtime
    monkeypatch.setitem(sys.modules, 'google', google)
    monkeypatch.setitem(sys.modules, 'google.colab', colab)

    def stop():
        calls.append('stop')
        return dict(state)

    scope = dict(startup_host=SimpleNamespace(stop=stop), StatusCard=Mock(), json=json, PUBLIC_API_URL='old')
    execute_current_stop(scope)
    assert calls == ['stop', 'release']
    assert scope['PUBLIC_API_URL'] is None
    assert scope['STOP_RECEIPT']['phase'] == 'stopped'
    assert '停止回执：' in capsys.readouterr().out
