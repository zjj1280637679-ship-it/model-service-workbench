"""Real local process-tree ownership and clean-terminal restart; no GPU/network."""
import json
import os
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import Mock

import psutil
import pytest

import notebook_console as console
import notebook_ui as ui


TREE_SCRIPT = '''import json, os, pathlib, subprocess, sys, time
import psutil
role, directory = sys.argv[1], pathlib.Path(sys.argv[2])
flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
if role != "grandchild":
    child_role = "child" if role == "parent" else "grandchild"
    subprocess.Popen([sys.executable, "-u", __file__, child_role, str(directory)],
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, creationflags=flags)
record = {"pid": os.getpid(), "created_at": psutil.Process().create_time()}
(directory / (role + ".json")).write_text(json.dumps(record), encoding="utf-8")
if role == "parent":
    deadline = time.monotonic() + 30
    while not (directory / "parent-may-exit").exists() and time.monotonic() < deadline:
        time.sleep(.02)
else:
    time.sleep(30)
'''


def same_live_process(identity):
    try:
        process = psutil.Process(identity['pid'])
        if (abs(process.create_time() - identity['created_at']) < .001
                and process.is_running() and process.status() != psutil.STATUS_ZOMBIE):
            return process
    except psutil.NoSuchProcess:
        pass
    return None


def await_tree(directory, controller, process):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        controller.process_progress(process)
        try:
            identities = [json.loads((directory / (role + '.json')).read_text(encoding='utf-8'))
                          for role in ('parent', 'child', 'grandchild')]
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(.02)
            continue
        if all(same_live_process(identity) for identity in identities):
            controller.process_progress(process)
            return identities
    raise AssertionError('Local fixture process tree did not become observable')


def cleanup_fixture(process, identities):
    """Only reap our exact process generations, including on a failed assertion."""
    for identity in identities:
        child = same_live_process(identity)
        if child:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
    if process.poll() is None:
        process.kill()
    process.wait(timeout=5)
    for identity in identities:
        child = same_live_process(identity)
        if child:
            try:
                child.wait(timeout=5)
            except psutil.NoSuchProcess:
                pass


@pytest.mark.parametrize('exit_path', ['explicit_stop', 'parent_exits_first'])
def test_real_console_process_tree_is_reaped_and_audited(tmp_path, exit_path):
    script = tmp_path / 'tree_fixture.py'
    script.write_text(TREE_SCRIPT, encoding='utf-8')
    controller = console.ConsoleController(tmp_path / 'controller', {})
    process = subprocess.Popen([sys.executable, '-u', str(script), 'parent', str(tmp_path)],
                               stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL,
                               creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    identities = []
    try:
        controller.process_started(process)
        identities = await_tree(tmp_path, controller, process)
        owner, _ = controller.process_owners[id(process)]
        observed = owner.observe()
        assert {item['pid'] for item in identities} <= set(observed['pids'])
        assert not observed['cleanup_complete']
        if exit_path == 'explicit_stop':
            controller._terminate_client_process(process)
        else:
            # An exited wrapper alone must not release live descendants.
            (tmp_path / 'parent-may-exit').write_text('exit', encoding='utf-8')
            process.wait(timeout=5)
            assert all(same_live_process(item) for item in identities[1:])
        controller.process_finished(process)
        assert process.poll() is not None
        assert all(same_live_process(item) is None for item in identities)
        assert id(process) not in controller.process_owners
        assert not controller.processes
        events = [json.loads(line) for line in controller.audit_path.read_text(encoding='utf-8').splitlines()]
        cleanups = [event for event in events if event['type'] == 'client.process_cleanup']
        assert cleanups and all(event['cleanup_complete'] for event in cleanups)
        assert {item['pid'] for item in identities} <= {item['pid'] for item in cleanups[-1]['identities']}
        assert not cleanups[-1]['pids'] and not cleanups[-1]['errors']
    finally:
        # Include late fixture records if setup failed before returning identities.
        for role in ('parent', 'child', 'grandchild'):
            try:
                identity = json.loads((tmp_path / (role + '.json')).read_text(encoding='utf-8'))
                if identity not in identities:
                    identities.append(identity)
            except (FileNotFoundError, json.JSONDecodeError):
                pass
        try:
            # Capture/reap any still-attached descendants even if fixture setup
            # failed before their PID files were read.
            controller._terminate_client_process(process)
        finally:
            cleanup_fixture(process, identities)
            controller.process_finished(process)
            controller.close()
    assert controller.closed
    assert ui._process_observer is not controller and ui._dashboard is not controller


def stopped_host(**changes):
    state = {'phase': 'stopped', 'cleanup_complete': True, 'observation_stale': True,
             'manager_alive': False, 'server_alive': False, 'model_alive': False, 'tunnel_alive': False,
             'process_observation_errors': [], **changes}
    return SimpleNamespace(snapshot=Mock(return_value=state), public=True, staged=False,
                           api_key='old-public-test-token', _control_key='old-private-test-token')


@pytest.mark.parametrize('phase', ['stopped', 'failed'])
def test_confirmed_clean_terminal_restarts_despite_stale_liveness(tmp_path, monkeypatch, phase):
    old_host = stopped_host(phase=phase)
    controller = console.ConsoleController(tmp_path, {}, scope={'startup_host': old_host})
    new_host = SimpleNamespace(api_key='new-host-token', instance_id='new-host-instance')
    request_id = 'clean-terminal-restart'
    # _bootstrap is private: submit() normally registers the request before it runs.
    controller.operations[request_id] = {'action': 'bootstrap', 'request_id': request_id}
    executed = []

    def execute(stage, parameters=None):
        executed.append(stage)
        if stage == 'preflight':
            assert 'startup_host' not in controller.scope
            controller.scope['PREFLIGHT_OK'] = True
        elif stage == 'install':
            controller.scope['INSTALL_OK'] = True
        elif stage == 'start':
            controller.scope['startup_host'] = new_host

    awaited = Mock()
    monkeypatch.setattr(controller, '_execute', execute)
    monkeypatch.setattr(controller, '_await_bootstrap', awaited)
    try:
        controller._bootstrap({'public': True, 'request_id': request_id})
        assert [stage for stage in executed if stage != 'advanced'] == ['preflight', 'install', 'start']
        awaited.assert_called_once_with(new_host, True)
        assert controller.scope['startup_host'] is new_host
        assert controller._bootstrap_hosts[request_id] == (new_host, new_host.instance_id)
        assert len(controller.scope['BOOTSTRAP_API_KEY']) >= 32
    finally:
        controller.close()


@pytest.mark.parametrize('changes', [
    {'cleanup_complete': False},
    {'server_alive': None},
    {'process_observation_errors': ['unconfirmed descendant']},
    {'phase': 'preparing'},
])
def test_unconfirmed_terminal_cleanup_cannot_start_a_new_owner(tmp_path, monkeypatch, changes):
    old_host = stopped_host(**changes)
    controller = console.ConsoleController(tmp_path, {}, scope={'startup_host': old_host})
    execute, awaited = Mock(), Mock()
    monkeypatch.setattr(controller, '_execute', execute)
    monkeypatch.setattr(controller, '_await_bootstrap', awaited)
    try:
        with pytest.raises(RuntimeError, match='尚未确认清理完成'):
            controller._bootstrap({'public': True})
        execute.assert_not_called()
        awaited.assert_not_called()
        assert controller.scope['startup_host'] is old_host
        assert 'BOOTSTRAP_API_KEY' not in controller.scope
    finally:
        controller.close()
