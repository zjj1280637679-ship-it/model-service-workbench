"""Launcher tests; process/job supervision now belongs to global_manager tests."""
import json
import os
from pathlib import Path
import sqlite3
import sys
import threading
import pytest
import startup
from control_protocol import ControlError


class Process:
    pid = 99123
    def __init__(self): self.code = None
    def poll(self): return self.code
    def terminate(self): self.code = 0
    def kill(self): self.code = -9
    def wait(self, timeout=None): self.code = 0; return 0


class Owner:
    def __init__(self, process, *, errors=None, pids=None):
        self.process, self.errors, self.pids = process, errors or [], pids

    def observe(self):
        pids = self.pids if self.pids is not None else ([self.process.pid] if self.process.poll() is None else [])
        return {"alive": bool(pids) or bool(self.errors), "cleanup_complete": not pids and not self.errors,
                "pids": pids, "errors": self.errors, "identities": [], "observed_at": 20}

    def terminate_all(self, timeout):
        self.process.terminate()
        return self.observe()


def session(tmp_path):
    return startup.StartupSession(tmp_path, env={}, poll_interval=.01)


def active(tmp_path):
    s = session(tmp_path)
    s.manager = Process()
    s.manager_url = "http://127.0.0.1:1"
    s._started_at = s._clock()
    s._last.update(phase="preparing", cleanup_complete=False)
    return s


def test_initial_state_does_not_start_resources(tmp_path):
    s = session(tmp_path)
    assert s.snapshot()["cleanup_complete"] and not s.snapshot()["manager_alive"]
    assert s.api_key != s._control_key
    assert not s.run_directory.exists()


@pytest.mark.parametrize('start_model', [True, False])
def test_start_only_waits_for_control_not_model(tmp_path, monkeypatch, start_model):
    # Omitting the option must preserve the normal model autostart default.
    options = {} if start_model else {'start_model': False}
    s = startup.StartupSession(tmp_path, env={}, poll_interval=.01, **options)
    commands = []
    monkeypatch.setattr(startup, "assert_port_available", lambda x: None)
    def spawn(command, **kw):
        commands.append((command, kw))
        return Process()
    monkeypatch.setattr(startup.subprocess, "Popen", spawn)
    monkeypatch.setattr(startup, "ProcessOwner", Owner)
    monkeypatch.setattr(startup, "rpc_call", lambda *a, **k: dict(s._last, instance_id=s.instance_id, model_ready=False, api_ready=True))
    s.start()
    assert not s.snapshot()["model_ready"] and s.snapshot()["api_ready"]
    assert "global_manager.py" in commands[0][0][2]
    assert ("--no-model-autostart" in commands[0][0]) is (not start_model)
    assert s.api_key not in " ".join(commands[0][0])
    with pytest.raises(RuntimeError): s.start()
    s.manager.terminate()
    s._output.close()


def test_stale_ready_is_revoked_when_control_unavailable(tmp_path, monkeypatch):
    s = active(tmp_path)
    s._last.update(ready=True, model_ready=True, public_ready=True, cleanup_complete=True)
    def unavailable(*a, **k): raise ControlError("offline", 503)
    monkeypatch.setattr(startup, "rpc_call", unavailable)
    result = s.snapshot()
    assert not result["ready"] and not result["cleanup_complete"] and result["observation_stale"]


def test_rpc_loss_marks_download_progress_historical_without_mutating_history(tmp_path, monkeypatch):
    s = active(tmp_path)
    s._last["downloads"] = {"schema_version": 1, "model_instance_id": "m" * 32,
                            "items": {"model": {"phase": "loaded", "percent": 100}},
                            "model_ready": True, "observation_stale": False, "lifecycle": "ready"}
    def unavailable(*args, **kwargs):
        raise ControlError("offline", 503)
    monkeypatch.setattr(startup, "rpc_call", unavailable)
    state = s.snapshot()
    assert state["downloads"]["items"]["model"]["percent"] == 100
    assert state["downloads"]["observation_stale"] and state["downloads"]["model_ready"] is False
    assert state["downloads"]["lifecycle"] == "unavailable"
    assert s._last["downloads"]["observation_stale"] is False
    assert state["last_manager_observation"]["downloads"]["model_ready"] is True


def test_wrong_manager_identity_cannot_publish_ready(tmp_path, monkeypatch):
    s = active(tmp_path)
    monkeypatch.setattr(startup, "rpc_call", lambda *a, **k: {"instance_id": "old", "ready": True})
    assert not s.snapshot()["ready"]


def test_failed_model_wait_keeps_manager_queryable(tmp_path, monkeypatch):
    s = active(tmp_path)
    monkeypatch.setattr(startup, "rpc_call", lambda *a, **k: dict(s._last, phase="failed", error="model load failed", api_ready=True))
    with pytest.raises(RuntimeError): s.wait_ready()
    assert s.manager.poll() is None
    assert s.snapshot()["api_ready"]


def test_stop_waits_for_confirmed_owned_child_cleanup(tmp_path, monkeypatch):
    s = active(tmp_path)
    calls = []
    def rpc(url, key, op, payload=None, **kw):
        calls.append(op)
        if op == "status":
            s.manager.code = 0
        return dict(s._last, cleanup_complete=True, phase="stopped", server_alive=False, model_alive=False, tunnel_alive=False)
    monkeypatch.setattr(startup, "rpc_call", rpc)
    result = s.stop()
    assert calls == ["control", "status"]
    assert result["cleanup_complete"] and not result["manager_alive"]
    assert s.manager.poll() is not None


def test_unacknowledged_dead_manager_does_not_imply_cleanup(tmp_path):
    s = active(tmp_path)
    s.manager.code = -9
    result = s.stop()
    assert result["phase"] == "failed" and not result["cleanup_complete"]


def test_prelaunch_empty_snapshot_does_not_certify_post_crash_cleanup(tmp_path):
    s = active(tmp_path)
    s._last.update(phase="new", cleanup_complete=True)
    s.manager.code = -9
    assert not s.snapshot()["cleanup_complete"]
    assert not s.stop()["cleanup_complete"]


def test_old_persisted_checkpoint_cannot_certify_new_instance(tmp_path):
    s = active(tmp_path)
    s.manager.code = -9
    s.run_directory.mkdir(parents=True)
    s.state_path.write_text(json.dumps({"instance_id": "old", "cleanup_complete": True}))
    assert not s.stop()["cleanup_complete"]


def test_public_private_key_must_not_be_reused(tmp_path):
    key = "a" * 40
    with pytest.raises(ValueError):
        startup.StartupSession(tmp_path, env={"H3_API_KEY": key, "H3_CONTROL_KEY": key})


def ready_observation(s):
    s._last.update(phase="ready", observed_at=10, manager_pid=s.manager.pid,
                   manager_alive=True, server_alive=True, model_alive=True, tunnel_alive=True,
                   ready=True, api_ready=True, model_ready=True, public_ready=True, public_api_ready=True,
                   tunnel_connected=True, supervisor_alive=True, cleanup_complete=False,
                   modules={"manager": {"alive": True, "supervisor_alive": True},
                            "api": {"alive": True, "ready": True},
                            "model": {"alive": True, "ready": True}},
                   owned_processes={"api": {"alive": True, "cleanup_complete": False,
                                             "identities": [{"pid": 123, "created_at": 1}]}})


def assert_readiness_revoked(state):
    for name in ("ready", "api_ready", "model_ready", "public_ready", "public_api_ready", "tunnel_connected"):
        assert state[name] is False
    assert all(module.get("ready") is False for module in state["modules"].values())


def test_normal_stop_updates_nested_projection_and_preserves_checkpoint(tmp_path, monkeypatch):
    s = active(tmp_path)
    ready_observation(s)
    s._manager_owner = Owner(s.manager)
    def rpc(url, key, operation, *args, **kwargs):
        if operation == "status":
            s.manager.code = 0
        return dict(s._last, phase="stopped", cleanup_complete=True,
                    server_alive=False, model_alive=False, tunnel_alive=False)
    monkeypatch.setattr(startup, "rpc_call", rpc)
    result = s.stop()
    assert result["cleanup_complete"] is True
    assert result["manager_alive"] is result["modules"]["manager"]["alive"] is False
    assert result["supervisor_alive"] is result["modules"]["manager"]["supervisor_alive"] is False
    assert all(result[key] is False for key in ("server_alive", "model_alive", "tunnel_alive"))
    assert result["last_manager_observation"]["modules"]["manager"]["alive"] is True
    assert result["manager_observed_at"] == 10 and result["observed_at"] > 10
    assert result["owner_observed_at"] == 20
    assert result["observation_age_seconds"] == round(result["observed_at"] - 20, 3)
    assert result["manager_observation_age_seconds"] == round(result["observed_at"] - 10, 3)
    assert_readiness_revoked(result)
    assert s.snapshot()["modules"]["manager"]["alive"] is False
    assert "last_manager_observation" not in s._last


def test_unexpected_exit_revokes_all_projections_without_certifying_job_cleanup(tmp_path):
    s = active(tmp_path)
    ready_observation(s)
    s._manager_owner = Owner(s.manager)
    s.manager.code = -9
    result = s.snapshot()
    assert result["manager_alive"] is result["modules"]["manager"]["alive"] is False
    assert result["model_alive"] is result["modules"]["model"]["alive"] is False
    assert result["phase"] == "failed" and result["cleanup_complete"] is False
    assert result["last_manager_observation"]["ready"] is True
    assert_readiness_revoked(result)


def test_unresolved_subtree_stays_unknown_even_with_clean_manager_checkpoint(tmp_path, monkeypatch):
    s = active(tmp_path)
    ready_observation(s)
    s._last.update(phase="stopped", cleanup_complete=True,
                   server_alive=False, model_alive=False, tunnel_alive=False)
    s.manager.code = -9
    s._manager_owner = Owner(s.manager, errors=["123.subtree_exit_unconfirmed"], pids=[])
    def unavailable(*a, **kw): raise ControlError("offline", 503)
    monkeypatch.setattr(startup, "rpc_call", unavailable)
    result = s.stop()
    assert result["manager_alive"] is result["modules"]["manager"]["alive"] is None
    assert result["server_alive"] is result["modules"]["api"]["alive"] is None
    assert result["owned_processes"]["api"]["cleanup_complete"] is False
    assert result["owned_processes"]["api"]["pids"] is None
    assert result["cleanup_complete"] is False and result["phase"] == "failed"
    assert result["process_observation_errors"]["session_owner"] == ["123.subtree_exit_unconfirmed"]
    assert not s._cleanup_confirmed(dict(result, cleanup_complete=True, process_observation_errors={}))
    assert_readiness_revoked(result)


def test_rpc_loss_keeps_last_manager_status_explicitly_historical(tmp_path, monkeypatch):
    s = active(tmp_path)
    ready_observation(s)
    s._manager_owner = Owner(s.manager)
    def unavailable(*a, **kw): raise ControlError("offline", 503)
    monkeypatch.setattr(startup, "rpc_call", unavailable)
    result = s.snapshot()
    assert result["manager_alive"] is result["modules"]["manager"]["alive"] is True
    assert result["model_alive"] is result["modules"]["model"]["alive"] is None
    assert result["phase"] == "degraded" and not result["cleanup_complete"]
    assert result["last_manager_observation"]["model_alive"] is True
    assert_readiness_revoked(result)


@pytest.mark.parametrize("clean_checkpoint", [False, True])
def test_manager_exit_with_surviving_descendant_is_not_live_manager(tmp_path, monkeypatch, clean_checkpoint):
    s = active(tmp_path)
    ready_observation(s)
    if clean_checkpoint:
        s._last.update(phase="stopped", cleanup_complete=True,
                       server_alive=False, model_alive=False, tunnel_alive=False)
    s.manager.code = -9
    s._manager_owner = Owner(s.manager, pids=[123])
    monkeypatch.setattr(startup, "rpc_call", lambda *a, **kw: dict(s._last))
    result = s.snapshot()
    assert result["manager_alive"] is result["modules"]["manager"]["alive"] is False
    assert result["manager_process_tree"]["alive"] is True
    assert result["model_alive"] is None and not result["cleanup_complete"]
    assert_readiness_revoked(result)


def test_wait_ready_retries_unknown_liveness_instead_of_treating_it_as_exit(tmp_path, monkeypatch):
    s = active(tmp_path)
    observations = iter([dict(phase="degraded", error="observation unavailable", model_alive=None, ready=False),
                         dict(phase="ready", error=None, model_alive=True, ready=True)])
    monkeypatch.setattr(s, "snapshot", lambda: next(observations))
    monkeypatch.setattr(startup.time, "sleep", lambda _: None)
    assert s.wait_ready()["ready"] is True


def test_stop_retries_lost_ack_and_waits_beyond_clean_child_snapshot(tmp_path, monkeypatch):
    s = active(tmp_path)
    s._manager_owner = Owner(s.manager)
    s.run_directory.mkdir(parents=True)
    controls, statuses = [], []
    clean = dict(s._last, phase="stopped", cleanup_complete=True, shutdown_complete=False,
                 server_alive=False, model_alive=False, tunnel_alive=False)

    def rpc(url, key, operation, payload=None, **kwargs):
        if operation == "control":
            controls.append(payload)
            if len(controls) == 1:
                raise ControlError("ack lost after acceptance", 503)
            return {"status": "accepted"}
        statuses.append(operation)
        if len(statuses) == 1:
            return clean
        if len(statuses) == 2:
            raise ControlError("temporary observation loss", 503)
        s.state_path.write_text(json.dumps(dict(clean, shutdown_complete=True)))
        s.manager.code = 0
        return dict(clean, shutdown_complete=True)

    monkeypatch.setattr(startup, "rpc_call", rpc)
    monkeypatch.setattr(startup.time, "sleep", lambda _: None)
    monkeypatch.setattr(s._manager_owner, "terminate_all", lambda **_: pytest.fail("graceful finalization interrupted"))
    result = s.stop()
    assert len(controls) == 2 and controls[0]["command_id"] == controls[1]["command_id"]
    assert len(statuses) == 3
    assert result["cleanup_complete"] and result["shutdown_complete"]
    assert s.stop()["shutdown_complete"] and s.stop()["cleanup_complete"]


def test_forced_exit_without_final_receipt_cannot_certify_shutdown(tmp_path, monkeypatch):
    s = active(tmp_path)
    s._manager_owner = Owner(s.manager)
    s.run_directory.mkdir(parents=True)
    s._last.update(phase="stopped", cleanup_complete=True, shutdown_complete=False,
                   server_alive=False, model_alive=False, tunnel_alive=False)
    s.state_path.write_text(json.dumps(s._last))
    now = [0.0]
    monkeypatch.setattr(startup.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(startup.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + 31))
    monkeypatch.setattr(startup, "rpc_call", lambda *a, **kw: dict(s._last))
    result = s.stop()
    assert result["manager_process_tree"]["cleanup_complete"]
    assert result["phase"] == "failed" and result["cleanup_complete"] is False
    assert result["shutdown_complete"] is False


def test_late_ui_snapshot_cannot_replace_final_stop_receipt(tmp_path, monkeypatch):
    s = active(tmp_path)
    s._manager_owner = Owner(s.manager)
    s.run_directory.mkdir(parents=True)
    old = dict(s._last, shutdown_complete=False)
    final = dict(old, phase="stopped", shutdown_complete=True, cleanup_complete=True,
                 server_alive=False, model_alive=False, tunnel_alive=False)
    waiting, release = threading.Event(), threading.Event()
    results = []

    def rpc(url, key, operation, payload=None, **kwargs):
        if operation == "control":
            return {"status": "accepted"}
        if threading.current_thread().name == "late-ui-snapshot":
            waiting.set()
            assert release.wait(3)
            return old
        s.state_path.write_text(json.dumps(final))
        s.manager.code = 0
        return final

    monkeypatch.setattr(startup, "rpc_call", rpc)
    poller = threading.Thread(target=lambda: results.append(s.snapshot()), name="late-ui-snapshot")
    poller.start()
    try:
        assert waiting.wait(3)
        assert s.stop()["shutdown_complete"]
    finally:
        release.set()
        poller.join(timeout=3)
    assert results[0]["cleanup_complete"] and results[0]["shutdown_complete"]
    assert s.snapshot()["cleanup_complete"] and s._last["shutdown_complete"]


def test_new_stop_attempt_uses_new_command_after_previous_failure(tmp_path, monkeypatch):
    s = active(tmp_path)
    s._manager_owner = Owner(s.manager)
    s._last.update(shutdown_complete=False, phase="failed")
    now, command_ids = [0.0], []
    monkeypatch.setattr(startup.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(startup.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + 31))
    # First cleanup cannot reclaim the owner; the user can repair the condition
    # and issue a second stop without replaying the first failed command record.
    monkeypatch.setattr(s._manager_owner, "terminate_all", lambda **_: s._manager_owner.observe())

    def rpc(url, key, operation, payload=None, **kwargs):
        if operation == "control":
            command_ids.append(payload["command_id"])
            return {"status": "failed" if len(command_ids) == 1 else "accepted"}
        if len(command_ids) == 1:
            return dict(s._last)
        s.manager.code = 0
        return dict(s._last, phase="stopped", shutdown_complete=True, cleanup_complete=True,
                    server_alive=False, model_alive=False, tunnel_alive=False)

    monkeypatch.setattr(startup, "rpc_call", rpc)
    assert not s.stop()["cleanup_complete"]
    assert s.stop()["cleanup_complete"]
    assert len(command_ids) == 2 and command_ids[0] != command_ids[1]


def test_dead_manager_snapshot_recovers_matching_durable_final_receipt(tmp_path):
    s = active(tmp_path)
    s._manager_owner = Owner(s.manager)
    s.run_directory.mkdir(parents=True)
    s._last["shutdown_complete"] = False
    s.state_path.write_text(json.dumps(dict(s._last, phase="stopped", cleanup_complete=True,
        shutdown_complete=True, server_alive=False, model_alive=False, tunnel_alive=False)))
    s.manager.code = 0
    assert s.snapshot()["cleanup_complete"] and s.snapshot()["shutdown_complete"]


def test_real_manager_exits_after_final_audit_without_launcher_signal(tmp_path, monkeypatch):
    # Real control HTTP, SQLite, manager and owned CPU-only processes. The fixture
    # substitutes only the expensive API/model payloads, never lifecycle code.
    project = str(Path(startup.__file__).parent)
    wrapper = "\n".join([
        "import sys,time,uuid",
        f"sys.path.insert(0, {project!r})",
        "import global_manager as gm",
        "def launch_api(self):",
        "    self._spawn('api',[sys.executable,'-c','import time; time.sleep(60)'],self.env)",
        "gm.GlobalManager._launch_api = launch_api",
        "original = gm.GlobalManager._checkpoint",
        "def delayed_final(self):",
        "    if self._shutdown_complete: time.sleep(.5)",
        "    return original(self)",
        "gm.GlobalManager._checkpoint = delayed_final",
        "gm.main()",
    ])
    (tmp_path / "global_manager.py").write_text(wrapper, encoding="utf-8")
    s = startup.StartupSession(tmp_path, port=startup._free_port(set()), env=dict(os.environ),
                               start_model=False, probe_timeout=.2, poll_interval=.05)
    try:
        s.start()
        assert s.manager.poll() is None
        monkeypatch.setattr(s._manager_owner, "terminate_all", lambda **_: pytest.fail("normal stop sent a process-tree signal"))
        result = s.stop()
        assert result["shutdown_complete"] is result["cleanup_complete"] is True
        assert all(result[name] is False for name in ("manager_alive", "server_alive", "model_alive", "tunnel_alive"))
        assert s.manager.poll() == 0
        assert s.stop()["cleanup_complete"] is True
        events = [json.loads(line) for line in s.events_path.read_text().splitlines()]
        kinds = [event["type"] for event in events]
        assert kinds.index("command.succeeded") < kinds.index("manager.shutdown_complete")
        with sqlite3.connect(s.root / "control.sqlite3") as database:
            records = [json.loads(row[0]) for row in database.execute("SELECT record FROM commands")]
        assert len(records) == 1 and records[0]["status"] == "succeeded"
        from global_manager import RootLock
        RootLock(s.root / "owner.lock").close()
    finally:
        # Preserve cleanup even when an assertion detects a regression.
        monkeypatch.undo()
        if s._manager_owner:
            s._manager_owner.terminate_all(timeout=.5)
        if s._output and not s._output.closed:
            s._output.close()
