"""Durable scheduling/ownership tests; no production fake model or CUDA use."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import uuid

import psutil
import pytest

import global_manager as gm
from control_protocol import ControlError


class Process:
    next_pid = 910000

    def __init__(self):
        Process.next_pid += 1
        self.pid = Process.next_pid
        self.alive = True
        self.poll_error = False
        self.stubborn = False

    def poll(self):
        if self.poll_error:
            raise OSError("cannot inspect")
        return None if self.alive else 0

    def terminate(self):
        if not self.stubborn:
            self.alive = False

    def wait(self, timeout=None):
        if self.alive:
            raise subprocess.TimeoutExpired("fixture", timeout)
        return 0

    def kill(self):
        if self.stubborn:
            raise OSError("cannot terminate")
        self.alive = False


class FixtureOwner:
    def __init__(self, process): self.process = process

    def observe(self):
        try:
            alive, errors = self.process.poll() is None, []
        except OSError:
            alive, errors = True, ["OSError"]
        return {"alive": alive, "cleanup_complete": not alive, "pids": [self.process.pid] if alive else [],
                "identities": [], "errors": errors}

    def request_stop(self):
        try: self.process.terminate()
        except OSError: pass
        return self.observe()

    def wait_stopped(self, timeout=15):
        try:
            if self.process.poll() is None: self.process.wait(timeout=timeout)
        except (OSError, subprocess.SubprocessError):
            try: self.process.kill()
            except OSError: pass
        return self.observe()

    def terminate_all(self, timeout=15):
        self.request_stop()
        return self.wait_stopped(timeout)


class Rig:
    def __init__(self, directory, **options):
        self.calls = []
        self.model_ready = False
        self.model_phase = "loading"
        self.active = None
        self.inflight = {}
        self.results = []
        self.capacity = 1
        self.ack_error = False
        self.result = None
        self.executions = []
        self.execute_error = False
        self.status_error = False
        self.terminated = []
        self.api_ready = True
        self.fail_spawn = None
        self.manager = gm.GlobalManager(directory, root=directory / "outputs", instance_id=uuid.uuid4().hex,
            port=19201, api_port=19202, model_port=19203, env={"H3_API_KEY": "a"*40, "H3_CONTROL_KEY": "b"*40},
            popen=self.spawn, rpc=self.rpc, probe=self.probe, owner_factory=FixtureOwner, **options)

    def spawn(self, command, **kwargs):
        name = "model" if any("model_container.py" in part for part in command) else "tunnel" if any("tunnel.py" in part for part in command) else "api"
        self.calls.append(("spawn", name))
        if self.fail_spawn == name:
            raise OSError("spawn failed")
        return Process()

    def probe(self, url, instance, timeout):
        self.calls.append(("probe", url))
        return self.api_ready

    def rpc(self, url, key, operation, payload, timeout):
        self.calls.append(("rpc", operation))
        if operation == "status":
            if self.status_error:
                raise TimeoutError()
            return {"instance_id": self.manager.model_instance_id, "phase": self.model_phase,
                    "observed_at": time.time(),
                    "ready": self.model_ready, "accepting": len(self.inflight) < self.capacity, "worker_alive": self.model_phase != "failed",
                    "attempts": [{**p, "status": "gpu_wait"} for p in self.inflight.values()],
                    "results": self.results.copy(),
                    "last_result": self.result, "active_attempt_id": self.active["attempt_id"] if self.active else None,
                    "active_job_id": self.active["job_id"] if self.active else None,
                    "active_phase": {"phase": "fixture", "elapsed_seconds": 1} if self.active else None}
        if operation == "execute":
            if len(self.inflight) >= self.capacity:
                raise ControlError(json.dumps({"guarantees_not_accepted": True, "reason": "resource_budget_exhausted"}), 429)
            self.executions.append(payload.copy())
            self.inflight[payload["attempt_id"]] = payload.copy()
            self.active = next(iter(self.inflight.values()))
            if self.execute_error:
                if self.execute_error == "unreconcilable": self.status_error = True
                raise TimeoutError("ack lost after execution accepted")
            return {"accepted": True, **{k: payload[k] for k in ("job_id", "attempt_id", "instance_id")}}
        if operation == "ack_results":
            if self.ack_error: raise TimeoutError("ack lost")
            self.results = [r for r in self.results if r["attempt_id"] not in payload["attempt_ids"]]
            return {"acknowledged": payload["attempt_ids"]}
        if operation == "cancel":
            return {"cancel_requested": True}
        raise AssertionError(operation)

    def start(self):
        self.manager.start(background=False)
        return self

    def ready(self):
        self.model_ready, self.model_phase = True, "ready"
        self.manager.refresh()

    def submit(self, idem=None, fingerprint="same"):
        job_id = uuid.uuid4().hex
        path = self.manager.root / job_id / "inputs" / "reference.png"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"input")
        payload = {"job_id": job_id, "parameters": {"references": [{"kind": "image", "path": str(path)}],
                   "prompt": "A test", "seconds": 5, "width": 512, "height": 288, "steps": 8, "seed": 42},
                   "idempotency_key": idem, "fingerprint": fingerprint}
        return self.manager.submit(payload)

    def complete(self, status="succeeded", *, target_attempt=None, **updates):
        active = self.inflight[target_attempt] if target_attempt else self.active
        path = self.manager.root / active["job_id"] / "result.mp4"
        content = b"validated fixture (not a production video)"
        path.write_bytes(content)
        self.result = {"instance_id": self.manager.model_instance_id, "job_id": active["job_id"], "attempt_id": active["attempt_id"],
                       "status": status, "error": None, "metrics": {"fixture": True}, "cleanup_complete": True,
                       "artifact": {"path": str(path), "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(),
                                    "media": {"full_decode_passed": True}}}
        self.result.update(updates)
        self.results.append(self.result)
        self.inflight.pop(active["attempt_id"])
        self.active = next(iter(self.inflight.values()), None)


@pytest.fixture
def rig(tmp_path):
    fixture = Rig(tmp_path).start()
    try:
        yield fixture
    finally:
        for child in fixture.manager.children.values():
            if child:
                child.stubborn = child.poll_error = False
        fixture.manager.close()


def job(rig, job_id):
    return rig.manager.dispatch("job", {"job_id": job_id})


def test_archive_receipt_is_committed_only_for_matching_successful_artifact(rig):
    accepted = rig.submit()
    job_id = accepted["job_id"]
    payload = {"job_id": job_id, "file_id": "private_Drive-id", "sha256": "a" * 64, "bytes": 42}
    with pytest.raises(ControlError) as queued:
        rig.manager.dispatch("archive", payload)
    assert queued.value.status == 409

    rig.ready()
    rig.complete()
    rig.manager.refresh()
    artifact = job(rig, job_id)["artifact"]
    assert "archive" not in job(rig, job_id)
    with pytest.raises(ControlError) as mismatch:
        rig.manager.dispatch("archive", {**payload, "sha256": "0" * 64, "bytes": artifact["bytes"]})
    assert mismatch.value.status == 409

    payload.update(sha256=artifact["sha256"], bytes=artifact["bytes"])
    with pytest.raises(ControlError) as invalid_id:
        rig.manager.dispatch("archive", {**payload, "file_id": "https://other.invalid/video"})
    assert invalid_id.value.status == 422
    with pytest.raises(ControlError) as different_instance:
        rig.manager.dispatch("archive", {**payload, "expected_instance_id": "0" * 32})
    assert different_instance.value.status == 409

    first = rig.manager.dispatch("archive", payload)
    assert first["archive"] == {
        "provider": "google_drive", "file_id": payload["file_id"],
        "sha256": payload["sha256"], "bytes": payload["bytes"],
    }
    assert first["artifact"] == artifact
    assert rig.manager.dispatch("archive", payload) == job(rig, job_id) == first
    with pytest.raises(ControlError) as replacement:
        rig.manager.dispatch("archive", {**payload, "file_id": "different-id"})
    assert replacement.value.status == 409
    events = rig.manager.dispatch("events", {})["events"]
    assert len([entry for entry in events if entry["type"] == "job.archived"]) == 1


def progress_fixture(manager, identity=None):
    return {"schema_version": 1, "model_instance_id": identity or manager.model_instance_id,
            "observed_at": time.time(), "observation_stale": False,
            "items": {"model": {"phase": "downloading", "percent": 40, "bytes_completed": 40, "bytes_total": 100}}}


def test_progress_projects_current_model_without_granting_readiness(rig):
    rig.manager.refresh()
    original = progress_fixture(rig.manager)
    rig.manager._model["downloads"] = original
    state = rig.manager.snapshot()
    assert state["downloads"]["items"]["model"]["percent"] == 40
    assert state["downloads"]["model_ready"] is False
    assert state["downloads"]["observation_stale"] is False
    assert state["modules"]["model"]["downloads"] == state["downloads"]
    assert "lifecycle" not in original  # Projection must not mutate the owned model snapshot.


def test_progress_from_previous_model_is_discarded(rig):
    rig.manager.refresh()
    rig.manager._model["downloads"] = progress_fixture(rig.manager, uuid.uuid4().hex)
    downloads = rig.manager.snapshot()["downloads"]
    assert downloads["model_instance_id"] == rig.manager.model_instance_id
    assert downloads["items"] == {} and downloads["observation_stale"]


def test_progress_is_stale_when_model_control_cannot_be_observed(rig):
    rig.manager.refresh()
    rig.manager._model["downloads"] = progress_fixture(rig.manager)
    rig.status_error = True
    state = rig.manager.refresh()
    assert state["downloads"]["items"]["model"]["percent"] == 40
    assert state["downloads"]["observation_stale"] is True
    assert state["downloads"]["model_ready"] is False


@pytest.mark.parametrize('reason,lifecycle', [('model_stopped', 'stopped'), ('model_worker_failed', 'failed')])
def test_stopped_or_failed_model_keeps_progress_as_history(rig, reason, lifecycle):
    rig.manager.refresh()
    rig.manager._model["downloads"] = progress_fixture(rig.manager)
    rig.manager._stop_model(force=True, reason=reason)
    state = rig.manager.snapshot()["downloads"]
    assert state["items"]["model"]["percent"] == 40
    assert state["observation_stale"] and state["model_ready"] is False
    assert state["lifecycle"] == lifecycle
    assert rig.manager.shutdown()["downloads"]["lifecycle"] == "stopped"


def test_each_model_launch_has_a_new_owned_progress_path(rig):
    first = rig.manager.model_instance_id
    launches = []
    def capture(command, **options):
        launches.append((command, options))
        return rig.spawn(command, **options)
    rig.manager._popen = capture
    rig.manager._stop_model(force=True)
    rig.manager._launch_model()
    command, options = launches[-1]
    env = options["env"]
    assert env["H3_MODEL_INSTANCE_ID"] == rig.manager.model_instance_id != first
    assert env["H3_INSTANCE_ID"] == rig.manager.instance_id
    path = Path(env["H3_DOWNLOAD_PROGRESS_PATH"])
    assert path.parent == rig.manager.run_directory and rig.manager.model_instance_id in path.name
    assert first not in path.name
    assert rig.manager.snapshot()["downloads"]["items"] == {}


def wait_command(manager, command, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = manager.dispatch("command", {"command_id": command["command_id"]})
        if result["status"] in {"succeeded", "failed"}:
            return result
        time.sleep(.005)
    raise AssertionError(result)


def test_independent_branches_launch_before_any_probe(tmp_path):
    fixture = Rig(tmp_path, public=True)
    try:
        fixture.start()
        assert fixture.calls == [("spawn", "api"), ("spawn", "model"), ("spawn", "tunnel")]
    finally:
        fixture.manager.close()


def test_explicit_no_model_autostart_keeps_api_queue_and_manual_launch_available(tmp_path):
    fixture = Rig(tmp_path, public=True, start_model=False)
    try:
        fixture.start()
        assert fixture.calls == [("spawn", "api"), ("spawn", "tunnel")]
        state = fixture.manager.refresh()
        assert state['api_ready'] and not state['model_alive']
        assert not state['model_ready'] and not state['ready']
        assert state['phase'] == 'preparing' and state['error'] is None
        assert state['modules']['model']['phase'] == 'not_started'
        assert state['downloads']['lifecycle'] == 'not_started'
        assert state['queue']['admission_open']
        accepted = fixture.submit()
        for _ in range(4):
            fixture.manager.refresh()
        assert job(fixture, accepted['job_id'])['status'] == 'queued'
        assert job(fixture, accepted['job_id'])['waiting_for'] == 'model_ready'
        assert not fixture.executions
        assert not any(call == ('rpc', 'status') for call in fixture.calls)
        assert fixture.manager.model_instance_id is None
        events = fixture.manager.dispatch('events', {})['events']
        assert any(event['type'] == 'model.autostart_skipped' for event in events)

        # The normal managed control is sufficient; no dependency on tunnel readiness.
        command = fixture.manager.control({'action': 'start_model'})
        assert wait_command(fixture.manager, command)['status'] == 'succeeded'
        assert fixture.calls.count(('spawn', 'model')) == 1
        assert fixture.manager.model_instance_id is not None
        assert not fixture.manager.snapshot()['public_api_ready']
        fixture.ready()
        assert job(fixture, accepted['job_id'])['status'] == 'running'
        assert len(fixture.executions) == 1
        # Once launched, a real exit still fails owned work normally.
        fixture.manager.children['model'].alive = False
        state = fixture.manager.refresh()
        assert state['error'] == 'model_process_exited'
        assert job(fixture, accepted['job_id'])['status'] == 'failed'
    finally:
        fixture.manager.close()


def test_api_query_available_during_model_loading(rig):
    state = rig.manager.refresh()
    assert state["api_ready"] and not state["model_ready"] and not state["ready"]
    accepted = rig.submit()
    rig.manager.refresh()
    assert job(rig, accepted["job_id"])["status"] == "queued"
    assert job(rig, accepted["job_id"])["waiting_for"] == "model_ready"
    assert not rig.executions


@pytest.mark.parametrize("field,value", [("ready", "true"), ("accepting", 1), ("worker_alive", "alive")])
def test_dispatch_requires_explicit_ready_capabilities(rig, field, value):
    accepted = rig.submit()
    model = {"ready": True, "accepting": True, "worker_alive": True, field: value}
    rig.manager._dispatch_next(model)
    assert not rig.executions
    assert job(rig, accepted["job_id"])["status"] == "queued"


def test_wait_reason_is_current_not_a_stale_dispatch_rejection(rig):
    first, second = rig.submit(), rig.submit()
    rig.ready()
    deferred = job(rig, second["job_id"])
    assert deferred["dispatch_reason"] == "resource_budget_exhausted"
    # The authoritative model snapshot reports its capacity on the next pass.
    rig.manager.refresh()
    assert job(rig, second["job_id"])["waiting_for"] == "model_capacity"
    rig.complete()
    rig.manager.refresh()
    running = job(rig, second["job_id"])
    assert running["status"] == "running"
    assert running["waiting_for"] is None and running["dispatch_reason"] is None


def test_durable_submit_and_single_model_execution_lease(rig):
    rig.ready()
    first, second = rig.submit(), rig.submit()
    rig.manager.refresh()
    rig.manager.refresh()
    assert len(rig.executions) == 1
    assert job(rig, first["job_id"])["status"] == "running"
    assert job(rig, second["job_id"])["status"] == "queued"
    rig.complete()
    rig.manager.refresh()
    assert job(rig, first["job_id"])["status"] == "succeeded"
    assert len(rig.executions) == 2
    assert job(rig, second["job_id"])["status"] == "running"


def test_same_idempotency_key_replays_without_second_execution(rig):
    first = rig.submit("once")
    second = rig.submit("once")
    assert first["job_id"] == second["job_id"]
    rig.ready()
    rig.manager.refresh()
    assert len(rig.executions) == 1
    with pytest.raises(ControlError) as failure:
        rig.submit("once", fingerprint="different")
    assert failure.value.status == 409


def test_durable_queue_has_no_task_count_limit(rig):
    accepted = [rig.submit() for _ in range(20)]
    assert [job(rig, item["job_id"])["queue_position"] for item in accepted] == list(range(1, 21))
    assert rig.manager.snapshot()["queue"]["queued"] == 20
    assert rig.manager.snapshot()["queue"]["job_count_limit"] is None


@pytest.mark.parametrize("mutation", ["submit", "cancel", "control"])
def test_old_manager_instance_cannot_mutate(rig, mutation):
    with pytest.raises(ControlError) as failure:
        rig.manager.dispatch(mutation, {"expected_instance_id": uuid.uuid4().hex})
    assert failure.value.status == 409


@pytest.mark.parametrize("field", ["attempt_id", "instance_id", "job_id"])
def test_old_completion_cannot_advance_current_main_event(rig, field):
    rig.ready()
    accepted = rig.submit()
    rig.manager.refresh()
    rig.complete(**{field: uuid.uuid4().hex})
    rig.manager.refresh()
    assert job(rig, accepted["job_id"])["status"] == "running"
    assert len(rig.executions) == 1


@pytest.mark.parametrize("corruption", ["bytes", "sha256", "decode", "path"])
def test_success_requires_artifact_commit_evidence(rig, corruption):
    rig.ready()
    accepted = rig.submit()
    rig.manager.refresh()
    rig.complete()
    artifact = rig.result["artifact"]
    if corruption == "bytes": artifact["bytes"] += 1
    elif corruption == "sha256": artifact["sha256"] = "0"*64
    elif corruption == "decode": artifact["media"]["full_decode_passed"] = False
    else: artifact["path"] = str(rig.manager.root / "another.mp4")
    rig.manager.refresh()
    result = job(rig, accepted["job_id"])
    assert result["status"] == "failed" and result["artifact"] is None


def test_cancel_queued_does_not_execute_and_cleans_inputs(rig):
    accepted = rig.submit()
    result = rig.manager.cancel(accepted["job_id"])
    assert result["status"] == "cancelled"
    assert not (rig.manager.root / accepted["job_id"] / "inputs").exists()
    rig.ready()
    assert not rig.executions


def test_cancel_running_keeps_lease_until_terminal_feedback(rig):
    rig.ready()
    accepted = rig.submit()
    rig.manager.refresh()
    rig.manager.cancel(accepted["job_id"])
    waiting = rig.submit()
    rig.manager.refresh()
    assert job(rig, accepted["job_id"])["status"] == "cancel_requested"
    assert job(rig, waiting["job_id"])["status"] == "queued"
    rig.complete()
    rig.manager.refresh()
    assert job(rig, accepted["job_id"])["status"] == "cancelled"
    assert job(rig, accepted["job_id"])["artifact"] is None
    assert len(rig.executions) == 2


def test_drain_rejects_new_admission_but_finishes_existing_queue(rig):
    accepted = rig.submit()
    command = rig.manager.control({"action": "drain"})
    assert wait_command(rig.manager, command)["status"] == "succeeded"
    with pytest.raises(ControlError): rig.submit()
    rig.ready()
    assert job(rig, accepted["job_id"])["status"] == "running"
    assert not rig.manager.snapshot()["ready"]


def test_model_failure_keeps_api_and_queries_alive(rig):
    accepted = rig.submit()
    rig.model_phase = "failed"
    state = rig.manager.refresh()
    assert state["api_ready"] and state["server_alive"]
    assert not state["model_alive"] and not state["ready"]
    assert job(rig, accepted["job_id"])["status"] == "failed"


def test_ack_timeout_does_not_resubmit_uncertain_execution(rig):
    rig.ready()
    accepted = rig.submit()
    rig.execute_error = "unreconcilable"
    rig.manager.refresh()
    assert len(rig.executions) == 1
    assert job(rig, accepted["job_id"])["status"] == "failed"
    assert not rig.manager.children["model"].alive
    assert rig.manager.children["api"].alive
    rig.manager.refresh()
    assert len(rig.executions) == 1


def test_unreachable_model_is_revoked_and_boundedly_stopped(rig):
    rig.ready()
    rig.status_error = True
    state = rig.manager.refresh()
    assert state["api_ready"] and not state["model_ready"]
    rig.manager.refresh()
    rig.manager.refresh()
    assert not rig.manager.children["model"].alive


def test_no_active_stop_without_explicit_force(rig):
    rig.ready()
    accepted = rig.submit()
    rig.manager.refresh()
    command = rig.manager.control({"action": "stop_model"})
    assert wait_command(rig.manager, command)["status"] == "failed"
    assert rig.manager.children["model"].alive
    assert job(rig, accepted["job_id"])["status"] == "running"
    forced = rig.manager.control({"action": "stop_model", "force": True})
    assert wait_command(rig.manager, forced)["status"] == "succeeded"
    assert job(rig, accepted["job_id"])["status"] == "failed"
    assert rig.manager.children["api"].alive


def test_stop_restart_model_uses_new_identity_and_preserves_jobs(rig):
    rig.ready()
    accepted = rig.submit()
    rig.manager.refresh()
    rig.complete()
    rig.manager.refresh()
    previous = rig.manager.model_instance_id
    stopped = rig.manager.control({"action": "stop_model"})
    assert wait_command(rig.manager, stopped)["status"] == "succeeded"
    restarted = rig.manager.control({"action": "start_model"})
    assert wait_command(rig.manager, restarted)["status"] == "succeeded"
    assert rig.manager.model_instance_id != previous
    assert job(rig, accepted["job_id"])["status"] == "succeeded"


def test_control_idempotency_and_conflicting_reuse(rig):
    first = rig.manager.control({"action": "drain", "command_id": "drain-once"})
    assert wait_command(rig.manager, first)["status"] == "succeeded"
    again = rig.manager.control({"action": "drain", "command_id": "drain-once"})
    assert again["status"] == "succeeded"
    with pytest.raises(ControlError):
        rig.manager.control({"action": "resume", "command_id": "drain-once"})


def test_close_joins_late_command_before_publishing_final_receipt(tmp_path, monkeypatch):
    fixture = Rig(tmp_path).start()
    manager = fixture.manager
    reached, release, closed = threading.Event(), threading.Event(), threading.Event()
    original = manager._checkpoint
    errors = []

    def checkpoint():
        if threading.current_thread().name == "main-control-shutdown":
            with manager._lock:
                row = manager._db.execute("SELECT record FROM commands").fetchone()
            if row and json.loads(row[0])["status"] == "succeeded":
                reached.set()
                assert release.wait(3)
        return original()

    def close():
        try:
            manager.close()
        except BaseException as error:
            errors.append(error)
        finally:
            closed.set()

    monkeypatch.setattr(manager, "_checkpoint", checkpoint)
    try:
        manager.control({"action": "shutdown"})
        assert reached.wait(3)
        closer = threading.Thread(target=close)
        closer.start()
        assert not closed.wait(.05)
        assert not json.loads(manager.state_path.read_text())["shutdown_complete"]
        release.set()
        closer.join(timeout=3)
        assert closed.is_set() and not errors
        final = json.loads(manager.state_path.read_text())
        assert final["cleanup_complete"] and final["shutdown_complete"]
        events = [json.loads(line)["type"] for line in manager.events_path.read_text().splitlines()]
        assert events.index("command.succeeded") < events.index("manager.shutdown_complete")
    finally:
        release.set()
        manager.close()


def test_final_audit_export_failure_cannot_publish_receipt_or_release_root(tmp_path, monkeypatch):
    manager = Rig(tmp_path).start().manager
    original = Path.open

    def open_path(path, *args, **kwargs):
        if path == manager.events_path and manager._shutdown_complete:
            raise OSError("audit storage unavailable")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_path)
    try:
        with pytest.raises(OSError, match="audit storage unavailable"):
            manager.close()
        assert manager.snapshot()["cleanup_complete"]
        assert not manager.snapshot()["shutdown_complete"]
        assert not json.loads(manager.state_path.read_text())["shutdown_complete"]
        with pytest.raises(RuntimeError, match="Another manager"):
            gm.RootLock(manager.root / "owner.lock")
    finally:
        monkeypatch.undo()
        manager.close()


def test_cleanup_failure_retains_ownership_and_blocks_model_restart(rig):
    rig.ready()
    child = rig.manager.children["model"]
    child.stubborn = True
    result = rig.manager.shutdown()
    assert not result["cleanup_complete"] and result["model_alive"]
    assert result["cleanup_errors"]
    assert not result["server_alive"]
    with pytest.raises(ControlError): rig.manager._launch_model()


def test_shutdown_command_failure_remains_queryable_and_retryable(rig):
    child = rig.manager.children["model"]
    child.stubborn = True
    command = rig.manager.control({"action": "shutdown", "command_id": "failed-shutdown"})
    assert wait_command(rig.manager, command)["status"] == "failed"
    assert not rig.manager._exit_requested
    state = rig.manager.dispatch("status")
    assert state["model_alive"] and not state["cleanup_complete"] and not state["shutdown_complete"]
    child.stubborn = False
    retried = rig.manager.control({"action": "shutdown", "command_id": "retry-shutdown"})
    assert wait_command(rig.manager, retried)["status"] == "succeeded"
    assert rig.manager.control({"action": "shutdown", "command_id": "failed-shutdown"})["status"] == "failed"
    assert rig.manager.snapshot()["cleanup_complete"] and rig.manager.snapshot()["error"] is None


def test_close_seals_command_admission_before_final_receipt(tmp_path, monkeypatch):
    manager = Rig(tmp_path).start().manager
    existing = manager.control({"action": "drain", "command_id": "prior-command"})
    assert wait_command(manager, existing)["status"] == "succeeded"
    reached, release = threading.Event(), threading.Event()
    original = manager._probe_pool.shutdown
    errors = []

    def pool_shutdown(*args, **kwargs):
        reached.set()
        assert release.wait(3)
        return original(*args, **kwargs)

    def close():
        try:
            manager.close()
        except BaseException as error:
            errors.append(error)

    monkeypatch.setattr(manager._probe_pool, "shutdown", pool_shutdown)
    closer = threading.Thread(target=close)
    closer.start()
    try:
        assert reached.wait(3)
        assert manager.control({"action": "drain", "command_id": "prior-command"})["status"] == "succeeded"
        with pytest.raises(ControlError, match="Manager is closing"):
            manager.control({"action": "drain", "command_id": "late-command"})
        with pytest.raises(ControlError, match="Command not found"):
            manager.dispatch("command", {"command_id": "late-command"})
    finally:
        release.set()
        closer.join(timeout=3)
        manager.close()
    assert not errors and not closer.is_alive()
    assert not any(thread.is_alive() for thread in manager._control_threads)
    assert json.loads(manager.state_path.read_text())["shutdown_complete"]


def test_accepted_command_registration_is_atomic_with_close(tmp_path, monkeypatch):
    manager = Rig(tmp_path).start().manager
    starting, release, closed = threading.Event(), threading.Event(), threading.Event()
    original = threading.Thread.start
    errors = []

    def start(thread):
        if thread.name == "main-control-drain":
            starting.set()
            assert release.wait(3)
        return original(thread)

    def submit():
        try:
            manager.control({"action": "drain", "command_id": "starting-command"})
        except BaseException as error:
            errors.append(error)

    def close():
        try:
            manager.close()
        except BaseException as error:
            errors.append(error)
        finally:
            closed.set()

    monkeypatch.setattr(threading.Thread, "start", start)
    submitter = threading.Thread(target=submit)
    submitter.start()
    closer = threading.Thread(target=close)
    try:
        assert starting.wait(3)
        closer.start()
        assert not closed.wait(.05)
        assert not manager._control_admission_closed
    finally:
        release.set()
        submitter.join(timeout=3)
        if closer.ident is not None:
            closer.join(timeout=3)
        manager.close()
    assert not errors and closed.is_set()
    events = [json.loads(line) for line in manager.events_path.read_text().splitlines()]
    assert any(event["type"] == "command.succeeded" and event["data"]["command_id"] == "starting-command" for event in events)
    assert not any(thread.is_alive() for thread in manager._control_threads)


def test_os_query_error_cannot_claim_exit(rig):
    rig.manager.children["model"].poll_error = True
    snapshot = rig.manager.snapshot()
    assert snapshot["model_alive"] and not snapshot["cleanup_complete"]
    assert snapshot["process_observation_errors"]["model"] == "OSError"
    assert not snapshot["ready"]


def test_partial_spawn_failure_rolls_back_api(tmp_path):
    fixture = Rig(tmp_path)
    fixture.fail_spawn = "model"
    try:
        with pytest.raises(OSError): fixture.start()
        assert not fixture.manager.children["api"].alive
        assert fixture.manager.snapshot()["cleanup_complete"]
    finally:
        fixture.manager.close()


def test_monitor_start_exception_rolls_back_children(tmp_path, monkeypatch):
    fixture = Rig(tmp_path)
    def fail(_): raise RuntimeError("thread start failed")
    monkeypatch.setattr(threading.Thread, "start", fail)
    try:
        with pytest.raises(RuntimeError): fixture.manager.start()
        assert fixture.manager.snapshot()["cleanup_complete"]
    finally:
        fixture.manager.close()


def test_os_lock_rejects_second_manager(tmp_path):
    first = Rig(tmp_path)
    try:
        with pytest.raises(RuntimeError, match="Another manager"):
            Rig(tmp_path)
    finally:
        first.manager.close()


def test_terminal_jobs_and_monotonic_events_survive_manager_recreation(tmp_path):
    first = Rig(tmp_path).start()
    first.ready()
    accepted = first.submit()
    first.manager.refresh()
    first.complete()
    first.manager.refresh()
    job_id = accepted["job_id"]
    artifact = job(first, job_id)["artifact"]
    archived = first.manager.dispatch("archive", {
        "job_id": job_id, "file_id": "private_Drive-id", "sha256": artifact["sha256"], "bytes": artifact["bytes"],
    })
    before = first.manager.dispatch("events", {})
    first.manager.close()
    second = Rig(tmp_path)
    try:
        restored = job(second, job_id)
        assert restored["status"] == "succeeded"
        assert restored["archive"] == archived["archive"]
        after = second.manager.dispatch("events", {"after": before["next_cursor"]})
        assert after["events"] and all(e["sequence"] > before["next_cursor"] for e in after["events"])
    finally:
        second.manager.close()


def test_recovery_never_replays_uncertain_running_job(tmp_path):
    first = Rig(tmp_path).start()
    first.capacity = 2
    first.ready()
    accepted = [first.submit() for _ in range(5)]
    first.manager.refresh()
    assert len(first.manager._active_jobs()) == 2
    # Simulate durable DB left by a crashed owner without graceful terminalization.
    for proc in first.manager.children.values():
        if proc: proc.alive = False
    first.manager._db.close()
    first.manager._owner.close()
    for output in first.manager._files: output.close()
    first.manager._closed = True
    second = Rig(tmp_path).start()
    try:
        second.ready()
        for item in accepted:
            restored = job(second, item["job_id"])
            assert restored["status"] == "failed"
            assert "not replayed" in restored["error"]
        assert not second.executions
    finally:
        second.manager.close()


def test_previous_live_model_identity_blocks_new_gpu_owner_but_keeps_api(tmp_path):
    fixture = Rig(tmp_path)
    current = psutil.Process()
    with fixture.manager._db:
        fixture.manager._db.execute("INSERT INTO owned_processes VALUES (?,?,?,?,0)",
                                   (uuid.uuid4().hex, "model", current.pid, current.create_time()))
    try:
        fixture.start()
        assert fixture.manager.children["api"].alive
        assert fixture.manager.children["model"] is None
        assert fixture.manager.snapshot()["error"] == "previous_model_exit_unconfirmed"
        stopped = fixture.manager.shutdown()
        assert not stopped["cleanup_complete"]
        assert stopped["historical_ownership"][0]["pid"] == current.pid
    finally:
        # This fixture deliberately used the test runner as the hypothetical old
        # model; release its synthetic registry ownership without killing it.
        with fixture.manager._db:
            fixture.manager._db.execute("UPDATE owned_processes SET exited=1 WHERE pid=?", (current.pid,))
        fixture.manager.close()


def test_checkpoint_process_fields_match_fresh_exit_observation(rig):
    rig.ready()
    rig.manager.children["model"].alive = False
    rig.manager.refresh()
    persisted = json.loads(rig.manager.state_path.read_text())
    events = [json.loads(line) for line in rig.manager.events_path.read_text().splitlines()]
    state_event = [x for x in events if x["type"] == "manager.state"][-1]
    assert persisted["model_alive"] is False
    assert state_event["data"]["model_alive"] is False
    assert persisted["server_alive"] is True


def test_status_does_not_depend_on_slow_model_rpc(rig):
    rig.ready()
    rig.manager._rpc = lambda *a, **k: (_ for _ in ()).throw(AssertionError("status must not make network calls"))
    assert rig.manager.dispatch("status")["api_ready"]


def test_generation_deadline_reclaims_model_without_replay(rig):
    rig.ready()
    accepted = rig.submit()
    rig.manager.refresh()
    with rig.manager._lock, rig.manager._db:
        record = rig.manager._get_job(accepted["job_id"])
        record["started_at"] = time.time() - 6000
        rig.manager._store_job(record)
    rig.manager.refresh()
    assert job(rig, accepted["job_id"])["status"] == "failed"
    assert not rig.manager.children["model"].alive
    assert len(rig.executions) == 1


def test_independent_api_and_model_probes_really_overlap(rig):
    api_entered, model_entered = threading.Event(), threading.Event()
    original_rpc = rig.manager._rpc
    def api_probe(*args):
        api_entered.set()
        assert model_entered.wait(1), "model probe was unnecessarily serialized"
        return True
    def model_probe(*args, **kwargs):
        model_entered.set()
        assert api_entered.wait(1), "API probe was unnecessarily serialized"
        return original_rpc(*args, **kwargs)
    rig.manager._probe, rig.manager._rpc = api_probe, model_probe
    assert rig.manager.refresh()["api_ready"]


def test_stale_model_probe_cannot_mutate_replacement_instance(rig):
    rig.ready()
    captured, release = threading.Event(), threading.Event()
    original_rpc = rig.manager._rpc
    def delayed_probe(*args, **kwargs):
        value = original_rpc(*args, **kwargs)
        captured.set()
        assert release.wait(3)
        return value
    rig.manager._rpc = delayed_probe
    refresh = threading.Thread(target=rig.manager.refresh)
    refresh.start()
    assert captured.wait(1)
    with rig.manager._control_lock:
        rig.manager._stop_model(force=True)
        rig.manager._launch_model()
    new_id = rig.manager.model_instance_id
    release.set()
    refresh.join(3)
    assert not refresh.is_alive()
    state = rig.manager.snapshot()
    assert state["model_instance_id"] == new_id
    assert state["modules"]["model"]["phase"] == "loading"
    assert not state["model_ready"]
    assert rig.manager._model_unreachable == 0


def test_all_stop_requests_are_issued_before_waiting_for_one_child(tmp_path):
    fixture = Rig(tmp_path, public=True).start()
    children = list(fixture.manager.children.values())
    for child in children:
        child.terminate_requested = False
        def terminate(child=child): child.terminate_requested = True
        def wait(timeout=None, child=child):
            assert all(other.terminate_requested for other in children)
            child.alive = False
            return 0
        child.terminate, child.wait = terminate, wait
    try:
        assert fixture.manager.shutdown()["cleanup_complete"]
    finally:
        fixture.manager.close()


def test_cleanup_failure_result_does_not_release_lease_before_process_exit(rig):
    rig.ready()
    accepted = rig.submit()
    rig.manager.refresh()
    rig.complete(status="failed", cleanup_complete=False)
    rig.manager.children["model"].stubborn = True
    with pytest.raises(ControlError): rig.manager.refresh()
    pending = job(rig, accepted["job_id"])
    assert pending["status"] == "running"
    assert pending["model_lease_released"] is False
    assert not pending["cleanup_complete"]
    rig.manager.children["model"].stubborn = False
    with rig.manager._control_lock:
        rig.manager._stop_model(force=True)
    terminal = job(rig, accepted["job_id"])
    assert terminal["status"] == "failed" and terminal["cleanup_complete"]


def test_main_cancel_discards_a_late_successful_result(rig):
    rig.ready()
    accepted = rig.submit()
    rig.manager.refresh()
    rig.manager.cancel(accepted["job_id"])
    rig.complete()
    path = Path(rig.result["artifact"]["path"])
    rig.manager.refresh()
    assert job(rig, accepted["job_id"])["status"] == "cancelled"
    assert not path.exists()


def test_model_first_ready_disarms_cold_start_deadline(rig):
    rig.ready()
    rig.manager._model_started -= 4000
    rig.status_error = True
    rig.manager.refresh()
    assert rig.manager.children["model"].alive
    assert rig.manager._model_unreachable == 1


def test_public_deadline_preserves_local_modules(tmp_path):
    now = [0.0]
    fixture = Rig(tmp_path, public=True, tunnel_timeout=60, clock=lambda: now[0]).start()
    try:
        now[0] = 61.0
        fixture.ready()
        status = fixture.manager.snapshot()
        assert status["api_ready"] and status["model_ready"]
        assert not status["ready"]
        assert "public_preparation_deadline" in status["error"]
        assert all(child.alive for child in fixture.manager.children.values())
    finally:
        fixture.manager.close()


def test_explicit_api_stop_pauses_public_delivery_budget_until_restart(tmp_path):
    now = [0.0]
    fixture = Rig(tmp_path, public=True, tunnel_timeout=60, clock=lambda: now[0]).start()
    manager = fixture.manager
    try:
        fixture.ready()
        now[0] = 20.0
        stopped = manager.control({"action": "stop_api"})
        assert wait_command(manager, stopped)["status"] == "succeeded"
        now[0] = 1000.0
        state = manager.refresh()
        assert state["error"] is None
        assert state["modules"]["api"]["lifecycle"] == "stopped"
        assert not state["api_ready"] and not state["public_api_ready"]
        assert manager._public_ever_ready is False

        restarted = manager.control({"action": "start_api"})
        assert wait_command(manager, restarted)["status"] == "succeeded"
        now[0] = 1039.0  # 20 before stop + 39 after restart, not wall time.
        assert manager.refresh()["error"] is None
        now[0] = 1041.0
        state = manager.refresh()
        assert state["error"].startswith("public_preparation_deadline_exceeded")
        assert state["api_ready"] and state["model_ready"]
    finally:
        manager.close()


def test_unconfirmed_api_stop_cannot_promote_residual_public_probe(tmp_path):
    now = [0.0]
    fixture = Rig(tmp_path, public=True, tunnel_timeout=60, clock=lambda: now[0]).start()
    manager = fixture.manager
    try:
        fixture.ready()
        manager._error = "public_fixture_pending"
        manager.children["api"].stubborn = True
        now[0] = 20.0
        stopped = manager.control({"action": "stop_api"})
        assert wait_command(manager, stopped)["status"] == "failed"
        manager._transport = lambda: ("https://fixture.invalid", True)
        now[0] = 1000.0
        state = manager.refresh()
        assert state["modules"]["api"]["alive"] is True
        assert state["modules"]["api"]["lifecycle"] == "stop_unconfirmed"
        assert state["modules"]["api"]["expected_running"] is False
        assert not state["api_ready"] and not state["public_api_ready"]
        assert manager._public_ever_ready is False
        assert state["error"] == "public_fixture_pending"

        manager.children["api"].stubborn = False
        retry = manager.control({"action": "stop_api"})
        assert wait_command(manager, retry)["status"] == "succeeded"
        restarted = manager.control({"action": "start_api"})
        assert wait_command(manager, restarted)["status"] == "succeeded"
        state = manager.refresh()
        assert state["public_api_ready"] is True
        assert manager._public_ever_ready is True
        assert state["error"] is None
    finally:
        manager.children["api"].stubborn = False
        manager.close()


def test_api_restart_preserves_model_and_completed_record(rig):
    rig.ready()
    accepted = rig.submit()
    rig.manager.refresh()
    rig.complete()
    rig.manager.refresh()
    model_pid = rig.manager.children["model"].pid
    first_api_pid = rig.manager.children["api"].pid
    stopped = rig.manager.control({"action": "stop_api"})
    assert wait_command(rig.manager, stopped)["status"] == "succeeded"
    assert rig.manager.children["model"].alive
    observed = rig.manager.refresh()
    assert not observed["api_ready"] and not observed["public_api_ready"]
    assert not observed["error"]
    assert observed["modules"]["api"]["expected_running"] is False
    assert observed["modules"]["api"]["lifecycle"] == "stopped"
    started = rig.manager.control({"action": "start_api"})
    assert wait_command(rig.manager, started)["status"] == "succeeded"
    assert rig.manager.children["api"].pid != first_api_pid
    assert rig.manager.children["model"].pid == model_pid
    assert job(rig, accepted["job_id"])["status"] == "succeeded"
    assert rig.manager.refresh()["modules"]["api"]["lifecycle"] == "ready"


def test_unrequested_api_exit_still_reports_failed_expected_module(rig):
    rig.ready()
    rig.manager.children["api"].alive = False
    observed = rig.manager.refresh()
    assert observed["error"].startswith("api_process_exited")
    assert observed["modules"]["api"]["expected_running"] is True
    assert observed["modules"]["api"]["lifecycle"] == "failed"
    assert observed["model_alive"] and observed["model_ready"]


def test_probe_from_before_stop_intent_cannot_republish_api_ready(rig):
    entered, release = threading.Event(), threading.Event()
    original = rig.manager._probe
    def probe(*args):
        entered.set()
        assert release.wait(3)
        return True
    rig.manager._probe = probe
    worker = threading.Thread(target=rig.manager.refresh)
    worker.start()
    try:
        assert entered.wait(1)
        stopped = rig.manager.control({"action": "stop_api"})
        assert wait_command(rig.manager, stopped)["status"] == "succeeded"
        release.set()
        worker.join(timeout=2)
        assert not worker.is_alive()
        state = rig.manager.snapshot()
        assert state["api_ready"] is False
        assert state["modules"]["api"]["lifecycle"] == "stopped"
        assert not state["error"]
    finally:
        release.set()
        worker.join(timeout=3)
        rig.manager._probe = original


def test_queued_job_keeps_file_cleanup_separate_from_unused_model_lease(rig):
    queued = rig.submit()
    assert queued["model_lease_released"] is True
    assert queued["cleanup_complete"] is False


def test_background_control_requests_have_a_global_bound(rig):
    with rig.manager._control_lock:
        accepted = [rig.manager.control({"action": "drain", "command_id": "bounded-" + str(i)}) for i in range(4)]
        with pytest.raises(ControlError) as failure:
            rig.manager.control({"action": "drain", "command_id": "overflow"})
        assert failure.value.status == 429
        with pytest.raises(ControlError) as absent:
            rig.manager.dispatch("command", {"command_id": "overflow"})
        assert absent.value.status == 404
    assert all(wait_command(rig.manager, command)["status"] == "succeeded" for command in accepted)
    resumed = rig.manager.control({"action": "resume", "command_id": "after-bound-released"})
    assert wait_command(rig.manager, resumed)["status"] == "succeeded"


def test_control_thread_start_failure_is_queryable_and_releases_slot(rig, monkeypatch):
    original = threading.Thread.start
    def fail(_): raise RuntimeError("cannot start command worker")
    monkeypatch.setattr(threading.Thread, "start", fail)
    with pytest.raises(RuntimeError):
        rig.manager.control({"action": "drain", "command_id": "failed-start"})
    result = rig.manager.dispatch("command", {"command_id": "failed-start"})
    assert result["status"] == "failed"
    monkeypatch.setattr(threading.Thread, "start", original)
    resumed = rig.manager.control({"action": "resume"})
    assert wait_command(rig.manager, resumed)["status"] == "succeeded"


def test_pipeline_fifo_with_many_jobs_and_cancelled_middle(rig):
    rig.capacity = 2
    submitted = [rig.submit() for _ in range(7)]
    cancelled_id = submitted[3]["job_id"]
    assert rig.manager.cancel(cancelled_id)["status"] == "cancelled"
    rig.ready()
    assert len(rig.executions) == 2
    assert len(rig.manager._active_jobs()) == 2
    assert rig.manager.snapshot()["queue"]["queued"] == 4
    while rig.manager.snapshot()["queue"]["unfinished"]:
        for attempt in list(rig.inflight):
            rig.complete(target_attempt=attempt)
        rig.manager.refresh()
    assert [x["job_id"] for x in rig.executions] == [j["job_id"] for j in submitted if j["job_id"] != cancelled_id]
    assert all(job(rig, item["job_id"])["status"] == ("cancelled" if item["job_id"] == cancelled_id else "succeeded") for item in submitted)
    assert not (rig.manager.root / cancelled_id / "inputs").exists()


@pytest.mark.parametrize("dispatch_slice_exhausted", [False, True])
def test_multiple_results_commit_before_ack_and_retry_ack_without_duplicate_events(rig, monkeypatch, dispatch_slice_exhausted):
    rig.capacity = 4
    submitted = [rig.submit() for _ in range(4)]
    if dispatch_slice_exhausted:
        class DispatchClock:
            value = 0

            def monotonic(self):
                self.value += 1
                return self.value

            def __getattr__(self, name):
                return getattr(time, name)

        # Exhaust each dispatch slice without delaying the test or changing the
        # real clock used by the bounded wait below.
        monkeypatch.setattr(gm, "time", DispatchClock())
    rig.ready()
    if dispatch_slice_exhausted:
        assert len(rig.inflight) == 1
    # One refresh has a time budget; model readiness does not imply all queued
    # jobs have already reached execute. Drive the foreground fixture until the
    # actual admissions needed by this result-batch test have been observed.
    deadline = time.monotonic() + 5
    expected_jobs = {item["job_id"] for item in submitted}
    while {item["job_id"] for item in rig.inflight.values()} != expected_jobs and time.monotonic() < deadline:
        rig.manager.refresh()
    assert {item["job_id"] for item in rig.inflight.values()} == expected_jobs
    for attempt in reversed(list(rig.inflight)):
        rig.complete(target_attempt=attempt)
    rig.ack_error = True
    rig.manager.refresh()
    assert all(job(rig, item["job_id"])["status"] == "succeeded" for item in submitted)
    assert len(rig.results) == 4
    before = rig.manager._db.execute("SELECT COUNT(*) FROM events WHERE record LIKE '%job.succeeded%'").fetchone()[0]
    rig.ack_error = False
    rig.manager.refresh()
    assert rig.results == []
    after = rig.manager._db.execute("SELECT COUNT(*) FROM events WHERE record LIKE '%job.succeeded%'").fetchone()[0]
    assert before == after == 4 and len(rig.executions) == 4


def test_busy_offer_keeps_fifo_head_and_does_not_execute_it(rig):
    rig.ready()
    first, second = rig.submit(), rig.submit()
    rig.manager.refresh()
    waiting = job(rig, second["job_id"])
    assert waiting["status"] == "queued" and waiting["dispatch_state"] == "deferred"
    assert waiting["model_lease_released"] and waiting["attempt_id"] is None
    assert [x["job_id"] for x in rig.executions] == [first["job_id"]]
    assert rig.manager.cancel(second["job_id"])["status"] == "cancelled"
    rig.complete()
    rig.manager.refresh()
    assert len(rig.executions) == 1


def test_lost_execute_response_reconciles_exact_attempt_without_resubmission(rig):
    rig.ready()
    accepted = rig.submit()
    rig.execute_error = True
    rig.manager.refresh()
    state = job(rig, accepted["job_id"])
    assert state["status"] == "running" and state["dispatch_state"] == "reconciled"
    assert rig.manager.children["model"].alive
    rig.complete()
    rig.manager.refresh()
    assert job(rig, accepted["job_id"])["status"] == "succeeded"
    assert len(rig.executions) == 1


def test_disk_budget_rejects_bytes_not_job_count(rig, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(gm.shutil, "disk_usage", lambda _: SimpleNamespace(free=rig.manager.disk_reserve_bytes))
    with pytest.raises(ControlError) as error:
        rig.submit()
    assert error.value.status == 507
    assert rig.manager.snapshot()["queue"]["queued"] == 0


def test_deadline_stops_all_pipeline_attempts_without_offering_queued_work(rig):
    rig.capacity = 2
    submitted = [rig.submit() for _ in range(5)]
    rig.ready()
    with rig.manager._lock, rig.manager._db:
        record = rig.manager._get_job(submitted[0]["job_id"])
        record["started_at"] = time.time() - 6000
        rig.manager._store_job(record)
    rig.capacity = 5
    rig.manager.refresh()
    assert len(rig.executions) == 2
    assert all(job(rig, item["job_id"])["status"] == "failed" for item in submitted)


def test_model_failure_terminalizes_every_active_and_queued_job(rig):
    rig.capacity = 3
    submitted = [rig.submit() for _ in range(6)]
    rig.ready()
    rig.manager.cancel(submitted[1]["job_id"])
    rig.model_phase = "failed"
    rig.manager.refresh()
    assert [job(rig, item["job_id"])["status"] for item in submitted] == ["failed", "cancelled", "failed", "failed", "failed", "failed"]
    assert rig.manager.snapshot()["api_ready"]


def test_unfit_request_is_terminal_and_does_not_block_fifo_forever(rig):
    submitted = [rig.submit(), rig.submit()]
    original = rig.manager._rpc
    def reject_first(url, key, operation, payload, timeout):
        if operation == "execute" and payload["job_id"] == submitted[0]["job_id"]:
            raise ControlError(json.dumps({"guarantees_not_accepted": True, "reason": "request_exceeds_total_budget"}), 507)
        return original(url, key, operation, payload, timeout)
    rig.manager._rpc = reject_first
    rig.ready()
    assert job(rig, submitted[0]["job_id"])["status"] == "failed"
    assert job(rig, submitted[1]["job_id"])["status"] == "running"
    assert len(rig.executions) == 1


def test_actual_private_rpc_pipeline_public_api_and_cancelled_queue(rig, monkeypatch):
    """Real API + RPC + scheduler threads; injected CPU stage barriers, no CUDA."""
    from fastapi.testclient import TestClient
    from control_protocol import make_server, rpc_call
    from model_container import ModelContainer
    from server import create_app
    from test_model_container import eventually, fixture_artifact
    from test_pipeline_scheduler import StageBackend

    class FeedbackBackend(StageBackend):
        def encode(self, result, output_path):
            if result.name == "C":
                raise RuntimeError("fixture encoder failed")
            return super().encode(result, output_path)

    backend = FeedbackBackend()
    backend.load_release.clear()
    backend.gate("gpu", "A", blocked=True)
    backend.gate("encode", "A", blocked=True)
    backend.gate("gpu", "B", blocked=True)
    model = ModelContainer(rig.manager.root, rig.manager.model_instance_id, backend=backend,
                           artifact_validator=fixture_artifact,
                           resource_limits={"ram_bytes": 256, "disk_bytes": 4096, "cpu_threads": 2})
    http = make_server("127.0.0.1", 0, rig.manager.key, model.dispatch)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{http.server_port}"
    offers = []
    def real_rpc(_url, key, operation, payload, timeout):
        if operation == "execute": offers.append(payload.copy())
        return rpc_call(url, key, operation, payload, timeout)
    rig.manager._rpc = real_rpc
    monkeypatch.setenv("H3_INSTANCE_ID", rig.manager.instance_id)
    monkeypatch.delenv("H3_PARENT_PID", raising=False)
    model.start()
    try:
        app = create_app(controller=rig.manager.dispatch, api_key="a" * 40, output_dir=rig.manager.root)
        headers = {"Authorization": "Bearer " + "a" * 40}
        with TestClient(app) as client:
            jobs = {}
            for name in "ABCDEF":
                response = client.post("/v1/jobs", headers=headers,
                    data={"prompt": name, "kinds": '["image"]', "width": 512, "height": 288, "steps": 8},
                    files=[("files", ("ref.png", b"test reference"))])
                assert response.status_code == 202
                accepted = response.json()
                jobs[name] = accepted["id"]
                assert accepted["status"] == "queued" and accepted["waiting_for"] == "current_observation"
                assert accepted["queue_position"] == len(jobs)
                assert accepted["cancel_url"] == accepted["status_url"] + "/cancel"
                assert accepted["poll_after_seconds"] == 2
                assert "file_url" not in accepted
            # Real API and manager are usable throughout the loading barrier.
            # No request reaches any model stage just because files/process exist.
            for _ in range(3):
                rig.manager.refresh()
                health = client.get("/health").json()
                assert health["api_ready"] and not health["model_ready"]
                assert not offers and not any(event.is_set() for event in backend.started.values())
            assert client.get(f"/v1/jobs/{jobs['A']}/file", headers=headers).status_code == 409
            cancelled = client.post(f"/v1/jobs/{jobs['D']}/cancel", headers=headers)
            assert cancelled.json()["status"] == "cancelled"
            backend.load_release.set()
            eventually(lambda: model.status()["ready"])
            rig.manager.refresh()
            assert backend.started["gpu", "A"].wait(2)
            eventually(lambda: ("prepare", "B") in backend.started and backend.started["prepare", "B"].is_set())
            assert len(rig.manager._active_jobs()) == 2
            rig.manager.refresh()
            running = client.get(f"/v1/jobs/{jobs['A']}", headers=headers).json()
            assert running["status"] == "running" and running["active_phase"]
            assert running["waiting_for"] is None
            backend.gates["gpu", "A"].set()
            assert backend.started["encode", "A"].wait(2)
            assert backend.started["gpu", "B"].wait(2)
            assert backend.max_gpu_active == 1
            backend.release_all()
            def finished():
                rig.manager.refresh()
                return rig.manager.snapshot()["queue"]["unfinished"] == 0
            eventually(finished, timeout=6)
            assert model.status()["pending_results"] == 0
            for name, identity in jobs.items():
                result = client.get(f"/v1/jobs/{identity}", headers=headers).json()
                expected = "cancelled" if name == "D" else "failed" if name == "C" else "succeeded"
                assert result["status"] == expected
                assert "poll_after_seconds" not in result
                if expected == "succeeded":
                    media = client.get(result["file_url"], headers=headers)
                    assert media.status_code == 200 and media.content == name.encode()
                    assert media.headers["content-type"] == "video/mp4"
                else:
                    assert "file_url" not in result
                    assert client.get(f"/v1/jobs/{identity}/file", headers=headers).status_code == 409
                if expected == "failed":
                    assert result["error"] and result["cleanup_complete"]
            assert not any(p["job_id"] == jobs["D"] for p in offers)
            events = [json.loads(line) for line in model.events_path.read_text().splitlines()]
            assert len([e for e in events if e["type"] == "execution.accepted"]) == 5
            assert not any(e.get("job_id") == jobs["D"] for e in events)
    finally:
        backend.load_release.set()
        backend.release_all()
        model.stop()
        assert model.join(5)
        http.shutdown()
        http.server_close()
        thread.join(2)
