"""Container lifecycle contract tests; no CUDA, weights or production fake mode."""
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import uuid

import pytest

from control_protocol import ControlError, make_server, rpc_call
from model_container import ModelContainer, verify_artifact


def eventually(check, timeout=4):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = check()
        if value:
            return value
        time.sleep(.01)
    raise AssertionError("Condition did not become true")


class Backend:
    def __init__(self, *, loading=False, blocked=False, load_error=None, execute_error=None):
        self.load_release = threading.Event()
        self.generate_release = threading.Event()
        self.generate_started = threading.Event()
        if not loading:
            self.load_release.set()
        if not blocked:
            self.generate_release.set()
        self.load_error, self.execute_error = load_error, execute_error
        self.calls = []
        self.unloaded = False
        self.last_metrics = None

    def load(self):
        assert self.load_release.wait(5), "Test must release load"
        if self.load_error:
            raise self.load_error

    def phase_snapshot(self):
        return {"phase": "fixture", "elapsed_seconds": .5, "kind": "generation"}

    def generate(self, **parameters):
        self.calls.append(parameters)
        self.generate_started.set()
        assert self.generate_release.wait(5), "Test must release execution"
        if self.execute_error:
            raise self.execute_error
        parameters["output_path"].write_bytes(b"unit artifact, not generated media")
        self.last_metrics = {"status": "succeeded", "wall_seconds": .1}

    def unload(self):
        self.unloaded = True


def fixture_artifact(path, parameters):
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "media": {"full_decode_passed": True, "test_only_fixture": True}}


@pytest.fixture
def containers(tmp_path):
    values = []

    def factory(backend=None, validator=fixture_artifact):
        backend = backend or Backend()
        model = ModelContainer(tmp_path, uuid.uuid4().hex, backend=backend, artifact_validator=validator)
        values.append((model, backend))
        model.start()
        return model, backend

    yield factory
    for model, backend in values:
        backend.load_release.set()
        backend.generate_release.set()
        try:
            model.stop()
        except ControlError:
            model._stop_requested = True
        assert model.join(5)


def request(model, job_id=None, attempt_id=None):
    job_id = job_id or uuid.uuid4().hex
    directory = model.root / job_id / "inputs"
    directory.mkdir(parents=True)
    source = directory / "0.png"
    source.write_bytes(b"fixture reference")
    return {"job_id": job_id, "attempt_id": attempt_id or uuid.uuid4().hex,
            "instance_id": model.instance_id,
            "parameters": {"prompt": "A reference", "references": [{"kind": "image", "path": str(source)}],
                           "seconds": 5, "width": 512, "height": 288, "steps": 8, "seed": 42}}


def ready(model):
    eventually(lambda: model.status()["accepting"])


def result(model):
    return eventually(lambda: model.status()["last_result"])


def test_controller_queries_while_model_loads(containers):
    model, backend = containers(Backend(loading=True))
    state = model.status()
    assert state["phase"] == "loading" and state["worker_alive"]
    assert not state["ready"] and not state["cleanup_complete"]
    with pytest.raises(ControlError):
        model.execute(request(model))
    backend.load_release.set()
    ready(model)


def write_download_progress(model, *, identity=None, percent=25):
    model.progress_path.write_text(json.dumps({
        "schema_version": 1, "model_instance_id": identity or model.instance_id,
        "observed_at": time.time(), "observation_stale": False,
        "items": {"model": {"phase": "downloading", "bytes_completed": percent, "bytes_total": 100,
                              "percent": percent, "files_completed": 0, "files_total": 1,
                              "active_files": 1, "scope": "observed transfer", "detail": "fixture"},
                  "acceleration": {"phase": "disabled"}, "style": {"phase": "disabled"}},
    }), encoding="utf-8")


def test_download_completion_is_not_model_readiness(containers):
    model, backend = containers(Backend(loading=True))
    write_download_progress(model, percent=100)
    state = model.status()
    assert state["downloads"]["items"]["model"]["percent"] == 100
    assert state["downloads"]["model_ready"] is False
    assert state["downloads"]["lifecycle"] == "loading"
    assert state["ready"] is False
    backend.load_release.set()
    ready(model)
    assert model.status()["downloads"]["model_ready"] is True


def test_progress_from_another_model_generation_is_not_reused(containers):
    model, backend = containers(Backend(loading=True))
    write_download_progress(model, identity=uuid.uuid4().hex)
    state = model.status()["downloads"]
    assert state["model_instance_id"] == model.instance_id
    assert state["items"] == {} and state["observation_stale"] is True
    assert state["model_ready"] is False


def test_model_exit_turns_progress_into_historical_observation(containers):
    model, backend = containers()
    ready(model)
    write_download_progress(model, percent=100)
    model.stop()
    assert model.join(5)
    downloads = model.status()["downloads"]
    assert downloads["items"]["model"]["percent"] == 100
    assert downloads["lifecycle"] == "stopped" and downloads["observation_stale"]
    assert downloads["model_ready"] is False


def test_actual_private_http_remains_queryable_during_loading(containers):
    model, backend = containers(Backend(loading=True))
    key = "test-only-private-" + "x" * 40
    http = make_server("127.0.0.1", 0, key, model.dispatch)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{http.server_address[1]}"
        assert rpc_call(url, key, "status")["phase"] == "loading"
        with pytest.raises(ControlError):
            rpc_call(url, "incorrect", "status")
        backend.load_release.set()
        eventually(lambda: rpc_call(url, key, "status")["ready"])
    finally:
        http.shutdown()
        http.server_close()
        thread.join(2)


def test_load_failure_revokes_ready_but_state_remains_readable(containers):
    model, backend = containers(Backend(load_error=RuntimeError("secret-internal")))
    eventually(lambda: model.status()["phase"] == "failed")
    assert model.join(2)
    state = model.dispatch("status")
    assert not state["ready"] and not state["worker_alive"]
    assert not state["cleanup_complete"] and state["worker_cleanup_complete"]
    assert "secret-internal" not in json.dumps(state)
    assert backend.unloaded


def test_single_attempt_fencing_and_idempotent_retry_after_input_cleanup(containers):
    model, backend = containers(Backend(blocked=True))
    ready(model)
    payload = request(model)
    wrong = dict(payload, instance_id=uuid.uuid4().hex)
    with pytest.raises(ControlError):
        model.execute(wrong)
    accepted = model.execute(payload)
    assert accepted["accepted"] and accepted["attempt_id"] == payload["attempt_id"]
    assert backend.generate_started.wait(2)
    assert model.execute(payload)["accepted"]
    assert model.status()["active_attempt_id"] == payload["attempt_id"]
    assert not model.status()["accepting"]
    with pytest.raises(ControlError):
        model.execute(request(model))
    backend.generate_release.set()
    completed = result(model)
    assert completed["status"] == "succeeded" and completed["artifact"]["sha256"]
    assert completed["cleanup_complete"] is True and model.status()["cleanup_complete"] is False
    assert completed["metrics"]["status"] == "succeeded"
    assert not (model.root / payload["job_id"] / "inputs").exists()
    assert model.execute(payload)["result"]["status"] == "succeeded"
    assert len(backend.calls) == 1


def test_duplicate_attempt_rejects_parameter_change(containers):
    model, _ = containers(Backend(blocked=True))
    ready(model)
    payload = request(model)
    model.execute(payload)
    payload["parameters"]["prompt"] = "changed"
    with pytest.raises(ControlError):
        model.execute(payload)


def test_cancel_ack_is_not_execution_or_resource_release(containers):
    model, backend = containers(Backend(blocked=True))
    ready(model)
    payload = request(model)
    model.execute(payload)
    assert backend.generate_started.wait(2)
    cancelled = model.cancel(payload)
    assert cancelled["status"] == "cancel_requested"
    state = model.status()
    assert state["active_job_id"] == payload["job_id"]
    assert not state["accepting"] and not state["cleanup_complete"]
    assert state["last_result"] is None
    backend.generate_release.set()
    terminal = result(model)
    assert terminal["status"] == "cancelled" and terminal["artifact"] is None
    assert terminal["cleanup_complete"] is True
    assert not (model.root / payload["job_id"] / "result.mp4").exists()
    assert model.cancel(payload)["status"] == "cancelled"


def test_cancel_during_artifact_verification_wins_before_terminal_commit(containers):
    entered, release = threading.Event(), threading.Event()

    def blocked_validator(path, parameters):
        entered.set()
        assert release.wait(3)
        return fixture_artifact(path, parameters)

    model, _ = containers(validator=blocked_validator)
    ready(model)
    payload = request(model)
    model.execute(payload)
    assert entered.wait(2)
    model.cancel(payload)
    release.set()
    assert result(model)["status"] == "cancelled"
    assert not (model.root / payload["job_id"] / "result.mp4").exists()


@pytest.mark.parametrize("change", ["outside", "other_job", "relative", "symlink"])
def test_input_paths_must_belong_to_assigned_job(containers, tmp_path, change):
    model, _ = containers()
    ready(model)
    payload = request(model)
    source = Path(payload["parameters"]["references"][0]["path"])
    if change == "outside":
        target = tmp_path.parent / "outside-fixture"
        target.write_bytes(b"outside")
    elif change == "other_job":
        target = Path(request(model)["parameters"]["references"][0]["path"])
    elif change == "relative":
        target = source.relative_to(model.root)
    else:
        target = source.with_name("link.png")
        try:
            target.symlink_to(source)
        except OSError:
            pytest.skip("OS account cannot create symbolic links")
    payload["parameters"]["references"][0]["path"] = str(target)
    with pytest.raises(ControlError):
        model.execute(payload)


@pytest.mark.parametrize("field,value", [("width", 513), ("width", 100000), ("seconds", float("nan")),
                                         ("seconds", 15), ("steps", True), ("steps", 100), ("seed", -1)])
def test_private_boundary_revalidates_resource_limits(containers, field, value):
    model, _ = containers()
    ready(model)
    payload = request(model)
    payload["parameters"][field] = value
    with pytest.raises(ControlError):
        model.execute(payload)


def test_stopping_drains_active_execution_and_does_not_claim_process_release(containers):
    model, backend = containers(Backend(blocked=True))
    ready(model)
    payload = request(model)
    model.execute(payload)
    assert backend.generate_started.wait(2)
    state = model.stop()
    assert state["phase"] == "draining" and not state["ready"]
    assert not model.join(.02)
    backend.generate_release.set()
    assert model.join(3)
    state = model.status()
    assert state["phase"] == "stopped" and not state["worker_alive"]
    assert state["worker_cleanup_complete"] and not state["cleanup_complete"]
    assert state["last_result"]["status"] == "succeeded"


def test_worker_baseexception_records_terminal_failure_and_revokes_capability(containers):
    model, _ = containers(Backend(execute_error=SystemExit("fatal")))
    ready(model)
    model.execute(request(model))
    assert result(model)["status"] == "failed"
    assert model.join(3)
    assert model.status()["phase"] == "failed"
    assert not model.status()["ready"]


def test_two_execution_failures_open_circuit(containers):
    model, backend = containers(Backend(execute_error=RuntimeError("execution failure")))
    ready(model)
    first = request(model)
    model.execute(first)
    assert result(model)["status"] == "failed"
    ready(model)
    second = request(model)
    model.execute(second)
    eventually(lambda: (model.status()["last_result"] or {}).get("attempt_id") == second["attempt_id"])
    assert model.join(3)
    assert model.status()["phase"] == "failed"
    assert len(backend.calls) == 2


def test_bad_media_request_does_not_poison_container(containers):
    model, _ = containers(Backend(execute_error=ValueError("invalid reference")))
    ready(model)
    model.execute(request(model))
    assert result(model)["status"] == "failed"
    assert model.status()["ready"]


def test_failed_media_validation_never_publishes_artifact(containers):
    def reject(*_):
        raise ValueError("not a valid movie")
    model, _ = containers(validator=reject)
    ready(model)
    payload = request(model)
    model.execute(payload)
    terminal = result(model)
    assert terminal["status"] == "failed" and terminal["artifact"] is None
    assert not (model.root / payload["job_id"] / "result.mp4").exists()


def test_cleanup_failure_revokes_ready_even_with_generated_file(containers, monkeypatch):
    import model_container
    model, backend = containers(Backend(blocked=True))
    ready(model)
    payload = request(model)
    model.execute(payload)
    assert backend.generate_started.wait(2)
    monkeypatch.setattr(model_container.shutil, "rmtree", lambda *_: (_ for _ in ()).throw(OSError("denied")))
    backend.generate_release.set()
    terminal = result(model)
    assert terminal["status"] == "failed" and terminal["artifact"] is None
    assert model.join(3)
    assert not model.status()["ready"]
    assert terminal["cleanup_complete"] is False
    assert (model.root / payload["job_id"] / "inputs").exists()


def test_events_are_ordered_and_identify_execution(containers):
    model, _ = containers()
    ready(model)
    payload = request(model)
    model.execute(payload)
    assert result(model)["status"] == "succeeded"
    model.stop()
    assert model.join(3)
    events = [json.loads(line) for line in model.events_path.read_text().splitlines()]
    assert [e["sequence"] for e in events] == list(range(1, len(events) + 1))
    assert all(e["instance_id"] == model.instance_id for e in events)
    assert len({e["event_id"] for e in events}) == len(events)
    executed = [e for e in events if e["type"].startswith("execution.")]
    assert [e["type"] for e in executed] == ["execution.accepted", "execution.running", "execution.succeeded"]
    assert all(e["job_id"] == payload["job_id"] and e["attempt_id"] == payload["attempt_id"] for e in executed)


@pytest.mark.parametrize("load_fails", [False, True])
def test_main_defers_repeated_signals_during_event_construction(tmp_path, monkeypatch, load_fails):
    """Exercise real Python signals at the observed event-ID interruption point."""
    import model_container
    backend = Backend(load_error=RuntimeError("load failed") if load_fails else None)
    model = ModelContainer(tmp_path, uuid.uuid4().hex, backend=backend)
    original_start, original_event, original_uuid = model.start, model._event, uuid.uuid4
    control_key = "signal-regression-private-key-0000"
    private_servers, observations = [], []
    nested_signal_sent = False

    def create_server(*args, **kwargs):
        server = make_server(*args, **kwargs)
        private_servers.append(server)
        return server

    def event_with_signal(kind, **data):
        def interrupted_uuid():
            nonlocal nested_signal_sent
            if not nested_signal_sent:
                nested_signal_sent = True
                # The outer event already owns its sequence and RLock. The old
                # handler reentered stop here and persisted sequence N+1 first.
                signal.raise_signal(signal.SIGTERM)
                signal.raise_signal(signal.SIGINT)
            return original_uuid()
        if kind == "model.stop_requested":
            with monkeypatch.context() as patch:
                patch.setattr(model_container.uuid, "uuid4", interrupted_uuid)
                return original_event(kind, **data)
        return original_event(kind, **data)

    def start_then_signal():
        original_start()
        eventually(lambda: model.status()["phase"] in {"ready", "failed"})
        url = "http://127.0.0.1:" + str(private_servers[0].server_port)
        state = rpc_call(url, control_key, "status")
        assert state["phase"] == ("failed" if load_fails else "ready")
        assert state["ready"] is not load_fails
        signal.raise_signal(signal.SIGTERM)
        observations.append(not model._stop_requested)
        return model

    monkeypatch.setenv("H3_CONTROL_KEY", control_key)
    monkeypatch.setattr(model_container, "ModelContainer", lambda *args, **kwargs: model)
    monkeypatch.setattr(model_container, "make_server", create_server)
    monkeypatch.setattr(model, "start", start_then_signal)
    monkeypatch.setattr(model, "_event", event_with_signal)
    monkeypatch.setattr(sys, "argv", ["model_container.py", "--port", "0", "--instance-id",
                                     model.instance_id, "--root", str(tmp_path)])
    handlers = {value: signal.getsignal(value) for value in (signal.SIGTERM, signal.SIGINT)}
    try:
        model_container.main()
    finally:
        for value, handler in handlers.items():
            signal.signal(value, handler)
        backend.load_release.set()
        model.stop()
        assert model.join(3)
    events = [json.loads(line) for line in model.events_path.read_text().splitlines()]
    assert observations == [True], "Signal callbacks must not execute lifecycle transitions"
    assert nested_signal_sent
    assert [event["sequence"] for event in events] == list(range(1, len(events) + 1))
    assert sum(event["type"] == "model.stop_requested" for event in events) == 1
    assert backend.unloaded


def test_main_parent_loss_applies_stop_and_keeps_bounded_join(tmp_path, monkeypatch):
    import model_container
    backend = Backend(loading=True)
    model = ModelContainer(tmp_path, uuid.uuid4().hex, backend=backend)
    original_join, join_timeouts = model.join, []

    def join(timeout=None):
        join_timeouts.append(timeout)
        assert model._stop_requested
        backend.load_release.set()
        return original_join(timeout)

    def exit_process(code):
        raise SystemExit(code)

    monkeypatch.setenv("H3_CONTROL_KEY", "parent-loss-regression-key-000000")
    monkeypatch.setattr(model_container, "ModelContainer", lambda *args, **kwargs: model)
    monkeypatch.setattr(model_container, "parent_identity", lambda *_: 100)
    monkeypatch.setattr(model_container, "parent_is_alive", lambda *_: False)
    monkeypatch.setattr(model_container.os, "_exit", exit_process)
    monkeypatch.setattr(model, "join", join)
    monkeypatch.setattr(sys, "argv", ["model_container.py", "--port", "0", "--instance-id",
                                     model.instance_id, "--root", str(tmp_path), "--parent-pid", "12345"])
    handlers = {value: signal.getsignal(value) for value in (signal.SIGTERM, signal.SIGINT)}
    try:
        with pytest.raises(SystemExit) as stopped:
            model_container.main()
        assert stopped.value.code == 70
    finally:
        for value, handler in handlers.items():
            signal.signal(value, handler)
        backend.load_release.set()
        model.stop()
        assert original_join(3)
    assert join_timeouts == [2, 2]
    events = [json.loads(line) for line in model.events_path.read_text().splitlines()]
    assert sum(event["type"] == "model.stop_requested" for event in events) == 1


def test_verify_artifact_checks_full_decode_not_just_header(tmp_path, monkeypatch):
    import model_container
    path = tmp_path / "result.mp4"
    path.write_bytes(b"video fixture")
    calls = []
    media = {"streams": [{"codec_type": "video", "codec_name": "h264", "nb_read_frames": "124",
                           "avg_frame_rate": "24/1", "duration": "5.166667", "width": 512, "height": 288},
                          {"codec_type": "audio", "codec_name": "aac", "channels": 2, "sample_rate": "32000"}]}
    def command(args, **kwargs):
        calls.append(args)
        if args[0] == "ffprobe":
            return subprocess.CompletedProcess(args, 0, json.dumps(media), "")
        raise subprocess.CalledProcessError(1, args, stderr="corrupt frame")
    monkeypatch.setattr(model_container.subprocess, "run", command)
    with pytest.raises(subprocess.CalledProcessError):
        verify_artifact(path, {"seconds": 5, "width": 512, "height": 288})
    assert len(calls) == 2 and "-xerror" in calls[1]


def test_thread_start_failure_has_queryable_failed_state(tmp_path, monkeypatch):
    model = ModelContainer(tmp_path, uuid.uuid4().hex, backend=Backend(), artifact_validator=fixture_artifact)
    monkeypatch.setattr(threading.Thread, "start", lambda *_: (_ for _ in ()).throw(RuntimeError("no thread")))
    with pytest.raises(RuntimeError):
        model.start()
    state = model.status()
    assert state["phase"] == "failed" and not state["ready"] and not state["worker_alive"]


@pytest.mark.parametrize("running,created,zombie,expected", [
    (True, 100, False, True), (False, 100, False, False),
    (True, 101, False, False), (True, 100, True, False),
])
def test_parent_watch_checks_liveness_and_process_generation(monkeypatch, running, created, zombie, expected):
    import model_container
    import psutil
    class Process:
        def is_running(self): return running
        def create_time(self): return created
        def status(self): return psutil.STATUS_ZOMBIE if zombie else psutil.STATUS_RUNNING
    monkeypatch.setattr(psutil, "Process", lambda pid: Process())
    assert model_container.parent_is_alive(12345, 100) is expected


def test_unknown_parent_state_fails_closed(monkeypatch):
    import model_container
    import psutil
    def denied(pid):
        raise psutil.AccessDenied(pid)
    monkeypatch.setattr(psutil, "Process", denied)
    assert not model_container.parent_is_alive(12345, 100)


def test_authorized_owner_may_differ_from_windows_redirector_parent(monkeypatch):
    import model_container
    import psutil
    class Process:
        def is_running(self): return True
        def create_time(self): return 123.0
        def status(self): return psutil.STATUS_RUNNING
    monkeypatch.setattr(psutil, "Process", lambda pid: Process())
    monkeypatch.setattr(model_container.os, "getppid", lambda: 56789)
    assert model_container.parent_identity(12345) == 123.0
    assert model_container.parent_is_alive(12345, 123.0)


def test_audit_failure_rejects_dispatch_before_gpu_execution(containers, monkeypatch):
    model, backend = containers()
    ready(model)
    payload = request(model)
    original = Path.open
    def refused(path, *args, **kwargs):
        if path == model.events_path:
            raise OSError("audit disk unavailable")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", refused)
    with pytest.raises(ControlError):
        model.execute(payload)
    assert not backend.calls
    assert model.status()["phase"] == "failed" and not model.status()["ready"]
