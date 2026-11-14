"""Collector lifecycle, redaction and completeness counterexamples."""
import importlib.util
import json
from pathlib import Path
import sqlite3
import threading
import time
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location("research_collection", Path(__file__).resolve().parents[1] / "research/collect_runtime.py")
collection = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collection)
MANAGER, MODEL, JOB = "a" * 32, "b" * 32, "c" * 32


def host(tmp_path, state=None):
    root = tmp_path / "runtime"
    root.mkdir(exist_ok=True)
    current = {"instance_id": MANAGER, "model_instance_id": MODEL, **(state or {})}
    return SimpleNamespace(instance_id=MANAGER, root=root, snapshot=lambda: dict(current),
                           api_key="private-public-key-123", _control_key="private-control-key-456")


def test_redaction_covers_nested_fields_and_secret_echoes():
    value = {"api_key": "unknown", "nested": {"H3_CONTROL_KEY": "unknown", "text": "error with actual-secret"},
             "list": ["Bearer abc.def", "ordinary"]}
    rendered = json.dumps(collection.redact(value, ("actual-secret",)))
    assert "actual-secret" not in rendered and "unknown" not in rendered and "abc.def" not in rendered
    assert "ordinary" in rendered


def test_two_independent_probes_and_stop_never_control_service(tmp_path, monkeypatch):
    seen_snapshot, seen_gpu = threading.Event(), threading.Event()
    owner = host(tmp_path)

    def snapshot():
        seen_snapshot.set()
        return {"instance_id": MANAGER, "api_key": owner.api_key, "message": owner._control_key}

    def gpu():
        seen_gpu.set()
        return {"devices": []}

    owner.snapshot = snapshot
    monkeypatch.setattr(collection, "gpu_observation", gpu)
    collector = collection.RuntimeCollector(owner, tmp_path / "observations", interval=.1).start()
    assert collector.start() is collector
    assert seen_snapshot.wait(1) and seen_gpu.wait(1)
    state = collector.stop(timeout=2)
    assert state["phase"] == "stopped" and state["threads_alive"] == 0
    rows = collection.read_jsonl(collector.directory / "system_snapshots.jsonl")
    assert rows[0]["finished_monotonic_ns"] >= rows[0]["monotonic_ns"]
    assert rows[0]["probe_duration_seconds"] >= 0
    all_text = "".join(path.read_text() for path in collector.directory.iterdir() if path.is_file())
    assert owner.api_key not in all_text and owner._control_key not in all_text
    with pytest.raises(RuntimeError, match="stopped"):
        collector.start()


def test_hung_snapshot_stop_is_bounded_and_reports_pending(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    owner = host(tmp_path)

    def slow():
        entered.set()
        release.wait(3)
        return {"instance_id": MANAGER}

    owner.snapshot = slow
    monkeypatch.setattr(collection, "gpu_observation", lambda: {"devices": []})
    collector = collection.RuntimeCollector(owner, tmp_path / "observations").start()
    assert entered.wait(1)
    started = time.monotonic()
    state = collector.stop(timeout=.03)
    assert time.monotonic() - started < .5
    assert state["phase"] == "stop_pending" and state["threads_alive"] >= 1
    with pytest.raises(RuntimeError, match="Stop collector"):
        collector.export_snapshot()
    release.set()
    assert collector.stop(timeout=2)["phase"] == "stopped"


def test_duplicate_output_ownership_is_rejected(tmp_path):
    owner = host(tmp_path)
    collection.RuntimeCollector(owner, tmp_path / "observations")
    with pytest.raises(FileExistsError):
        collection.RuntimeCollector(owner, tmp_path / "observations")


def populate(owner, *, closed=True, gap=False):
    manager_events = [
        {"instance_id": MANAGER, "sequence": 1, "event_id": "1", "type": "manager.created"},
        {"instance_id": MANAGER, "sequence": 2, "event_id": "2", "type": "job.queued", "job_id": JOB},
        {"instance_id": MANAGER, "sequence": 3, "event_id": "3", "type": "manager.shutdown_complete" if closed else "manager.state"},
    ]
    with sqlite3.connect(owner.root / "control.sqlite3") as db:
        db.execute("CREATE TABLE events (sequence INTEGER, record TEXT)")
        db.execute("CREATE TABLE jobs (id TEXT, record TEXT)")
        db.executemany("INSERT INTO events VALUES (?,?)", [(event["sequence"], json.dumps(event)) for event in manager_events])
        db.execute("INSERT INTO jobs VALUES (?,?)", (JOB, json.dumps({"job_id": JOB, "status": "cancelled",
                      "attempt_id": None, "cleanup_complete": True, "parameters": {"prompt": "must not export"}})))
    directory = owner.root / "_model_runtime" / MODEL
    directory.mkdir(parents=True)
    model_events = [
        {"instance_id": MODEL, "sequence": 1, "event_id": "4", "type": "model.loading"},
        {"instance_id": MODEL, "sequence": 3 if gap else 2, "event_id": "5", "type": "model.worker_exited",
         "worker_cleanup_complete": True},
    ]
    (directory / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in model_events))


def receipt():
    return {"instance_id": MANAGER, "state": {"instance_id": MANAGER, "shutdown_complete": True,
        "cleanup_complete": True, "manager_alive": False, "server_alive": False, "model_alive": False, "tunnel_alive": False},
        "owner_observation": {"alive": False, "cleanup_complete": True, "errors": []}}


def test_terminal_events_alone_do_not_prove_complete_lifetime(tmp_path):
    owner = host(tmp_path)
    populate(owner)
    collector = collection.RuntimeCollector(owner, tmp_path / "observations")
    collector.stop()
    result = collector.export_snapshot(queued_cancel_job_ids=[JOB])
    assert not result["collection"]["model_events_complete"]
    assert not result["collection"]["manager_events_complete"]
    assert result["collection"]["model_event_sequence_final"] is None


def test_export_requires_owner_receipt_and_sequences_and_strips_parameters(tmp_path):
    owner = host(tmp_path)
    populate(owner)
    collector = collection.RuntimeCollector(owner, tmp_path / "observations")
    collector.stop()
    result = collector.export_snapshot(queued_cancel_job_ids=[JOB], cleanup_receipt=receipt())
    assert result["collection"]["model_events_complete"] and result["collection"]["manager_events_complete"]
    trace = json.loads((collector.directory / "trace.json").read_text())
    assert trace["jobs"][0]["instance_id"] == MANAGER
    assert "parameters" not in trace["jobs"][0]
    assert trace["queued_cancel_job_ids"] == [JOB]
    assert "platform_release" not in trace
    assert result["public_safe"] is False
    assert not list(collector.directory.glob("*.sqlite3"))


@pytest.mark.parametrize("mutation", ["gap", "alive_owner", "wrong_owner", "not_terminal"])
def test_incomplete_conditions_remain_partial(tmp_path, mutation):
    owner = host(tmp_path)
    populate(owner, gap=mutation == "gap", closed=mutation != "not_terminal")
    final = receipt()
    if mutation == "alive_owner":
        final["owner_observation"]["alive"] = True
    if mutation == "wrong_owner":
        final["instance_id"] = "d" * 32
    collector = collection.RuntimeCollector(owner, tmp_path / "observations")
    collector.stop()
    result = collector.export_snapshot(cleanup_receipt=final)
    assert not (result["collection"]["model_events_complete"] and result["collection"]["manager_events_complete"])


def test_missing_export_sources_record_problems_instead_of_success(tmp_path):
    collector = collection.RuntimeCollector(host(tmp_path), tmp_path / "observations")
    collector.stop()
    result = collector.export_snapshot(cleanup_receipt=receipt())
    assert result["collection"]["problems"]
    assert not result["collection"]["model_events_complete"]


def test_gpu_telemetry_is_bounded_and_unknown_values_are_null(monkeypatch):
    def run(command, **options):
        assert options["timeout"] == 2
        assert command[0] == "nvidia-smi"
        return SimpleNamespace(stdout="2026/11/15 12:00, 0, NVIDIA A100, 100, 81920, N/A, 55.4\n")

    monkeypatch.setattr(collection.subprocess, "run", run)
    result = collection.gpu_observation()
    assert result["synchronizes_cuda"] is False
    assert result["devices"][0]["utilization_gpu_percent"] is None
    assert result["devices"][0]["memory_used_mib"] == 100


def test_owned_path_cannot_escape(tmp_path):
    outside = tmp_path / "private.json"
    outside.write_text("{}")
    base = tmp_path / "base"
    base.mkdir()
    with pytest.raises(ValueError):
        collection.owned_file(base, "../private.json")


def test_source_manifest_hashes_only_exact_code_allowlist(tmp_path):
    (tmp_path / "startup.py").write_text("source")
    (tmp_path / "private-connection.json").write_text("secret")
    (tmp_path / "api_key.py").write_text("secret")
    result = collection.source_identities(tmp_path)
    assert set(result["files"]) == {"startup.py"}
    assert len(result["files"]["startup.py"]["sha256"]) == 64
    assert result["unavailable"]
    assert "secret" not in json.dumps(result)
