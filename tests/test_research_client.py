from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest
import requests


spec = importlib.util.spec_from_file_location("research_client", Path(__file__).resolve().parents[1] / "research/run_experiment.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Response:
    def __init__(self, status, body=None):
        self.status_code = status
        self.value = body

    def json(self):
        return self.value

    def close(self):
        pass


class Session:
    def __init__(self, handler):
        self.handler, self.calls = handler, []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return self.handler(method, url, kwargs)


def manifest(tmp_path):
    reference = tmp_path / "new-reference.png"
    reference.write_bytes(b"new-input-identity")
    value = {"jobs": [{"name": name, "prompt": "new scene " + name, "reference_file": reference.name,
                       "seed": index, "disposition": "cancel" if name == "c" else "retain"}
                      for index, name in enumerate(("a", "b", "c"))],
             "parameters": {"seconds": 5, "width": 512, "height": 288, "steps": 5,
                            "acceleration_lora": "lightx2v-ref2va-4step", "attention_backend": "native"}}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def client(tmp_path, handler):
    session = Session(handler)
    c = module.ExperimentClient({"url": "https://private.example", "api_key": "private-key", "instance_id": "instance-1"},
                               tmp_path / "evidence", session=session, timeout_seconds=.2, poll_seconds=.001)
    return c, session


def state_for_submit(c):
    c.state = {"instance_id": "instance-1", "state": "submitting", "jobs": [{"name": "a", "state": "not_submitted",
        "disposition": "retain", "idempotency_key": "stable-key", "job_id": None}], "handled_commands": {}}
    return c.state["jobs"][0]


def test_submit_persists_intent_before_network_and_uses_one_stable_key(tmp_path):
    def handler(method, url, kwargs):
        persisted = json.loads((tmp_path / "evidence/state.json").read_text())
        assert persisted["jobs"][0]["state"] == "submission_started"
        assert kwargs["headers"]["Idempotency-Key"] == "stable-key"
        assert kwargs["data"]["kinds"] == '["image"]'
        assert kwargs["allow_redirects"] is False
        return Response(202, {"id": "job-a", "instance_id": "instance-1", "status": "queued"})
    c, session = client(tmp_path, handler)
    row = state_for_submit(c)
    value, _ = module.validate_manifest(manifest(tmp_path))
    assert c.submit(row, value["jobs"][0], value["parameters"]) == "job-a"
    assert len(session.calls) == 1
    assert row["state"] == "queued"


def test_503_with_id_preserves_queryable_identity_without_retry(tmp_path):
    c, session = client(tmp_path, lambda *_: Response(503, {"detail": {"job_id": "uncertain-job"}}))
    row = state_for_submit(c)
    value, _ = module.validate_manifest(manifest(tmp_path))
    assert c.submit(row, value["jobs"][0], value["parameters"]) == "uncertain-job"
    assert row["state"] == "submission_outcome_unknown"
    assert len(session.calls) == 1


def test_connection_loss_never_blindly_replays_submission(tmp_path):
    def handler(*_):
        raise requests.Timeout("https://private.example key=private-key")
    c, session = client(tmp_path, handler)
    row = state_for_submit(c)
    value, _ = module.validate_manifest(manifest(tmp_path))
    assert c.submit(row, value["jobs"][0], value["parameters"]) is None
    assert row["state"] == "submission_outcome_unknown"
    assert len(session.calls) == 1
    audit = (c.output / "request-audit.jsonl").read_text()
    assert "private.example" not in audit and "private-key" not in audit


def test_wrong_system_instance_prevents_submission(tmp_path):
    c, session = client(tmp_path, lambda *_: Response(200, {"instance_id": "other", "model_ready": False}))
    with pytest.raises(ValueError, match="identity changed"):
        c.run(manifest(tmp_path))
    assert all(method == "GET" for method, _, _ in session.calls)


def test_already_ready_system_does_not_fake_cold_queue_experiment(tmp_path):
    c, session = client(tmp_path, lambda *_: Response(200, {"instance_id": "instance-1", "model_ready": True, "public_api_ready": True}))
    assert c.run(manifest(tmp_path)) == 2
    assert all(method == "GET" for method, _, _ in session.calls)
    assert c.state["reason"] == "cold_public_queue_precondition_not_observed"


def test_resume_never_submits_unknown_or_not_submitted_jobs(tmp_path):
    path = manifest(tmp_path)
    value, fingerprint = module.validate_manifest(path)
    def handler(method, url, kwargs):
        assert method == "GET"
        if "/v1/system" in url:
            return Response(200, {"instance_id": "instance-1", "model_ready": True})
        return Response(200, {"events": [], "next_cursor": 0})
    c, session = client(tmp_path, handler)
    state = {"instance_id": "instance-1", "manifest_sha256": fingerprint, "state": "interrupted",
        "event_cursor": 0, "handled_commands": {}, "jobs": [{"name": job["name"], "disposition": job["disposition"],
            "state": "submission_outcome_unknown", "job_id": None} for job in value["jobs"]]}
    module.atomic_json(c.output / "state.json", state)
    assert c.run(path, resume=True) == 2
    assert all(method == "GET" for method, _, _ in session.calls)


def test_cancel_unknown_response_is_persisted_and_not_replayed(tmp_path):
    c, session = client(tmp_path, lambda *_: Response(503))
    row = state_for_submit(c)
    row.update(job_id="job-a", state="queued")
    directory = c.output / "commands"
    directory.mkdir()
    module.atomic_json(directory / "one.json", {"command_id": "cmd1", "action": "cancel", "job_id": "job-a"})
    c.drain_control()
    c.drain_control()
    assert len(session.calls) == 1
    assert row["cancel_outcome"] == "unknown_reconcile_by_job_id"


def test_mismatched_job_identity_cannot_replace_receipt(tmp_path):
    c, _ = client(tmp_path, lambda *_: None)
    row = state_for_submit(c)
    row.update(job_id="job-a", state="queued")
    with pytest.raises(ValueError, match="Job identity"):
        c.record_job(row, {"id": "job-b", "instance_id": "instance-1", "status": "succeeded"})
    assert not (c.output / "job-job-a.json").exists()


def test_hash_mismatch_cannot_count_as_verified_download(tmp_path):
    c, session = client(tmp_path, lambda *_: None)
    row = state_for_submit(c)
    row.update(job_id="job-a", state="succeeded")
    (c.output / "job-a.mp4").write_bytes(b"wrong data")
    assert c.download(row, {"artifact": {"sha256": "0" * 64}}, {"seconds": 5}) is False
    assert row["download"]["state"] == "hash_or_size_mismatch"
    assert not session.calls


def test_safe_status_never_contains_connection_or_prompt(tmp_path):
    state = {"state": "running", "api_key": "secret", "url": "https://secret.invalid", "jobs": [
        {"name": "a", "job_id": "job-a", "state": "running", "prompt": "private prompt", "idempotency_key": "key"}]}
    rendered = json.dumps(module.safe_status(state))
    for value in ("secret", "private prompt", "idempotency_key"):
        assert value not in rendered


def test_reference_identity_and_parameters_are_required(tmp_path):
    path = manifest(tmp_path)
    value, digest = module.validate_manifest(path)
    assert len(digest) == 64
    assert Path(value["jobs"][0]["reference_file"]).is_absolute()
    raw = json.loads(path.read_text())
    raw["parameters"].pop("steps")
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="Explicit generation"):
        module.validate_manifest(path)
