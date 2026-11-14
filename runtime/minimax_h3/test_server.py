"""Public API boundary tests. No weights, CUDA, or production mock mode."""
import builtins
import copy
import hashlib
import json
import os
from pathlib import Path
import threading

import pytest
from fastapi.testclient import TestClient

import server
from control_protocol import ControlError

KEY = "api-test-" + "x" * 40
PRIVATE = "private-test-" + "y" * 40
INSTANCE = "a" * 32
HEADERS = {"Authorization": "Bearer " + KEY}


class TestController:
    __test__ = False

    def __init__(self, root):
        self.root = root
        self.calls = []
        self.records = {}
        self.idempotency = {}
        self.model_ready = False
        self.phase = "preparing"
        self.unavailable = False
        self.drop_submit_response = False
        self.drop_before_accept = False
        self.invalid_submit_response = False
        self.admission_error = None
        self.state = {"instance_id": INSTANCE, "ready": False, "api_ready": True,
                      "model_ready": False, "phase": "preparing"}
        self.lock = threading.RLock()

    def __call__(self, operation, payload):
        with self.lock:
            self.calls.append((operation, copy.deepcopy(payload)))
            if self.unavailable:
                raise OSError("private connection " + PRIVATE)
            if operation == "status":
                return copy.deepcopy(self.state)
            if operation == "events":
                return {"events": [{"sequence": 1, "type": "test.observed"}], "next_cursor": 1}
            if operation == "command":
                return {"command_id": payload["command_id"], "status": "succeeded"}
            if operation == "control":
                return {"command_id": payload.get("command_id", "test-command"), "status": "accepted",
                        "action": payload["action"]}
            if operation == "submit":
                if self.drop_before_accept:
                    raise OSError("connection dropped before acceptance")
                if payload["idempotency_key"] in self.idempotency:
                    job_id, fingerprint = self.idempotency[payload["idempotency_key"]]
                    if fingerprint != payload["fingerprint"]:
                        raise ControlError("Idempotency-Key already used for different input", 409)
                    return copy.deepcopy(self.records[job_id])
                if self.phase in {"failed", "draining"}:
                    raise ControlError("Admission is closed", 409)
                if self.admission_error:
                    raise ControlError("Resource admission rejected", self.admission_error)
                job_id = payload["job_id"]
                self.records[job_id] = {"id": job_id, "status": "queued", "created_at": 1,
                    "started_at": None, "finished_at": None, "error": None,
                    "lifecycle": [{"state": "queued", "observed_at": 1}],
                    "input": payload["parameters"], "api_key": KEY}
                self.idempotency[payload["idempotency_key"]] = (job_id, payload["fingerprint"])
                if self.drop_submit_response:
                    raise OSError("lost committed response")
                if self.invalid_submit_response:
                    return {"status": "queued"}
                return copy.deepcopy(self.records[job_id])
            job_id = payload.get("job_id")
            if job_id not in self.records:
                raise ControlError("Task not found", 404)
            if operation == "cancel":
                record = self.records[job_id]
                record["status"] = "cancelled" if record["status"] == "queued" else "cancel_requested"
                return copy.deepcopy(record)
            if operation == "job":
                return copy.deepcopy(self.records[job_id])
            if operation == "archive":
                record = self.records[job_id]
                receipt = {"provider": "google_drive", **{key: payload[key] for key in ("file_id", "sha256", "bytes")}}
                artifact = record.get("artifact") or {}
                if record["status"] != "succeeded" or any(receipt[key] != artifact.get(key) for key in ("sha256", "bytes")):
                    raise ControlError("Archive does not match a verified result", 409)
                if record.get("archive", receipt) != receipt:
                    raise ControlError("Archive already recorded", 409)
                record["archive"] = receipt
                return copy.deepcopy(record)
            raise ControlError("Unknown operation", 422)


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setenv("H3_INSTANCE_ID", INSTANCE)
    monkeypatch.setenv("H3_CONTROL_KEY", PRIVATE)
    monkeypatch.delenv("H3_PARENT_PID", raising=False)
    controller = TestController(tmp_path)
    app = server.create_app(controller=controller, api_key=KEY, output_dir=tmp_path)
    with TestClient(app) as client:
        yield client, controller, tmp_path


def submit(client, kinds=None, names=None, contents=None, headers=None, **parameters):
    kinds = ["image"] if kinds is None else kinds
    names = names or ["ref.png"]
    contents = contents if contents is not None else [b"reference"] * len(names)
    return client.post("/v1/jobs", headers=headers or HEADERS,
                       data={"prompt": "An astronaut waves", "kinds": json.dumps(kinds), **parameters},
                       files=[("files", (name, data)) for name, data in zip(names, contents)])


def test_private_drive_receipt_is_authenticated_and_bound_to_the_committed_video(api):
    client, owner, _ = api
    job_id = submit(client).json()["id"]
    route = f"/v1/jobs/{job_id}/drive-archive"
    receipt = {"file_id": "Drive-test_ID", "sha256": "a" * 64, "bytes": 42}

    assert client.post(route, json=receipt).status_code == 401
    assert client.post(route, headers=HEADERS, json={**receipt, "file_id": "../invalid"}).status_code == 422
    assert client.post(route, headers=HEADERS, json={**receipt, "bytes": "42"}).status_code == 422
    assert client.post(route, headers=HEADERS, json=receipt).status_code == 409

    owner.records[job_id].update(status="succeeded", artifact={"sha256": receipt["sha256"], "bytes": 42})
    assert client.post(route, headers=HEADERS, json={**receipt, "sha256": "b" * 64}).status_code == 409
    assert "download_url" not in client.get(f"/v1/jobs/{job_id}", headers=HEADERS).json()
    stored = client.post(route, headers=HEADERS, json=receipt)
    assert stored.status_code == 200
    assert next(payload for operation, payload in reversed(owner.calls)
                if operation == "archive")["expected_instance_id"] == INSTANCE
    result = stored.json()
    assert result["file_url"] == f"/v1/jobs/{job_id}/file"
    assert result["download_url"] == "https://drive.google.com/uc?export=download&id=Drive-test_ID"
    assert result["archive"] == {"provider": "google_drive", **receipt,
                                 "download_url": result["download_url"],
                                 "view_url": "https://drive.google.com/file/d/Drive-test_ID/view"}
    assert client.get(f"/v1/jobs/{job_id}", headers=HEADERS).json() == result
    assert client.post(route, headers=HEADERS, json=receipt).json() == result
    assert client.post(route, headers=HEADERS, json={**receipt, "file_id": "Other-file"}).status_code == 409


def test_api_control_and_status_alive_while_model_loading_and_failed(api):
    client, owner, _ = api
    assert client.get("/health/live").json() == {"api_ready": True, "alive": True, "instance_id": INSTANCE}
    for phase in ("preparing", "failed", "stopped"):
        owner.state.update(phase=phase, model_ready=False, ready=False)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["api_ready"] and not response.json()["ready"]
        assert client.get("/v1/system", headers=HEADERS).json()["phase"] == phase
        result = client.post("/v1/control", headers=HEADERS, json={"action": "start_model"})
        assert result.status_code == 202 and result.json()["status"] == "accepted"


def test_health_fences_wrong_manager_instance(api):
    client, owner, _ = api
    owner.state.update(instance_id="b" * 32, ready=True, model_ready=True)
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["ready"] is False


def test_api_liveness_survives_manager_unavailable(api):
    client, owner, _ = api
    owner.unavailable = True
    assert client.get("/health/live").status_code == 200
    assert client.get("/health").status_code == 503
    response = client.get("/v1/system", headers=HEADERS)
    assert response.status_code == 503 and PRIVATE not in response.text


@pytest.mark.parametrize("method,path", [("get", "/v1/system"), ("get", "/v1/events"),
    ("get", "/v1/commands/op-1"), ("post", "/v1/control"), ("post", "/v1/jobs"), ("get", "/v1/jobs/" + "b" * 32),
    ("post", "/v1/jobs/" + "b" * 32 + "/cancel"), ("get", "/v1/jobs/" + "b" * 32 + "/file")])
def test_all_business_endpoints_authenticate_before_parsing(api, method, path):
    client, owner, _ = api
    response = client.request(method, path, content=b"bad multipart/json", headers={"Content-Type": "multipart/form-data"})
    assert response.status_code == 401
    assert not owner.calls


def test_invalid_api_key_rejected(tmp_path):
    with pytest.raises(RuntimeError):
        server.create_app(controller=TestController(tmp_path), api_key="short", output_dir=tmp_path)


def test_api_does_not_import_or_load_model(api, monkeypatch):
    client, _, _ = api
    original = builtins.__import__
    def protected(name, *args, **kwargs):
        assert name.split(".")[0] not in {"torch", "diffusers", "model_backend", "staged_backend"}
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", protected)
    assert client.get("/health/live").status_code == 200
    assert client.get("/v1/system", headers=HEADERS).status_code == 200
    assert submit(client).status_code == 202


def test_ordered_multimodal_upload_and_queued_before_model_ready(api):
    client, owner, root = api
    response = submit(client, kinds=["video", "image", "audio"], names=["a.mp4", "b.png", "c.wav"],
                      contents=[b"video", b"image", b"audio"], width=512, height=288, steps=8)
    assert response.status_code == 202
    result = response.json()
    assert result["status"] == "queued" and result["actual_seconds"] == 124 / 24
    assert "input" not in result and "api_key" not in result
    references = owner.records[result["id"]]["input"]["references"]
    assert [r["kind"] for r in references] == ["video", "image", "audio"]
    assert [Path(r["path"]).read_bytes() for r in references] == [b"video", b"image", b"audio"]
    assert all(Path(r["path"]).parent == root / result["id"] / "inputs" for r in references)


@pytest.mark.parametrize("parameters", [{"width": 513}, {"height": 287}, {"width": 1344, "height": 1344},
    {"seconds": 15}, {"seconds": "nan"}, {"prompt": "   "}, {"steps": 1}, {"seed": -1}])
def test_invalid_generation_parameters_never_reach_manager(api, parameters):
    client, owner, root = api
    assert submit(client, **parameters).status_code == 422
    assert not owner.records and not list(root.iterdir())


@pytest.mark.parametrize("kinds,names,contents", [(["audio"], ["x.wav"], [b"sound"]),
    (["other"], ["x.png"], [b"x"]), (["image"], ["x.exe"], [b"x"]),
    (["image"], ["x.png"], [b""]), ([], ["x.png"], [b"x"]),
    ({"kind": "image"}, ["x.png"], [b"x"])])
def test_reference_contract_and_failed_upload_cleanup(api, kinds, names, contents):
    client, owner, root = api
    assert submit(client, kinds=kinds, names=names, contents=contents).status_code == 422
    assert not owner.records and not list(root.iterdir())


def test_file_and_total_byte_limits_cleanup(api, monkeypatch):
    client, owner, root = api
    monkeypatch.setattr(server, "MAX_FILE_BYTES", 5)
    assert submit(client, contents=[b"123456"]).status_code == 413
    monkeypatch.setattr(server, "MAX_FILE_BYTES", 10)
    monkeypatch.setattr(server, "MAX_TOTAL_BYTES", 7)
    assert submit(client, kinds=["image", "image"], names=["a.png", "b.png"], contents=[b"1234", b"5678"]).status_code == 413
    assert not owner.records and not list(root.iterdir())


def test_whole_request_and_control_json_size_limits(api, monkeypatch):
    client, owner, root = api
    monkeypatch.setattr(server, "MAX_REQUEST_BYTES", 16)
    assert submit(client).status_code == 413
    monkeypatch.setattr(server, "MAX_CONTROL_BYTES", 16)
    assert client.post("/v1/control", headers=HEADERS, json={"action": "start_model"}).status_code == 413
    assert not owner.calls and not list(root.iterdir())


def test_idempotent_replay_one_job_and_redundant_directory_removed(api):
    client, owner, root = api
    headers = {**HEADERS, "Idempotency-Key": "same-request"}
    one = submit(client, headers=headers)
    two = submit(client, headers=headers, names=["renamed.png"])
    assert one.status_code == two.status_code == 202
    assert one.json()["id"] == two.json()["id"]
    assert len(owner.records) == 1 and len(list(root.iterdir())) == 1
    changed = submit(client, headers=headers, prompt="Different command")
    assert changed.status_code == 409
    assert len(list(root.iterdir())) == 1


def test_content_order_participates_in_idempotency_identity(api):
    client, _, _ = api
    headers = {**HEADERS, "Idempotency-Key": "ordered"}
    options = dict(kinds=["image", "image"], names=["a.png", "b.png"], headers=headers)
    assert submit(client, contents=[b"first", b"second"], **options).status_code == 202
    assert submit(client, contents=[b"second", b"first"], **options).status_code == 409


def test_durable_queue_accepts_more_than_three_tasks(api):
    client, owner, root = api
    for _ in range(12):
        assert submit(client).status_code == 202
    assert len(list(root.iterdir())) == len(owner.records) == 12


def test_response_loss_reconciles_without_repeating_submission(api):
    client, owner, root = api
    owner.drop_submit_response = True
    response = submit(client)
    assert response.status_code == 202
    assert len(owner.records) == 1
    assert len([c for c in owner.calls if c[0] == "submit"]) == 1
    assert (root / response.json()["id"] / "inputs" / "00.png").exists()


def test_unknown_submission_preserves_inputs_and_returns_recovery_identity(api):
    client, owner, root = api
    owner.drop_before_accept = True
    response = submit(client)
    assert response.status_code == 503
    identity = response.json()["detail"]["job_id"]
    assert (root / identity / "submission-uncertain.json").exists()
    assert (root / identity / "inputs" / "00.png").exists()
    assert len([c for c in owner.calls if c[0] == "submit"]) == 1


def test_malformed_committed_reply_never_deletes_owned_inputs(api):
    client, owner, root = api
    owner.invalid_submit_response = True
    assert submit(client).status_code == 503
    assert len(owner.records) == 1
    job_id = next(iter(owner.records))
    assert (root / job_id / "inputs" / "00.png").exists()


def test_job_queries_cancel_and_cursor_forwarding(api):
    client, owner, _ = api
    job_id = submit(client).json()["id"]
    assert client.get(f"/v1/jobs/{job_id}", headers=HEADERS).json()["status"] == "queued"
    response = client.post(f"/v1/jobs/{job_id}/cancel", headers=HEADERS)
    assert response.status_code == 202 and response.json()["status"] == "cancelled"
    owner.records[job_id]["status"] = "running"
    assert client.post(f"/v1/jobs/{job_id}/cancel", headers=HEADERS).json()["status"] == "cancel_requested"
    assert client.get("/v1/events?after=7&limit=20", headers=HEADERS).json()["next_cursor"] == 1
    assert owner.calls[-1] == ("events", {"after": 7, "limit": 20})
    assert client.get("/v1/events?after=-1", headers=HEADERS).status_code == 422
    assert client.get("/v1/events?limit=1001", headers=HEADERS).status_code == 422


@pytest.mark.parametrize("body", [{"action": "shell"}, {"action": "stop_model", "force": "true"},
    {"action": "stop_model", "extra": 1}, {"action": "resume", "command_id": "has space"}])
def test_control_validation_rejects_unsupported_commands(api, body):
    client, owner, _ = api
    assert client.post("/v1/control", headers=HEADERS, json=body).status_code == 422
    assert not owner.calls


def test_control_ack_is_not_completion(api):
    client, owner, _ = api
    request = {"action": "stop_model", "force": True, "command_id": "op-1"}
    response = client.post("/v1/control", headers=HEADERS, json=request)
    assert response.status_code == 202 and response.json()["status"] == "accepted"
    assert owner.calls[-1] == ("control", {**request, "expected_instance_id": INSTANCE})
    assert client.get("/v1/commands/op-1", headers=HEADERS).json()["status"] == "succeeded"


def ready_file(client, owner, root):
    job_id = submit(client).json()["id"]
    path = root / job_id / "result.mp4"
    content = b"API fixture bytes; actual media decode belongs to the model container"
    path.write_bytes(content)
    owner.records[job_id].update(status="succeeded", artifact={"path": str(path), "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(), "media": {"full_decode_passed": True}})
    return job_id, path, content


def test_only_committed_matching_file_is_served_and_private_path_hidden(api):
    client, owner, root = api
    job_id, _, content = ready_file(client, owner, root)
    status = client.get(f"/v1/jobs/{job_id}", headers=HEADERS).json()
    assert "path" not in status["artifact"] and "input" not in status
    response = client.get(status["file_url"], headers=HEADERS)
    assert response.status_code == 200 and response.content == content
    assert response.headers["etag"] == '"' + hashlib.sha256(content).hexdigest() + '"'
    assert client.get(status["file_url"]).status_code == 401


def test_file_manifest_drift_rejected(api):
    client, owner, root = api
    job_id, path, original = ready_file(client, owner, root)
    path.write_bytes(b"different" + original)
    assert client.get(f"/v1/jobs/{job_id}/file", headers=HEADERS).status_code == 409
    path.write_bytes(b"x" * len(original))
    assert client.get(f"/v1/jobs/{job_id}/file", headers=HEADERS).status_code == 409
    path.unlink()
    assert client.get(f"/v1/jobs/{job_id}/file", headers=HEADERS).status_code == 404


def test_file_cannot_escape_owned_job_directory(api):
    client, owner, root = api
    job_id, _, content = ready_file(client, owner, root)
    outside = root / "private.txt"
    outside.write_bytes(content)
    owner.records[job_id]["artifact"]["path"] = str(outside)
    assert client.get(f"/v1/jobs/{job_id}/file", headers=HEADERS).status_code == 409
    assert client.get("/v1/jobs/not-a-uuid/file", headers=HEADERS).status_code == 404


def test_parent_death_watch_exits_without_model_or_blocking_shutdown():
    import psutil
    class Immediate:
        def wait(self, _):
            return False
    def missing(_):
        raise psutil.NoSuchProcess(1234)
    exited = []
    server._watch_parent(1234, Immediate(), exited.append, missing)
    assert exited == [70]


def test_parent_watch_allows_redirector_but_rejects_pid_reuse():
    class StopAfterOne:
        count = 0
        def wait(self, _):
            self.count += 1
            return self.count > 1
    class Process:
        def __init__(self, created):
            self.created = created
        def is_running(self):
            return True
        def status(self):
            return "running"
        def create_time(self):
            return self.created
    exited = []
    server._watch_parent(1234, StopAfterOne(), exited.append, lambda _: Process(3.0), expected_creation=3.0)
    assert not exited
    server._watch_parent(1234, StopAfterOne(), exited.append, lambda _: Process(4.0), expected_creation=3.0)
    assert exited == [70]


def test_api_lifecycle_commands_keep_client_recovery_identity(api):
    client, owner, _ = api
    assert client.post("/v1/control", headers=HEADERS, json={"action":"start_api"}).status_code == 202
    for action in ("stop_api", "shutdown"):
        assert client.post("/v1/control", headers=HEADERS, json={"action":action}).status_code == 422
        request = {"action":action,"command_id":"client-"+action}
        response = client.post("/v1/control", headers=HEADERS, json=request)
        assert response.status_code == 202
        assert response.json()["command_id"] == request["command_id"]


def test_state_output_redacts_private_tokens(api):
    client, owner, _ = api
    owner.state.update(control_key=PRIVATE, api_key=KEY, error="failed " + PRIVATE)
    response = client.get("/v1/system", headers=HEADERS)
    assert PRIVATE not in response.text and KEY not in response.text
    assert "control_key" not in response.json()


def test_unknown_uploads_do_not_create_a_hidden_task_count_limit(api):
    client, owner, root = api
    owner.drop_before_accept = True
    for _ in range(6):
        assert submit(client).status_code == 503
    assert len(list(root.iterdir())) == 6
    state = client.get("/v1/system", headers=HEADERS).json()["api_local"]
    assert state["unknown_submissions"] == 6 and state["active_uploads"] == 0
    assert state["durable_submission_count_limit"] is None
    owner.unavailable = True
    response = client.get("/v1/system", headers=HEADERS)
    assert response.status_code == 503
    assert response.json()["api_local"]["unknown_submissions"] == 6


def test_manager_confirmation_releases_only_uncertain_transport_marker(api):
    client, owner, root = api
    owner.invalid_submit_response = True
    response = submit(client)
    job_id = response.json()["detail"]["job_id"]
    marker = root / job_id / "submission-uncertain.json"
    assert marker.exists()
    assert client.get(f"/v1/jobs/{job_id}", headers=HEADERS).status_code == 200
    assert not marker.exists()
    assert (root / job_id / "inputs" / "00.png").exists()


@pytest.mark.parametrize("phase", ["failed", "draining"])
def test_closed_admission_is_definite_and_never_accumulates_unknown_uploads(api, phase):
    client, owner, root = api
    owner.phase = phase
    for _ in range(5):
        assert submit(client).status_code == 409
    assert not list(root.iterdir())
    assert client.get("/v1/system", headers=HEADERS).json()["api_local"]["unknown_submissions"] == 0


def test_manager_disk_rejection_preserves_507_and_removes_unowned_upload(api):
    client, owner, root = api
    owner.admission_error = 507
    assert submit(client).status_code == 507
    assert not list(root.iterdir()) and not owner.records


def test_disk_bytes_reject_upload_before_manager_submission(api, monkeypatch):
    from types import SimpleNamespace
    client, owner, root = api
    monkeypatch.setattr(server.shutil, "disk_usage", lambda _: SimpleNamespace(free=1))
    assert submit(client).status_code == 507
    assert not owner.calls and not list(root.iterdir())


def test_unknown_outbox_does_not_block_future_valid_submission(api):
    client, owner, root = api
    owner.drop_before_accept = True
    for _ in range(5):
        assert submit(client).status_code == 503
    owner.drop_before_accept = False
    accepted = submit(client)
    assert accepted.status_code == 202
    assert len(owner.records) == 1 and len(list(root.iterdir())) == 6


def test_real_manager_queue_cancel_and_closed_admission_boundary(tmp_path, monkeypatch):
    import global_manager
    monkeypatch.setenv("H3_INSTANCE_ID", INSTANCE)
    monkeypatch.delenv("H3_PARENT_PID", raising=False)
    root = tmp_path / "outputs"
    manager = global_manager.GlobalManager(tmp_path, root=root, instance_id=INSTANCE,
        port=19301, api_port=19302, model_port=19303,
        env={"H3_API_KEY": KEY, "H3_CONTROL_KEY": PRIVATE})
    try:
        manager._phase = "preparing"
        app = server.create_app(controller=manager.dispatch, api_key=KEY, output_dir=root)
        with TestClient(app) as client:
            responses = [submit(client) for _ in range(8)]
            assert all(response.status_code == 202 for response in responses)
            ids = [response.json()["id"] for response in responses]
            cancelled = client.post(f"/v1/jobs/{ids[3]}/cancel", headers=HEADERS)
            assert cancelled.status_code == 202 and cancelled.json()["status"] == "cancelled"
            assert not (root / ids[3] / "inputs").exists()
            assert client.get(f"/v1/jobs/{ids[4]}", headers=HEADERS).json()["queue_position"] == 4
            manager._draining = True
            for _ in range(4):
                assert submit(client).status_code == 409
            assert not list(root.glob("*/submission-uncertain.json"))
            assert manager.snapshot()["queue"]["unfinished"] == 7
    finally:
        manager.close()


def test_non_ascii_bearer_is_rejected_without_server_error(api):
    client, owner, _ = api
    response = client.get("/v1/system", headers={b"Authorization": b"Bearer invalid-\xff"})
    assert response.status_code == 401 and not owner.calls
