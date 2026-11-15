"""Public condition evidence remains available without publishing private context."""
import copy

from fastapi.testclient import TestClient
import pytest

from condition_guard import evaluate_gate
import server
from test_global_manager import rig
from test_server import api, submit, HEADERS, KEY, PRIVATE


def receipt(gate_id="model.stage.gpu", **observations):
    required = server.PUBLIC_GATE_CONDITIONS[gate_id]
    return evaluate_gate(gate_id, required, {**dict.fromkeys(required, True), **observations},
                         {"instance_id": "a" * 32, "job_id": "b" * 32,
                          "attempt_id": "c" * 32, "source_file": "model_container.py",
                          "source_sha256": "d" * 64})


def test_public_terminal_receipts_survive_actual_manager_ack(rig, monkeypatch):
    accepted = rig.submit()
    rig.ready()
    rig.manager.cancel(accepted["job_id"])
    blocked = receipt(predecessor_completed=None, cancel_flag_clear=[True, False])
    blocked["context"].update(instance_id=rig.manager.model_instance_id,
        job_id=accepted["job_id"], attempt_id=rig.active["attempt_id"])
    completions = {"prepare": None, "gpu": None, "encode_validate": None}
    rig.complete(status="failed", error="condition_gate_blocked",
                 condition_gates={"model.stage.gpu": blocked}, stage_completions=completions)
    rig.manager.refresh()
    assert rig.results == []  # Real manager acknowledged and removed the model result.
    monkeypatch.setenv("H3_INSTANCE_ID", rig.manager.instance_id)
    monkeypatch.delenv("H3_PARENT_PID", raising=False)
    app = server.create_app(controller=rig.manager.dispatch, api_key=KEY, output_dir=rig.manager.root)
    with TestClient(app) as client:
        response = client.get("/v1/jobs/" + accepted["job_id"], headers=HEADERS)
        assert response.status_code == 200
        public = response.json()
        assert public["status"] == "failed" and public["error"] == "condition_gate_blocked"
        assert public["cancellation_requested"] is True
        assert public["stage_completions"] == completions
        assert public["condition_gate"]["allowed"] is True
        evidence = public["execution_condition_gates"]["model.stage.gpu"]
        assert evidence["allowed"] is False
        assert evidence["conditions"]["predecessor_completed"] == "unknown"
        assert evidence["conditions"]["cancel_flag_clear"] == "conflict"
        assert evidence["decision_id"] == blocked["decision_id"]
        assert evidence["context"]["attempt_id"] == blocked["context"]["attempt_id"]
        assert client.get("/v1/jobs/" + accepted["job_id"], headers=HEADERS).json() == public
    assert rig.results == []


def test_receipt_projection_drops_private_and_unrecognized_nested_fields(api):
    client, owner, _ = api
    job_id = submit(client).json()["id"]
    private_path = "/private/model/weights/secret.bin"
    evidence = receipt(predecessor_completed=None, cancel_event_clear=[True, False])
    evidence.update(parameters={"prompt": "private-input"}, api_key=KEY, private_path=private_path)
    evidence["conditions"][PRIVATE] = "true"
    evidence["blocked"].extend([private_path, PRIVATE])
    evidence["context"].update(parameters={"references": [private_path]}, path=private_path,
        api_key=KEY, control_key=PRIVATE, source_file=private_path, job_id=private_path,
        stage=private_path, budget_scope=PRIVATE, unknown="private-input")
    owner.records[job_id].update(status="failed", parameters={"secret": private_path},
        execution_condition_gates={"model.stage.gpu": evidence, PRIVATE: evidence},
        stage_completions={"prepare": True, "gpu": None, "encode_validate": False,
                           "private": private_path}, cancellation_requested=True)
    response = client.get("/v1/jobs/" + job_id, headers=HEADERS)
    assert response.status_code == 200
    assert all(secret not in response.text for secret in (KEY, PRIVATE, private_path, "private-input"))
    public = response.json()
    assert "parameters" not in public
    projected = public["execution_condition_gates"]["model.stage.gpu"]
    assert projected["conditions"]["predecessor_completed"] == "unknown"
    assert projected["conditions"]["cancel_event_clear"] == "conflict"
    assert "source_file" not in projected["context"] and "job_id" not in projected["context"]
    assert set(projected["context"]) == {"instance_id", "attempt_id", "source_sha256"}
    assert public["stage_completions"] == {"prepare": True, "gpu": None, "encode_validate": False}


@pytest.mark.parametrize("bad", [None, "true", 1, {"private": PRIVATE}, [True, False]])
def test_malformed_public_completion_and_cancel_do_not_become_true(api, bad):
    client, owner, _ = api
    job_id = submit(client).json()["id"]
    owner.records[job_id].update(stage_completions={"prepare": bad},
                                 cancellation_requested=bad, condition_gate=bad)
    response = client.get("/v1/jobs/" + job_id, headers=HEADERS)
    assert response.status_code == 200 and PRIVATE not in response.text
    public = response.json()
    assert public["cancellation_requested"] is None
    assert public["condition_gate"] is None
    assert public["stage_completions"]["prepare"] is None


def test_public_gate_must_match_its_declared_slot_and_known_schema(api):
    client, owner, _ = api
    job_id = submit(client).json()["id"]
    wrong = receipt()
    owner.records[job_id].update(condition_gate=wrong,
                                 execution_condition_gates={"model.execute": wrong})
    public = client.get("/v1/jobs/" + job_id, headers=HEADERS).json()
    assert public["condition_gate"] is None
    assert public["execution_condition_gates"]["model.execute"] is None


def test_receipts_do_not_bypass_parent_instance_fence(api):
    client, owner, _ = api
    job_id = submit(client).json()["id"]
    owner.records[job_id].update(instance_id="f" * 32,
                                 execution_condition_gates={"model.stage.gpu": receipt()})
    response = client.get("/v1/jobs/" + job_id, headers=HEADERS)
    assert response.status_code == 503
    assert "execution_condition_gates" not in response.json()


def test_public_projection_does_not_mutate_durable_receipt():
    original = receipt(predecessor_completed=None)
    original["context"]["parameters"] = {"private": "retained only in owner"}
    before = copy.deepcopy(original)
    result = server.public_gate_receipt(original, "model.stage.gpu")
    assert original == before
    assert "parameters" not in result["context"]


@pytest.mark.parametrize("state", [None, "unknown", "conflict", "false", 1])
def test_inconsistent_positive_receipt_never_publishes_permission(state):
    original = receipt()
    original["conditions"]["model_loaded"] = state
    result = server.public_gate_receipt(original, "model.stage.gpu")
    assert result["allowed"] is None


def test_public_condition_names_track_owning_runtime_contracts():
    from global_manager import DISPATCH_CONDITIONS
    from model_container import EXECUTE_CONDITIONS, STAGE_CONDITIONS
    assert server.PUBLIC_GATE_CONDITIONS["manager.dispatch"] == DISPATCH_CONDITIONS
    assert server.PUBLIC_GATE_CONDITIONS["model.execute"] == EXECUTE_CONDITIONS
    assert all(server.PUBLIC_GATE_CONDITIONS["model.stage." + stage] == STAGE_CONDITIONS
               for stage in ("prepare", "gpu", "encode_validate"))
