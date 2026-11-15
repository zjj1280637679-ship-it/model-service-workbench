"""Public HTTP contracts against real GlobalManager SQLite queue behavior.

Only child-process/GPU transports use the harmless Rig fixtures. Job records,
idempotency, instance projection, cancellation and server serialization are real.
"""
from fastapi.testclient import TestClient
import pytest

import server
from test_global_manager import rig
from test_server import KEY, HEADERS, submit


def bridge(rig, monkeypatch, *, legacy=False):
    manager = rig.manager
    monkeypatch.setenv("H3_INSTANCE_ID", manager.instance_id)
    monkeypatch.setenv("H3_CONTROL_KEY", "b" * 40)
    monkeypatch.delenv("H3_PARENT_PID", raising=False)
    calls = []
    def controller(operation, payload):
        calls.append(operation)
        result = manager.dispatch(operation, payload)
        # Reproduce the deployed older manager omission without fabricating a
        # record or bypassing queue ownership/admission/cancellation.
        if legacy and operation in {"submit", "job", "cancel", "archive"}:
            result = {key: value for key, value in result.items() if key != "instance_id"}
        return result
    return TestClient(server.create_app(controller=controller, api_key=KEY, output_dir=manager.root)), calls


@pytest.mark.parametrize("legacy", [False, True])
def test_queued_query_replay_and_cancel_stay_bound_to_real_manager_instance(rig, monkeypatch, legacy):
    client, calls = bridge(rig, monkeypatch, legacy=legacy)
    expected = rig.manager.instance_id
    with client:
        first = submit(client, headers={**HEADERS, "Idempotency-Key": "real-bridge-job"})
        assert first.status_code == 202
        task = first.json()
        assert task["instance_id"] == expected and task["status"] == "queued"
        assert task["waiting_for"] == "current_observation"
        assert task["condition_summary"]["status"] == "stale"
        assert task["generation_options"]["requested"]["acceleration_lora"] == "none"
        replay = submit(client, headers={**HEADERS, "Idempotency-Key": "real-bridge-job"})
        assert replay.status_code == 202 and replay.json()["id"] == task["id"]
        status = client.get(task["status_url"], headers=HEADERS)
        assert status.status_code == 200 and status.json()["instance_id"] == expected
        # A loading wait is known only after a current, identity-bound probe.
        # Merely having started the process does not establish ready=False.
        rig.manager.refresh()
        observed = client.get(task["status_url"], headers=HEADERS)
        assert observed.status_code == 200
        observed = observed.json()
        assert observed["status"] == "queued" and observed["waiting_for"] == "model_ready"
        assert observed["condition_summary"]["status"] == "current"
        assert observed["condition_summary"]["conditions"]["model_ready"] == "false"
        assert observed["condition_summary"]["model_instance_id"] == rig.manager.model_instance_id
        cancelled = client.post(task["cancel_url"], headers=HEADERS)
        assert cancelled.status_code == 202 and cancelled.json()["status"] == "cancelled"
        assert cancelled.json()["instance_id"] == expected
        record = rig.manager.dispatch("job", {"job_id": task["id"]})
        assert record["instance_id"] == expected and record["status"] == "cancelled"
        assert not rig.executions
        assert rig.manager._db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1
    assert ("status" in calls) is legacy


@pytest.mark.parametrize("legacy", [False, True])
def test_changed_real_manager_instance_cannot_be_relabelled_by_old_api(rig, monkeypatch, legacy):
    client, _ = bridge(rig, monkeypatch, legacy=legacy)
    with client:
        task = submit(client).json()
        old_identity = rig.manager.instance_id
        try:
            rig.manager.instance_id = "0" * 32
            response = client.get(task["status_url"], headers=HEADERS)
            assert response.status_code == 503
            assert "instance_id" not in response.json()
        finally:
            rig.manager.instance_id = old_identity
