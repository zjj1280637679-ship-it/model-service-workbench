"""Live-owner dispatch evidence, separate from finite Boolean model tests."""
from copy import deepcopy

import pytest

from test_global_manager import Rig


@pytest.fixture
def rig(tmp_path):
    value = Rig(tmp_path).start()
    try:
        yield value
    finally:
        for child in value.manager.children.values():
            if child:
                child.poll_error = child.stubborn = False
        value.manager.close()


def ready_observation(rig):
    rig.ready()
    return deepcopy(rig.manager._model)


@pytest.mark.parametrize("field", ["ready", "worker_alive", "accepting"])
@pytest.mark.parametrize("value", [None, "true", 1])
def test_malformed_observation_cannot_borrow_previous_true(rig, field, value):
    model = ready_observation(rig)
    accepted = rig.submit()
    model[field] = value
    rig.manager._dispatch_next(model)
    receipt = rig.manager.snapshot()["condition_gates"]["manager.dispatch"]
    condition = {"ready": "model_ready", "accepting": "worker_accepting"}.get(field, field)
    assert receipt["conditions"][condition] == "unknown"
    assert receipt["allowed"] is False and rig.executions == []
    assert rig.manager.dispatch("job", {"job_id": accepted["job_id"]})["status"] == "queued"


def test_conflicting_readiness_blocks_then_new_decision_can_dispatch(rig):
    model = ready_observation(rig)
    accepted = rig.submit()
    rig.manager._model["ready"] = False
    rig.manager._dispatch_next(model)
    first = rig.manager.snapshot()["condition_gates"]["manager.dispatch"]
    assert first["conditions"]["model_ready"] == "conflict" and not first["allowed"]
    assert not rig.executions
    rig.manager._model["ready"] = True
    rig.manager._dispatch_next(model)
    second = rig.manager.dispatch("job", {"job_id": accepted["job_id"]})["condition_gate"]
    assert second["allowed"] and second["decision_id"] != first["decision_id"]
    events = rig.manager.dispatch("events", {})["events"]
    dispatches = [event for event in events if event["type"] == "job.dispatch_intent"]
    assert len(dispatches) == 1
    assert dispatches[0]["data"]["condition_decision_id"] == second["decision_id"]
    assert second["context"]["attempt_id"] == rig.executions[0]["attempt_id"]


@pytest.mark.parametrize("missing", [True, False])
def test_missing_or_wrong_model_identity_blocks(rig, missing):
    model = ready_observation(rig)
    rig.submit()
    model["instance_id"] = None if missing else "different-current-instance"
    rig.manager._dispatch_next(model)
    receipt = rig.manager.snapshot()["condition_gates"]["manager.dispatch"]
    assert receipt["conditions"]["model_identity_current"] == ("unknown" if missing else "false")
    assert not rig.executions


def test_expired_or_absent_observation_does_not_dispatch(rig):
    model = ready_observation(rig)
    rig.submit()
    rig.manager._last_observation = None
    rig.manager._dispatch_next(model)
    assert rig.manager.snapshot()["condition_gates"]["manager.dispatch"]["conditions"]["model_observation_fresh"] == "unknown"
    rig.manager._last_observation = rig.manager._clock() - 100
    rig.manager._dispatch_next(model)
    assert rig.manager.snapshot()["condition_gates"]["manager.dispatch"]["conditions"]["model_observation_fresh"] == "false"
    assert not rig.executions


def test_uninspectable_owned_process_is_unknown_not_alive_permission(rig):
    model = ready_observation(rig)
    rig.submit()
    rig.manager.children["model"].poll_error = True
    rig.manager._dispatch_next(model)
    receipt = rig.manager.snapshot()["condition_gates"]["manager.dispatch"]
    assert receipt["conditions"]["model_process_alive"] == "unknown"
    assert not rig.executions


def test_gate_keeps_independent_pipeline_offers_parallel(rig):
    rig.capacity = 2
    first, second = rig.submit(), rig.submit()
    rig.ready()
    assert len(rig.executions) == 2
    assert {execution["job_id"] for execution in rig.executions} == {first["job_id"], second["job_id"]}
    assert rig.manager._model["ready"] is True
    assert all("gpu_idle" not in rig.manager.dispatch("job", {"job_id": item["job_id"]})["condition_gate"]["conditions"]
               for item in (first, second))


def test_loading_queue_records_block_without_claiming_execution(rig):
    accepted = rig.submit()
    rig.manager.refresh()
    receipt = rig.manager.snapshot()["condition_gates"]["manager.dispatch"]
    assert receipt["conditions"]["model_ready"] == "false" and not receipt["allowed"]
    queued = rig.manager.dispatch("job", {"job_id": accepted["job_id"]})
    assert queued["status"] == "queued" and queued["attempt_id"] is None
    assert not rig.executions


def test_cancel_does_not_hide_container_gate_failure_after_result_ack(rig):
    accepted = rig.submit()
    rig.ready()
    rig.manager.cancel(accepted["job_id"])
    condition_gates = {"model.stage.gpu": {"allowed": False, "conditions": {"prior_stage_succeeded": "unknown"}}}
    completions = {"prepare": None, "gpu": None, "encode_validate": None}
    rig.complete(status="failed", error="condition_gate_blocked", condition_gates=condition_gates,
                 stage_completions=completions)
    rig.manager.refresh()
    result = rig.manager.dispatch("job", {"job_id": accepted["job_id"]})
    assert result["status"] == "failed" and result["cancellation_requested"] is True
    assert result["execution_condition_gates"] == condition_gates
    assert result["stage_completions"] == completions
    assert result["error"] == "condition_gate_blocked" and result["artifact"] is None
    assert rig.results == []  # Receipt survived acknowledgement/removal from container status.
