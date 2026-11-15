"""Counterexamples for runtime-owned gates; CPU fixtures never generate media."""
from contextlib import contextmanager
import copy
import json
import threading

import pytest

from control_protocol import ControlError
from model_container import EXECUTE_CONDITIONS, STAGE_CONDITIONS, SOURCE_SHA256
from test_model_container import eventually, request
from test_pipeline_scheduler import pipeline, submit, terminal


def events(model):
    return [json.loads(line) for line in model.events_path.read_text().splitlines()]


def before_stage(monkeypatch, model, name, mutate):
    original = model._pipeline_stage

    @contextmanager
    def wrapped(attempt, stage, kind):
        if stage == name:
            with model._lock:
                mutate(attempt)
        with original(attempt, stage, kind):
            yield

    monkeypatch.setattr(model, "_pipeline_stage", wrapped)


@pytest.mark.parametrize("observation,expected", [
    (None, "unknown"), (1, "unknown"), ("true", "unknown"),
    ([True, False], "conflict"), (False, "false"),
])
def test_unconfirmed_loaded_state_cannot_admit(pipeline, observation, expected):
    model, backend = pipeline()
    payload = request(model)
    with model._lock:
        model._loaded = observation
        with pytest.raises(ControlError):
            model.execute(payload)
        receipt = model.status()["last_condition_gate"]
        assert model.status()["ready"] is False
        assert model.status()["accepting"] is False
        model._loaded = True
    assert receipt["conditions"]["model_loaded"] == expected
    assert receipt["allowed"] is False
    assert payload["attempt_id"] not in model._attempts
    assert not backend.started
    assert model._budget.snapshot()["admitted_attempts"] == 0
    assert events(model)[-1]["receipt"] == receipt


def test_admission_rechecks_drain_after_unlocked_estimation(pipeline, monkeypatch):
    model, backend = pipeline()
    entered, release = threading.Event(), threading.Event()
    estimate = backend.estimate_resources

    def paused(parameters):
        entered.set()
        assert release.wait(3)
        return estimate(parameters)

    monkeypatch.setattr(backend, "estimate_resources", paused)
    payload, replies = request(model), []

    def execute():
        try:
            replies.append(model.execute(payload))
        except ControlError as error:
            replies.append(error)

    thread = threading.Thread(target=execute)
    thread.start()
    assert entered.wait(2)
    model.drain()
    release.set()
    thread.join(3)
    assert not thread.is_alive() and isinstance(replies[0], ControlError)
    assert model.status()["last_condition_gate"]["conditions"]["not_draining"] == "false"
    assert not backend.started and model._budget.snapshot()["admitted_attempts"] == 0


@pytest.mark.parametrize("stage,previous", [("gpu", "prepare"), ("encode_validate", "gpu")])
@pytest.mark.parametrize("observation,expected", [
    (None, "unknown"), (False, "false"), (1, "unknown"), ([True, False], "conflict"),
])
def test_only_confirmed_predecessor_completion_allows_effect(
        pipeline, monkeypatch, stage, previous, observation, expected):
    model, backend = pipeline()
    before_stage(monkeypatch, model, stage,
                 lambda attempt: attempt["stage_completions"].__setitem__(previous, observation))
    payload = submit(model, "predecessor-counterexample")
    result = terminal(model, payload["attempt_id"])
    receipt = result["condition_gates"]["model.stage." + stage]
    assert result["status"] == "failed"
    assert receipt["conditions"]["predecessor_completed"] == expected
    assert receipt["allowed"] is False
    assert ("gpu" if stage == "gpu" else "encode", "predecessor-counterexample") not in backend.started
    assert model._budget.snapshot()["cpu_threads_in_use"] == 0
    assert not model._budget.snapshot()["active_leases"]


@pytest.mark.parametrize("observation,expected", [(None, "unknown"), ([True, False], "conflict")])
def test_unknown_cancel_is_failure_not_confirmed_cancellation(pipeline, monkeypatch, observation, expected):
    model, backend = pipeline()
    before_stage(monkeypatch, model, "gpu",
                 lambda attempt: attempt.__setitem__("cancel_requested", observation))
    payload = submit(model, "cancel-evidence")
    result = terminal(model, payload["attempt_id"])
    assert result["status"] == "failed"
    assert result["error"] == "condition_gate_blocked"
    assert result["condition_gates"]["model.stage.gpu"]["conditions"]["cancel_flag_clear"] == expected
    assert ("gpu", "cancel-evidence") not in backend.started


def test_changed_attempt_instance_blocks_prepare(pipeline, monkeypatch):
    model, backend = pipeline()
    before_stage(monkeypatch, model, "prepare",
                 lambda attempt: attempt.__setitem__("instance_id", "0" * 32))
    payload = submit(model, "foreign-owner")
    result = terminal(model, payload["attempt_id"])
    assert result["status"] == "failed"
    assert result["condition_gates"]["model.stage.prepare"]["conditions"]["instance_matches"] == "false"
    assert not backend.started


@pytest.mark.parametrize("mutation,expected", [("missing", "unknown"), ("unowned", "false"),
                                              ("foreign", "false"), ("conflict", "conflict"),
                                              ("foreign_budget", "false"), ("foreign_reservation", "false")])
def test_grant_without_matching_live_lease_cannot_execute(pipeline, monkeypatch, mutation, expected):
    model, backend = pipeline()
    original = model._budget.lease_witness

    def altered(identity, kind):
        witness = original(identity, kind)
        if kind == "gpu" and witness:
            if mutation == "missing":
                return None
            if mutation == "unowned":
                witness["owned"] = False
            if mutation == "foreign":
                witness["attempt_id"] = "0" * 32
            if mutation == "conflict":
                witness["owned"] = [True, False]
            if mutation == "foreign_budget":
                witness["budget_instance_id"] = "0" * 32
            if mutation == "foreign_reservation":
                witness["reservation_id"] = "0" * 32
        return witness

    monkeypatch.setattr(model._budget, "lease_witness", altered)
    payload = submit(model, "lease-counterexample")
    result = terminal(model, payload["attempt_id"])
    assert result["status"] == "failed"
    assert result["condition_gates"]["model.stage.gpu"]["conditions"]["budget_grant_held"] == expected
    assert ("gpu", "lease-counterexample") not in backend.started
    assert not model._budget.snapshot()["active_leases"]


def test_foreign_reservation_cannot_admit(pipeline, monkeypatch):
    model, backend = pipeline()
    original = model._budget.reservation_witness

    def foreign(identity):
        witness = original(identity)
        witness["attempt_id"] = "0" * 32
        return witness

    monkeypatch.setattr(model._budget, "reservation_witness", foreign)
    payload = request(model)
    with pytest.raises(ControlError):
        model.execute(payload)
    assert model.status()["last_condition_gate"]["conditions"]["resources_reserved"] == "false"
    assert payload["attempt_id"] not in model._attempts
    assert model._budget.snapshot()["admitted_attempts"] == 0
    assert not backend.started


@pytest.mark.parametrize("failure", ["condition.gate", "stage.finished"])
def test_audit_failure_cannot_unlock_gpu(pipeline, monkeypatch, failure):
    model, backend = pipeline()
    original = model._event

    def rejecting(event_type, **details):
        target = (details.get("gate_id") == "model.stage.prepare" if failure == "condition.gate"
                  else details.get("stage") == "prepare")
        if event_type == failure and target:
            raise OSError("Injected audit persistence failure")
        return original(event_type, **details)

    monkeypatch.setattr(model, "_event", rejecting)
    payload = submit(model, "audit-counterexample")
    result = terminal(model, payload["attempt_id"])
    assert result["status"] == "failed"
    assert result["stage_completions"]["prepare"] is None
    assert ("gpu", "audit-counterexample") not in backend.started
    if failure == "condition.gate":
        assert not backend.started
    assert not model._budget.snapshot()["active_leases"]


def test_admission_audit_failure_releases_reservation_without_enqueue(pipeline, monkeypatch):
    model, backend = pipeline()
    original = model._event

    def rejecting(event_type, **details):
        if event_type == "condition.gate" and details.get("gate_id") == "model.execute":
            raise OSError("Injected admission audit failure")
        return original(event_type, **details)

    monkeypatch.setattr(model, "_event", rejecting)
    payload = request(model)
    with pytest.raises(OSError):
        model.execute(payload)
    assert payload["attempt_id"] not in model._attempts
    assert model._budget.snapshot()["admitted_attempts"] == 0
    assert not backend.started


def test_drain_preserves_accepted_work_and_audit_identity(pipeline):
    model, backend = pipeline()
    backend.gate("prepare", "draining-work", blocked=True)
    payload = submit(model, "draining-work")
    assert backend.started["prepare", "draining-work"].wait(2)
    model.drain()
    with pytest.raises(ControlError):
        model.execute(request(model))
    backend.release_all()
    result = terminal(model, payload["attempt_id"])
    assert result["status"] == "succeeded"
    assert all(value is True for value in result["stage_completions"].values())
    selected = [event for event in events(model) if event.get("attempt_id") == payload["attempt_id"]]
    receipts = {event["receipt"]["decision_id"]: event
                for event in selected if event["type"] == "condition.gate"}
    reserve = next(event for event in selected if event["type"] == "reservation.acquired")
    assert reserve["reservation_receipt"]["allowed"] is True
    assert reserve["reservation_witness"]["attempt_id"] == payload["attempt_id"]
    for event in selected:
        if event["type"] not in {"execution.accepted", "stage.started"}:
            continue
        decision = receipts[event["condition_decision_id"]]
        receipt = decision["receipt"]
        assert decision["sequence"] < event["sequence"]
        assert receipt["allowed"] is True
        assert set(receipt["conditions"]) == set(
            EXECUTE_CONDITIONS if event["type"] == "execution.accepted" else STAGE_CONDITIONS)
        assert receipt["context"]["source_sha256"] == SOURCE_SHA256
        assert receipt["context"]["instance_id"] == model.instance_id
        assert receipt["context"]["job_id"] == payload["job_id"]
        assert receipt["context"]["attempt_id"] == payload["attempt_id"]
        if event["type"] == "stage.started":
            lease = event["lease_witness"]
            assert lease["decision_id"] == lease["grant_receipt"]["decision_id"]
            assert lease["grant_receipt"]["allowed"] is True
            assert lease["attempt_id"] == payload["attempt_id"]
            release = next(item for item in selected
                           if item["type"] == "lease.released" and item["lease_id"] == lease["lease_id"])
            assert release["lease_witness"]["owned"] is False
            assert release["lease_witness"]["released_monotonic_ns"] >= lease["granted_monotonic_ns"]
    before = copy.deepcopy(events(model))
    assert model.execute(payload)["result"] == result
    assert events(model) == before


def test_worker_exit_persists_final_budget_history_after_all_releases(pipeline):
    model, backend = pipeline()
    backend.gate("gpu", "history-first", blocked=True)
    first = submit(model, "history-first")
    assert backend.started["gpu", "history-first"].wait(2)
    second = submit(model, "history-second")
    eventually(lambda: ("prepare", "history-second") in backend.started)
    model.stop()
    backend.release_all()
    assert model.join(5)
    final = events(model)[-1]
    assert final["type"] == "model.worker_exited"
    assert final["worker_cleanup_complete"] is True
    snapshot = final["budget_snapshot"]
    assert snapshot["budget_instance_id"] == model._budget.instance_id
    assert snapshot["active_leases"] == []
    assert snapshot["cpu_threads_in_use"] == 0
    assert snapshot["gpu_attempt_id"] is None
    assert snapshot["admitted_attempts"] == 0
    history = snapshot["lease_history"]
    assert history["history_complete"] is True and history["dropped"] == 0
    assert history["count"] == 12
    assert [event["sequence"] for event in history["events"]] == list(range(1, 13))
    assert {event["attempt_id"] for event in history["events"]} == {first["attempt_id"], second["attempt_id"]}
    leases = {event["lease_id"] for event in history["events"]}
    assert len(leases) == 6
    for identity in leases:
        pair = [event for event in history["events"] if event["lease_id"] == identity]
        assert [event["type"] for event in pair] == ["lease.acquired", "lease.released"]
        assert pair[1]["monotonic_ns"] <= final["monotonic_ns"]
    assert all(result["status"] == "succeeded" for result in model.status()["results"])
