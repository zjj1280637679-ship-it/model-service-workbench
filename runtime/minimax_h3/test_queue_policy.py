"""Actual manager admission counterexamples; no CUDA or generated media."""
import json
import threading

import pytest

from control_protocol import ControlError
from queue_policy import QueuePolicy, guaranteed_resource_deferral
from test_global_manager import Rig, job


@pytest.fixture
def rig(tmp_path):
    value = Rig(tmp_path).start()
    value.capacity = 100
    try:
        yield value
    finally:
        value.manager.close()


def block(rig, identities, *, status=429, reason="resource_budget_exhausted", guarantee=True):
    original = rig.rpc
    offers = []
    def rpc(url, key, operation, payload, timeout):
        if operation == "execute":
            offers.append(payload.copy())
            if payload["job_id"] in identities:
                raise ControlError(json.dumps({"guarantees_not_accepted": guarantee, "reason": reason}), status)
        return original(url, key, operation, payload, timeout)
    rig.manager._rpc = rpc
    return offers


def refresh_until(rig, predicate):
    for _ in range(12):
        rig.manager.refresh()
        if predicate():
            return
    assert predicate()


def test_known_resource_block_allows_only_two_successful_overtakes_and_head_recovers(rig):
    items = [rig.submit() for _ in range(5)]
    blocked = {items[0]["job_id"]}
    offers = block(rig, blocked)
    rig.ready()
    refresh_until(rig, lambda: len(rig.executions) == 2)
    assert [p["job_id"] for p in rig.executions] == [i["job_id"] for i in items[1:3]]
    for _ in range(3):
        rig.manager.refresh()
    assert len(rig.executions) == 2
    assert job(rig, items[0]["job_id"])["queue_policy"]["bypasses"] == 2
    blocked.clear()
    refresh_until(rig, lambda: len(rig.executions) == 5)
    assert rig.executions[2]["job_id"] == items[0]["job_id"]
    assert len({p["attempt_id"] for p in offers}) == len(offers)


def test_fifo_keeps_order_and_does_not_claim_successor_was_not_runnable(rig):
    rig.manager.queue_policy = QueuePolicy(mode="fifo")
    head, tail = rig.submit(), rig.submit()
    blocked = {head["job_id"]}
    offers = block(rig, blocked)
    rig.ready()
    assert not rig.executions and {p["job_id"] for p in offers} == blocked
    assert job(rig, tail["job_id"])["status"] == "queued"
    blocked.clear()
    refresh_until(rig, lambda: len(rig.executions) == 2)
    assert rig.executions[0]["job_id"] == head["job_id"]


@pytest.mark.parametrize("status,reason", [(409,"model_not_accepting"), (429,"reservation_conditions_unconfirmed"), (429,"live_host_memory_pressure")])
def test_non_resource_or_unknown_conditions_never_enable_bypass(rig, status, reason):
    head, tail = rig.submit(), rig.submit()
    offers = block(rig, {head["job_id"]}, status=status, reason=reason)
    rig.ready()
    assert not rig.executions and [p["job_id"] for p in offers] == [head["job_id"]]
    assert job(rig, tail["job_id"])["status"] == "queued"


def test_age_budget_stops_new_bypasses_without_claiming_finite_execution_time(rig):
    now = [1.0]
    rig.manager._clock = lambda: now[0]
    rig.manager.queue_policy = QueuePolicy(mode="fifo")
    head, tail = rig.submit(), rig.submit()
    block(rig, {head["job_id"]})
    rig.ready()
    rig.manager.queue_policy = QueuePolicy()
    now[0] = 32.0
    rig.manager.refresh()
    assert not rig.executions and job(rig, tail["job_id"])["status"] == "queued"


def test_candidate_window_is_not_a_queue_count_limit(rig):
    rig.manager.queue_policy = QueuePolicy(lookahead=3)
    items = [rig.submit() for _ in range(12)]
    offers = block(rig, {item["job_id"] for item in items[:3]})
    rig.ready()
    assert not rig.executions and len(offers) <= 3
    assert rig.manager.snapshot()["queue"]["queued"] == 12
    assert rig.manager.snapshot()["queue"]["job_count_limit"] is None


def test_reconciled_ack_counts_against_same_bypass_budget(rig):
    items = [rig.submit() for _ in range(4)]
    blocked = {items[0]["job_id"]}
    block(rig, blocked)
    rig.execute_error = True
    rig.ready()
    refresh_until(rig, lambda: len(rig.executions) == 2)
    for _ in range(3):
        rig.manager.refresh()
    assert len(rig.executions) == 2
    assert job(rig, items[0]["job_id"])["queue_policy"]["bypasses"] == 2


@pytest.mark.parametrize("unknown", [None, 1, "true", "conflict"])
def test_rejection_witness_must_be_true(unknown):
    assert not guaranteed_resource_deferral(429, {"guarantees_not_accepted": unknown, "reason": "resource_budget_exhausted"})


@pytest.mark.parametrize("fields", [{"max_bypasses":0},{"max_bypasses":True},{"lookahead":1},{"max_head_age_seconds":float("nan")},{"mode":"unbounded"}])
def test_invalid_policy_is_not_silently_replaced(fields):
    with pytest.raises(ValueError):
        QueuePolicy(**fields)


@pytest.mark.parametrize("first,now,bypasses", [(None,3,0),(1,None,0),(3,1,0),(1,3,None),(1,3,True)])
def test_unknown_or_conflicting_fairness_evidence_disallows_bypass(first, now, bypasses):
    assert not QueuePolicy().may_bypass(first_deferred_at=first,now=now,bypasses=bypasses)


@pytest.mark.parametrize('lost_ack', [False, True])
def test_every_skipped_queued_job_is_charged_even_before_it_becomes_head(rig, lost_ack):
    items = [rig.submit() for _ in range(7)]
    blocked = {items[0]['job_id'], items[1]['job_id']}
    offers = block(rig, blocked)
    rig.execute_error = lost_ack
    rig.ready()
    refresh_until(rig, lambda: len(rig.executions) == 2)
    assert [p['job_id'] for p in rig.executions] == [items[2]['job_id'], items[3]['job_id']]
    assert [job(rig, item['job_id'])['queue_policy']['bypasses'] for item in items[:2]] == [2, 2]
    blocked.remove(items[0]['job_id'])
    refresh_until(rig, lambda: len(rig.executions) == 3)
    for _ in range(3):
        rig.manager.refresh()
    assert [p['job_id'] for p in rig.executions] == [items[i]['job_id'] for i in (2, 3, 0)]
    assert job(rig, items[1]['job_id'])['queue_policy']['bypasses'] == 2
    blocked.clear()
    refresh_until(rig, lambda: len(rig.executions) == 7)
    assert [p['job_id'] for p in rig.executions] == [items[i]['job_id'] for i in (2, 3, 0, 1, 4, 5, 6)]
    assert len({p['attempt_id'] for p in rig.executions}) == 7
    assert len({p['attempt_id'] for p in offers}) == len(offers)


def test_cancel_of_skipped_queued_job_remains_serialized_with_inflight_offer(rig):
    items = [rig.submit() for _ in range(5)]
    blocked = {items[0]['job_id'], items[1]['job_id']}
    block(rig, blocked)
    original = rig.manager._rpc
    offered, resume, cancel_started = threading.Event(), threading.Event(), threading.Event()
    errors = []
    def rpc(url, key, operation, payload, timeout):
        if operation == 'execute' and payload['job_id'] == items[2]['job_id']:
            offered.set()
            if not resume.wait(5):
                raise RuntimeError('test offer barrier expired')
        return original(url, key, operation, payload, timeout)
    def ready():
        try:
            rig.ready()
        except BaseException as error:
            errors.append(error)
    def cancel():
        cancel_started.set()
        try:
            rig.manager.cancel(items[1]['job_id'])
        except BaseException as error:
            errors.append(error)
    rig.manager._rpc = rpc
    dispatcher = threading.Thread(target=ready)
    canceller = threading.Thread(target=cancel)
    dispatcher.start()
    try:
        assert offered.wait(5)
        canceller.start()
        assert cancel_started.wait(5)
        assert rig.manager.snapshot()['queue']['queued'] >= 2  # State query does not await the RPC.
    finally:
        resume.set()
        dispatcher.join(5)
        if canceller.ident is not None:
            canceller.join(5)
    assert not dispatcher.is_alive() and not canceller.is_alive() and not errors
    assert job(rig, items[1]['job_id'])['status'] == 'cancelled'
    assert items[1]['job_id'] not in {p['job_id'] for p in rig.executions}
    assert job(rig, items[0]['job_id'])['queue_policy']['bypasses'] <= 2


@pytest.mark.parametrize('middle', [None, {}, {'bypasses': True, 'first_deferred_at': 1},
    {'bypasses': 2, 'first_deferred_at': 1}, {'bypasses': 0, 'first_deferred_at': None}])
def test_head_budget_cannot_substitute_for_unknown_or_exhausted_middle_job(middle):
    assert not QueuePolicy().may_bypass_all([{'bypasses': 0, 'first_deferred_at': 1}, middle], now=2)
