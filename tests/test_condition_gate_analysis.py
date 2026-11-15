"""Counterexamples for offline gate and actual lease receipt analysis."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ROOT = Path(__file__).resolve().parents[1]
analyzer = load("condition_gate_analysis_test", ROOT / "research/analyze_condition_gates.py")
baseline = load("condition_gate_baseline_fixture", ROOT / "tests/test_research_trace.py")


def fixture(tmp_path):
    trace = baseline.example()
    limits = {"cpu_threads": 4, "ram_bytes": 100, "disk_bytes": 100}
    capacity_version = hashlib.sha256(json.dumps(
        {"limits": limits, "gpu_host_cpu_threads": 1}, sort_keys=True).encode()).hexdigest()
    hashes = {}
    for name in ("condition_guard.py", "global_manager.py", "model_container.py", "pipeline_scheduler.py"):
        path = tmp_path / name
        path.write_text("synthetic receipt source: " + name, encoding="utf-8")
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    trace["runtime_source_identity"] = {"files": {name: {"sha256": digest} for name, digest in hashes.items()}}
    decisions, lease_ids, ledger_pairs = {}, {}, []
    def receipt(gate, job, stamp, decision=None):
        source, required = analyzer.CONTRACTS[gate]
        context = {"job_id": job, "attempt_id": job + "-attempt", "source_file": source, "source_sha256": hashes[source]}
        if gate.startswith("manager."):
            context.update(manager_instance_id="manager-1", model_instance_id="model-1")
        elif gate.startswith("model."):
            context["instance_id"] = "model-1"
        else:
            context.update(budget_instance_id="budget-1", capacity_version=capacity_version, model_instance_id="model-1",
                           reservation_id=job + "-reservation", resource_kind=gate.removeprefix("pipeline."))
        result = {"schema_version": "1.0", "gate_version": "1.0", "gate_id": gate,
                  "decision_id": decision or f"{gate}-{job}-{stamp}", "observed_at": 100 + stamp / 1e9,
                  "monotonic_ns": stamp, "context": context,
                  "conditions": {name: "true" for name in required}, "allowed": True, "blocked": []}
        return result
    def event(kind, job, stamp, **data):
        return {"type": kind, "event_id": f"{kind}-{job}-{stamp}", "instance_id": "model-1", "job_id": job,
                "attempt_id": job + "-attempt" if job else None, "monotonic_ns": stamp,
                "observed_at": 100 + stamp / 1e9, **data}
    rows = []
    for original in trace["model_events"]:
        kind, job, stage = original["type"], original.get("job_id"), original.get("stage")
        stamp = original["monotonic_ns"]
        if kind == "execution.accepted":
            reserve = receipt("pipeline.reserve", job, stamp - 30_000_000)
            rows.append(event("reservation.acquired", job, stamp - 20_000_000, reservation_receipt=reserve,
                reservation_witness={"owned": True, "budget_instance_id": "budget-1", "attempt_id": job + "-attempt", "reservation_id": job + "-reservation", "capacity_version": capacity_version}))
            admission = receipt("model.execute", job, stamp - 10_000_000)
            admission["context"].update(budget_instance_id="budget-1", reservation_id=job + "-reservation")
            rows.append(event("condition.gate", job, stamp - 5_000_000, receipt=admission))
            original["condition_decision_id"] = admission["decision_id"]
        elif kind == "stage.started":
            resource = "gpu" if stage == "gpu" else "cpu"
            grant = receipt("pipeline.stage." + resource, job, stamp - 40_000_000)
            lease = {"owned": True, "lease_id": f"lease-{job}-{stage}", "attempt_id": job + "-attempt", "kind": resource,
                "budget_instance_id": "budget-1", "reservation_id": job + "-reservation", "cpu_threads": 1,
                "capacity_version": capacity_version, "decision_id": grant["decision_id"], "grant_receipt": grant,
                "granted_at": 100 + (stamp - 30_000_000) / 1e9, "granted_monotonic_ns": stamp - 30_000_000}
            lease_ids[(job, stage)] = lease
            ledger_pairs.append(("lease.acquired", deepcopy(lease), stamp - 29_000_000))
            rows.append(event("lease.acquired", job, stamp - 20_000_000, lease_witness=deepcopy(lease)))
            decision = receipt("model.stage." + stage, job, stamp - 10_000_000)
            decision["context"].update(stage=stage, lease_id=lease["lease_id"], budget_instance_id="budget-1", budget_decision_id=lease["decision_id"])
            decisions[(job, stage)] = decision["decision_id"]
            rows.append(event("condition.gate", job, stamp - 5_000_000, receipt=decision))
            original.update(condition_decision_id=decision["decision_id"], lease_id=lease["lease_id"],
                            budget_instance_id="budget-1", lease_witness=deepcopy(lease))
        elif kind == "stage.finished":
            lease = lease_ids[(job, stage)]
            original.update(condition_decision_id=decisions[(job, stage)], lease_id=lease["lease_id"],
                            budget_instance_id="budget-1", lease_witness=deepcopy(lease))
        rows.append(original)
        if kind == "stage.finished":
            release = {**lease, "owned": False, "released_monotonic_ns": stamp + 10_000_000,
                       "released_at": 100 + (stamp + 10_000_000) / 1e9}
            rows.append(event("lease.released", job, stamp + 20_000_000, lease_witness=deepcopy(release)))
            ledger_pairs.append(("lease.released", deepcopy(release), stamp + 11_000_000))
    trace["model_events"] = rows
    manager = []
    for original in trace["manager_events"]:
        stamp = original["sequence"] * 1_000_000_000
        original["monotonic_ns"] = stamp
        if original["type"] == "job.dispatch_accepted":
            job = original["job_id"]
            gate = receipt("manager.dispatch", job, stamp - 20_000_000)
            manager.append({"instance_id": "manager-1", "event_id": "manager-gate-" + job,
                "type": "condition.checked", "job_id": job, "attempt_id": None,
                "monotonic_ns": stamp - 10_000_000, "observed_at": 100, "data": {"receipt": gate}})
            manager.append({"instance_id": "manager-1", "event_id": "dispatch-intent-" + job,
                "type": "job.dispatch_intent", "job_id": job, "attempt_id": job + "-attempt",
                "monotonic_ns": stamp - 5_000_000, "observed_at": 100,
                "data": {"condition_decision_id": gate["decision_id"]}})
        manager.append(original)
    trace["manager_events"] = manager
    ledger, cpu, gpu = [], 0, None
    for kind, lease, stamp in sorted(ledger_pairs, key=lambda value: value[2]):
        cpu += 1 if kind == "lease.acquired" else -1
        if lease["kind"] == "gpu":
            gpu = lease["attempt_id"] if kind == "lease.acquired" else None
        row = {key: lease[key] for key in ("lease_id", "budget_instance_id", "attempt_id", "kind", "cpu_threads", "capacity_version", "decision_id", "reservation_id")}
        row.update(event_id=f"budget-event-{len(ledger)+1}", sequence=len(ledger)+1, type=kind,
                   monotonic_ns=stamp, observed_at=100 + stamp / 1e9, cpu_threads_in_use=cpu,
                   gpu_attempt_id=gpu, limits=deepcopy(limits))
        ledger.append(row)
    budget = trace["system_snapshots"][0]["modules"]["model"]["budget"]
    budget.update(budget_instance_id="budget-1", active_leases=[], capacity_version=capacity_version,
                  gpu_host_cpu_threads=1,
                  lease_history={"events": ledger, "count": len(ledger), "capacity": 512, "dropped": 0, "history_complete": True})
    trace["model_events"][-1]["budget_snapshot"] = deepcopy(budget)
    for stream in ("model", "manager"):
        for sequence, row in enumerate(trace[stream + "_events"], 1):
            row["sequence"] = sequence
        trace["collection"][stream + "_event_sequence_final"] = len(trace[stream + "_events"])
    return trace


def check(trace, tmp_path, name):
    return analyzer.analyze(trace, tmp_path)["checks"][name]


def receipts(trace):
    return [event["receipt"] for event in trace["model_events"] if event["type"] == "condition.gate"]


def test_complete_gated_trace_passes_positive_checks_but_unexercised_rejections_are_unknown(tmp_path):
    result = analyzer.analyze(fixture(tmp_path), tmp_path)
    assert result["counts"] == {"pass": 7, "fail": 0, "unknown": 2}
    assert result["checks"]["unknown_blocked_without_effect"]["observed_blocked_decisions"] == 0
    assert result["checks"]["actual_stage_lease_ownership"]["paired_leases"] == 6
    assert result["checks"]["atomic_budget_ledger"]["observed_leases"] == 6


@pytest.mark.parametrize("state", ["unknown", "conflict", "false"])
def test_allowed_decision_must_not_hide_a_nontrue_condition(tmp_path, state):
    trace = fixture(tmp_path)
    receipts(trace)[0]["conditions"]["model_loaded"] = state
    assert check(trace, tmp_path, "gate_receipt_contract")["status"] == "fail"


def test_empty_conditions_cannot_prove_allowance(tmp_path):
    trace = fixture(tmp_path)
    receipts(trace)[0]["conditions"] = {}
    assert check(trace, tmp_path, "gate_receipt_contract")["status"] == "fail"


@pytest.mark.parametrize("field", ["source_sha256", "attempt_id"])
def test_missing_source_or_attempt_evidence_stays_unknown(tmp_path, field):
    trace = fixture(tmp_path)
    receipts(trace)[0]["context"].pop(field)
    result = analyzer.analyze(trace, tmp_path)
    assert result["counts"]["pass"] < 7
    assert result["checks"]["effects_have_allowed_gate"]["status"] != "pass"


def test_changed_source_file_cannot_match_old_runtime_receipt(tmp_path):
    trace = fixture(tmp_path)
    (tmp_path / "condition_guard.py").write_text("changed", encoding="utf-8")
    assert check(trace, tmp_path, "gate_source_bytes")["status"] == "fail"
    assert check(trace, tmp_path, "effects_have_allowed_gate")["status"] == "unknown"


def test_source_receipt_hash_must_match_manifest(tmp_path):
    trace = fixture(tmp_path)
    receipts(trace)[0]["context"]["source_sha256"] = "0" * 64
    assert check(trace, tmp_path, "gate_receipt_contract")["status"] == "fail"


def test_effect_cannot_reference_another_attempts_gate(tmp_path):
    trace = fixture(tmp_path)
    accepted = [event for event in trace["model_events"] if event["type"] == "execution.accepted"]
    accepted[1]["condition_decision_id"] = accepted[0]["condition_decision_id"]
    assert check(trace, tmp_path, "effects_have_allowed_gate")["status"] == "fail"


def test_effect_before_persisted_gate_is_rejected(tmp_path):
    trace = fixture(tmp_path)
    gate = next(event for event in trace["model_events"] if event["type"] == "condition.gate")
    effect = next(event for event in trace["model_events"] if event["type"] == "execution.accepted")
    gate["sequence"], effect["sequence"] = effect["sequence"], gate["sequence"]
    assert check(trace, tmp_path, "effects_have_allowed_gate")["status"] == "fail"


def add_blocked(trace, state):
    first = next(event for event in trace["manager_events"] if event["type"] == "condition.checked")
    blocked = deepcopy(first)
    blocked["event_id"] = "manager-blocked-" + state
    blocked["monotonic_ns"] -= 1_000_000
    receipt = blocked["data"]["receipt"]
    receipt["decision_id"] = "blocked-" + state
    receipt["monotonic_ns"] -= 1_000_000
    receipt["conditions"]["model_ready"] = state
    receipt.update(allowed=False, blocked=["model_ready"])
    trace["manager_events"].insert(trace["manager_events"].index(first), blocked)
    for sequence, event in enumerate(trace["manager_events"], 1):
        event["sequence"] = sequence
    trace["collection"]["manager_event_sequence_final"] = len(trace["manager_events"])
    return receipt


@pytest.mark.parametrize("state", ["unknown", "conflict"])
def test_blocked_then_new_allowed_decision_is_valid_progress(tmp_path, state):
    trace = fixture(tmp_path)
    add_blocked(trace, state)
    assert check(trace, tmp_path, state + "_blocked_without_effect")["status"] == "pass"
    assert check(trace, tmp_path, "effects_have_allowed_gate")["status"] == "pass"


@pytest.mark.parametrize("state", ["unknown", "conflict"])
def test_same_blocked_decision_cannot_authorize_dispatch(tmp_path, state):
    trace = fixture(tmp_path)
    gate = add_blocked(trace, state)
    effect = next(event for event in trace["manager_events"] if event["type"] == "job.dispatch_intent")
    effect["data"]["condition_decision_id"] = gate["decision_id"]
    assert check(trace, tmp_path, state + "_blocked_without_effect")["status"] == "fail"


def test_partial_stream_does_not_prove_no_blocked_effect(tmp_path):
    trace = fixture(tmp_path)
    add_blocked(trace, "unknown")
    trace["collection"]["manager_events_complete"] = False
    assert check(trace, tmp_path, "unknown_blocked_without_effect")["status"] == "unknown"


def test_stage_after_release_is_not_owned(tmp_path):
    trace = fixture(tmp_path)
    stage = next(event for event in trace["model_events"] if event["type"] == "stage.finished")
    release = next(event for event in trace["model_events"] if event["type"] == "lease.released")
    release["lease_witness"]["released_monotonic_ns"] = stage["monotonic_ns"] - 1
    assert check(trace, tmp_path, "actual_stage_lease_ownership")["status"] == "fail"


def test_stage_cannot_borrow_another_jobs_lease(tmp_path):
    trace = fixture(tmp_path)
    stages = [event for event in trace["model_events"] if event["type"] == "stage.started" and event["stage"] == "gpu"]
    stages[1]["lease_id"] = stages[0]["lease_id"]
    assert check(trace, tmp_path, "actual_stage_lease_ownership")["status"] == "fail"


def test_missing_release_is_unknown_not_assumed_success(tmp_path):
    trace = fixture(tmp_path)
    index = next(index for index, event in enumerate(trace["model_events"]) if event["type"] == "lease.released")
    trace["model_events"].pop(index)
    assert check(trace, tmp_path, "actual_stage_lease_ownership")["status"] == "unknown"


def test_periodic_samples_cannot_replace_final_budget_history(tmp_path):
    trace = fixture(tmp_path)
    trace["model_events"][-1].pop("budget_snapshot")
    assert check(trace, tmp_path, "atomic_budget_ledger")["status"] == "unknown"


def test_truncated_final_budget_history_cannot_prove_lease_coverage(tmp_path):
    trace = fixture(tmp_path)
    history = trace["model_events"][-1]["budget_snapshot"]["lease_history"]
    history.update(history_complete=False, dropped=1)
    history["events"].pop(0)
    assert check(trace, tmp_path, "atomic_budget_ledger")["status"] == "unknown"


def test_final_budget_cannot_keep_active_leases(tmp_path):
    trace = fixture(tmp_path)
    trace["model_events"][-1]["budget_snapshot"]["active_leases"] = [{"lease_id": "orphan"}]
    assert check(trace, tmp_path, "atomic_budget_ledger")["status"] == "fail"


def test_ledger_reported_cpu_count_cannot_disagree_with_real_grants(tmp_path):
    trace = fixture(tmp_path)
    trace["model_events"][-1]["budget_snapshot"]["lease_history"]["events"][0]["cpu_threads_in_use"] = 0
    assert check(trace, tmp_path, "atomic_budget_ledger")["status"] == "fail"


def test_budget_ledger_event_identity_must_be_unique(tmp_path):
    trace = fixture(tmp_path)
    ledger = trace["model_events"][-1]["budget_snapshot"]["lease_history"]["events"]
    ledger[1]["event_id"] = ledger[0]["event_id"]
    assert check(trace, tmp_path, "atomic_budget_ledger")["status"] == "fail"


def test_missing_reservation_event_cannot_be_inferred_from_all_true_gate(tmp_path):
    trace = fixture(tmp_path)
    trace["model_events"] = [event for event in trace["model_events"] if event["type"] != "reservation.acquired"]
    for sequence, event in enumerate(trace["model_events"], 1):
        event["sequence"] = sequence
    trace["collection"]["model_event_sequence_final"] = len(trace["model_events"])
    assert check(trace, tmp_path, "effects_have_allowed_gate")["status"] == "unknown"


def test_admission_cannot_borrow_another_reservation_id(tmp_path):
    trace = fixture(tmp_path)
    receipts(trace)[0]["context"]["reservation_id"] = "other-reservation"
    assert check(trace, tmp_path, "effects_have_allowed_gate")["status"] == "fail"


def test_stage_finish_cannot_change_the_gate_identity(tmp_path):
    trace = fixture(tmp_path)
    end = next(event for event in trace["model_events"] if event["type"] == "stage.finished")
    end["condition_decision_id"] = "different-decision"
    assert check(trace, tmp_path, "actual_stage_lease_ownership")["status"] == "fail"


def omit_capacity_versions(value):
    if isinstance(value, dict):
        value.pop("capacity_version", None)
        for item in value.values():
            omit_capacity_versions(item)
    elif isinstance(value, list):
        for item in value:
            omit_capacity_versions(item)


def test_all_missing_capacity_versions_are_unknown_not_equal_evidence(tmp_path):
    trace = fixture(tmp_path)
    omit_capacity_versions(trace)
    report = analyzer.analyze(trace, tmp_path)
    for name in ("gate_receipt_contract", "effects_have_allowed_gate", "actual_stage_lease_ownership", "atomic_budget_ledger"):
        assert report["checks"][name]["status"] == "unknown"


@pytest.mark.parametrize("field", ["capacity_version", "gpu_host_cpu_threads", "limits"])
def test_capacity_payload_must_be_observed_not_assumed(tmp_path, field):
    trace = fixture(tmp_path)
    for snapshot in trace["system_snapshots"]:
        snapshot["modules"]["model"]["budget"].pop(field, None)
    trace["model_events"][-1]["budget_snapshot"].pop(field, None)
    result = check(trace, tmp_path, "atomic_budget_ledger")
    assert result["status"] == "unknown"


@pytest.mark.parametrize("field", ["cpu_threads", "ram_bytes", "disk_bytes"])
def test_ledger_limits_cannot_change_under_same_capacity_version(tmp_path, field):
    trace = fixture(tmp_path)
    ledger = trace["model_events"][-1]["budget_snapshot"]["lease_history"]["events"]
    ledger[0]["limits"][field] *= 2
    result = check(trace, tmp_path, "atomic_budget_ledger")
    assert result["status"] == "fail"
    assert "budget_event_limits_disagree_with_capacity_version" in result["errors"]


@pytest.mark.parametrize("limits", [None, [], {"cpu_threads": 4}, {"cpu_threads": True, "ram_bytes": 100, "disk_bytes": 100}])
def test_missing_or_malformed_ledger_capacity_is_unknown(tmp_path, limits):
    trace = fixture(tmp_path)
    trace["model_events"][-1]["budget_snapshot"]["lease_history"]["events"][0]["limits"] = limits
    result = check(trace, tmp_path, "atomic_budget_ledger")
    assert result["status"] == "unknown"


@pytest.mark.parametrize("field", ["cpu_threads", "gpu_host_cpu_threads"])
def test_snapshot_capacity_cannot_change_under_same_version(tmp_path, field):
    trace = fixture(tmp_path)
    budget = trace["model_events"][-1]["budget_snapshot"]
    if field == "cpu_threads":
        budget["limits"][field] *= 2
    else:
        budget[field] *= 2
    result = check(trace, tmp_path, "atomic_budget_ledger")
    assert result["status"] == "fail"
    assert "capacity_version_payload_hash_mismatch" in result["errors"]


def test_self_consistent_opaque_capacity_label_does_not_replace_payload_hash(tmp_path):
    trace = fixture(tmp_path)
    def rewrite(value):
        if isinstance(value, dict):
            if "capacity_version" in value:
                value["capacity_version"] = "f" * 64
            for child in value.values():
                rewrite(child)
        elif isinstance(value, list):
            for child in value:
                rewrite(child)
    rewrite(trace)
    result = check(trace, tmp_path, "atomic_budget_ledger")
    assert result["status"] == "fail"
    assert "capacity_version_payload_hash_mismatch" in result["errors"]


def test_gpu_host_charge_must_match_capacity_payload_even_if_row_is_consistent(tmp_path):
    trace = fixture(tmp_path)
    # The capacity payload is valid in isolation but is incompatible with the
    # unchanged one-lane GPU grants. Rebind every version consistently so the
    # rejection must come from the resource charge, not an opaque label mismatch.
    limits = trace["model_events"][-1]["budget_snapshot"]["limits"]
    replacement = hashlib.sha256(json.dumps(
        {"limits": limits, "gpu_host_cpu_threads": 2}, sort_keys=True).encode()).hexdigest()
    def rewrite(value):
        if isinstance(value, dict):
            if "capacity_version" in value:
                value["capacity_version"] = replacement
            if "gpu_host_cpu_threads" in value:
                value["gpu_host_cpu_threads"] = 2
            for child in value.values():
                rewrite(child)
        elif isinstance(value, list):
            for child in value:
                rewrite(child)
    rewrite(trace)
    result = check(trace, tmp_path, "atomic_budget_ledger")
    assert result["status"] == "fail"
    assert "gpu_lease_cpu_charge_disagrees_with_capacity_version" in result["errors"]


def test_actual_published_trace_still_passes_and_missing_versions_do_not():
    trace = json.loads((ROOT / "evidence/runtime-conditions-2026-11-15/trace.json").read_text(encoding="utf-8"))
    binding = load('condition_source_binding', ROOT / 'research/source_bindings.py')
    source = binding.bound_source_root('research/source-bindings/runtime-conditions-v040.json', trace=trace)
    assert analyzer.analyze(trace, source)["counts"] == {"pass": 7, "fail": 0, "unknown": 2}
    omit_capacity_versions(trace)
    assert analyzer.analyze(trace, source)["checks"]["atomic_budget_ledger"]["status"] == "unknown"


def paginated_fixture(tmp_path):
    trace = fixture(tmp_path)
    budget = trace['model_events'][-1]['budget_snapshot']
    rows = deepcopy(budget['lease_history']['events'])
    previous = '0' * 64
    for row in rows:
        row['previous_sha256'] = previous
        row['sha256'] = previous = hashlib.sha256(json.dumps(row, sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    budget['lease_journal'] = {'schema_version': 1, 'budget_instance_id': 'budget-1',
                              'count': len(rows), 'tail_sha256': previous, 'complete': True, 'error': None}
    budget['lease_history'].update(events=rows[-2:], dropped=len(rows)-2, history_complete=False)
    pages = []
    for after in range(0, len(rows), 4):
        next_seq = min(len(rows), after + 4)
        pages.append({key: deepcopy(budget[key]) for key in
                      ('budget_instance_id', 'capacity_version', 'limits', 'gpu_host_cpu_threads')})
        pages[-1].update(schema_version=1, model_instance_id='model-1', after_sequence=after,
            through_sequence=len(rows), next_sequence=next_seq, after_sha256=rows[after-1]['sha256'] if after else '0'*64,
            tail_sha256=previous, events=rows[after:next_seq], page_complete=True, more=next_seq < len(rows))
    trace['budget_audit_pages'] = pages
    return trace


def test_complete_pages_recover_truncated_tail_and_identical_overlap_is_allowed(tmp_path):
    trace = paginated_fixture(tmp_path)
    assert check(trace, tmp_path, 'atomic_budget_ledger')['status'] == 'pass'
    trace['budget_audit_pages'].append(deepcopy(trace['budget_audit_pages'][0]))
    trace['budget_audit_pages'].reverse()
    assert check(trace, tmp_path, 'atomic_budget_ledger')['status'] == 'pass'


@pytest.mark.parametrize('change', ['missing_middle', 'missing_release', 'no_final', 'incomplete_page'])
def test_incomplete_durable_pages_remain_unknown(tmp_path, change):
    trace = paginated_fixture(tmp_path)
    if change == 'missing_middle':
        trace['budget_audit_pages'].pop(1)
    elif change == 'missing_release':
        trace['budget_audit_pages'].pop()
    elif change == 'no_final':
        trace['model_events'][-1].pop('budget_snapshot')
    else:
        trace['budget_audit_pages'][1]['page_complete'] = False
    assert check(trace, tmp_path, 'atomic_budget_ledger')['status'] == 'unknown'


@pytest.mark.parametrize('change', ['duplicate_conflict', 'hash', 'cutoff', 'model_identity', 'budget_identity', 'final_tail', 'final_count'])
def test_conflicting_durable_pages_fail(tmp_path, change):
    trace = paginated_fixture(tmp_path)
    if change == 'duplicate_conflict':
        trace['budget_audit_pages'].append(deepcopy(trace['budget_audit_pages'][0]))
        trace['budget_audit_pages'][-1]['events'][0]['cpu_threads'] = 99
    elif change == 'hash':
        trace['budget_audit_pages'][1]['events'][0]['previous_sha256'] = 'f' * 64
    elif change == 'cutoff':
        trace['budget_audit_pages'][1]['tail_sha256'] = 'f' * 64
    elif change == 'model_identity':
        trace['budget_audit_pages'][1]['model_instance_id'] = 'other-model'
    elif change == 'budget_identity':
        trace['budget_audit_pages'][1]['budget_instance_id'] = 'other-budget'
    elif change == 'final_tail':
        trace['model_events'][-1]['budget_snapshot']['lease_journal']['tail_sha256'] = 'f'*64
    else:
        trace['model_events'][-1]['budget_snapshot']['lease_journal']['count'] += 1
    assert check(trace, tmp_path, 'atomic_budget_ledger')['status'] == 'fail'
