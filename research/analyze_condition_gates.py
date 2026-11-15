"""Audit recorded condition gates without changing the historical trace checks.

Consumes the existing collected trace plus condition/lease events. Optional
source_root names the exact runtime source tree used by this run (defaults to the
repository reference). A source hash identifies bytes, not authentic execution.
Only complete, identity-bound streams can support absence claims. Unknown or
conflict rejection requires an actual recorded witness; an unexercised path is
unknown. A blocked decision does not prohibit a later, newly allowed decision.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re


STATES = {"true", "false", "unknown", "conflict"}
MANAGER_REQUIRED = ("manager_open", "job_queued", "attempt_unassigned", "model_identity_current",
    "model_observation_fresh", "model_process_alive", "model_ready", "worker_alive", "worker_accepting", "supervisor_alive")
EXECUTE_REQUIRED = ("instance_matches", "request_validated", "model_loaded", "worker_alive", "capability_available",
    "not_draining", "not_stopping", "attempt_available", "job_unowned", "artifact_absent", "generation_options_valid", "resources_reserved")
STAGE_REQUIRED = ("instance_matches", "attempt_owned", "attempt_live", "model_loaded", "worker_alive", "capability_available",
    "admission_confirmed", "predecessor_completed", "stage_not_started", "cancel_flag_clear", "cancel_event_clear",
    "reservation_owned", "budget_grant_held")
CPU_REQUIRED = ("reservation_owned", "cancellation_clear", "not_cancelled", "attempt_stage_idle", "cpu_grant_available")
CONTRACTS = {
    "manager.dispatch": ("global_manager.py", MANAGER_REQUIRED),
    "model.execute": ("model_container.py", EXECUTE_REQUIRED),
    **{"model.stage." + stage: ("model_container.py", STAGE_REQUIRED) for stage in ("prepare", "gpu", "encode_validate")},
    "pipeline.reserve": ("pipeline_scheduler.py", ("estimate_within_limits", "working_set_available", "host_memory_available", "disk_available", "result_identity_unused")),
    "pipeline.reserve.existing": ("pipeline_scheduler.py", ("reservation_matches",)),
    "pipeline.stage.cpu": ("pipeline_scheduler.py", CPU_REQUIRED),
    "pipeline.stage.gpu": ("pipeline_scheduler.py", (*CPU_REQUIRED, "gpu_grant_available")),
}


def _result(status, reason, **extra):
    return {"status": status, "reason": reason, **extra}


def _finite(value):
    return type(value) in (float, int) and math.isfinite(value)


def _identity(value):
    return isinstance(value, str) and bool(re.fullmatch(r"[A-Za-z0-9._:-]{1,256}", value))


def _capacity_version(value):
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))


def _capacity_limits(value):
    return (isinstance(value, dict) and set(value) == {"cpu_threads", "ram_bytes", "disk_bytes"}
            and all(type(item) is int and item > 0 for item in value.values()))


def _nested_receipts(value):
    if isinstance(value, dict):
        if "gate_id" in value and "conditions" in value and "allowed" in value:
            yield value
        for child in value.values():
            yield from _nested_receipts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _nested_receipts(child)


def _base_analysis(trace):
    spec = importlib.util.spec_from_file_location("condition_base_trace", Path(__file__).with_name("analyze_trace.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.analyze(trace)


def collect_budget_ledgers(trace):
    """Reassemble identity-bound audit pages without restoring execution ownership.

    Exact duplicate pages are harmless. Missing evidence is unknown; conflicting
    sequences, hashes, cutoffs or final receipts are errors. Resource semantics
    are deliberately checked separately by ``analyze``.
    """
    ledgers, finals, observations, errors, missing = {}, {}, [], [], []
    model_id, manager_id = trace.get('model_instance_id'), trace.get('manager_instance_id')
    for snapshot in trace.get('system_snapshots', []):
        model = (snapshot.get('modules') or {}).get('model') or {}
        if snapshot.get('instance_id') == manager_id and model.get('instance_id') == model_id:
            budget = model.get('budget') or {}
            if budget:
                observations.append(budget)
            identity, history = budget.get('budget_instance_id'), budget.get('lease_history')
            if _identity(identity) and isinstance(history, dict) and type(history.get('count')) is int:
                if identity not in ledgers or history['count'] >= ledgers[identity]['count']:
                    ledgers[identity] = history
    for event in trace.get('model_events', []):
        if event.get('type') != 'model.worker_exited' or event.get('instance_id') != model_id:
            continue
        budget = event.get('budget_snapshot') or {}
        if budget:
            observations.append(budget)
        identity, history = budget.get('budget_instance_id'), budget.get('lease_history')
        if not _identity(identity) or not isinstance(history, dict) or type(history.get('count')) is not int:
            continue
        if budget.get('active_leases') != []:
            errors.append('final_budget_retains_active_leases')
        if identity in finals and finals[identity] != budget:
            errors.append('conflicting_final_budget_receipts')
        if identity in ledgers and history['count'] < ledgers[identity]['count']:
            errors.append('final_budget_history_regressed')
        finals[identity], ledgers[identity] = budget, history

    by_budget, headers, cuts = defaultdict(dict), {}, {}
    pages = trace.get('budget_audit_pages', [])
    if not isinstance(pages, list):
        missing.append('budget_audit_pages_missing')
        pages = []
    for page in pages:
        if not isinstance(page, dict):
            missing.append('budget_audit_page_missing')
            continue
        identity = page.get('budget_instance_id')
        if page.get('model_instance_id') != model_id or not _identity(identity):
            errors.append('budget_audit_page_identity_mismatch')
            continue
        observations.append(page)
        header = {key: page.get(key) for key in ('schema_version', 'capacity_version', 'limits', 'gpu_host_cpu_threads')}
        if identity in headers and headers[identity] != header:
            errors.append('budget_audit_page_metadata_conflict')
        headers[identity] = header
        after, cutoff, next_seq = (page.get(key) for key in ('after_sequence', 'through_sequence', 'next_sequence'))
        rows = page.get('events')
        if (header['schema_version'] != 1 or any(type(value) is not int for value in (after, cutoff, next_seq))
                or not 0 <= after <= next_seq <= cutoff or not isinstance(rows, list)
                or any(not isinstance(row, dict) for row in rows)):
            missing.append('budget_audit_page_contract_missing')
            continue
        if page.get('page_complete') is not True:
            missing.append('budget_audit_page_incomplete')
            continue
        if (len(rows) > 256 or next_seq - after != len(rows) or (next_seq == after and next_seq < cutoff)
                or page.get('more') is not (next_seq < cutoff)):
            errors.append('budget_audit_page_range_conflict')
            continue
        if not _capacity_version(page.get('after_sha256')) or not _capacity_version(page.get('tail_sha256')):
            missing.append('budget_audit_page_hash_missing')
            continue
        if after == 0 and page['after_sha256'] != '0' * 64:
            errors.append('budget_audit_genesis_conflict')
        for sequence, digest in ((after, page['after_sha256']), (cutoff, page['tail_sha256'])):
            key = identity, sequence
            if key in cuts and cuts[key] != digest:
                errors.append('budget_audit_cutoff_hash_conflict')
            cuts[key] = digest
        previous = page['after_sha256']
        for sequence, row in enumerate(rows, after + 1):
            try:
                digest = hashlib.sha256(json.dumps({key: value for key, value in row.items() if key != 'sha256'},
                    sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
            except (ValueError, TypeError):
                errors.append('budget_audit_row_not_canonical_json')
                continue
            if (row.get('sequence') != sequence or row.get('budget_instance_id') != identity
                    or row.get('previous_sha256') != previous or row.get('sha256') != digest):
                errors.append('budget_audit_row_hash_or_identity_conflict')
            previous = row.get('sha256')
            if sequence in by_budget[identity] and by_budget[identity][sequence] != row:
                errors.append('budget_audit_sequence_conflict')
            by_budget[identity][sequence] = row
        if next_seq == cutoff and previous != page['tail_sha256']:
            errors.append('budget_audit_page_tail_conflict')
    for identity in set(headers) | {key for key, value in finals.items() if 'lease_journal' in value}:
        final = finals.get(identity)
        journal = (final or {}).get('lease_journal') or {}
        if not final or journal.get('complete') is not True or journal.get('error') is not None:
            missing.append('final_budget_journal_unconfirmed')
            continue
        count = journal.get('count')
        if (journal.get('schema_version') != 1 or journal.get('budget_instance_id') != identity
                or type(count) is not int or count < 0 or not _capacity_version(journal.get('tail_sha256'))):
            missing.append('final_budget_journal_contract_missing')
            continue
        if count != final['lease_history']['count']:
            errors.append('final_budget_journal_count_conflict')
        rows_by_seq = by_budget[identity]
        if any(sequence > count for sequence in rows_by_seq):
            errors.append('budget_audit_exceeds_final_cutoff')
        if set(rows_by_seq) != set(range(1, count + 1)):
            missing.append('budget_audit_sequence_gap')
            continue
        rows = [rows_by_seq[index] for index in range(1, count + 1)]
        previous = '0' * 64
        for row in rows:
            if row.get('previous_sha256') != previous:
                errors.append('budget_audit_segment_chain_conflict')
            previous = row.get('sha256')
        if previous != journal['tail_sha256']:
            errors.append('budget_audit_final_tail_conflict')
        for (pool, sequence), digest in cuts.items():
            if pool == identity and (sequence > count or digest != ('0' * 64 if sequence == 0 else rows_by_seq[sequence].get('sha256'))):
                errors.append('budget_audit_cutoff_hash_conflict')
        for row in final['lease_history'].get('events', []):
            if not isinstance(row, dict) or rows_by_seq.get(row.get('sequence')) != row:
                errors.append('budget_audit_final_tail_rows_conflict')
        ledgers[identity] = {'count': count, 'events': rows, 'dropped': 0, 'history_complete': True}
    return {'ledgers': ledgers, 'errors': sorted(set(errors)), 'missing': sorted(set(missing)),
            'final_budget_ids': sorted(finals), 'budget_observations': observations}


def analyze(trace, source_root=None):
    base = _base_analysis(trace)
    complete = base["checks"]["trace_integrity"]["model_complete"] and base["checks"]["trace_integrity"]["manager_complete"]
    source_root = Path(source_root) if source_root is not None else Path(__file__).resolve().parents[1] / "runtime/minimax_h3"
    identities = {"manager": trace.get("manager_instance_id"), "model": trace.get("model_instance_id")}
    manifest = trace.get("runtime_source_identity", {}).get("files", {})
    source_states = {}
    for name in ("condition_guard.py", "global_manager.py", "model_container.py", "pipeline_scheduler.py"):
        digest = manifest.get(name, {}).get("sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            source_states[name] = "unknown"
        else:
            try:
                source_states[name] = "pass" if hashlib.sha256((source_root / name).read_bytes()).hexdigest() == digest else "fail"
            except OSError:
                source_states[name] = "unknown"
    events = {name: sorted(trace.get(name + "_events", []), key=lambda event: event.get("sequence", 0)) for name in identities}
    receipts, origins = {}, defaultdict(list)
    errors, unknown = [], []
    for stream, rows in events.items():
        for event in rows:
            if event.get("type") not in {"condition.checked", "condition.gate", "reservation.acquired", "reservation.blocked",
                    "lease.acquired", "lease.released", "stage.started", "stage.finished"}:
                continue
            for receipt in _nested_receipts(event):
                identity = receipt.get("decision_id")
                if not _identity(identity):
                    unknown.append("gate_decision_identity_missing")
                    continue
                if identity in receipts and receipts[identity] != receipt:
                    errors.append("conflicting_receipts_for_same_decision")
                    continue
                receipts[identity] = receipt
                origins[identity].append((stream, event))

    valid = {}
    for decision, receipt in receipts.items():
        receipt_errors, receipt_unknown = [], []
        gate = receipt.get("gate_id")
        contract = CONTRACTS.get(gate)
        conditions = receipt.get("conditions")
        context = receipt.get("context")
        if not contract or not isinstance(conditions, dict) or not isinstance(context, dict):
            unknown.append("unrecognized_or_incomplete_gate_contract")
            valid[decision] = False
            continue
        source_file, required = contract
        if receipt.get("schema_version") != "1.0" or receipt.get("gate_version") != "1.0":
            receipt_unknown.append("unsupported_gate_version")
        missing = set(required) - set(conditions)
        extra = set(conditions) - set(required)
        if missing:
            (receipt_errors if receipt.get("allowed") is True else receipt_unknown).append("required_condition_missing")
        if extra:
            receipt_unknown.append("undeclared_condition_contract")
        if any(state not in STATES for state in conditions.values() if isinstance(state, str)) or any(not isinstance(state, str) for state in conditions.values()):
            receipt_errors.append("invalid_four_valued_condition")
        blocked = {name for name, state in conditions.items() if state != "true"} | missing
        if type(receipt.get("allowed")) is not bool or receipt["allowed"] != (not blocked):
            receipt_errors.append("allowed_is_not_all_required_true")
        if not isinstance(receipt.get("blocked"), list) or any(not isinstance(value, str) for value in receipt.get("blocked", [])):
            receipt_unknown.append("blocked_condition_identity_missing")
        elif set(receipt["blocked"]) != blocked or len(receipt["blocked"]) != len(set(receipt["blocked"])):
            receipt_errors.append("blocked_conditions_mismatch")
        if context.get("source_file") != source_file:
            receipt_errors.append("gate_source_identity_mismatch")
        expected_sha = manifest.get(source_file, {}).get("sha256")
        if not context.get("source_sha256") or not expected_sha:
            receipt_unknown.append("gate_source_hash_missing")
        elif context["source_sha256"] != expected_sha:
            receipt_errors.append("gate_source_hash_mismatch")
        if source_states[source_file] != "pass" or source_states["condition_guard.py"] != "pass":
            receipt_unknown.append("gate_source_bytes_not_confirmed")
        if not _identity(context.get("attempt_id")):
            receipt_unknown.append("attempt_identity_missing")
        if gate.startswith(("manager.", "model.")) and not _identity(context.get("job_id")):
            receipt_unknown.append("job_identity_missing")
        if not _finite(receipt.get("observed_at")) or type(receipt.get("monotonic_ns")) is not int or receipt["monotonic_ns"] < 0:
            receipt_unknown.append("gate_observation_time_missing")
        for stream, event in origins[decision]:
            if event.get("instance_id") != identities[stream] or not _identity(event.get("event_id")) or type(event.get("sequence")) is not int:
                receipt_errors.append("gate_event_identity_mismatch")
            if event.get("attempt_id") is not None and event["attempt_id"] != context.get("attempt_id"):
                receipt_errors.append("gate_event_attempt_mismatch")
            if type(receipt.get("monotonic_ns")) is int and type(event.get("monotonic_ns")) is int and receipt["monotonic_ns"] > event["monotonic_ns"]:
                receipt_errors.append("gate_timestamp_after_recording_event")
            if context.get("job_id") is not None and event.get("job_id") != context["job_id"]:
                receipt_errors.append("gate_event_job_mismatch")
            if gate.startswith("manager."):
                if stream != "manager" or context.get("manager_instance_id") != identities["manager"]:
                    receipt_errors.append("manager_gate_instance_mismatch")
                if receipt.get("allowed") is True and context.get("model_instance_id") != identities["model"]:
                    receipt_errors.append("allowed_manager_gate_model_mismatch")
            elif gate.startswith("model."):
                if stream != "model" or context.get("instance_id") != identities["model"]:
                    receipt_errors.append("model_gate_instance_mismatch")
            elif stream != "model" or not _identity(context.get("budget_instance_id")):
                receipt_unknown.append("budget_gate_owner_not_bound")
            if gate.startswith("pipeline.") and not _capacity_version(context.get("capacity_version")):
                receipt_unknown.append("budget_gate_capacity_version_missing")
        errors.extend(receipt_errors)
        unknown.extend(receipt_unknown)
        valid[decision] = not receipt_errors and not receipt_unknown

    effects, used = [], set()
    effect_errors, effect_unknown = [], []
    def effect(event, stream, expected_gate, decision, *, lease=None, reservation=False):
        effects.append({"event_id": event.get("event_id"), "gate_id": expected_gate, "decision_id": decision})
        if not _identity(decision) or decision not in receipts:
            effect_unknown.append("effect_has_no_matching_gate_receipt")
            return
        used.add(decision)
        receipt = receipts[decision]
        context = receipt.get("context", {})
        if receipt.get("allowed") is not True:
            effect_errors.append("blocked_decision_produced_effect")
        if receipt.get("gate_id") != expected_gate:
            effect_errors.append("effect_gate_kind_mismatch")
        if context.get("attempt_id") != event.get("attempt_id") or (context.get("job_id") is not None and context["job_id"] != event.get("job_id")):
            effect_errors.append("effect_job_attempt_mismatch")
        if not valid.get(decision):
            effect_unknown.append("effect_receipt_not_fully_verified")
        if type(event.get("monotonic_ns")) is not int or type(receipt.get("monotonic_ns")) is not int:
            effect_unknown.append("effect_clock_evidence_missing")
        elif receipt["monotonic_ns"] > event["monotonic_ns"]:
            effect_errors.append("effect_timestamp_precedes_gate")
        if expected_gate.startswith("model.stage."):
            if context.get("stage") != event.get("stage") or context.get("lease_id") != event.get("lease_id") or context.get("budget_instance_id") != event.get("budget_instance_id"):
                effect_errors.append("stage_gate_resource_identity_mismatch")
        if reservation:
            witness = event.get("reservation_witness") or {}
            if witness.get("owned") is not True or witness.get("attempt_id") != event.get("attempt_id") or witness.get("budget_instance_id") != context.get("budget_instance_id") or not _identity(witness.get("reservation_id")):
                effect_errors.append("reservation_gate_identity_mismatch")
            if not all(_capacity_version(value) for value in (witness.get("capacity_version"), context.get("capacity_version"))):
                effect_unknown.append("reservation_capacity_version_missing")
            elif witness["capacity_version"] != context["capacity_version"]:
                effect_errors.append("reservation_capacity_version_mismatch")
            if type(receipt.get("monotonic_ns")) is not int or type(event.get("monotonic_ns")) is not int:
                effect_unknown.append("reservation_gate_time_missing")
            elif receipt["monotonic_ns"] > event["monotonic_ns"]:
                effect_errors.append("reservation_observed_before_gate")
        elif lease is None:
            prior = any(origin_stream == stream and origin.get("sequence", 0) < event.get("sequence", 0)
                        for origin_stream, origin in origins[decision])
            if not prior:
                effect_errors.append("effect_precedes_durable_gate")
        else:
            if context.get("budget_instance_id") != lease.get("budget_instance_id"):
                effect_errors.append("lease_gate_resource_identity_mismatch")
            if not all(_capacity_version(value) for value in (context.get("capacity_version"), lease.get("capacity_version"))):
                effect_unknown.append("lease_gate_capacity_version_missing")
            elif context["capacity_version"] != lease["capacity_version"]:
                effect_errors.append("lease_gate_resource_identity_mismatch")
            before = receipt.get("monotonic_ns")
            granted = lease.get("granted_monotonic_ns")
            if type(before) is not int or type(granted) is not int:
                effect_unknown.append("lease_gate_time_missing")
            elif before > granted:
                effect_errors.append("lease_granted_before_decision")
    for stream, rows in events.items():
        for event in rows:
            kind = event.get("type")
            if kind == "job.dispatch_intent":
                effect(event, stream, "manager.dispatch", event.get("data", {}).get("condition_decision_id"))
            elif kind == "execution.accepted":
                effect(event, stream, "model.execute", event.get("condition_decision_id"))
            elif kind == "stage.started":
                effect(event, stream, "model.stage." + str(event.get("stage")), event.get("condition_decision_id"))
            elif kind == "lease.acquired":
                witness = event.get("lease_witness")
                if not isinstance(witness, dict):
                    effect_unknown.append("lease_grant_witness_missing")
                else:
                    effect(event, stream, "pipeline.stage." + str(witness.get("kind")), witness.get("decision_id"), lease=witness)
            elif kind == "reservation.acquired":
                receipt = event.get("reservation_receipt") or {}
                effect(event, stream, "pipeline.reserve", receipt.get("decision_id"), reservation=True)

    reservations = {}
    for event in events["model"]:
        if event.get("type") == "reservation.acquired":
            reservations[event.get("attempt_id")] = event
    for event in events["model"]:
        if event.get("type") != "execution.accepted":
            continue
        reservation = reservations.get(event.get("attempt_id"))
        receipt = receipts.get(event.get("condition_decision_id"), {})
        context = receipt.get("context") or {}
        if reservation is None:
            effect_unknown.append("model_admission_reservation_not_recorded")
            continue
        witness = reservation.get("reservation_witness") or {}
        if reservation.get("sequence", 0) >= event.get("sequence", 0) or reservation.get("job_id") != event.get("job_id"):
            effect_errors.append("model_admission_precedes_its_reservation")
        if context.get("budget_instance_id") != witness.get("budget_instance_id") or context.get("reservation_id") != witness.get("reservation_id"):
            effect_errors.append("model_admission_reservation_identity_mismatch")

    leases = defaultdict(dict)
    lease_errors, lease_unknown = [], []
    for event in events["model"]:
        if event.get("type") not in {"lease.acquired", "lease.released"}:
            continue
        witness = event.get("lease_witness")
        if not isinstance(witness, dict) or not _identity(witness.get("lease_id")):
            lease_unknown.append("lease_witness_identity_missing")
            continue
        side = "grant" if event["type"] == "lease.acquired" else "release"
        if side in leases[witness["lease_id"]]:
            lease_errors.append("duplicate_lease_lifecycle_event")
        leases[witness["lease_id"]][side] = (event, witness)
    intervals = []
    for lease_id, sides in leases.items():
        if set(sides) != {"grant", "release"}:
            lease_unknown.append("unpaired_lease_lifecycle")
            continue
        grant_event, grant = sides["grant"]
        release_event, release = sides["release"]
        immutable = ("lease_id", "attempt_id", "budget_instance_id", "reservation_id", "kind", "cpu_threads", "decision_id", "granted_monotonic_ns")
        if any(grant.get(key) != release.get(key) for key in immutable):
            lease_errors.append("lease_identity_changed_before_release")
        if not all(_capacity_version(value) for value in (grant.get("capacity_version"), release.get("capacity_version"))):
            lease_unknown.append("lease_capacity_version_missing")
        elif grant["capacity_version"] != release["capacity_version"]:
            lease_errors.append("lease_capacity_version_changed_before_release")
        if grant.get("owned") is not True or release.get("owned") is not False:
            lease_errors.append("lease_owned_flag_contradiction")
        if grant_event.get("attempt_id") != grant.get("attempt_id") or release_event.get("attempt_id") != grant.get("attempt_id") or grant_event.get("job_id") != release_event.get("job_id"):
            lease_errors.append("lease_event_owner_mismatch")
        reservation = reservations.get(grant.get("attempt_id"))
        if reservation is None:
            lease_unknown.append("lease_reservation_not_recorded")
        elif any((reservation.get("reservation_witness") or {}).get(key) != grant.get(key) for key in ("budget_instance_id", "reservation_id", "attempt_id")):
            lease_errors.append("lease_reservation_identity_mismatch")
        start, end = grant.get("granted_monotonic_ns"), release.get("released_monotonic_ns")
        if type(start) is not int or type(end) is not int or not _identity(grant.get("budget_instance_id")) or type(grant.get("cpu_threads")) is not int:
            lease_unknown.append("lease_time_or_resource_missing")
        elif end < start or grant["cpu_threads"] <= 0:
            lease_errors.append("invalid_lease_interval_or_charge")
        else:
            intervals.append({"lease_id": lease_id, "job_id": grant_event.get("job_id"), "attempt_id": grant["attempt_id"],
                "start": start, "end": end, "kind": grant.get("kind"), "cpu_threads": grant["cpu_threads"],
                "budget_instance_id": grant["budget_instance_id"], "capacity_version": grant.get("capacity_version")})
    starts = {(event.get("attempt_id"), event.get("stage")): event for event in events["model"] if event.get("type") == "stage.started"}
    for event in events["model"]:
        if event.get("type") not in {"stage.started", "stage.finished"}:
            continue
        if event["type"] == "stage.finished":
            start = starts.get((event.get("attempt_id"), event.get("stage")))
            if start is None or event.get("condition_decision_id") is None:
                lease_unknown.append("stage_finish_gate_identity_missing")
            elif start.get("condition_decision_id") != event["condition_decision_id"]:
                lease_errors.append("stage_finish_gate_changed")
        row = next((lease for lease in intervals if lease["lease_id"] == event.get("lease_id")), None)
        if row is None:
            lease_unknown.append("stage_lease_lifecycle_not_proven")
            continue
        expected_kind = "gpu" if event.get("stage") == "gpu" else "cpu"
        if row["kind"] != expected_kind or row["attempt_id"] != event.get("attempt_id") or row["job_id"] != event.get("job_id") or row["budget_instance_id"] != event.get("budget_instance_id"):
            lease_errors.append("stage_lease_identity_mismatch")
        stamp = event.get("monotonic_ns")
        if type(stamp) is not int:
            lease_unknown.append("stage_time_missing")
        elif not row["start"] <= stamp <= row["end"]:
            lease_errors.append("stage_outside_owned_lease_interval")
    gpu = [lease for lease in intervals if lease["kind"] == "gpu"]
    for index, first in enumerate(gpu):
        for second in gpu[index + 1:]:
            # One model instance owns the one physical GPU. Different budget
            # aliases cannot manufacture another physical pool.
            if max(first["start"], second["start"]) < min(first["end"], second["end"]):
                lease_errors.append("exclusive_gpu_lease_overlap")

    collected = collect_budget_ledgers(trace)
    ledgers = collected["ledgers"]
    ledger_errors, ledger_unknown = collected["errors"], collected["missing"]
    # A repeated opaque label is not capacity evidence. Bind each version to the
    # exact payload hashed by PipelineBudget, then reconcile every ledger row.
    capacities = {}
    def capacity_observation(budget):
        identity, version = budget.get("budget_instance_id"), budget.get("capacity_version")
        limits, gpu_cpu = budget.get("limits"), budget.get("gpu_host_cpu_threads")
        if not _identity(identity) or not _capacity_version(version):
            ledger_unknown.append("budget_capacity_identity_missing")
            return
        if not _capacity_limits(limits) or type(gpu_cpu) is not int or gpu_cpu <= 0:
            ledger_unknown.append("budget_capacity_payload_missing")
            return
        payload = {"limits": limits, "gpu_host_cpu_threads": gpu_cpu}
        key = (identity, version)
        if key in capacities and capacities[key] != payload:
            ledger_errors.append("capacity_version_has_conflicting_payloads")
        expected = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        if version != expected:
            ledger_errors.append("capacity_version_payload_hash_mismatch")
            return
        if gpu_cpu > limits["cpu_threads"]:
            ledger_errors.append("gpu_host_cpu_charge_exceeds_capacity")
            return
        capacities[key] = payload

    final_budget_ids = set(collected["final_budget_ids"])
    for budget in collected["budget_observations"]:
        capacity_observation(budget)
    ledger_ids = set()
    peak_cpu = 0
    for budget_id, history in ledgers.items():
        rows = history.get("events")
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            ledger_unknown.append("budget_event_history_missing")
            continue
        ordered = sorted(rows, key=lambda row: row.get("sequence", 0))
        if history.get("history_complete") is not True or history.get("dropped") != 0 or history["count"] != len(rows) or [row.get("sequence") for row in ordered] != list(range(1, len(rows) + 1)):
            ledger_unknown.append("budget_history_incomplete_or_truncated")
            continue
        if any(not _identity(row.get("event_id")) for row in rows) or len({row["event_id"] for row in rows}) != len(rows):
            ledger_errors.append("budget_event_identity_invalid")
            continue
        active, cpu_used, gpu_owner = {}, 0, None
        for row in ordered:
            identity = row.get("lease_id")
            if row.get("budget_instance_id") != budget_id or not _identity(identity):
                ledger_errors.append("budget_event_pool_identity_mismatch")
                continue
            ledger_ids.add(identity)
            recorded = leases.get(identity, {})
            side = "grant" if row.get("type") == "lease.acquired" else "release" if row.get("type") == "lease.released" else None
            if side is None:
                ledger_unknown.append("unsupported_budget_event")
                continue
            if side not in recorded:
                ledger_unknown.append("budget_event_has_no_model_owned_witness")
            elif any(row.get(key) != recorded[side][1].get(key) for key in ("attempt_id", "kind", "cpu_threads", "decision_id", "reservation_id")):
                ledger_errors.append("budget_and_model_witness_disagree")
            version = row.get("capacity_version")
            witness_version = recorded.get(side, ({}, {}))[1].get("capacity_version")
            if not _capacity_version(version) or not _capacity_version(witness_version):
                ledger_unknown.append("budget_event_capacity_version_missing")
            elif version != witness_version:
                ledger_errors.append("budget_and_model_capacity_versions_disagree")
            capacity = capacities.get((budget_id, version)) if _capacity_version(version) else None
            row_limits = row.get("limits")
            if capacity is None:
                ledger_unknown.append("budget_event_capacity_payload_not_observed")
            if not _capacity_limits(row_limits):
                ledger_unknown.append("budget_event_capacity_limits_missing")
            elif capacity is not None and row_limits != capacity["limits"]:
                ledger_errors.append("budget_event_limits_disagree_with_capacity_version")
            if capacity is not None and row.get("kind") == "gpu" and row.get("cpu_threads") != capacity["gpu_host_cpu_threads"]:
                ledger_errors.append("gpu_lease_cpu_charge_disagrees_with_capacity_version")
            charge = row.get("cpu_threads")
            limit = row_limits.get("cpu_threads") if isinstance(row_limits, dict) else None
            if type(charge) is not int or charge < 1:
                ledger_unknown.append("budget_cpu_charge_missing")
                # Subsequent totals cannot be reconstructed after an unknown
                # charge; do not manufacture a release-without-grant failure.
                break
            if type(limit) is not int or limit < 1:
                ledger_unknown.append("budget_cpu_limit_missing")
                limit = None
            if side == "grant":
                if identity in active:
                    ledger_errors.append("budget_duplicate_live_lease")
                active[identity] = row
                cpu_used += charge
                if row.get("kind") == "gpu":
                    if gpu_owner is not None:
                        ledger_errors.append("budget_gpu_owner_overlap")
                    gpu_owner = row.get("attempt_id")
            else:
                prior = active.pop(identity, None)
                if prior is None:
                    ledger_errors.append("budget_release_without_grant")
                cpu_used -= charge
                if row.get("kind") == "gpu":
                    if gpu_owner != row.get("attempt_id"):
                        ledger_errors.append("budget_gpu_release_owner_mismatch")
                    gpu_owner = None
            peak_cpu = max(peak_cpu, cpu_used)
            if cpu_used < 0 or (limit is not None and cpu_used > limit):
                ledger_errors.append("budget_cpu_lease_limit_violated")
            if row.get("cpu_threads_in_use") != cpu_used or row.get("gpu_attempt_id") != gpu_owner:
                ledger_errors.append("budget_reported_ownership_disagrees_with_events")
        if active:
            ledger_unknown.append("budget_lease_release_not_observed")
    if set(leases) != ledger_ids:
        ledger_unknown.append("budget_and_model_lease_coverage_incomplete")
    if not final_budget_ids or any(item["budget_instance_id"] not in final_budget_ids for item in intervals):
        ledger_unknown.append("final_owner_budget_history_missing")
    if len({item["budget_instance_id"] for item in intervals}) > 1:
        ledger_unknown.append("multiple_budget_identities_need_physical_pool_reconciliation")

    checks = {
        "complete_base_trace": _result("pass" if complete else "unknown", "Uses unchanged historical identity/sequence/final-owner checks."),
        "gate_source_bytes": _result("fail" if "fail" in source_states.values() else "pass" if all(value == "pass" for value in source_states.values()) else "unknown",
            "Recorded source hashes compared with the supplied runtime source tree; not an execution attestation.", files=source_states),
        "gate_receipt_contract": _result("fail" if errors else "pass" if receipts and not unknown else "unknown",
            "Known gate contract, required four-valued conditions, strict all-true decision, source, instance and event identities.",
            observed_decisions=len(receipts), errors=sorted(set(errors)), missing=sorted(set(unknown))),
        "effects_have_allowed_gate": _result("fail" if effect_errors else "pass" if effects and complete and not effect_unknown else "unknown",
            "Each observed dispatch, model admission, stage start and budget grant is linked to its own allowed decision before the effect.",
            observed_effects=len(effects), errors=sorted(set(effect_errors)), missing=sorted(set(effect_unknown))),
        "actual_stage_lease_ownership": _result("fail" if lease_errors else "pass" if intervals and complete and not lease_unknown else "unknown",
            "Paired model-persisted lease witnesses cover every recorded stage; GPU leases share one physical model pool. Lease time is not kernel busy time.",
            paired_leases=len(intervals), gpu_leases=len(gpu), errors=sorted(set(lease_errors)), missing=sorted(set(lease_unknown))),
        "atomic_budget_ledger": _result("fail" if ledger_errors else "pass" if ledgers and leases and complete and not ledger_unknown else "unknown",
            "Complete budget-domain event sequence reconciled to model-persisted leases; replayed CPU charges and GPU owner must match each atomic record.",
            budget_instances=len(ledgers), observed_leases=len(ledger_ids), peak_cpu_lease_charge=peak_cpu,
            errors=sorted(set(ledger_errors)), missing=sorted(set(ledger_unknown))),
        "existing_stage_overlap": base["checks"]["cross_task_cpu_gpu_overlap"],
    }
    for state in ("unknown", "conflict"):
        blocked_ids = [identity for identity, receipt in receipts.items() if isinstance(receipt.get("conditions"), dict)
                       and state in receipt["conditions"].values() and receipt.get("allowed") is False]
        violations = sorted(set(blocked_ids) & used)
        evidence = all(valid.get(identity) for identity in blocked_ids)
        checks[state + "_blocked_without_effect"] = _result("fail" if violations else "pass" if blocked_ids and evidence and complete else "unknown",
            "Only this blocked decision is prohibited; a subsequent fresh allowed decision may legitimately progress. Unexercised paths stay unknown.",
            observed_blocked_decisions=len(blocked_ids), effect_violations=violations)
    return {"schema_version": 1, "scope": "Finite auditable gate/lease observations for a single model/GPU instance; no causal completeness, live authorization, universal fault tolerance or throughput optimality proof.",
            "manager_instance_id": identities["manager"], "model_instance_id": identities["model"], "checks": checks,
            "counts": {state: sum(check["status"] == state for check in checks.values()) for state in ("pass", "fail", "unknown")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(json.loads(args.trace.read_text(encoding="utf-8")), args.source_root)
    rendered = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    else:
        print(rendered, end="")
    return 1 if result["counts"]["fail"] else 2 if result["counts"]["unknown"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
