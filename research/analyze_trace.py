"""Analyze one observed manager/model lifetime, without importing the GPU runtime.

Input JSON schema (all event/job/snapshot bodies are the runtime's existing JSON):
  schema_version: 1; manager_instance_id: str; model_instance_id: str
  model_events: list; manager_events: list; jobs: list; system_snapshots: list
  collection: {model_events_complete: bool, model_event_sequence_final: int,
               manager_events_complete: bool, manager_event_sequence_final: int}
  queued_cancel_job_ids: list[str]  # declared experiment targets, not inferred
  cleanup_receipt: {instance_id, state, owner_observation}  # optional
  platform_release: {released: bool, source: str, observed_at: number,
                     manager_instance_id: str}  # optional external observation

Stage ordering and interval intersection use the single model stream's sequence.
Durations use monotonic_ns only within that model instance. Wall timestamps and
client polling timestamps are never subtracted. Stages are observed host scopes,
not CUDA kernel timers or proofs of device occupancy. A complete recorded stream
is not independent proof that the producer persisted every physical event.

The tool deliberately does not copy raw inputs into its output: secrets, prompts,
local paths and URLs do not belong in a public research summary. Caller remains
responsible for reviewing IDs and any source text before publishing a report.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path


STAGES = ("prepare", "gpu", "encode_validate")
TERMINAL = {"succeeded", "failed", "cancelled"}


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _check(status, reason, **data):
    return {"status": status, "reason": reason, **data}


def _cleanup_observation(trace, manager_id):
    receipt = trace.get("cleanup_receipt") or {}
    state, owner = receipt.get("state") or {}, receipt.get("owner_observation") or {}
    if not isinstance(state, dict) or not isinstance(owner, dict):
        raise ValueError("cleanup state and owner_observation must be objects")
    matching = receipt.get("instance_id") == manager_id and state.get("instance_id") == manager_id and bool(manager_id)
    live_fields = ("manager_alive", "server_alive", "model_alive", "tunnel_alive")
    pids, identities = owner.get("pids"), owner.get("identities")
    manager_pid = state.get("manager_launcher_pid")
    observed_at = owner.get("observed_at")
    owner_bound = (type(manager_pid) is int and manager_pid > 0 and isinstance(identities, list)
                   and _number(observed_at) and observed_at > 0
                   and any(isinstance(item, dict) and item.get("pid") == manager_pid
                           and _number(item.get("created_at")) and 0 < item["created_at"] <= observed_at for item in identities))
    clean = (state.get("cleanup_complete") is True and state.get("shutdown_complete") is True
             and all(state.get(key) is False for key in live_fields)
             and owner.get("alive") is False and owner.get("cleanup_complete") is True
             and pids == [] and owner.get("errors") == [] and owner_bound)
    contradicted = matching and (state.get("cleanup_complete") is False or owner.get("alive") is True
        or any(state.get(key) is True for key in live_fields) or bool(owner.get("errors"))
        or (isinstance(pids, list) and bool(pids)))
    return _check("fail" if contradicted else "pass" if matching and clean else "unknown",
        "Matching final service state and process-owner exit, bound to manager_launcher_pid and a recorded process generation; not platform deallocation.",
        owner_identity_bound=bool(owner_bound))


def _stream(trace, name, instance, final_owner_confirmed):
    records = trace.get(name + "_events", [])
    if not isinstance(records, list) or any(not isinstance(x, dict) for x in records):
        raise ValueError(name + "_events must be an array of objects")
    errors = []
    valid = []
    seen_sequences, seen_ids = set(), set()
    for item in records:
        sequence = item.get("sequence")
        if not instance or item.get("instance_id") != instance:
            errors.append("instance_identity_mismatch")
            continue
        if type(sequence) is not int or sequence < 1:
            errors.append("invalid_sequence")
            continue
        if sequence in seen_sequences:
            errors.append("duplicate_sequence")
        if not isinstance(item.get("event_id"), str) or not item["event_id"]:
            errors.append("missing_event_identity")
        elif item["event_id"] in seen_ids:
            errors.append("duplicate_event_identity")
        seen_sequences.add(sequence)
        if isinstance(item.get("event_id"), str):
            seen_ids.add(item["event_id"])
        valid.append(item)
    valid.sort(key=lambda x: x["sequence"])
    collection = trace.get("collection", {})
    final = collection.get(name + "_event_sequence_final")
    declared = collection.get(name + "_events_complete") is True
    contiguous = bool(valid) and [x["sequence"] for x in valid] == list(range(1, len(valid) + 1))
    terminal = bool(valid) and valid[-1].get("type") == ("model.worker_exited" if name == "model" else "manager.shutdown_complete")
    if name == "model":
        terminal = terminal and valid[-1].get("worker_cleanup_complete") is True
    complete = (declared and contiguous and type(final) is int and final == len(valid) and not errors
                and terminal and final_owner_confirmed)
    if declared and not complete:
        errors.append("declared_complete_but_sequence_terminal_event_or_owner_receipt_incomplete")
    return valid, complete, sorted(set(errors))


def analyze(trace):
    if not isinstance(trace, dict) or type(trace.get("schema_version")) is not int or trace["schema_version"] != 1:
        raise ValueError("Expected a schema_version=1 trace object")
    for field in ("collection",):
        if field in trace and not isinstance(trace[field], dict):
            raise ValueError(field + " must be an object")
    for field in ("jobs", "system_snapshots", "queued_cancel_job_ids"):
        if field in trace and not isinstance(trace[field], list):
            raise ValueError(field + " must be an array")
    for field in ("cleanup_receipt", "platform_release"):
        if field in trace and not isinstance(trace[field], dict):
            raise ValueError(field + " must be an object")
    if any(not isinstance(value, str) or not value for value in trace.get("queued_cancel_job_ids", [])):
        raise ValueError("queued_cancel_job_ids must contain nonempty strings")
    manager_id, model_id = trace.get("manager_instance_id"), trace.get("model_instance_id")
    final_cleanup = _cleanup_observation(trace, manager_id)
    model, model_complete, model_errors = _stream(trace, "model", model_id, final_cleanup["status"] == "pass")
    manager, manager_complete, manager_errors = _stream(trace, "manager", manager_id, final_cleanup["status"] == "pass")
    checks = {}
    errors = model_errors + manager_errors
    checks["trace_integrity"] = _check("fail" if errors else "pass" if model_complete and manager_complete else "unknown",
        "Recorded identity and contiguous-sequence checks; missing or partial streams cannot prove absence.",
        errors=errors, model_complete=model_complete, manager_complete=manager_complete)

    stages = defaultdict(dict)
    stage_errors = []
    for event in model:
        if event.get("type") not in {"stage.started", "stage.finished"}:
            continue
        identity = event.get("attempt_id")
        stage, job = event.get("stage"), event.get("job_id")
        if not isinstance(identity, str) or not identity or not isinstance(job, str) or not job or stage not in STAGES:
            stage_errors.append("invalid_stage_identity")
            continue
        record = stages[identity].setdefault(stage, {"job_id": job})
        if record["job_id"] != job:
            stage_errors.append("attempt_job_identity_changed")
        side = "start" if event["type"] == "stage.started" else "end"
        if side in record:
            stage_errors.append("duplicate_stage_" + side)
        record[side] = event
    intervals = []
    missing = 0
    for identity, by_stage in stages.items():
        if len({value["job_id"] for value in by_stage.values()}) != 1:
            stage_errors.append("attempt_job_identity_changed")
        for index, stage in enumerate(STAGES):
            item = by_stage.get(stage)
            if item is None:
                continue
            start, end = item.get("start"), item.get("end")
            if not start or not end:
                missing += 1
                if model_complete:
                    stage_errors.append("unpaired_stage_event")
                continue
            if end["sequence"] <= start["sequence"]:
                stage_errors.append("stage_end_before_start")
                continue
            if index:
                previous = by_stage.get(STAGES[index - 1], {}).get("end")
                if previous is None:
                    missing += 1
                    if model_complete:
                        stage_errors.append("missing_predecessor_stage")
                elif previous["sequence"] >= start["sequence"] or previous.get("succeeded") is not True:
                    stage_errors.append("stage_started_without_successful_predecessor")
            intervals.append({"attempt_id": identity, "job_id": item["job_id"], "stage": stage,
                              "start": start, "end": end})
    job_records = {}
    job_errors = []
    for job in trace.get("jobs", []):
        if not isinstance(job, dict):
            raise ValueError("jobs must contain objects")
        identity = job.get("id", job.get("job_id"))
        if not isinstance(identity, str) or not identity or job.get("instance_id") != manager_id:
            job_errors.append("job_instance_identity_mismatch")
            continue
        if identity in job_records:
            job_errors.append("duplicate_final_job_record")
        job_records[identity] = job
        if job.get("status") == "succeeded":
            by_stage = stages.get(job.get("attempt_id"), {})
            if any(value["job_id"] != identity for value in by_stage.values()):
                stage_errors.append("job_receipt_bound_to_other_job_attempt")
            for stage in STAGES:
                finish = by_stage.get(stage, {}).get("end")
                if not finish:
                    missing += 1
                    if model_complete:
                        stage_errors.append("successful_job_missing_stage")
                elif finish.get("succeeded") is not True:
                    stage_errors.append("successful_job_with_failed_stage")
    stage_errors.extend(job_errors)
    checks["stage_causality"] = _check("fail" if stage_errors else "pass" if intervals and model_complete and not missing else "unknown",
        "Each observed attempt must prepare, infer and encode/validate in causal order; failed/cancelled prefixes may stop early.",
        errors=sorted(set(stage_errors)), paired_stages=len(intervals), unproven_pairs_or_predecessors=missing)

    expected_jobs = {event["job_id"] for event in model + manager
                     if isinstance(event.get("job_id"), str) and event["job_id"]}
    model_attempts = {(event["job_id"], event["attempt_id"]) for event in model
                      if isinstance(event.get("job_id"), str) and isinstance(event.get("attempt_id"), str)
                      and (event.get("type", "").startswith("execution.") or event.get("type", "").startswith("stage."))}
    terminal_attempts = {(event.get("job_id"), event.get("attempt_id")) for event in model
                         if event.get("type") in {"execution.succeeded", "execution.failed", "execution.cancelled"}}
    absent_receipts = sorted(expected_jobs - set(job_records))
    receipt_missing_terminal = sorted(identity for identity, job in job_records.items() if job.get("status") not in TERMINAL)
    unreconciled_attempts, identity_conflicts = [], []
    for job_id, attempt_id in sorted(model_attempts):
        receipt = job_records.get(job_id)
        if receipt is None or receipt.get("status") not in TERMINAL or (job_id, attempt_id) not in terminal_attempts:
            unreconciled_attempts.append({"job_id": job_id, "attempt_id": attempt_id})
        elif receipt.get("attempt_id") != attempt_id:
            identity_conflicts.append({"job_id": job_id, "attempt_id": attempt_id})
    checks["terminal_receipt_coverage"] = _check("fail" if identity_conflicts or job_errors else "pass" if expected_jobs and model_complete and manager_complete and not absent_receipts and not receipt_missing_terminal and not unreconciled_attempts else "unknown",
        "All event-observed jobs and model execution attempts must reconcile to terminal job receipts; omitted receipts are unknown, not success.",
        expected_jobs=len(expected_jobs), final_job_receipts=len(job_records), missing_job_receipts=absent_receipts,
        nonterminal_job_receipts=receipt_missing_terminal, unreconciled_attempts=unreconciled_attempts,
        identity_conflicts=identity_conflicts)

    ready = False
    ready_violations, unproven_ready = [], []
    seen_gpu = 0
    for event in model:
        kind = event.get("type")
        if kind == "model.ready":
            ready = True
        elif kind in {"model.loading", "model.failed", "model.worker_exited", "model.stopped"} or (
                kind == "stage.finished" and event.get("stage") == "gpu" and event.get("succeeded") is False):
            ready = False
        elif kind == "stage.started" and event.get("stage") == "gpu":
            seen_gpu += 1
            if not ready:
                (ready_violations if model_complete else unproven_ready).append(event.get("job_id"))
    checks["model_ready_gate"] = _check("fail" if ready_violations else "pass" if seen_gpu and model_complete and not unproven_ready and not model_errors else "unknown",
        "GPU stages require preceding same-instance model.ready without observed capability revocation. Drain/stop requests close admission but do not revoke already-admitted work by themselves.",
        gpu_starts=seen_gpu, violations=ready_violations, unproven=unproven_ready)

    gpu = [item for item in intervals if item["stage"] == "gpu"]
    collisions, overlaps = [], []
    for index, left in enumerate(intervals):
        for right in intervals[index + 1:]:
            if left["attempt_id"] == right["attempt_id"]:
                continue
            intersect = max(left["start"]["sequence"], right["start"]["sequence"]) < min(left["end"]["sequence"], right["end"]["sequence"])
            if not intersect:
                continue
            pair = {"jobs": [left["job_id"], right["job_id"]], "stages": [left["stage"], right["stage"]]}
            if left["stage"] == right["stage"] == "gpu":
                collisions.append(pair)
            elif left["job_id"] != right["job_id"] and (left["stage"] == "gpu") != (right["stage"] == "gpu"):
                overlaps.append(pair)
    usable = not model_errors and not stage_errors and not missing
    checks["gpu_stage_exclusivity"] = _check("fail" if collisions else "pass" if gpu and model_complete and usable else "unknown",
        "Observed GPU stage scopes do not intersect; this is not a CUDA kernel occupancy measurement.",
        observed_gpu_stages=len(gpu), collisions=collisions)
    checks["cross_task_cpu_gpu_overlap"] = _check("pass" if overlaps and usable else "unknown",
        "Positive interval intersection is witnessed by the single model event order. No overlap in a finite workload does not disprove the capability.",
        witnessed_overlaps=overlaps if usable else [])

    mono = [item.get("monotonic_ns") for item in model]
    monotonic_ok = bool(mono) and all(type(x) is int and x >= 0 for x in mono) and all(a <= b for a, b in zip(mono, mono[1:]))
    durations = []
    if monotonic_ok and not model_errors:
        for item in intervals:
            durations.append({"job_id": item["job_id"], "attempt_id": item["attempt_id"], "stage": item["stage"],
                              "seconds": (item["end"]["monotonic_ns"] - item["start"]["monotonic_ns"]) / 1e9})
    checks["stage_durations"] = _check("pass" if durations else "unknown",
        "Same model-instance monotonic host scopes only; wall clocks and different hosts are not subtracted.",
        clock="model_instance_monotonic_ns" if durations else None, stages=durations)

    snapshots = trace.get("system_snapshots", [])
    budgets, invalid_snapshots = [], []
    excluded_snapshots = defaultdict(int)
    unidentified_budget_samples = 0
    for snapshot in snapshots:
        if not isinstance(snapshot, dict):
            raise ValueError("system_snapshots must contain objects")
        modules = snapshot.get("modules") or {}
        if not isinstance(modules, dict):
            raise ValueError("snapshot modules must be an object")
        component = modules.get("model") or {}
        if not isinstance(component, dict):
            raise ValueError("snapshot model component must be an object")
        if snapshot.get("instance_id") != manager_id:
            invalid_snapshots.append("snapshot_instance_identity_mismatch")
            continue
        if snapshot.get("observation_stale") is not False:
            excluded_snapshots["stale_or_freshness_unknown"] += 1
            continue
        budget = component.get("budget")
        component_id = component.get("instance_id")
        # A normal startup/shutdown projection may intentionally contain only
        # alive/ready/observation_unknown. No model identity or budget was
        # observed: this is an excluded measurement, not a conflicting model.
        if component_id is None or component_id == "":
            if isinstance(budget, dict):
                unidentified_budget_samples += 1
                excluded_snapshots["budget_model_identity_unknown"] += 1
            else:
                excluded_snapshots["model_identity_and_budget_not_observed"] += 1
            continue
        if component_id != model_id:
            invalid_snapshots.append("snapshot_instance_identity_mismatch")
            continue
        if component.get("observation_unknown") is True:
            excluded_snapshots["model_observation_unknown"] += 1
            continue
        if not isinstance(budget, dict):
            excluded_snapshots["budget_not_observed"] += 1
            continue
        budgets.append(budget)
    budget_failures, usable_budgets = [], 0
    for budget in budgets:
        limits, reserved = budget.get("limits") or {}, budget.get("reserved") or {}
        if not isinstance(limits, dict) or not isinstance(reserved, dict):
            continue
        values = [(budget.get("cpu_threads_in_use"), limits.get("cpu_threads"))]
        values += [(reserved.get(name), limits.get(name)) for name in ("ram_bytes", "disk_bytes")]
        if not all(_number(used) and _number(limit) and used >= 0 and limit > 0 for used, limit in values):
            continue
        usable_budgets += 1
        if any(used > limit for used, limit in values):
            budget_failures.append("observed_reservation_exceeds_limit")
    checks["sampled_resource_budgets"] = _check("fail" if budget_failures or invalid_snapshots else "pass" if usable_budgets and not unidentified_budget_samples else "unknown",
        "Sampled scheduler reservations versus declared CPU/RAM/disk limits; not physical utilization or proof between samples.",
        usable_snapshots=usable_budgets, excluded_snapshots=dict(excluded_snapshots),
        errors=sorted(set(budget_failures + invalid_snapshots)))

    cpu_limits = {b["limits"]["cpu_threads"] for b in budgets if isinstance(b.get("limits"), dict) and type(b["limits"].get("cpu_threads")) is int}
    cpu_known = len(cpu_limits) == 1 and all(type(item["start"].get("resources", {}).get("cpu_threads")) is int and item["start"]["resources"]["cpu_threads"] > 0 for item in intervals)
    peak_cpu = None
    if intervals and cpu_known and usable:
        actions = [(item[side]["sequence"], (1 if side == "start" else -1) * item["start"]["resources"]["cpu_threads"])
                   for item in intervals for side in ("start", "end")]
        current = 0
        peak_cpu = 0
        for _, delta in sorted(actions):
            current += delta
            peak_cpu = max(peak_cpu, current)
    cpu_limit = next(iter(cpu_limits)) if len(cpu_limits) == 1 else None
    checks["observed_stage_cpu_budget"] = _check("fail" if peak_cpu is not None and peak_cpu > cpu_limit else "pass" if peak_cpu is not None and model_complete else "unknown",
        "Concurrent stage-declared CPU lanes; event scopes can omit acquire/release margins and are not sampled physical CPU use.",
        peak_declared_lanes=peak_cpu, declared_limit=cpu_limit)

    cancel_rows = []
    for job_id in trace.get("queued_cancel_job_ids", []):
        job = job_records.get(job_id)
        offers = [event for event in manager if event.get("job_id") == job_id and event.get("type") in {"job.dispatch_intent", "job.dispatch_accepted", "job.dispatch_reconciled"}]
        executions = [event for event in model if event.get("job_id") == job_id and event.get("type") in {"execution.accepted", "stage.started"}]
        if offers or executions or (job and job.get("attempt_id") is not None):
            status = "fail"
        elif job and job.get("status") == "cancelled" and model_complete and manager_complete:
            status = "pass"
        else:
            status = "unknown"
        cancel_rows.append({"job_id": job_id, "status": status, "dispatch_observations": len(offers), "execution_observations": len(executions)})
    checks["queued_cancel_not_dispatched"] = _check("fail" if any(x["status"] == "fail" for x in cancel_rows) else "pass" if cancel_rows and all(x["status"] == "pass" for x in cancel_rows) else "unknown",
        "Only explicitly declared queued-cancel targets are tested; cancellation after dispatch is a different contract.", jobs=cancel_rows)
    cleanup_values = [job.get("cleanup_complete") for job in job_records.values() if job.get("status") in TERMINAL]
    checks["terminal_job_cleanup"] = _check("fail" if any(x is False for x in cleanup_values) or job_errors else "pass" if cleanup_values and all(x is True for x in cleanup_values) and len(cleanup_values) == len(job_records) else "unknown",
        "Terminal API receipts attest attempt-local cleanup; they do not release the model process or cloud VM.",
        terminal_receipts=len(cleanup_values), jobs=len(job_records))

    checks["service_process_cleanup"] = final_cleanup
    release = trace.get("platform_release") or {}
    bindings = [release[key] for key in ("manager_instance_id", "instance_id") if release.get(key) is not None]
    wrong_release = bool(bindings) and any(value != manager_id for value in bindings)
    release_bound = bool(bindings) and bool(manager_id) and not wrong_release
    release_observed = (_number(release.get("observed_at")) and release["observed_at"] > 0
                        and isinstance(release.get("source"), str) and bool(release["source"].strip()))
    checks["platform_release"] = _check("fail" if wrong_release or (release_bound and release.get("released") is False) else "pass" if release_bound and release_observed and release.get("released") is True else "unknown",
        "Separate dated platform observation explicitly bound to this manager lifetime. The analyzer cannot authenticate the external observation or its evidence hash.",
        instance_bound=release_bound, dated_source_observed=bool(release_observed))
    return {"schema_version": 1, "scope": "Finite observed trace checks, not universal correctness, mathematical optimality, device utilization or independently authenticated evidence.",
            "manager_instance_id": manager_id, "model_instance_id": model_id, "checks": checks,
            "counts": {state: sum(value["status"] == state for value in checks.values()) for state in ("pass", "fail", "unknown")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(json.loads(args.trace.read_text(encoding="utf-8")))
    rendered = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    else:
        print(rendered, end="")
    # A zero exit code means every declared check has evidence, never merely
    # that no contradiction happened to be found in a partial trace.
    return 1 if result["counts"]["fail"] else 2 if result["counts"]["unknown"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
