"""Bounded public condition views, shared by HTTP and notebook diagnostics.

Observation summaries describe current evidence; decision receipts remain history.
No projection authorizes execution or imports model/framework dependencies.
"""
from __future__ import annotations

import hashlib
import json
import math
import re

JOB_ID = re.compile(r"^[a-f0-9]{32}$")

# Public receipts are an explicit projection, not arbitrary owner-provided JSON.
# Keep this contract independent of the model module: the API must not load it.
PUBLIC_STAGE_CONDITIONS = ("instance_matches", "attempt_owned", "attempt_live", "model_loaded", "worker_alive",
    "capability_available", "admission_confirmed", "predecessor_completed", "stage_not_started",
    "cancel_flag_clear", "cancel_event_clear", "reservation_owned", "budget_grant_held")
PUBLIC_GATE_CONDITIONS = {
    "manager.dispatch": ("manager_open", "job_queued", "attempt_unassigned", "model_identity_current",
        "model_observation_fresh", "model_process_alive", "model_ready", "worker_alive", "worker_accepting", "supervisor_alive"),
    "model.execute": ("instance_matches", "request_validated", "model_loaded", "worker_alive", "capability_available",
        "not_draining", "not_stopping", "attempt_available", "job_unowned", "artifact_absent",
        "generation_options_valid", "resources_reserved"),
    **{"model.stage." + stage: PUBLIC_STAGE_CONDITIONS for stage in ("prepare", "gpu", "encode_validate")},
}


def public_gate_receipt(value, expected_gate):
    """Expose known decision facts; no free-form context, parameters or paths."""
    if not isinstance(value, dict) or value.get("gate_id") != expected_gate or expected_gate not in PUBLIC_GATE_CONDITIONS:
        return None
    required = PUBLIC_GATE_CONDITIONS[expected_gate]
    raw_conditions = value.get("conditions")
    raw_conditions = raw_conditions if isinstance(raw_conditions, dict) else {}
    states = {"true", "false", "unknown", "conflict"}
    conditions = {name: raw_conditions.get(name) if type(raw_conditions.get(name)) is str
                  and raw_conditions[name] in states else "unknown" for name in required}
    result = {"gate_id": expected_gate, "conditions": conditions,
              "allowed": value.get("allowed") if type(value.get("allowed")) is bool else None}
    if result["allowed"] is True and any(state != "true" for state in conditions.values()):
        # Do not publish a positive authorization whose required evidence is
        # missing, malformed or contradictory; keep uncertainty visible.
        result["allowed"] = None
    for name in ("schema_version", "gate_version"):
        if isinstance(value.get(name), str) and re.fullmatch(r"[0-9]{1,3}\.[0-9]{1,3}", value[name]):
            result[name] = value[name]
    if isinstance(value.get("decision_id"), str) and JOB_ID.fullmatch(value["decision_id"]):
        result["decision_id"] = value["decision_id"]
    observed = value.get("observed_at")
    if type(observed) in {int, float} and math.isfinite(observed) and observed >= 0:
        result["observed_at"] = observed
    monotonic = value.get("monotonic_ns")
    if type(monotonic) is int and 0 <= monotonic < 2 ** 63:
        result["monotonic_ns"] = monotonic
    blocked = value.get("blocked")
    if isinstance(blocked, list):
        result["blocked"] = [name for name in required if name in blocked]
    raw = value.get("context")
    context = {}
    if isinstance(raw, dict):
        for name in ("manager_instance_id", "model_instance_id", "instance_id", "job_id", "attempt_id",
                     "budget_instance_id", "reservation_id", "lease_id", "budget_decision_id"):
            if isinstance(raw.get(name), str) and JOB_ID.fullmatch(raw[name]):
                context[name] = raw[name]
        expected_source = "global_manager.py" if expected_gate == "manager.dispatch" else "model_container.py"
        if raw.get("source_file") == expected_source:
            context["source_file"] = expected_source
        if isinstance(raw.get("source_sha256"), str) and re.fullmatch(r"[a-f0-9]{64}", raw["source_sha256"]):
            context["source_sha256"] = raw["source_sha256"]
        if type(raw.get("pipeline")) is bool:
            context["pipeline"] = raw["pipeline"]
        if type(raw.get("budget_scope")) is str and raw["budget_scope"] in ("stage_budget", "legacy_single_active"):
            context["budget_scope"] = raw["budget_scope"]
        if type(raw.get("stage")) is str and raw["stage"] in ("prepare", "gpu", "encode_validate"):
            context["stage"] = raw["stage"]
    result["context"] = context
    if result.get("schema_version") != "1.0" or result.get("gate_version") != "1.0":
        result["allowed"] = None
    return result


def public_condition_fields(record):
    fields = {}
    if "condition_gate" in record:
        fields["condition_gate"] = public_gate_receipt(record["condition_gate"], "manager.dispatch")
    if "execution_condition_gates" in record:
        raw = record["execution_condition_gates"]
        fields["execution_condition_gates"] = ({
            name: public_gate_receipt(raw[name], name) for name in PUBLIC_GATE_CONDITIONS
            if name.startswith("model.") and name in raw
        } if isinstance(raw, dict) else None)
    if "stage_completions" in record:
        raw = record["stage_completions"]
        fields["stage_completions"] = ({
            name: raw.get(name) if type(raw.get(name)) is bool else None
            for name in ("prepare", "gpu", "encode_validate")
        } if isinstance(raw, dict) else None)
    if "cancellation_requested" in record:
        fields["cancellation_requested"] = (record["cancellation_requested"]
            if type(record["cancellation_requested"]) is bool else None)
    if "execution_observation" in record:
        fields["execution_observation"] = public_execution_observation(record["execution_observation"])
    if "condition_summary" in record:
        fields["condition_summary"] = public_condition_summary(record["condition_summary"], record.get("instance_id"))
    return fields


def _identity(value):
    return isinstance(value, str) and JOB_ID.fullmatch(value) is not None


def _number(value):
    return type(value) in {int, float} and math.isfinite(value) and value >= 0


def public_queue_policy(value, manager_instance_id):
    if not isinstance(value, dict) or value.get("manager_instance_id") != manager_instance_id or not _identity(manager_instance_id):
        return None
    result = {"manager_instance_id": manager_instance_id}
    if type(value.get("mode")) is str and value["mode"] in {"fifo", "bounded_backfill"}:
        result["mode"] = value["mode"]
    if value.get("clock") == "manager_monotonic":
        result["clock"] = value["clock"]
    for key in ("bypasses", "deferrals"):
        if type(value.get(key)) is int and value[key] >= 0:
            result[key] = value[key]
    for key in ("first_deferred_at", "last_deferred_at", "admitted_at", "deferral_to_admission_seconds"):
        if _number(value.get(key)) or key in value and value[key] is None:
            result[key] = value[key]
    if _identity(value.get("bypassed_head_job_id")):
        result["bypassed_head_job_id"] = value["bypassed_head_job_id"]
    return result


def public_execution_summary(record, expected_manager):
    """One explicitly selected attempt, never a claim to cover every live job."""
    if not isinstance(record, dict) or record.get("instance_id") != expected_manager or not _identity(expected_manager):
        return None
    result = {"scope": "first_live_attempt", "instance_id": expected_manager}
    for key in ("job_id", "model_instance_id", "attempt_id"):
        if _identity(record.get(key)):
            result[key] = record[key]
    for key, value in public_condition_fields(record).items():
        if key in {"execution_observation", "execution_condition_gates", "stage_completions", "cancellation_requested"}:
            result[key] = value
    phase = record.get("active_phase")
    allowed = {"accepted", "prepare_wait", "preparing", "prepare", "gpu_wait", "gpu_running", "gpu",
               "encode_wait", "encode_validate", "validating", "running", "cancel_requested"}
    result["active_phase"] = ({"phase": phase["phase"]} if isinstance(phase, dict)
                              and type(phase.get("phase")) is str and phase["phase"] in allowed else None)
    observation = result.get("execution_observation") or {}
    if (observation.get("status") != "current" or observation.get("manager_instance_id") != expected_manager
            or any(observation.get(key) != result.get(key) or not _identity(result.get(key))
                   for key in ("job_id", "attempt_id", "model_instance_id"))):
        result["active_phase"] = None
        result.pop("execution_condition_gates", None)
        result.pop("stage_completions", None)
    return result


def public_execution_observation(value):
    """A current snapshot and a durable terminal result have different scope."""
    if not isinstance(value, dict):
        return None
    states = {"current", "historical", "stale", "unknown", "identity_mismatch"}
    result = {"status": value.get("status") if type(value.get("status")) is str and value["status"] in states else "unknown"}
    for key in ("manager_instance_id", "model_instance_id", "job_id", "attempt_id"):
        if _identity(value.get(key)):
            result[key] = value[key]
    for key in ("observed_at", "observation_age_seconds"):
        if _number(value.get(key)):
            result[key] = value[key]
    if result["status"] == "current" and (not all(key in result for key in
            ("manager_instance_id", "model_instance_id", "job_id", "attempt_id"))
            or "observed_at" not in result or "observation_age_seconds" not in result):
        result["status"] = "unknown"
    return result


def public_condition_summary(value, expected_manager=None):
    """Keep current read-only facts separate from the previous decision receipt.

    This bounded projection never provides an executable authorization. Unknown
    fields cannot turn into positive current evidence through UI coercion.
    """
    if not isinstance(value, dict):
        return None
    result = {"kind": "observation", "status": "unknown", "conditions": {}, "blocked": []}
    for key in ("manager_instance_id", "model_instance_id", "job_id"):
        if _identity(value.get(key)):
            result[key] = value[key]
    if expected_manager is not None and result.get("manager_instance_id") != expected_manager:
        result["status"] = "identity_mismatch"
        return result
    state = value.get("status")
    if type(state) is str and state in {"current", "historical", "stale", "unknown"}:
        result["status"] = state
    for key in ("observed_at", "observation_age_seconds"):
        if _number(value.get(key)):
            result[key] = value[key]
    if result["status"] == "current" and (not all(key in result for key in
            ("manager_instance_id", "model_instance_id", "job_id")) or "observed_at" not in result
            or "observation_age_seconds" not in result):
        result["status"] = "unknown"
    if result["status"] != "historical":
        raw = value.get("conditions")
        raw = raw if isinstance(raw, dict) else {}
        result["conditions"] = {key: raw.get(key) if type(raw.get(key)) is str and
            raw[key] in {"true", "false", "unknown", "conflict"} else "unknown"
            for key in PUBLIC_GATE_CONDITIONS["manager.dispatch"]}
        result["blocked"] = [key for key, state in result["conditions"].items() if state != "true"]
    receipt = public_gate_receipt(value.get("last_decision"), "manager.dispatch")
    context = receipt.get("context", {}) if receipt else {}
    if (receipt and _identity(receipt.get("decision_id")) and _number(receipt.get("observed_at"))
            and _identity(result.get("manager_instance_id")) and _identity(result.get("model_instance_id"))
            and context.get("manager_instance_id") == result.get("manager_instance_id")
            and context.get("model_instance_id") == result.get("model_instance_id")
            and ("job_id" not in result or context.get("job_id") == result["job_id"])):
        result["last_decision"] = receipt
    return result


def public_budget_page(value, request=None):
    """Reject malformed pagination; retain only fixed audit fields, never paths."""
    if not isinstance(value, dict) or type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        raise ValueError("Invalid budget audit schema")
    result = {"schema_version": 1}
    for key in ("model_instance_id", "budget_instance_id"):
        if not _identity(value.get(key)):
            raise ValueError("Invalid budget audit identity")
        result[key] = value[key]
    def sha(v):
        return isinstance(v, str) and re.fullmatch(r"[a-f0-9]{64}", v) is not None
    for key in ("capacity_version", "after_sha256", "tail_sha256"):
        if not sha(value.get(key)):
            raise ValueError("Invalid budget audit digest")
        result[key] = value[key]
    for key in ("after_sequence", "through_sequence", "next_sequence", "gpu_host_cpu_threads"):
        if type(value.get(key)) is not int or value[key] < 0:
            raise ValueError("Invalid budget audit cursor")
        result[key] = value[key]
    if not result["after_sequence"] <= result["next_sequence"] <= result["through_sequence"]:
        raise ValueError("Invalid budget audit range")
    def limits(raw):
        if not isinstance(raw, dict) or any(type(raw.get(k)) is not int or raw[k] < 0
                for k in ("ram_bytes", "disk_bytes", "cpu_threads")):
            raise ValueError("Invalid budget capacity")
        return {k: raw[k] for k in ("ram_bytes", "disk_bytes", "cpu_threads")}
    result["limits"] = limits(value.get("limits"))
    for key in ("page_complete", "more"):
        if type(value.get(key)) is not bool:
            raise ValueError("Invalid budget audit completeness")
        result[key] = value[key]
    events = value.get("events")
    if not isinstance(events, list) or len(events) > 256:
        raise ValueError("Invalid budget audit page size")
    result["events"] = []
    for row in events:
        if not isinstance(row, dict):
            raise ValueError("Invalid budget event")
        event = {}
        for key in ("lease_id", "budget_instance_id", "attempt_id", "decision_id", "reservation_id", "event_id"):
            if not _identity(row.get(key)):
                raise ValueError("Invalid budget event identity")
            event[key] = row[key]
        if row.get("gpu_attempt_id") is not None and not _identity(row["gpu_attempt_id"]):
            raise ValueError("Invalid GPU owner")
        event["gpu_attempt_id"] = row.get("gpu_attempt_id")
        for key in ("capacity_version", "previous_sha256", "sha256"):
            if not sha(row.get(key)):
                raise ValueError("Invalid budget event digest")
            event[key] = row[key]
        for key in ("cpu_threads", "cpu_threads_in_use", "sequence", "monotonic_ns"):
            if type(row.get(key)) is not int or row[key] < 0:
                raise ValueError("Invalid budget event number")
            event[key] = row[key]
        if (not _number(row.get("observed_at")) or type(row.get("kind")) is not str or row["kind"] not in {"cpu", "gpu"}
                or type(row.get("type")) is not str or row["type"] not in {"lease.acquired", "lease.released"}):
            raise ValueError("Invalid budget event type")
        event.update(observed_at=row["observed_at"], kind=row["kind"], type=row["type"], limits=limits(row.get("limits")))
        result["events"].append(event)
    previous = result["after_sha256"]
    for sequence, event in enumerate(result["events"], result["after_sequence"] + 1):
        content = {key: value for key, value in event.items() if key != "sha256"}
        digest = hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        if (event["sequence"] != sequence or event["previous_sha256"] != previous or event["sha256"] != digest
                or event["budget_instance_id"] != result["budget_instance_id"]
                or event["capacity_version"] != result["capacity_version"] or event["limits"] != result["limits"]):
            raise ValueError("Invalid budget audit continuity")
        previous = event["sha256"]
    if (result["next_sequence"] != result["after_sequence"] + len(result["events"])
            or result["more"] != (result["next_sequence"] < result["through_sequence"])
            or result["page_complete"] is not True
            or (result["more"] and result["next_sequence"] <= result["after_sequence"])
            or (not result["more"] and previous != result["tail_sha256"])
            or (result["after_sequence"] == 0 and result["after_sha256"] != "0" * 64)):
        raise ValueError("Invalid budget audit completeness")
    if request is not None and (result["after_sequence"] != request.get("after_sequence", 0)
            or (request.get("through_sequence") is not None and result["through_sequence"] != request["through_sequence"])
            or len(result["events"]) > request.get("limit", 100)):
        raise ValueError("Budget audit response differs from request")
    return result
