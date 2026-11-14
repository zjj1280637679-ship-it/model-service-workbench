"""Counterexamples for the research analyzer, without GPU or network access."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location("research_trace", Path(__file__).resolve().parents[1] / "research/analyze_trace.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
analyze = module.analyze


def example():
    trace = {"schema_version": 1, "manager_instance_id": "manager-1", "model_instance_id": "model-1",
             "model_events": [], "manager_events": [], "jobs": [], "system_snapshots": [],
             "queued_cancel_job_ids": ["cancel-job"], "collection": {}}

    def event(kind, job=None, stage=None, attempt=None):
        sequence = len(trace["model_events"]) + 1
        result = {"sequence": sequence, "event_id": f"model-event-{sequence}", "instance_id": "model-1",
                  "type": kind, "job_id": job, "attempt_id": attempt or (job + "-attempt" if job else None),
                  "monotonic_ns": sequence * 1_000_000_000, "observed_at": 100 + sequence}
        if stage:
            result.update(stage=stage, resources={"cpu_threads": 1, "ram_bytes": 10, "disk_bytes": 10})
        if kind == "stage.finished":
            result["succeeded"] = True
        trace["model_events"].append(result)

    event("model.loading")
    event("model.ready")
    event("execution.accepted", "a")
    event("stage.started", "a", "prepare")
    event("stage.finished", "a", "prepare")
    event("stage.started", "a", "gpu")
    event("execution.accepted", "b")
    event("stage.started", "b", "prepare")
    event("stage.finished", "b", "prepare")
    event("stage.finished", "a", "gpu")
    event("stage.started", "a", "encode_validate")
    event("stage.started", "b", "gpu")
    event("stage.finished", "a", "encode_validate")
    event("execution.succeeded", "a")
    event("stage.finished", "b", "gpu")
    event("stage.started", "b", "encode_validate")
    event("stage.finished", "b", "encode_validate")
    event("execution.succeeded", "b")
    event("model.worker_exited")
    trace["model_events"][-1]["worker_cleanup_complete"] = True
    for sequence, (kind, job) in enumerate([
        ("manager.created", None), ("job.queued", "a"), ("job.queued", "b"),
        ("job.queued", "cancel-job"), ("job.cancelled", "cancel-job"),
        ("job.dispatch_accepted", "a"), ("job.dispatch_accepted", "b"),
        ("job.succeeded", "a"), ("job.succeeded", "b"),
        ("manager.shutdown_complete", None),
    ], 1):
        trace["manager_events"].append({"sequence": sequence, "event_id": f"manager-event-{sequence}",
            "instance_id": "manager-1", "type": kind, "job_id": job, "observed_at": sequence})
    trace["jobs"] = [{"id": name, "instance_id": "manager-1", "status": "succeeded",
        "attempt_id": name + "-attempt", "cleanup_complete": True} for name in ("a", "b")]
    trace["jobs"].append({"id": "cancel-job", "instance_id": "manager-1", "status": "cancelled",
                          "attempt_id": None, "cleanup_complete": True})
    trace["system_snapshots"] = [{"instance_id": "manager-1", "observation_stale": False,
        "modules": {"model": {"instance_id": "model-1", "budget": {
            "limits": {"cpu_threads": 4, "ram_bytes": 100, "disk_bytes": 100},
            "reserved": {"ram_bytes": 20, "disk_bytes": 20}, "cpu_threads_in_use": 2}}}}]
    trace["collection"] = {"model_events_complete": True, "model_event_sequence_final": len(trace["model_events"]),
                           "manager_events_complete": True, "manager_event_sequence_final": len(trace["manager_events"])}
    trace["cleanup_receipt"] = {"instance_id": "manager-1", "state": {
        "instance_id": "manager-1", "cleanup_complete": True, "shutdown_complete": True,
        "manager_launcher_pid": 100,
        "manager_alive": False, "server_alive": False, "model_alive": False, "tunnel_alive": False},
        "owner_observation": {"alive": False, "cleanup_complete": True, "errors": [], "pids": [],
                              "identities": [{"pid": 100, "created_at": 10}], "observed_at": 80}}
    trace["platform_release"] = {"released": True, "source": "independent platform session observation",
                                 "manager_instance_id": "manager-1", "observed_at": 100}
    return trace


def pick(trace, kind, job, stage):
    return next(event for event in trace["model_events"] if event["type"] == kind and event.get("job_id") == job and event.get("stage") == stage)


def check(trace, name):
    return analyze(trace)["checks"][name]


def test_complete_finite_trace_reports_scoped_evidence():
    result = analyze(example())
    assert result["counts"]["fail"] == result["counts"]["unknown"] == 0
    assert result["checks"]["cross_task_cpu_gpu_overlap"]["witnessed_overlaps"]
    assert result["checks"]["observed_stage_cpu_budget"]["peak_declared_lanes"] == 2
    assert result["checks"]["stage_durations"]["stages"][0]["seconds"] == 1


def resequence(trace, name):
    for index, event in enumerate(trace[name + "_events"], 1):
        event["sequence"] = index
        if name == "model":
            event["monotonic_ns"] = index * 1_000_000_000
    trace["collection"][name + "_event_sequence_final"] = len(trace[name + "_events"])


def test_omitted_terminal_job_receipt_does_not_disappear_from_accounting():
    trace = example()
    trace["jobs"] = [job for job in trace["jobs"] if job["id"] != "b"]
    result = check(trace, "terminal_receipt_coverage")
    assert result["status"] == "unknown"
    assert result["missing_job_receipts"] == ["b"]
    assert analyze(trace)["counts"]["unknown"] > 0


def test_missing_model_attempt_terminal_event_is_not_inferred_from_job_success():
    trace = example()
    trace["model_events"] = [event for event in trace["model_events"]
                             if not (event["type"] == "execution.succeeded" and event.get("job_id") == "b")]
    resequence(trace, "model")
    assert check(trace, "terminal_receipt_coverage")["status"] == "unknown"


def test_model_attempt_must_match_final_job_receipt():
    trace = example()
    trace["jobs"][1]["attempt_id"] = "unrelated-attempt"
    assert check(trace, "terminal_receipt_coverage")["status"] == "fail"


@pytest.mark.parametrize("remove", [True, False])
def test_gpu_cannot_start_before_model_ready(remove):
    trace = example()
    ready = trace["model_events"].pop(1)
    if not remove:
        trace["model_events"].insert(8, ready)
    resequence(trace, "model")
    assert check(trace, "model_ready_gate")["status"] == "fail"


def test_missing_ready_in_partial_trace_remains_unknown():
    trace = example()
    trace["model_events"].pop(1)
    trace["collection"]["model_events_complete"] = False
    resequence(trace, "model")
    assert check(trace, "model_ready_gate")["status"] == "unknown"


def test_partial_trace_cannot_prove_absence_of_unobserved_ready_revocation():
    trace = example()
    trace["collection"]["model_events_complete"] = False
    assert check(trace, "model_ready_gate")["status"] == "unknown"


def test_model_failure_revokes_ready_for_later_gpu_stages():
    trace = example()
    revoked = deepcopy(trace["model_events"][1])
    revoked.update(type="model.failed", event_id="model-revoked")
    trace["model_events"].insert(5, revoked)
    resequence(trace, "model")
    assert check(trace, "model_ready_gate")["status"] == "fail"


def test_drain_does_not_revoke_inference_capability_of_already_admitted_work():
    trace = example()
    drain = deepcopy(trace["model_events"][1])
    drain.update(type="model.drain_requested", event_id="model-draining")
    trace["model_events"].insert(11, drain)
    resequence(trace, "model")
    assert check(trace, "model_ready_gate")["status"] == "pass"


@pytest.mark.parametrize("name", ["manager", "model"])
def test_declared_contiguous_prefix_without_terminal_event_is_not_complete(name):
    trace = example()
    trace[name + "_events"].pop()
    resequence(trace, name)
    result = check(trace, "trace_integrity")
    assert result["status"] == "fail"
    assert result[name + "_complete"] is False
    assert check(trace, "queued_cancel_not_dispatched")["status"] != "pass"


def test_model_exit_without_worker_cleanup_cannot_certify_full_lifetime():
    trace = example()
    trace["model_events"][-1]["worker_cleanup_complete"] = False
    assert check(trace, "trace_integrity")["model_complete"] is False


def test_final_sequence_claim_requires_owner_exit_evidence():
    trace = example()
    trace.pop("cleanup_receipt")
    assert check(trace, "trace_integrity")["status"] == "fail"
    assert check(trace, "gpu_stage_exclusivity")["status"] == "unknown"


def test_empty_evidence_is_unknown_not_success():
    result = analyze({"schema_version": 1})
    assert result["counts"]["pass"] == result["counts"]["fail"] == 0


def test_missing_stream_suffix_cannot_prove_exclusivity_or_queued_cancel():
    trace = example()
    trace["model_events"].pop()
    assert check(trace, "trace_integrity")["status"] == "fail"
    assert check(trace, "gpu_stage_exclusivity")["status"] == "unknown"
    assert check(trace, "queued_cancel_not_dispatched")["status"] == "unknown"


def test_missing_middle_event_is_not_hidden_by_last_sequence():
    trace = example()
    del trace["model_events"][0]
    assert check(trace, "trace_integrity")["status"] == "fail"


@pytest.mark.parametrize("field,value", [("instance_id", "old-model"), ("sequence", 1), ("event_id", "model-event-1")])
def test_identity_and_sequence_contamination_are_not_accepted(field, value):
    trace = example()
    trace["model_events"][5][field] = value
    assert check(trace, "trace_integrity")["status"] == "fail"
    assert check(trace, "gpu_stage_exclusivity")["status"] != "pass"


def test_gpu_overlap_is_detected_even_without_a_complete_trace():
    trace = example()
    # Extend A's GPU interval past the beginning of B's GPU scope.
    end = pick(trace, "stage.finished", "a", "gpu")
    end["sequence"] = 12.5
    # Integer sequences remain unique after inserting the new event position.
    for index, event in enumerate(sorted(trace["model_events"], key=lambda x: x["sequence"]), 1):
        event["sequence"] = index
        event["monotonic_ns"] = index * 1_000_000_000
    trace["collection"]["model_events_complete"] = False
    assert check(trace, "gpu_stage_exclusivity")["status"] == "fail"


def test_failed_predecessor_cannot_enable_next_stage():
    trace = example()
    pick(trace, "stage.finished", "b", "prepare")["succeeded"] = False
    assert check(trace, "stage_causality")["status"] == "fail"


def test_successful_job_cannot_skip_encoding():
    trace = example()
    trace["model_events"] = [e for e in trace["model_events"] if not (e.get("job_id") == "b" and e.get("stage") == "encode_validate")]
    for index, event in enumerate(trace["model_events"], 1):
        event["sequence"] = index
    trace["collection"]["model_event_sequence_final"] = len(trace["model_events"])
    assert check(trace, "stage_causality")["status"] == "fail"


def test_a_success_receipt_cannot_claim_another_jobs_attempt():
    trace = example()
    trace["jobs"][0]["attempt_id"] = "b-attempt"
    assert check(trace, "stage_causality")["status"] == "fail"


def test_malformed_event_identity_is_reported_not_used_as_hash_key():
    trace = example()
    trace["model_events"][1]["event_id"] = []
    assert check(trace, "trace_integrity")["status"] == "fail"


def test_wall_clock_never_becomes_precise_duration_but_sequence_can_show_overlap():
    trace = example()
    for event in trace["model_events"]:
        event.pop("monotonic_ns")
        event["observed_at"] = 10_000 - event["sequence"]
    assert check(trace, "stage_durations")["status"] == "unknown"
    assert check(trace, "cross_task_cpu_gpu_overlap")["status"] == "pass"


def test_regressing_monotonic_clock_is_not_accepted_as_duration():
    trace = example()
    trace["model_events"][5]["monotonic_ns"] = 1
    assert check(trace, "stage_durations")["status"] == "unknown"


def test_partial_stream_can_witness_overlap_but_not_prove_exclusivity():
    trace = example()
    trace["collection"]["model_events_complete"] = False
    assert check(trace, "cross_task_cpu_gpu_overlap")["status"] == "pass"
    assert check(trace, "gpu_stage_exclusivity")["status"] == "unknown"


@pytest.mark.parametrize("resource", ["cpu", "ram", "disk"])
def test_budget_overcommit_is_a_counterexample(resource):
    trace = example()
    budget = trace["system_snapshots"][0]["modules"]["model"]["budget"]
    if resource == "cpu":
        budget["cpu_threads_in_use"] = 5
    else:
        budget["reserved"][resource + "_bytes"] = 101
    assert check(trace, "sampled_resource_budgets")["status"] == "fail"


def test_stale_and_missing_budget_snapshots_do_not_become_zero_usage():
    trace = example()
    trace["system_snapshots"][0]["observation_stale"] = True
    assert check(trace, "sampled_resource_budgets")["status"] == "unknown"
    assert check(trace, "observed_stage_cpu_budget")["peak_declared_lanes"] is None


@pytest.mark.parametrize("stale", [False, True, None])
def test_real_stopped_host_projection_is_excluded_not_a_model_identity_conflict(stale):
    trace = example()
    trace["system_snapshots"].append({"instance_id": "manager-1", "phase": "stopped",
        "observation_stale": stale, "modules": {"model": {
            "alive": False, "ready": False, "observation_unknown": True}}})
    result = check(trace, "sampled_resource_budgets")
    assert result["status"] == "pass"
    assert result["usable_snapshots"] == 1
    assert sum(result["excluded_snapshots"].values()) == 1
    assert result["errors"] == []


def test_only_startup_or_stopped_projections_leave_budget_unknown():
    trace = example()
    trace["system_snapshots"] = [{"instance_id": "manager-1", "observation_stale": False,
        "modules": {"model": {"alive": False, "ready": False, "observation_unknown": True}}}]
    result = check(trace, "sampled_resource_budgets")
    assert result["status"] == "unknown"
    assert result["errors"] == []
    assert result["usable_snapshots"] == 0


def test_stale_snapshot_is_excluded_before_model_identity_comparison():
    trace = example()
    trace["system_snapshots"][0]["observation_stale"] = True
    trace["system_snapshots"][0]["modules"]["model"]["instance_id"] = "old-model"
    result = check(trace, "sampled_resource_budgets")
    assert result["status"] == "unknown"
    assert result["errors"] == []


def test_fresh_nonempty_wrong_model_identity_remains_a_failure():
    trace = example()
    trace["system_snapshots"][0]["modules"]["model"]["instance_id"] = "wrong-model"
    assert check(trace, "sampled_resource_budgets")["status"] == "fail"


@pytest.mark.parametrize("keep_valid_sample", [False, True])
def test_fresh_budget_with_no_model_identity_is_unknown_not_violation(keep_valid_sample):
    trace = example()
    unknown = deepcopy(trace["system_snapshots"][0])
    unknown["modules"]["model"].pop("instance_id")
    trace["system_snapshots"] = trace["system_snapshots"] + [unknown] if keep_valid_sample else [unknown]
    result = check(trace, "sampled_resource_budgets")
    assert result["status"] == "unknown"
    assert result["errors"] == []
    assert result["excluded_snapshots"]["budget_model_identity_unknown"] == 1


def test_stale_sample_from_wrong_manager_still_fails_manager_identity():
    trace = example()
    trace["system_snapshots"][0].update(instance_id="wrong-manager", observation_stale=True)
    assert check(trace, "sampled_resource_budgets")["status"] == "fail"


def test_stage_cpu_concurrency_is_checked_against_declared_capacity():
    trace = example()
    trace["system_snapshots"][0]["modules"]["model"]["budget"]["limits"]["cpu_threads"] = 1
    assert check(trace, "observed_stage_cpu_budget")["status"] == "fail"


def test_queued_cancel_dispatched_with_missing_response_is_detected():
    trace = example()
    event = deepcopy(trace["manager_events"][-1])
    event.update(type="job.dispatch_intent", job_id="cancel-job", event_id="offer-cancelled")
    trace["manager_events"].insert(-1, event)
    resequence(trace, "manager")
    assert check(trace, "queued_cancel_not_dispatched")["status"] == "fail"


def test_no_cancellation_target_is_unknown_not_vacuous_pass():
    trace = example()
    trace["queued_cancel_job_ids"] = []
    assert check(trace, "queued_cancel_not_dispatched")["status"] == "unknown"


def test_job_cleanup_process_cleanup_and_platform_release_are_independent():
    trace = example()
    trace.pop("cleanup_receipt")
    trace.pop("platform_release")
    assert check(trace, "terminal_job_cleanup")["status"] == "pass"
    assert check(trace, "service_process_cleanup")["status"] == "unknown"
    assert check(trace, "platform_release")["status"] == "unknown"


def test_old_process_cleanup_cannot_close_new_instance():
    trace = example()
    trace["cleanup_receipt"]["instance_id"] = "old-instance"
    assert check(trace, "service_process_cleanup")["status"] == "unknown"


def test_owner_still_alive_overrides_optimistic_cleanup_state():
    trace = example()
    trace["cleanup_receipt"]["owner_observation"]["alive"] = True
    assert check(trace, "service_process_cleanup")["status"] == "fail"


def test_nonempty_live_pid_list_contradicts_owner_clean_booleans():
    trace = example()
    trace["cleanup_receipt"]["owner_observation"]["pids"] = [12345]
    assert check(trace, "service_process_cleanup")["status"] == "fail"
    assert check(trace, "trace_integrity")["model_complete"] is False


@pytest.mark.parametrize("change", ["missing_identities", "wrong_pid", "missing_launcher_pid", "unknown_creation", "no_observation_time"])
def test_owner_cleanup_must_be_bound_to_manager_process_generation(change):
    trace = example()
    receipt = trace["cleanup_receipt"]
    owner = receipt["owner_observation"]
    if change == "missing_identities":
        owner.pop("identities")
    elif change == "wrong_pid":
        owner["identities"][0]["pid"] = 999
    elif change == "missing_launcher_pid":
        receipt["state"].pop("manager_launcher_pid")
    elif change == "unknown_creation":
        owner["identities"][0]["created_at"] = None
    else:
        owner.pop("observed_at")
    assert check(trace, "service_process_cleanup")["status"] == "unknown"


def test_previous_vm_release_cannot_certify_current_instance():
    trace = example()
    trace["platform_release"] = {"released": True, "source": "old VM", "instance_id": "different-vm", "observed_at": 100}
    assert check(trace, "platform_release")["status"] == "fail"


@pytest.mark.parametrize("field", ["manager_instance_id", "observed_at", "source"])
def test_platform_release_requires_current_binding_dated_source(field):
    trace = example()
    trace["platform_release"].pop(field)
    assert check(trace, "platform_release")["status"] == "unknown"


def test_platform_release_accepts_explicit_instance_id_alias():
    trace = example()
    trace["platform_release"]["instance_id"] = trace["platform_release"].pop("manager_instance_id")
    assert check(trace, "platform_release")["status"] == "pass"


def test_unobserved_platform_release_is_unknown():
    trace = example()
    trace["platform_release"] = {"released": True}
    assert check(trace, "platform_release")["status"] == "unknown"


def test_raw_secrets_prompts_and_paths_are_not_reflected_in_report():
    trace = example()
    trace["system_snapshots"][0].update(api_key="test-private-value", public_url="https://private.invalid")
    trace["jobs"][0].update(prompt="unpublished prompt", path="private-directory")
    rendered = str(analyze(trace))
    for forbidden in ("test-private-value", "https://private.invalid", "unpublished prompt", "private-directory"):
        assert forbidden not in rendered


@pytest.mark.parametrize("value", [None, [], {"schema_version": 9}, {"schema_version": True},
    {"schema_version": 1, "jobs": {}}, {"schema_version": 1, "cleanup_receipt": []},
    {"schema_version": 1, "queued_cancel_job_ids": [[]]}])
def test_invalid_top_level_input_is_rejected(value):
    with pytest.raises(ValueError):
        analyze(value)
