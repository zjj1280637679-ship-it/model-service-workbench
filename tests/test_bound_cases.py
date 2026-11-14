"""Analytic counterexamples for a tiny, exhaustively enumerated schedule model."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location("bound_cases", Path(__file__).resolve().parents[1] / "research/check_bound_cases.py")
bounds = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bounds)


def test_nonpreemptive_packing_has_strict_gap_above_resource_and_path_floors():
    result = bounds.solve(bounds.case_documents()["strict_lower_bound_gap"])
    # At time 0 at most two jobs can start; every length-2 job needs a whole
    # lane. Three jobs cannot complete by tick 3, but [0,2],[0,2],[2,4] can.
    assert result["resource_floor_ticks"] == 3
    assert result["critical_path_floor_ticks"] == 2
    assert result["exact_grid_optimum_ticks"] == 4
    assert result["gap_ticks"] == 1
    assert result["assignments_examined"] == 5 ** 3


def test_aliases_cannot_make_single_gpu_parallel_capacity():
    documents = bounds.case_documents()
    invented = bounds.solve(documents["alias_invented_capacity"])
    actual = bounds.solve(documents["alias_physical_pool"])
    assert invented["exact_grid_optimum_ticks"] == 2
    assert actual["resource_floor_ticks"] == actual["exact_grid_optimum_ticks"] == 4
    starts = {item["task"]: item["start"] for item in invented["witness"]}
    assert not bounds.schedule_feasible(bounds.normalize_model(documents["alias_physical_pool"]), starts)


def test_false_dependency_delays_independent_physical_resources():
    documents = bounds.case_documents()
    parallel = bounds.solve(documents["independent_parallel"])
    serial = bounds.solve(documents["artificial_serial_barrier"])
    assert parallel["exact_grid_optimum_ticks"] == 2
    assert serial["exact_grid_optimum_ticks"] == 4
    assert parallel["resource_floor_ticks"] == serial["resource_floor_ticks"] == 2
    assert serial["critical_path_floor_ticks"] == 4


def test_half_open_boundary_allows_immediate_resource_reuse():
    model = bounds.normalize_model(bounds.case_documents()["alias_physical_pool"])
    assert bounds.schedule_feasible(model, {"A": 0, "B": 2})
    assert not bounds.schedule_feasible(model, {"A": 0, "B": 1})


def test_aliases_on_same_task_are_aggregated_before_admission():
    document = {"pools": {"gpu": 1}, "aliases": {"a": "gpu", "b": "gpu"},
                "tasks": [{"id": "A", "duration": 1, "demand": {"a": 1, "b": 1}}]}
    with pytest.raises(ValueError, match="after alias resolution"):
        bounds.solve(document)


@pytest.mark.parametrize("durations", [(1, 1, 1), (1, 2, 3), (2, 2, 2)])
def test_exclusive_pool_equals_sum_and_independent_pools_equal_max(durations):
    names = list("ABC")
    exclusive = {"pools": {"one": 1}, "tasks": [bounds.task(name, duration, "one")
                 for name, duration in zip(names, durations)]}
    independent = {"pools": {name: 1 for name in names}, "tasks": [bounds.task(name, duration, name)
                   for name, duration in zip(names, durations)]}
    assert bounds.solve(exclusive)["exact_grid_optimum_ticks"] == sum(durations)
    assert bounds.solve(independent)["exact_grid_optimum_ticks"] == max(durations)


@pytest.mark.parametrize("mutation", ["cycle", "unknown_dependency", "duplicate_id", "unknown_pool", "bool_duration", "over_capacity", "alias_chain"])
def test_invalid_claims_are_rejected_instead_of_scored(mutation):
    document = deepcopy(bounds.case_documents()["alias_physical_pool"])
    if mutation == "cycle":
        document["tasks"][0]["depends_on"] = ["B"]
        document["tasks"][1]["depends_on"] = ["A"]
    elif mutation == "unknown_dependency":
        document["tasks"][0]["depends_on"] = ["missing"]
    elif mutation == "duplicate_id":
        document["tasks"][1]["id"] = "A"
    elif mutation == "unknown_pool":
        document["tasks"][0]["demand"] = {"missing": 1}
    elif mutation == "bool_duration":
        document["tasks"][0]["duration"] = True
    elif mutation == "over_capacity":
        document["tasks"][0]["demand"] = {"gpu_a": 2}
    else:
        document["aliases"]["gpu_b"] = "gpu_a"
    with pytest.raises(ValueError):
        bounds.solve(document)


def test_search_size_is_explicitly_bounded():
    with pytest.raises(ValueError, match="start assignments"):
        bounds.solve({"pools": {"one": 1}, "tasks": [bounds.task(str(i), 4, "one") for i in range(6)]})
    with pytest.raises(ValueError, match="horizon"):
        bounds.solve({"pools": {"one": 1}, "tasks": [bounds.task("A", 25, "one")]})


def test_report_keeps_model_results_separate_from_gpu_evidence():
    report = bounds.run_cases()
    assert report["evidence_layer"] == "finite_model_validation"
    assert report["gpu_used"] is False
    assert report["simulation_durations_are_declared_not_measured"] is True
    assert report["real_world_claims"]["gpu_architecture_validated"] is False
    assert report["real_world_claims"]["measured_cloud_speedup"] is None
    assert report["findings"]["aliases_cannot_create_capacity"]["incorrect_witness_violates_physical_pool"] is True
