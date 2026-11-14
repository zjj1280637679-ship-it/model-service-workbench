"""Tiny synthetic workloads test the algebra; none are MiniMax-H3 estimates."""
import copy
import pytest

from performance_model import compute_performance_bounds, hardware_template


def model(stages, coverage="complete", observed=10):
    return {"coverage": coverage, "observed_seconds": observed,
            "resources": {"compute": {"work_unit": "FLOP", "capacity_per_second": 10},
                          "memory": {"work_unit": "byte", "capacity_per_second": 20}},
            "stages": stages}


def node(name, compute=0, memory=0, deps=()):
    return {"id": name, "depends_on": list(deps),
            "work": {"compute": {"amount": compute, "unit": "FLOP"},
                     "memory": {"amount": memory, "unit": "byte"}}}


def test_perfect_resource_overlap_uses_max_not_sum():
    result = compute_performance_bounds(model([node("one", 20, 60)]))
    assert result["ideal_full_parallel_lower_bound_seconds"] == 3
    assert result["critical_path_lower_bound_seconds"] == 3
    assert result["overall_lower_bound_seconds"] == 3


def test_shared_resource_capacity_bounds_parallel_nodes():
    result = compute_performance_bounds(model([node("a", 20), node("b", 30)]))
    assert result["ideal_full_parallel_lower_bound_seconds"] == 5
    assert result["critical_path_lower_bound_seconds"] == 3
    assert result["overall_lower_bound_seconds"] == 5


def test_dependencies_prevent_overlap_across_different_resources():
    result = compute_performance_bounds(model([node("encode", 20), node("copy", memory=60, deps=["encode"])]))
    assert result["ideal_full_parallel_lower_bound_seconds"] == 3
    assert result["critical_path_lower_bound_seconds"] == 5
    assert result["overall_lower_bound_seconds"] == 5
    assert result["critical_path"] == ["encode", "copy"]


def test_diamond_graph_uses_longest_path_not_sum_of_branches():
    result = compute_performance_bounds(model([
        node("start", 10), node("short", memory=20, deps=["start"]),
        node("long", memory=60, deps=["start"]), node("end", 10, deps=["short", "long"]),
    ]))
    assert result["critical_path_lower_bound_seconds"] == 5
    assert result["critical_path"] == ["start", "long", "end"]


def test_missing_work_or_capacity_remains_partial_and_not_efficiency():
    data = model([node("one", compute=None, memory=20)])
    data["resources"]["memory"]["capacity_per_second"] = None
    result = compute_performance_bounds(data)
    assert result["coverage"] == "partial"
    assert result["overall_lower_bound_seconds"] is None
    assert len(result["missing"]) == 2
    assert result["comparison"]["overall_efficiency"] is None
    assert result["comparison"]["scope"] == "partial_workload_only"


def test_known_subset_can_give_bound_without_claiming_whole_task_efficiency():
    result = compute_performance_bounds(model([node("one", compute=None, memory=20)]))
    assert result["overall_lower_bound_seconds"] == 1
    assert result["coverage"] == "partial"
    assert result["comparison"]["time_over_lower_bound"] == 10
    assert result["comparison"]["overall_efficiency"] is None


def test_empty_hardware_template_does_not_predict_zero_second_generation():
    for profile, hbm in [("a100-sxm-40gb", 1555e9), ("a100-sxm-80gb", 2039e9)]:
        data = hardware_template(profile)
        assert data["resources"]["gpu_bf16_dense_flop"]["capacity_per_second"] == 312e12
        assert data["resources"]["gpu_hbm_bytes"]["capacity_per_second"] == hbm
        assert data["resources"]["gpu_fp32_flop"]["capacity_per_second"] is None
        assert data["resources"]["pcie_host_to_gpu_bytes"]["capacity_per_second"] is None
        result = compute_performance_bounds(data)
        assert result["overall_lower_bound_seconds"] is None
        assert result["coverage"] == "partial"
        assert result["missing"]


@pytest.mark.parametrize("change", [
    lambda data: data["stages"][0]["depends_on"].append("unknown"),
    lambda data: data["stages"][0]["depends_on"].append("one"),
    lambda data: data["stages"].append(copy.deepcopy(data["stages"][0])),
    lambda data: data["stages"][0]["work"]["memory"].update(unit="GB"),
    lambda data: data["stages"][0]["work"]["memory"].update(amount=-1),
    lambda data: data["stages"][0]["work"]["memory"].update(amount=float("nan")),
    lambda data: data["resources"]["compute"].update(capacity_per_second=0),
    lambda data: data["resources"]["compute"].update(capacity_per_second=float("inf")),
])
def test_invalid_graph_or_units_or_numbers_are_rejected(change):
    data = model([node("one", 10, 20)])
    change(data)
    with pytest.raises(ValueError):
        compute_performance_bounds(data)


def test_observation_faster_than_bound_is_marked_inconsistent():
    result = compute_performance_bounds(model([node("one", 20)], observed=1))
    assert result["comparison"]["inconsistent_observation"] is True


def test_explicit_partial_coverage_is_not_promoted_by_complete_numeric_subset():
    result = compute_performance_bounds(model([node("one", 10, 20)], coverage="partial"))
    assert result["coverage"] == "partial"
    assert result["comparison"]["scope"] == "partial_workload_only"
