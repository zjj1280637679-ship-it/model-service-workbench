"""CPU-only optimistic physical lower bounds, not a GPU benchmark or simulator.

Resource floor = max_r(sum_i(work[i,r]) / peak[r]).
Node floor = max_r(work[i,r] / peak[r]).
Critical-path floor = longest dependency path using node floors.
Overall floor = max(resource floor, critical-path floor).

Missing workload/capacity is reported, never invented. No PyTorch, CUDA,
downloads, or live runtime access is used. Numeric inputs use explicit work
units; capacities use the same unit per second. Decimal GB means 1e9 bytes.
"""
from __future__ import annotations

import argparse
from collections import deque
import json
import math
from pathlib import Path


DATASHEET = "https://images.nvidia.com/data-center/a100/a100-datasheet.pdf"
A100_40_SOURCE = "https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/"
BACKGROUND = "https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/"


def _number(value, path, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{path} must be a finite number")
    if value < 0 or (positive and value <= 0):
        raise ValueError(f"{path} must be {'positive' if positive else 'non-negative'}")
    return float(value)


def hardware_template(profile="a100-sxm-40gb"):
    """Return capacities only. The caller must supply a profiled workload DAG."""
    if profile not in {"a100-sxm-40gb", "a100-sxm-80gb"}:
        raise ValueError("Supported profiles: a100-sxm-40gb, a100-sxm-80gb")
    hbm = 1555e9 if profile == "a100-sxm-40gb" else 2039e9
    def resource(unit, capacity, note, source=None):
        return {"work_unit": unit, "capacity_per_second": capacity, "note": note, "source": source}
    return {
        "schema_version": 1,
        "hardware": profile,
        "coverage": "partial",
        "observed_seconds": None,
        "resources": {
            "gpu_bf16_dense_flop": resource("FLOP", 312e12,
                "BF16 dense Tensor Core work only; 624 TFLOP/s sparsity peak does not apply. One multiply-add counts as 2 FLOPs.", DATASHEET),
            "gpu_fp32_flop": resource("FLOP", None,
                "Keep FP32-layer work separate. Determine actual instruction path first; do not divide it by the BF16 peak. FP32 CUDA-core peak is 19.5e12 FLOP/s, TF32 is a different path.", DATASHEET),
            "gpu_hbm_bytes": resource("byte", hbm,
                "Total HBM traffic, not model-file size or PCIe traffic. Decimal GB/s; theoretical peak, not measured bandwidth.", A100_40_SOURCE if profile.endswith("40gb") else DATASHEET),
            "host_memory_bytes": resource("byte", None, "Host RAM topology and bandwidth are unknown."),
            "pcie_host_to_gpu_bytes": resource("byte", None, "Actual host-to-GPU topology/link width unknown. 64 GB/s bidirectional must not be used as one-way capacity."),
            "pcie_gpu_to_host_bytes": resource("byte", None, "Actual GPU-to-host topology/link width unknown. Keep transfer directions explicit."),
            "storage_read_bytes": resource("byte", None, "Backing-storage read bandwidth is unknown; distinguish page-cache hits from physical reads."),
            "network_bytes": resource("byte", None, "External network path and bandwidth are unknown."),
            "cpu_work": resource("operation", None, "Choose a measured CPU operation definition and matching peak before supplying work."),
        },
        "stages": [],
        "notes": [
            "This template contains no MiniMax-H3 FLOP or byte estimates and predicts no generation time.",
            "Add stage objects with id, depends_on, and work entries of {amount, unit}; null amount means unknown.",
            "Use distinct stages for non-overlappable work and preserve true dependencies. A fused stage assumes ideal overlap among its resources.",
            "Resource capacities describe shared resource pools. Do not duplicate one physical pool under different names to invent extra capacity.",
            "Omitted resource work is assumed zero for the declared model. coverage=complete is an explicit assertion by the profiler, not automatic discovery.",
            "Only use this full-GPU profile when the assigned GPU is the matching full A100, not a MIG slice.",
            "GPU matrix, memory, launch, decode, I/O and transfer work still need profiling. Available RAM and VRAM capacity are feasibility constraints, not bandwidth.",
        ],
        "method_source": BACKGROUND,
    }


def compute_performance_bounds(document):
    if not isinstance(document, dict):
        raise ValueError("Input must be a JSON object")
    coverage = document.get("coverage", "partial")
    if coverage not in {"partial", "complete"}:
        raise ValueError("coverage must be partial or complete")
    resources = document.get("resources")
    stages = document.get("stages")
    if not isinstance(resources, dict) or not resources:
        raise ValueError("resources must be a nonempty object")
    if not isinstance(stages, list):
        raise ValueError("stages must be a list")
    capacities, units = {}, {}
    for name, definition in resources.items():
        if not isinstance(name, str) or not name or not isinstance(definition, dict):
            raise ValueError("Every resource needs a nonempty name and an object definition")
        unit = definition.get("work_unit")
        if not isinstance(unit, str) or not unit.strip():
            raise ValueError(f"resources.{name}.work_unit must be explicit")
        units[name] = unit
        capacity = definition.get("capacity_per_second")
        capacities[name] = None if capacity is None else _number(capacity, f"resources.{name}.capacity_per_second", positive=True)

    nodes, dependencies, weights, totals, missing = {}, {}, {}, {}, []
    known_usable_items = 0
    for index, stage in enumerate(stages):
        if not isinstance(stage, dict) or not isinstance(stage.get("id"), str) or not stage["id"]:
            raise ValueError(f"stages[{index}] requires a nonempty string id")
        name = stage["id"]
        if name in nodes:
            raise ValueError(f"Duplicate stage id: {name}")
        nodes[name] = stage
        deps = stage.get("depends_on", [])
        if not isinstance(deps, list) or any(not isinstance(dep, str) for dep in deps) or len(set(deps)) != len(deps):
            raise ValueError(f"stages.{name}.depends_on must be a list of unique stage ids")
        dependencies[name] = deps
        work = stage.get("work")
        if not isinstance(work, dict):
            raise ValueError(f"stages.{name}.work must be an object")
        node_times = []
        if not work:
            missing.append({"path": f"stages.{name}.work", "reason": "no workload supplied"})
        for resource, item in work.items():
            path = f"stages.{name}.work.{resource}"
            if resource not in resources:
                raise ValueError(f"{path} names an undefined resource")
            if not isinstance(item, dict) or item.get("unit") != units[resource]:
                raise ValueError(f"{path}.unit must equal {units[resource]!r}; implicit conversion is not performed")
            if "amount" not in item:
                raise ValueError(f"{path}.amount must be a number or explicit null")
            total = totals.setdefault(resource, {"known_amount": 0.0, "unit": units[resource], "has_unknown_work": False})
            if item["amount"] is None:
                total["has_unknown_work"] = True
                missing.append({"path": path + ".amount", "reason": "unknown workload"})
                continue
            amount = _number(item["amount"], path + ".amount")
            total["known_amount"] += amount
            if capacities[resource] is None:
                if amount > 0:
                    missing.append({"path": "resources." + resource + ".capacity_per_second", "reason": "unknown capacity for nonzero work"})
                else:
                    node_times.append(0.0)
                    known_usable_items += 1
            else:
                node_times.append(amount / capacities[resource])
                known_usable_items += 1
        weights[name] = max(node_times) if node_times else None

    if not stages:
        missing.append({"path": "stages", "reason": "no workload DAG supplied"})
    followers = {name: [] for name in nodes}
    indegree = {name: len(deps) for name, deps in dependencies.items()}
    for name, deps in dependencies.items():
        for dep in deps:
            if dep not in nodes:
                raise ValueError(f"Stage {name!r} depends on unknown stage {dep!r}")
            followers[dep].append(name)
    ready = deque(name for name in nodes if indegree[name] == 0)
    ends, paths, order = {}, {}, []
    while ready:
        name = ready.popleft()
        order.append(name)
        predecessor = max(dependencies[name], key=lambda dep: ends[dep], default=None)
        ends[name] = (ends[predecessor] if predecessor is not None else 0.0) + (weights[name] or 0.0)
        paths[name] = (paths[predecessor] if predecessor is not None else []) + [name]
        for successor in followers[name]:
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
    if len(order) != len(nodes):
        raise ValueError("The stage graph contains a dependency cycle")

    resource_bounds = {
        name: total["known_amount"] / capacities[name]
        for name, total in totals.items() if capacities[name] is not None
    }
    usable = known_usable_items > 0
    ideal = max(resource_bounds.values(), default=0.0) if usable else None
    terminal = max(ends, key=ends.get, default=None)
    critical = ends[terminal] if terminal is not None and usable else None
    lower_bound = max(ideal, critical) if ideal is not None and critical is not None else None
    # Deduplicate shared missing capacity entries while keeping deterministic order.
    missing = [dict(path=path, reason=reason) for path, reason in dict.fromkeys((entry["path"], entry["reason"]) for entry in missing)]
    complete = coverage == "complete" and not missing and bool(stages)
    comparison = None
    observed = document.get("observed_seconds")
    if observed is not None:
        observed = _number(observed, "observed_seconds", positive=True)
        comparison = {
            "observed_seconds": observed,
            "scope": "declared_complete_workload" if complete else "partial_workload_only",
            "time_over_lower_bound": observed / lower_bound if lower_bound is not None and lower_bound > 0 else None,
            "lower_bound_fraction_of_observed": lower_bound / observed if lower_bound is not None else None,
            "inconsistent_observation": lower_bound is not None and observed + 1e-12 < lower_bound,
            "overall_efficiency": None,
            "note": "Ratios compare time with an optimistic bound, not measured device utilization or proven achievable efficiency. Partial workloads cannot establish whole-task efficiency.",
        }
    return {
        "schema_version": 1,
        "coverage": "declared_complete" if complete else "partial",
        "ideal_full_parallel_lower_bound_seconds": ideal,
        "critical_path_lower_bound_seconds": critical,
        "overall_lower_bound_seconds": lower_bound,
        "resource_lower_bounds_seconds": resource_bounds,
        "stage_lower_bounds_seconds": weights,
        "critical_path": paths[terminal] if terminal is not None and usable else [],
        "total_work_by_resource": totals,
        "missing": missing,
        "unavailable_capacities": [name for name, value in capacities.items() if value is None],
        "comparison": comparison,
        "interpretation": "Optimistic necessary lower bound under supplied resource pools and DAG, not an achievable runtime prediction. Independent resources and work inside each stage are assumed to overlap perfectly; dependencies cannot overlap.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, help="UTF-8 JSON workload and resource document")
    parser.add_argument("--template", choices=("a100-sxm-40gb", "a100-sxm-80gb"), help="Emit capacities only; no fake workload")
    parser.add_argument("--output", type=Path, help="Write UTF-8 JSON instead of stdout")
    args = parser.parse_args(argv)
    if bool(args.input) == bool(args.template):
        parser.error("Choose exactly one input JSON or --template")
    try:
        result = hardware_template(args.template) if args.template else compute_performance_bounds(json.loads(args.input.read_text(encoding="utf-8-sig")))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    rendered = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
