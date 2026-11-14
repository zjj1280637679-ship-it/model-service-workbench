"""Exhaust a tiny integer scheduling model; never execute or estimate a GPU.

Each operation has a declared fixed duration, starts on a nonnegative integer
tick, is non-preemptive, and holds its constant resource demand on [start,end).
There are no release times, setup costs, bandwidth effects or stochastic times.
The search is exact only in this explicitly finite model.
"""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
from itertools import product
import json
import math
from pathlib import Path


MAX_TASKS = 6
MAX_HORIZON = 24
MAX_ASSIGNMENTS = 2_000_000


def _positive_int(value, label):
    if type(value) is not int or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def normalize_model(document):
    """Resolve explicit aliases to one physical pool, never duplicate capacity."""
    if not isinstance(document, dict):
        raise ValueError("Model must be an object")
    raw_pools = document.get("pools")
    raw_tasks = document.get("tasks")
    aliases = document.get("aliases", {})
    if not isinstance(raw_pools, dict) or not raw_pools:
        raise ValueError("Physical pools must be a nonempty object")
    if not isinstance(aliases, dict):
        raise ValueError("Aliases must map directly to physical pool names")
    pools = {}
    for name, capacity in raw_pools.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Pool names must be nonempty strings")
        pools[name] = _positive_int(capacity, f"capacity of {name}")
    for name, pool in aliases.items():
        if not isinstance(name, str) or not name or name in pools or pool not in pools:
            raise ValueError("Aliases must be distinct names targeting one physical pool directly")
    if not isinstance(raw_tasks, list) or not 1 <= len(raw_tasks) <= MAX_TASKS:
        raise ValueError(f"Declare 1 to {MAX_TASKS} tasks")
    tasks = {}
    for raw in raw_tasks:
        if not isinstance(raw, dict) or not isinstance(raw.get("id"), str) or not raw["id"]:
            raise ValueError("Every task requires a nonempty id")
        identity = raw["id"]
        if identity in tasks:
            raise ValueError("Duplicate task identity")
        duration = _positive_int(raw.get("duration"), f"duration of {identity}")
        dependencies = raw.get("depends_on", [])
        if (not isinstance(dependencies, list)
                or any(not isinstance(dep, str) for dep in dependencies)
                or len(dependencies) != len(set(dependencies))):
            raise ValueError("Dependencies must be unique task names")
        raw_demand = raw.get("demand")
        if not isinstance(raw_demand, dict) or not raw_demand:
            raise ValueError("Every task requires explicit positive resource demands")
        demand = {}
        for name, value in raw_demand.items():
            canonical = aliases.get(name, name)
            if canonical not in pools:
                raise ValueError("Task refers to an undeclared resource pool")
            demand[canonical] = demand.get(canonical, 0) + _positive_int(value, "demand")
        if any(value > pools[name] for name, value in demand.items()):
            raise ValueError("Task demand exceeds physical capacity after alias resolution")
        tasks[identity] = {"id": identity, "duration": duration,
                           "depends_on": dependencies, "demand": demand}
    followers = {name: [] for name in tasks}
    indegree = {name: len(task["depends_on"]) for name, task in tasks.items()}
    for name, task in tasks.items():
        for dep in task["depends_on"]:
            if dep not in tasks:
                raise ValueError("Unknown dependency")
            followers[dep].append(name)
    ready = deque(sorted(name for name in tasks if not indegree[name]))
    order = []
    while ready:
        name = ready.popleft()
        order.append(name)
        for follower in sorted(followers[name]):
            indegree[follower] -= 1
            if not indegree[follower]:
                ready.append(follower)
    if len(order) != len(tasks):
        raise ValueError("Dependency graph contains a cycle")
    horizon = sum(task["duration"] for task in tasks.values())
    if horizon > MAX_HORIZON:
        raise ValueError(f"Finite search horizon exceeds {MAX_HORIZON} ticks")
    assignments = math.prod(horizon - task["duration"] + 1 for task in tasks.values())
    if assignments > MAX_ASSIGNMENTS:
        raise ValueError(f"Finite search exceeds {MAX_ASSIGNMENTS} start assignments")
    return {"pools": pools, "tasks": tasks, "order": order,
            "horizon": horizon, "search_assignments": assignments}


def lower_bounds(model):
    """Necessary bounds for fixed-duration tasks, not first-principles FLOPs."""
    resource = {
        name: Fraction(sum(task["duration"] * task["demand"].get(name, 0)
                           for task in model["tasks"].values()), capacity)
        for name, capacity in model["pools"].items()
    }
    ends = {}
    for name in model["order"]:
        task = model["tasks"][name]
        ends[name] = task["duration"] + max((ends[dep] for dep in task["depends_on"]), default=0)
    path_floor = max(ends.values())
    resource_floor = max(resource.values())
    return resource_floor, Fraction(path_floor), max(resource_floor, path_floor)


def schedule_feasible(model, starts):
    """Check one complete witness against half-open occupancy and dependencies."""
    tasks = model["tasks"]
    if set(starts) != set(tasks):
        return False
    if any(type(start) is not int or start < 0
           or start + tasks[name]["duration"] > model["horizon"]
           for name, start in starts.items()):
        return False
    for name, task in tasks.items():
        if any(starts[name] < starts[dep] + tasks[dep]["duration"] for dep in task["depends_on"]):
            return False
    for tick in range(model["horizon"]):
        for pool, capacity in model["pools"].items():
            occupied = sum(task["demand"].get(pool, 0) for name, task in tasks.items()
                           if starts[name] <= tick < starts[name] + task["duration"])
            if occupied > capacity:
                return False
    return True


def solve(document):
    model = normalize_model(document)
    names = sorted(model["tasks"])
    domains = [range(model["horizon"] - model["tasks"][name]["duration"] + 1) for name in names]
    best, witness, examined, feasible = None, None, 0, 0
    for values in product(*domains):
        examined += 1
        starts = dict(zip(names, values))
        if not schedule_feasible(model, starts):
            continue
        feasible += 1
        makespan = max(starts[name] + model["tasks"][name]["duration"] for name in names)
        if best is None or makespan < best:
            best, witness = makespan, starts
    # An acyclic graph whose individual demands fit always has a topological
    # serial witness within sum(duration). Exhausting this grid covers an optimum.
    if best is None or examined != model["search_assignments"]:
        raise AssertionError("Finite search did not cover the declared feasible model")
    resource, path, combined = lower_bounds(model)
    if combined > best:
        raise AssertionError("Necessary lower bound exceeded a feasible witness")
    return {
        "unit": "model_tick_not_seconds",
        "resource_floor_ticks": float(resource),
        "critical_path_floor_ticks": float(path),
        "combined_necessary_floor_ticks": float(combined),
        "exact_grid_optimum_ticks": best,
        "gap_ticks": float(Fraction(best) - combined),
        "integer_start_grid": {"min": 0, "latest_finish": model["horizon"]},
        "assignments_examined": examined,
        "feasible_assignments": feasible,
        "witness": [{"task": name, "start": witness[name],
                     "end": witness[name] + model["tasks"][name]["duration"],
                     "canonical_demand": model["tasks"][name]["demand"]} for name in names],
        "scope": "Exact only for the declared finite integer-start, fixed-duration, non-preemptive model; not real GPU or continuous/general scheduling optimality.",
    }


def task(identity, duration, pool, *, depends_on=()):
    return {"id": identity, "duration": duration, "depends_on": list(depends_on), "demand": {pool: 1}}


def case_documents():
    return {
        "strict_lower_bound_gap": {"pools": {"shared": 2}, "tasks": [task(name, 2, "shared") for name in "ABC"]},
        "alias_invented_capacity": {"pools": {"gpu_a": 1, "gpu_b": 1},
                                    "tasks": [task("A", 2, "gpu_a"), task("B", 2, "gpu_b")]},
        "alias_physical_pool": {"pools": {"gpu": 1}, "aliases": {"gpu_a": "gpu", "gpu_b": "gpu"},
                                "tasks": [task("A", 2, "gpu_a"), task("B", 2, "gpu_b")]},
        "independent_parallel": {"pools": {"cpu": 1, "gpu": 1},
                                 "tasks": [task("A", 2, "cpu"), task("B", 2, "gpu")]},
        "artificial_serial_barrier": {"pools": {"cpu": 1, "gpu": 1},
                                      "tasks": [task("A", 2, "cpu"), task("B", 2, "gpu", depends_on=("A",))]},
    }


def run_cases():
    cases = {name: {"declared_model": document, "result": solve(document)}
             for name, document in case_documents().items()}
    bad = cases["alias_invented_capacity"]["result"]
    physical_model = normalize_model(cases["alias_physical_pool"]["declared_model"])
    bad_starts = {item["task"]: item["start"] for item in bad["witness"]}
    alias_violation = not schedule_feasible(physical_model, bad_starts)
    if not alias_violation or cases["strict_lower_bound_gap"]["result"]["gap_ticks"] <= 0:
        raise AssertionError("Counterexamples were not exercised")
    return {
        "schema_version": 1,
        "evidence_layer": "finite_model_validation",
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "gpu_used": False,
        "simulation_durations_are_declared_not_measured": True,
        "model_assumptions": ["integer starts and fixed positive integer durations", "non-preemptive tasks",
                              "constant resource demand over half-open intervals", "all tasks released at zero",
                              "explicit acyclic dependencies", "no setup cost, memory residency, bandwidth contention or uncertainty"],
        "exhaustiveness": "All start assignments finishing by sum(durations) were enumerated; a topological serial schedule fits this horizon. Optimality is confined to this grid and these declared tasks.",
        "cases": cases,
        "findings": {
            "necessary_not_sufficient": "Three independent duration-2 tasks on capacity 2: resource floor 3, path floor 2, combined floor 3, enumerated optimum 4 ticks.",
            "aliases_cannot_create_capacity": {"incorrect_model_optimum_ticks": bad["exact_grid_optimum_ticks"],
                "physical_model_optimum_ticks": cases["alias_physical_pool"]["result"]["exact_grid_optimum_ticks"],
                "incorrect_witness_violates_physical_pool": alias_violation,
                "interpretation": "The solver is consistent with both declared models; the invented second pool is the modeling error, and its witness is physically inadmissible under the canonical pool."},
            "unnecessary_serial_dependency": {"independent_optimum_ticks": cases["independent_parallel"]["result"]["exact_grid_optimum_ticks"],
                "artificial_barrier_optimum_ticks": cases["artificial_serial_barrier"]["result"]["exact_grid_optimum_ticks"],
                "interpretation": "A false dependency doubles this tiny model's makespan; this is not a measured cloud speedup."},
        },
        "real_world_claims": {"gpu_architecture_validated": False, "hardware_efficiency": None,
                              "global_real_schedule_optimum": None, "measured_cloud_speedup": None},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_cases()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf8", newline="\n")
    print(json.dumps({"evidence_layer": result["evidence_layer"], "cases": len(result["cases"]),
                      "gpu_used": False, "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
