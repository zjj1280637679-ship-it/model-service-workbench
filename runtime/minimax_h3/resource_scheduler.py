"""Bounded CPU child events; the GPU owner and model stages remain serial.

The optimizer is exact within floating-point tolerance for its declared cost
model, not a claim of global hardware optimality. Admission uses calibrated
memory/thread estimates, not OS-enforced peak limits. Missing calibration selects
one lane. Native codec/library pools are not constrained by the Python executor.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time

MIB = 1024 ** 2
RESERVE_MIB = 2048


@dataclass(frozen=True)
class Allocation:
    lanes: tuple[tuple[int, ...], ...]
    predicted_seconds: float
    reason: str
    task_memory_mib: float | None = None
    threads_per_worker: int | None = None
    profile_metadata: dict | None = None


def optimal_allocation(costs, io_mib, *, cpu_slots, memory_slots, bandwidth_mib_s,
                       max_workers=4, overhead_seconds=0.002):
    """Minimize max(compute makespan, shared IO floor)+lane overhead.

    At most 12 children / 4 lanes: branch-and-bound enumerates legal assignments,
    removing symmetric empty lanes. Inputs are service estimates from calibration.
    Equal predicted costs favor fewer workers. No nested GPU concurrency is allowed.
    """
    if not costs:
        return Allocation((), 0.0, "no children")
    if len(costs) > 12 or any(not math.isfinite(x) or x <= 0 for x in costs):
        raise ValueError("Require 1..12 finite positive child costs")
    if len(io_mib) != len(costs) or any(not math.isfinite(x) or x < 0 for x in io_mib):
        raise ValueError("IO demand must match the child costs")
    if not math.isfinite(bandwidth_mib_s) or bandwidth_mib_s <= 0:
        raise ValueError("A positive measured bandwidth is required")
    if overhead_seconds < 0 or not math.isfinite(overhead_seconds):
        raise ValueError("Invalid dispatch overhead")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (cpu_slots, memory_slots, max_workers)):
        raise ValueError("CPU, memory, and worker lane caps must be integers")
    cap = min(4, max_workers, cpu_slots, memory_slots, len(costs))
    if cap < 1:
        raise MemoryError("No CPU/memory lane is available")
    order = sorted(range(len(costs)), key=lambda i: costs[i], reverse=True)
    io_floor = sum(io_mib) / bandwidth_mib_s
    winner = None
    for workers in range(1, cap + 1):
        loads, bins = [0.0] * workers, [[] for _ in range(workers)]
        # A valid upper bound from longest-processing-time list scheduling.
        for index in order:
            lane = min(range(workers), key=lambda k: loads[k])
            loads[lane] += costs[index]
            bins[lane].append(index)
        best_load, best_bins = max(loads), [row[:] for row in bins]
        lower_bound = max(sum(costs) / workers, max(costs))
        loads, bins = [0.0] * workers, [[] for _ in range(workers)]

        def visit(position):
            nonlocal best_load, best_bins
            if best_load <= lower_bound + 1e-10:
                return
            if position == len(order):
                best_load, best_bins = max(loads), [row[:] for row in bins]
                return
            index = order[position]
            seen = set()
            for lane in range(workers):
                previous = loads[lane]
                if previous in seen or previous + costs[index] >= best_load - 1e-10:
                    continue
                seen.add(previous)
                loads[lane] += costs[index]
                bins[lane].append(index)
                visit(position + 1)
                bins[lane].pop()
                loads[lane] = previous

        visit(0)
        predicted = max(best_load, io_floor) + overhead_seconds * workers
        candidate = Allocation(tuple(tuple(row) for row in best_bins), predicted,
                               "minimum predicted makespan under calibrated CPU/IO/memory limits")
        if winner is None or predicted < winner.predicted_seconds - 1e-10:
            winner = candidate
    return winner


def _cpu_slots():
    """Affinity-visible logical cores; not a cgroup CPU quota or thread limit."""
    import psutil
    try:
        return max(1, len(psutil.Process().cpu_affinity()))
    except (AttributeError, OSError, psutil.Error):
        return max(1, os.cpu_count() or 1)


def _phase_table(profile, key, phase):
    table = profile.get(key, {})
    if not isinstance(table, dict) or not isinstance(table.get(phase, {}), dict):
        raise ValueError(f"Calibration {key} must map phase names to modality estimates")
    return table.get(phase, {})


class MediaController:
    """Negative feedback applies between batches, never inside a GPU stage.

    A slow/low-memory observation reduces the next concurrency ceiling. Recovery
    increases it by one only after three healthy observations. A profile is an
    explicit local calibration artifact, never silently invented from model IO.
    """
    def __init__(self):
        self.ceiling = 4
        self.healthy = 0
        self.lock = threading.Lock()

    def plan(self, items, phase):
        import psutil
        requested = os.environ.get("H3_MEDIA_WORKERS", "auto")
        if requested not in {"auto", "1", "2", "3", "4"}:
            raise ValueError("H3_MEDIA_WORKERS must be auto or 1..4")
        cpu = _cpu_slots()
        free = psutil.virtual_memory().available
        # This coarse guard protects the uncalibrated serial path. It is not a
        # promise that a codec's true peak is bounded by one GiB.
        if free < (RESERVE_MIB + 1024) * MIB:
            raise MemoryError("Less than 3 GiB available RAM for reference preparation")
        with self.lock:
            ceiling = self.ceiling
        maximum = min(cpu, ceiling)
        if requested != "auto":
            maximum = min(maximum, int(requested))
        profile_path = os.environ.get("H3_MEDIA_PROFILE", "")
        if not profile_path:
            # A manual ceiling does not substitute for measured child costs.
            return Allocation((tuple(range(len(items))),), 0.0, "uncalibrated: serial; native threads/peak memory not bounded"), 1
        profile_bytes = Path(profile_path).read_bytes()
        profile = json.loads(profile_bytes)
        if not isinstance(profile, dict) or profile.get("schema_version") != 1:
            raise ValueError("Unsupported media calibration schema")
        if profile.get("cpu_slots") != cpu:
            raise ValueError("Media calibration CPU affinity differs from this machine")
        metadata = {"sha256": hashlib.sha256(profile_bytes).hexdigest(), "schema_version": 1,
                    "affinity_cpu_slots": cpu, "phase": phase}
        rates = _phase_table(profile, "seconds_per_mib", phase)
        if any(item["kind"] not in rates for item in items):
            return Allocation((tuple(range(len(items))),), 0.0, "missing modality calibration: serial",
                              profile_metadata=metadata), 1
        threads = profile.get("threads_per_worker")
        memory_rates = _phase_table(profile, "task_memory_mib", phase)
        if threads is None or any(item["kind"] not in memory_rates for item in items):
            return Allocation((tuple(range(len(items))),), 0.0, "missing thread/memory calibration: serial",
                              profile_metadata=metadata), 1
        if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
            raise ValueError("threads_per_worker must be a positive measured integer")
        demands = [float(memory_rates[item["kind"]]) for item in items]
        if any(not math.isfinite(value) or value <= 0 for value in demands):
            raise ValueError("task_memory_mib estimates must be finite and positive")
        if not items:
            return Allocation((), 0.0, "no children"), 0
        task_memory = max(demands)
        # Prepared objects survive until encoding. Conservatively reserve each
        # child's estimated peak again as its retained-result budget; then admit
        # temporary concurrent working sets. Estimates need matching input bounds.
        retained = sum(demands) if phase == "prepare" else 0
        memory_slots = max(0, math.floor((free / MIB - RESERVE_MIB - retained) / task_memory))
        cpu_budget = cpu // threads
        maximum = min(maximum, memory_slots, cpu_budget)
        if maximum < 1:
            raise MemoryError("Calibrated reference working sets exceed the available CPU/memory budget")
        sizes = [max(Path(item["path"]).stat().st_size / MIB, 0.01) for item in items]
        costs = [float(rates[item["kind"]]) * size for item, size in zip(items, sizes)]
        allocation = optimal_allocation(costs, sizes, cpu_slots=cpu_budget, memory_slots=memory_slots,
                                        bandwidth_mib_s=float(profile.get("bandwidth_mib_s", 0)),
                                        max_workers=maximum,
                                        overhead_seconds=float(profile.get("overhead_seconds", 0.002)))
        metadata.update(cpu_budget_slots=cpu_budget, memory_budget_slots=memory_slots,
                        estimated_child_seconds=costs, estimated_child_memory_mib=demands,
                        bandwidth_mib_s=float(profile["bandwidth_mib_s"]),
                        overhead_seconds=float(profile.get("overhead_seconds", 0.002)),
                        retained_result_reserve_mib=retained)
        return Allocation(allocation.lanes, allocation.predicted_seconds, allocation.reason,
                          task_memory, threads, metadata), memory_slots

    def observe(self, allocation, elapsed, available_slots, succeeded):
        with self.lock:
            slow = allocation.predicted_seconds > 0 and elapsed > allocation.predicted_seconds * 1.25
            if not succeeded or slow or available_slots < len(allocation.lanes):
                self.ceiling = max(1, min(self.ceiling - 1, len(allocation.lanes)))
                self.healthy = 0
            else:
                self.healthy += 1
                if self.healthy >= 3:
                    self.ceiling = min(4, self.ceiling + 1)
                    self.healthy = 0


CONTROLLER = MediaController()


def _children(items, function, phase, metrics):
    if not items:
        return []
    allocation, slots = CONTROLLER.plan(items, phase)
    attribution = _input_attribution(items)
    attribution["profile"] = allocation.profile_metadata
    results = [None] * len(items)
    stop = threading.Event()
    started, succeeded = time.perf_counter(), False
    _schedule_log("H3_SCHEDULE", {"phase": phase, "workers": len(allocation.lanes),
                  "predicted_seconds": allocation.predicted_seconds, "reason": allocation.reason,
                  "estimated_task_memory_mib": allocation.task_memory_mib,
                  "estimated_threads_per_worker": allocation.threads_per_worker,
                  **attribution}, metrics)

    def lane(indices):
        for index in indices:
            if stop.is_set():
                return
            try:
                if metrics is None:
                    results[index] = function(items[index])
                else:
                    with metrics.stage(f"child.{phase}.{index}"):
                        results[index] = function(items[index])
            except BaseException:
                stop.set()
                raise

    try:
        if len(allocation.lanes) == 1:
            lane(allocation.lanes[0])
        else:
            # Structured lifetime: cancel unstarted work, join started work before
            # returning or raising. No orphan children overlap the next GPU phase.
            with ThreadPoolExecutor(max_workers=len(allocation.lanes), thread_name_prefix="h3-media") as pool:
                futures = [pool.submit(lane, indices) for indices in allocation.lanes]
                try:
                    for future in as_completed(futures):
                        future.result()
                except BaseException:
                    stop.set()
                    for future in futures:
                        future.cancel()
                    raise
        succeeded = True
        return results
    finally:
        elapsed = time.perf_counter() - started
        if not succeeded:
            results.clear()  # Release completed child objects before re-raising.
        try:
            import psutil
            free_mib = psutil.virtual_memory().available / MIB
            estimate = allocation.task_memory_mib or 1024
            remaining = max(0, math.floor((free_mib - RESERVE_MIB) / estimate))
            CONTROLLER.observe(allocation, elapsed, remaining, succeeded)
            _schedule_log("H3_SCHEDULE_FEEDBACK", {"phase": phase, "wall_seconds": elapsed,
                          "succeeded": succeeded, "next_ceiling": CONTROLLER.ceiling,
                          "workers": len(allocation.lanes), **attribution}, metrics)
        except Exception as feedback_error:
            # A missing/denied counter or telemetry failure must never mask the
            # original child exception or discard successfully prepared media.
            _schedule_log("H3_SCHEDULE_FEEDBACK", {"phase": phase, "feedback": "unavailable",
                          "error_type": type(feedback_error).__name__, "wall_seconds": elapsed,
                          "succeeded": succeeded, **attribution}, metrics)


def _input_attribution(items):
    children = []
    for index, item in enumerate(items):
        kind = item.get("kind") if isinstance(item, dict) else None
        size = None
        if isinstance(item, dict) and "path" in item:
            try:
                size = Path(item["path"]).stat().st_size
            except (OSError, TypeError, ValueError):
                pass
        children.append({"index": index, "kind": kind, "input_bytes": size})
    complete = all(child["input_bytes"] is not None for child in children)
    return {"child_count": len(children), "children": children,
            "input_bytes": sum(child["input_bytes"] for child in children) if complete else None}


def _schedule_log(prefix, record, metrics=None):
    try:
        print("\n" + prefix + " " + json.dumps(record), flush=True)
    except (OSError, ValueError):
        pass
    try:
        recorder = getattr(metrics, "record_schedule", None)
        if recorder is not None:
            recorder({"event": prefix, **record})
    except Exception:
        pass  # Attribution must not alter the child outcome.


def prepare_references(references, classes, probe, normalize, metrics=None):
    """Probe barrier -> aggregate validation -> bounded decode barrier, in order."""
    normalized, refs = [], []
    lock = threading.Lock()
    try:
        durations = _children(references, lambda item: probe(Path(item["path"]), item["kind"]),
                              "probe", metrics)
        for kind in ("video", "audio"):
            if sum(duration for item, duration in zip(references, durations) if item["kind"] == kind) > 15:
                raise ValueError("Total reference duration per modality must be <=15 seconds")

        def prepare(item):
            path = Path(item["path"])
            if item["kind"] == "video":
                # Known target is registered before conversion, so failed ffmpeg
                # outputs are cleaned too. normalize uses this same target.
                target = path.with_name(path.stem + "_24fps.mp4")
                with lock:
                    normalized.append(target)
                path = normalize(path)
                with lock:
                    if path not in normalized:
                        normalized.append(path)
            return classes[item["kind"]].from_file(str(path))

        refs = _children(references, prepare, "prepare", metrics)
        return refs, normalized
    except BaseException:
        refs.clear()
        for path in normalized:
            path.unlink(missing_ok=True)
        raise
