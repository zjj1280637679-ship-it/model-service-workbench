"""Per-job working-set admission and stage leases; no global job-count cap."""
from __future__ import annotations

from contextlib import contextmanager
import math
import os
from pathlib import Path
import shutil
import sys
import threading

import psutil

GiB = 1024 ** 3


def cgroup_cpu_quotas(root=Path('/sys/fs/cgroup'), membership=Path('/proc/self/cgroup')):
    """Read v1/v2 quota limits, including stricter visible parent cgroups."""
    root = Path(root).resolve()
    candidates = {(root, 'v2'), (root / 'cpu', 'v1'), (root / 'cpu,cpuacct', 'v1')}
    try:
        records = Path(membership).read_text().splitlines()
    except OSError:
        records = []
    for record in records:
        parts = record.split(':', 2)
        if len(parts) != 3:
            continue
        controllers, relative = parts[1], parts[2].lstrip('/')
        bases = [(root, 'v2')] if not controllers else (
            [(root / 'cpu', 'v1'), (root / 'cpu,cpuacct', 'v1')] if 'cpu' in controllers.split(',') else [])
        for base, version in bases:
            leaf = (base / relative).resolve()
            if not leaf.is_relative_to(base):
                continue
            for directory in (leaf, *leaf.parents):
                if not directory.is_relative_to(base):
                    break
                candidates.add((directory, version))
    limits = []
    for directory, version in sorted(candidates):
        try:
            if version == 'v2':
                filename = directory / 'cpu.max'
                quota, period = filename.read_text().split()
                if quota == 'max':
                    continue
            else:
                filename = directory / 'cpu.cfs_quota_us'
                quota = filename.read_text().strip()
                period = (directory / 'cpu.cfs_period_us').read_text().strip()
            quota, period = int(quota), int(period)
            if quota > 0 and period > 0:
                limits.append({'source': str(filename), 'cores': quota / period})
        except (OSError, ValueError):
            continue
    return limits


def detect_cpu_capacity():
    logical = max(1, os.cpu_count() or 1)
    affinity = None
    try:
        affinity = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        try:
            affinity = len(psutil.Process().cpu_affinity())
        except (AttributeError, psutil.Error):
            pass
    quota_limits = cgroup_cpu_quotas() if sys.platform.startswith('linux') else []
    effective = min([logical] + ([affinity] if affinity else []) + [value['cores'] for value in quota_limits])
    return {'logical_cpus': logical, 'affinity_cpus': affinity, 'cgroup_quotas': quota_limits,
            'effective_cpu_cores': effective, 'usable_cpu_threads': max(1, math.floor(effective)),
            'fractional_quota_note': 'A quota below one CPU uses one throttled lane, not one full CPU'}


class BudgetUnavailable(Exception):
    def __init__(self, reason, permanent=False):
        super().__init__(reason)
        self.permanent = permanent


class PipelineBudget:
    def __init__(self, root, *, limits=None, model_peak_ram_bytes=0):
        self.root = Path(root)
        self._condition = threading.Condition()
        self._reservations = {}
        self._result_bytes = {}
        self._cpu_used = 0
        self._gpu_owner = None
        self._model_peak = int(model_peak_ram_bytes)
        self._system_checks = limits is None
        self._cpu_capacity = detect_cpu_capacity()
        self._os_reserve = int(os.environ.get("H3_PIPELINE_OS_RESERVE_MIB", "2048")) * 1024 ** 2
        defaults = {
            "ram_bytes": int(os.environ.get("H3_PIPELINE_RAM_MIB", "8192")) * 1024 ** 2,
            "disk_bytes": max(0, shutil.disk_usage(self.root).free - GiB),
            "cpu_threads": int(os.environ.get("H3_PIPELINE_CPU_THREADS", str(self._cpu_capacity['usable_cpu_threads']))),
        }
        self.limits = defaults if limits is None else {**defaults, **limits}
        if any(type(value) is not int or value <= 0 for value in self.limits.values()):
            raise ValueError("Pipeline budgets must be positive integers")
        self._requested_cpu_threads = self.limits['cpu_threads']
        self.limits['cpu_threads'] = min(self.limits['cpu_threads'], self._cpu_capacity['usable_cpu_threads'])
        self._requested_gpu_cpu_threads = int(os.environ.get('H3_PIPELINE_GPU_CPU_THREADS', '1'))
        if self._requested_gpu_cpu_threads < 1:
            raise ValueError('GPU host CPU reservation must be positive')
        self.gpu_host_cpu_threads = min(self._requested_gpu_cpu_threads, self.limits['cpu_threads'])
        self.native_cpu_status = {'configured': False, 'reason': 'Backend native CPU configuration pending'}

    def _used(self, name):
        return sum(value[name] for value in self._reservations.values()) + (
            sum(self._result_bytes.values()) if name == "ram_bytes" else 0)

    def reserve(self, identity, resources):
        charge = {name: resources[name] for name in ("ram_bytes", "disk_bytes", "cpu_threads")}
        if any(type(value) is not int or value <= 0 for value in charge.values()):
            raise BudgetUnavailable("invalid_backend_resource_estimate", permanent=True)
        with self._condition:
            if identity in self._reservations:
                return
            if any(charge[name] > self.limits[name] for name in charge):
                raise BudgetUnavailable("request_exceeds_working_set_budget", permanent=True)
            if any(self._used(name) + charge[name] > self.limits[name] for name in ("ram_bytes", "disk_bytes")):
                raise BudgetUnavailable("resource_budget_exhausted")
            if self._system_checks:
                # Existing job memory must not be misclassified as resident model
                # memory when allowing for the model's much larger inference RSS.
                rss = psutil.Process().memory_info().rss
                resident_model = max(0, rss - self._used("ram_bytes"))
                future_model = max(0, self._model_peak - resident_model)
                if psutil.virtual_memory().available < future_model + self._os_reserve + self._used("ram_bytes") + charge["ram_bytes"]:
                    raise BudgetUnavailable("live_host_memory_pressure")
                if shutil.disk_usage(self.root).free < GiB + self._used("disk_bytes") + charge["disk_bytes"]:
                    raise BudgetUnavailable("live_disk_pressure")
            self._reservations[identity] = charge

    def finish(self, identity, result_bytes=0):
        with self._condition:
            self._reservations.pop(identity, None)
            if result_bytes:
                self._result_bytes[identity] = result_bytes
            self._condition.notify_all()

    def acknowledge(self, identity):
        with self._condition:
            self._result_bytes.pop(identity, None)
            self._condition.notify_all()

    @contextmanager
    def stage(self, identity, kind, *, cpu_threads=1, cancelled=lambda: False):
        if kind not in {'cpu', 'gpu'}:
            raise ValueError('Unknown resource stage')
        charge = self.gpu_host_cpu_threads if kind == 'gpu' else cpu_threads
        if type(charge) is not int or not 1 <= charge <= self.limits['cpu_threads']:
            raise ValueError('Stage CPU requirement exceeds the available CPU budget')
        acquired = False
        with self._condition:
            while not cancelled():
                if kind == "gpu" and self._gpu_owner is None and self._cpu_used + charge <= self.limits['cpu_threads']:
                    self._gpu_owner, acquired = identity, True
                    self._cpu_used += charge
                    break
                if kind == "cpu" and self._cpu_used + charge <= self.limits["cpu_threads"]:
                    self._cpu_used += charge
                    acquired = True
                    break
                self._condition.wait(.05)
        try:
            yield acquired
        finally:
            if acquired:
                with self._condition:
                    if kind == "gpu":
                        self._gpu_owner = None
                    self._cpu_used -= charge
                    self._condition.notify_all()

    def snapshot(self):
        with self._condition:
            used = {name: self._used(name) for name in ("ram_bytes", "disk_bytes")}
            return {"limits": dict(self.limits), "reserved": used,
                    "free": {name: self.limits[name] - used[name] for name in used},
                    "cpu_threads_in_use": self._cpu_used, "gpu_attempt_id": self._gpu_owner,
                    "gpu_host_cpu_threads_in_use": self.gpu_host_cpu_threads if self._gpu_owner else 0,
                    "cpu_stage_threads_in_use": self._cpu_used - (self.gpu_host_cpu_threads if self._gpu_owner else 0),
                    "cpu_count_scope": "Reserved native execution lanes; not sampled CPU utilization",
                    "gpu_host_cpu_threads": self.gpu_host_cpu_threads,
                    "requested_cpu_threads": self._requested_cpu_threads,
                    "requested_gpu_host_cpu_threads": self._requested_gpu_cpu_threads,
                    "cpu_capacity": self._cpu_capacity, 'native_cpu': self.native_cpu_status,
                    "io_budget": {"scope": "disk_capacity_only", "bandwidth_calibrated": False,
                                  "bytes_per_second_limit": None,
                                  "optimality": "Conservative admission estimates; no mathematical optimum claimed"},
                    "admitted_attempts": len(self._reservations),
                    "unacknowledged_result_bytes": sum(self._result_bytes.values()),
                    "model_peak_ram_bytes": self._model_peak,
                    "os_reserve_bytes": self._os_reserve,
                    "live_pressure_checks": self._system_checks}

    def has_capacity(self):
        state = self.snapshot()
        return all(value > 0 for value in state["free"].values())
