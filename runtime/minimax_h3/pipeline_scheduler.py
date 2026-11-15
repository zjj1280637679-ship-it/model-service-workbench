"""Per-job working-set admission and stage leases; no global job-count cap."""
from __future__ import annotations

from contextlib import contextmanager
from collections import OrderedDict, deque
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import threading
import time
import uuid

import psutil

from condition_guard import evaluate_gate, negate_observation

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
        self._reservation_ids = {}
        self._cancellations = {}
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
        self.instance_id = uuid.uuid4().hex
        self._source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self._last_gates = {}
        self._attempt_gates = OrderedDict()
        self._active_leases = {}
        self._released_leases = OrderedDict()
        self._lease_history = deque(maxlen=512)
        self._lease_event_count = 0

    def _capacity_version(self):
        payload = {'limits': self.limits, 'gpu_host_cpu_threads': self.gpu_host_cpu_threads}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def _gate(self, identity, kind, observations, context=None):
        # Called with our Condition held. No callbacks into other ownership domains.
        receipt = evaluate_gate('pipeline.' + kind, list(observations), observations,
                                {**(context or {}), 'budget_instance_id': self.instance_id,
                                 'attempt_id': identity, 'resource_kind': kind,
                                 'reservation_id': self._reservation_ids.get(identity),
                                 'capacity_version': self._capacity_version(),
                                 'source_file': 'pipeline_scheduler.py',
                                 'source_sha256': self._source_sha256})
        self._last_gates[receipt['gate_id']] = receipt
        key = (identity, kind)
        self._attempt_gates[key] = receipt
        self._attempt_gates.move_to_end(key)
        while len(self._attempt_gates) > 128:
            self._attempt_gates.popitem(last=False)
        return receipt

    def gate_receipt(self, identity, kind):
        with self._condition:
            return copy.deepcopy(self._attempt_gates.get((identity, kind)))

    def reservation_witness(self, identity):
        with self._condition:
            return {'owned': identity in self._reservations,
                    'budget_instance_id': self.instance_id, 'attempt_id': identity,
                    'reservation_id': self._reservation_ids.get(identity),
                    'resources': copy.deepcopy(self._reservations.get(identity)),
                    'capacity_version': self._capacity_version()}

    def lease_witness(self, identity, kind):
        with self._condition:
            return copy.deepcopy(next((lease for lease in self._active_leases.values()
                                       if lease['attempt_id'] == identity and lease['kind'] == kind), None))

    def release_witness(self, lease_id):
        with self._condition:
            return copy.deepcopy(self._released_leases.get(lease_id))

    def set_cancelled(self, identity, observation):
        """Update cancellation evidence in the same domain as budget grants.

        Model cancellation also owns its Event and rechecks under its own lock.
        Updating this ledger does not interrupt a lease already granted.
        """
        with self._condition:
            if identity not in self._reservations:
                return False
            self._cancellations[identity] = copy.deepcopy(observation)
            self._condition.notify_all()
            return True

    def _lease_event(self, event_type, lease):
        self._lease_event_count += 1
        event = {key: copy.deepcopy(lease[key]) for key in
                 ('lease_id', 'budget_instance_id', 'attempt_id', 'kind', 'cpu_threads',
                  'capacity_version', 'decision_id', 'reservation_id')}
        event.update(event_id=uuid.uuid4().hex, sequence=self._lease_event_count,
                     type=event_type, observed_at=time.time(), monotonic_ns=time.monotonic_ns(),
                     cpu_threads_in_use=self._cpu_used, gpu_attempt_id=self._gpu_owner,
                     limits=dict(self.limits))
        self._lease_history.append(event)
        return event

    def _used(self, name):
        return sum(value[name] for value in self._reservations.values()) + (
            sum(self._result_bytes.values()) if name == "ram_bytes" else 0)

    def reserve(self, identity, resources):
        if type(identity) is not str or not 1 <= len(identity) <= 256:
            raise BudgetUnavailable('invalid_attempt_identity', permanent=True)
        charge = {name: resources[name] for name in ("ram_bytes", "disk_bytes", "cpu_threads")}
        if any(type(value) is not int or value <= 0 for value in charge.values()):
            raise BudgetUnavailable("invalid_backend_resource_estimate", permanent=True)
        with self._condition:
            if identity in self._reservations:
                same = self._reservations[identity] == charge
                self._gate(identity, 'reserve.existing', {'reservation_matches': same})
                if not same:
                    raise BudgetUnavailable('reservation_identity_conflict', permanent=True)
                return
            fits = all(charge[name] <= self.limits[name] for name in charge)
            room = all(self._used(name) + charge[name] <= self.limits[name]
                       for name in ('ram_bytes', 'disk_bytes'))
            host_available = disk_available = True
            if self._system_checks:
                try:
                    # Existing task memory is not part of the resident model.
                    rss = psutil.Process().memory_info().rss
                    resident_model = max(0, rss - self._used('ram_bytes'))
                    future_model = max(0, self._model_peak - resident_model)
                    host_available = psutil.virtual_memory().available >= (
                        future_model + self._os_reserve + self._used('ram_bytes') + charge['ram_bytes'])
                    disk_available = shutil.disk_usage(self.root).free >= GiB + self._used('disk_bytes') + charge['disk_bytes']
                except (OSError, psutil.Error):
                    host_available = disk_available = None
            receipt = self._gate(identity, 'reserve', {
                'estimate_within_limits': fits, 'working_set_available': room,
                'host_memory_available': host_available, 'disk_available': disk_available,
                'result_identity_unused': identity not in self._result_bytes})
            if not fits:
                raise BudgetUnavailable("request_exceeds_working_set_budget", permanent=True)
            if not room:
                raise BudgetUnavailable("resource_budget_exhausted")
            if host_available is False:
                raise BudgetUnavailable('live_host_memory_pressure')
            if disk_available is False:
                raise BudgetUnavailable('live_disk_pressure')
            if not receipt['allowed']:
                raise BudgetUnavailable('reservation_conditions_unconfirmed')
            self._reservations[identity] = charge
            self._reservation_ids[identity] = uuid.uuid4().hex
            self._cancellations[identity] = False

    def finish(self, identity, result_bytes=0):
        with self._condition:
            if any(lease['attempt_id'] == identity for lease in self._active_leases.values()):
                raise BudgetUnavailable('attempt_still_holds_stage_lease')
            self._reservations.pop(identity, None)
            self._reservation_ids.pop(identity, None)
            self._cancellations.pop(identity, None)
            if result_bytes:
                self._result_bytes[identity] = result_bytes
            self._condition.notify_all()

    def acknowledge(self, identity):
        with self._condition:
            self._result_bytes.pop(identity, None)
            self._condition.notify_all()

    @contextmanager
    def stage(self, identity, kind, *, cpu_threads=1, cancelled=lambda: False,
              cancellation_event=None, context=None):
        if kind not in {'cpu', 'gpu'}:
            raise ValueError('Unknown resource stage')
        charge = self.gpu_host_cpu_threads if kind == 'gpu' else cpu_threads
        if type(charge) is not int or not 1 <= charge <= self.limits['cpu_threads']:
            raise ValueError('Stage CPU requirement exceeds the available CPU budget')
        acquired = False
        lease_id = None
        with self._condition:
            reservation_id = self._reservation_ids.get(identity)
        while True:
            # Legacy arbitrary callbacks may take a model lock, so never invoke
            # them while holding the budget lock. Their sampled state does not
            # claim global atomicity; the model rechecks before actual effects.
            cancellation = None
            if cancellation_event is None:
                try:
                    cancellation = cancelled()
                except Exception:
                    pass
            with self._condition:
                if cancellation_event is not None:
                    # Event.is_set is a local flag read, never a foreign callback.
                    cancellation = (threading.Event.is_set(cancellation_event)
                                    if type(cancellation_event) is threading.Event else None)
                observations = {
                    'reservation_owned': reservation_id is not None and self._reservation_ids.get(identity) == reservation_id,
                    'cancellation_clear': negate_observation(self._cancellations.get(identity)),
                    'not_cancelled': negate_observation(cancellation),
                    'attempt_stage_idle': not any(lease['attempt_id'] == identity for lease in self._active_leases.values()),
                    'cpu_grant_available': self._cpu_used + charge <= self.limits['cpu_threads'],
                }
                if kind == 'gpu':
                    observations['gpu_grant_available'] = self._gpu_owner is None
                receipt = self._gate(identity, 'stage.' + kind, observations, context)
                if receipt['allowed']:
                    # The declaration and its resource effect share this lock.
                    self._cpu_used += charge
                    if kind == 'gpu':
                        self._gpu_owner = identity
                    lease_id = uuid.uuid4().hex
                    lease = {'owned': True, 'lease_id': lease_id, 'attempt_id': identity, 'kind': kind,
                             'budget_instance_id': self.instance_id, 'reservation_id': reservation_id,
                             'cpu_threads': charge, 'capacity_version': receipt['context']['capacity_version'],
                             'decision_id': receipt['decision_id'], 'grant_receipt': receipt,
                             'granted_at': time.time(), 'granted_monotonic_ns': time.monotonic_ns()}
                    self._active_leases[lease_id] = lease
                    self._lease_event('lease.acquired', lease)
                    acquired = True
                    break
                if any(receipt['conditions'][name] != 'true' for name in
                       ('reservation_owned', 'cancellation_clear', 'not_cancelled')):
                    break
                # Only actual contention waits; wait releases the lock and thus
                # cannot serialize unrelated CPU/GPU releases or admissions.
                self._condition.wait(.05)
        try:
            yield acquired
        finally:
            if acquired:
                with self._condition:
                    lease = self._active_leases.get(lease_id)
                    if lease is None or lease['attempt_id'] != identity or lease['kind'] != kind:
                        raise RuntimeError('Stage lease ownership was lost')
                    if kind == "gpu":
                        if self._gpu_owner != identity:
                            raise RuntimeError('GPU lease ownership was lost')
                        self._gpu_owner = None
                    self._cpu_used -= charge
                    self._active_leases.pop(lease_id)
                    lease = {**lease, 'owned': False, 'released_at': time.time(),
                             'released_monotonic_ns': time.monotonic_ns()}
                    self._released_leases[lease_id] = lease
                    while len(self._released_leases) > 256:
                        self._released_leases.popitem(last=False)
                    self._lease_event('lease.released', lease)
                    self._condition.notify_all()

    def snapshot(self):
        with self._condition:
            used = {name: self._used(name) for name in ("ram_bytes", "disk_bytes")}
            return {"limits": dict(self.limits), "reserved": used,
                    'budget_instance_id': self.instance_id,
                    'capacity_version': self._capacity_version(),
                    'last_gate_receipts': copy.deepcopy(self._last_gates),
                    'active_leases': copy.deepcopy(list(self._active_leases.values())),
                    'lease_history': {'capacity': self._lease_history.maxlen,
                                      'events': copy.deepcopy(list(self._lease_history)),
                                      'count': self._lease_event_count,
                                      'dropped': self._lease_event_count - len(self._lease_history),
                                      'history_complete': self._lease_event_count == len(self._lease_history)},
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
