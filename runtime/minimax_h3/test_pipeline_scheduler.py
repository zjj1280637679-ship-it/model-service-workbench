"""CPU stage barriers verify real overlap/ownership without pretending to generate."""
import json
from pathlib import Path
import threading
from types import SimpleNamespace
import uuid

import pytest

from control_protocol import ControlError
from model_container import ModelContainer
from pipeline_scheduler import BudgetUnavailable, PipelineBudget
from test_model_container import Backend, eventually, fixture_artifact, ready, request


class StageBackend(Backend):
    pipeline_model_peak_ram_bytes = 0

    def __init__(self):
        super().__init__()
        self.gates = {}
        self.started = {}
        self.lock = threading.Lock()
        self.gpu_active = 0
        self.max_gpu_active = 0
        self.cleanup_jobs = []

    def gate(self, stage, name, *, blocked=False):
        key = stage, name
        if key not in self.gates:
            self.gates[key] = threading.Event()
            self.started[key] = threading.Event()
            if not blocked:
                self.gates[key].set()
        return self.gates[key]

    def enter(self, stage, name):
        gate = self.gate(stage, name)
        self.started[stage, name].set()
        assert gate.wait(5), "Test must release stage barrier"

    def estimate_resources(self, parameters):
        return {"ram_bytes": 128, "disk_bytes": 64, "cpu_threads": 1}

    def prepare(self, **parameters):
        name = parameters["prompt"]
        self.enter("prepare", name)
        return SimpleNamespace(name=name, metrics={"job_marker": name})

    def infer(self, prepared):
        with self.lock:
            self.gpu_active += 1
            self.max_gpu_active = max(self.max_gpu_active, self.gpu_active)
        try:
            self.enter("gpu", prepared.name)
            return SimpleNamespace(name=prepared.name, metrics=prepared.metrics, cpu_handoff_complete=True)
        finally:
            with self.lock:
                self.gpu_active -= 1

    def encode(self, result, output_path):
        self.enter("encode", result.name)
        Path(output_path).write_bytes(result.name.encode())
        return result.metrics

    def cleanup(self, prepared=None, result=None):
        self.cleanup_jobs.append((prepared or result).name if prepared or result else None)

    def release_all(self):
        for event in self.gates.values():
            event.set()


@pytest.fixture
def pipeline(tmp_path):
    values = []

    def factory(backend=None, **limits):
        backend = backend or StageBackend()
        model = ModelContainer(tmp_path, uuid.uuid4().hex, backend=backend, artifact_validator=fixture_artifact,
                               resource_limits={"ram_bytes": 16_384, "disk_bytes": 4096, "cpu_threads": 2, **limits})
        values.append((model, backend))
        model.start()
        ready(model)
        return model, backend

    yield factory
    for model, backend in values:
        backend.release_all()
        model.stop()
        assert model.join(5)


def submit(model, name):
    payload = request(model)
    payload["parameters"]["prompt"] = name
    model.execute(payload)
    return payload


def terminal(model, identity):
    return eventually(lambda: next((result for result in model.status()["results"]
                                    if result["attempt_id"] == identity), None))


def test_cross_job_overlap_single_gpu_and_results_survive_until_ack(pipeline):
    model, backend = pipeline()
    backend.gate("gpu", "A", blocked=True)
    backend.gate("encode", "A", blocked=True)
    backend.gate("gpu", "B", blocked=True)
    first = submit(model, "A")
    assert backend.started["gpu", "A"].wait(2)
    second = submit(model, "B")
    eventually(lambda: ("prepare", "B") in backend.started and backend.started["prepare", "B"].is_set())
    assert backend.gpu_active == 1
    assert not backend.started["gpu", "B"].is_set()
    backend.gates["gpu", "A"].set()
    assert backend.started["encode", "A"].wait(2)
    assert backend.started["gpu", "B"].wait(2)
    assert not model.status()["results"]
    backend.release_all()
    one = terminal(model, first["attempt_id"])
    two = terminal(model, second["attempt_id"])
    assert one["status"] == two["status"] == "succeeded"
    assert one["metrics"]["job_marker"] == "A" and two["metrics"]["job_marker"] == "B"
    assert backend.max_gpu_active == 1
    assert len(model.status()["results"]) == 2
    with pytest.raises(ControlError):
        model.ack_results({"instance_id": "0" * 32, "attempt_ids": [first["attempt_id"]]})
    ack = {"instance_id": model.instance_id, "attempt_ids": [first["attempt_id"]]}
    assert model.ack_results(ack)["acknowledged"] == [first["attempt_id"]]
    assert model.ack_results(ack)["acknowledged"] == [first["attempt_id"]]
    assert first["attempt_id"] not in model._attempts
    assert model.execute(first)["result"] == one
    assert [result["attempt_id"] for result in model.status()["results"]] == [second["attempt_id"]]
    events = [json.loads(line) for line in model.events_path.read_text().splitlines()]
    gpu_events = [event for event in events if event.get("stage") == "gpu"]
    assert [event["type"] for event in gpu_events] == ["stage.started", "stage.finished"] * 2
    assert [event["sequence"] for event in events] == list(range(1, len(events) + 1))


@pytest.mark.parametrize("phase", ["preparing", "gpu_wait", "gpu_running", "encoding"])
def test_cancel_at_each_pipeline_boundary_never_runs_later_gpu(pipeline, phase):
    model, backend = pipeline()
    if phase in {"preparing", "gpu_wait"}:
        backend.gate("gpu", "owner", blocked=True)
        owner = submit(model, "owner")
        assert backend.started["gpu", "owner"].wait(2)
    if phase == "preparing":
        backend.gate("prepare", "target", blocked=True)
    if phase == "gpu_running":
        backend.gate("gpu", "target", blocked=True)
    if phase == "encoding":
        backend.gate("encode", "target", blocked=True)
    target = submit(model, "target")
    eventually(lambda: any(a["attempt_id"] == target["attempt_id"] and a["status"] == phase
                           for a in model.status()["attempts"]))
    result = model.cancel({key: target[key] for key in ("instance_id", "job_id", "attempt_id")})
    assert result["status"] == "cancel_requested"
    if phase != "gpu_wait":
        assert not any(r["attempt_id"] == target["attempt_id"] for r in model.status()["results"])
    backend.release_all()
    result = terminal(model, target["attempt_id"])
    assert result["status"] == "cancelled" and result["artifact"] is None and result["cleanup_complete"]
    assert not (model.root / target["job_id"] / "result.mp4").exists()
    if phase in {"preparing", "gpu_wait"}:
        events = [json.loads(line) for line in model.events_path.read_text().splitlines()]
        assert not any(event.get("stage") == "gpu" and event["attempt_id"] == target["attempt_id"] for event in events)


def test_budget_rejection_does_not_accept_or_consume_inputs(pipeline):
    model, backend = pipeline(ram_bytes=256)
    backend.gate("gpu", "one", blocked=True)
    first = submit(model, "one")
    assert backend.started["gpu", "one"].wait(2)
    submit(model, "two")
    third = request(model)
    with pytest.raises(ControlError) as rejected:
        model.execute(third)
    assert rejected.value.status in {409, 429}
    assert json.loads(str(rejected.value))["guarantees_not_accepted"] is True
    assert third["attempt_id"] not in model._attempts
    assert (model.root / third["job_id"] / "inputs").is_dir()
    assert model.status()["budget"]["admitted_attempts"] == 2


def test_stop_joins_all_cpu_and_gpu_stages_before_unload(pipeline):
    model, backend = pipeline()
    backend.gate("encode", "A", blocked=True)
    backend.gate("gpu", "B", blocked=True)
    first = submit(model, "A")
    assert backend.started["encode", "A"].wait(2)
    second = submit(model, "B")
    assert backend.started["gpu", "B"].wait(2)
    model.stop()
    assert not model.join(.05) and not backend.unloaded
    backend.gates["gpu", "B"].set()
    terminal(model, second["attempt_id"])
    assert not model.join(.05) and not backend.unloaded
    backend.gates["encode", "A"].set()
    assert model.join(3) and backend.unloaded
    assert terminal(model, first["attempt_id"])["status"] == "succeeded"
    assert model.status()["cleanup_complete"] is False


def test_unconfirmed_gpu_handoff_revokes_waiters(pipeline):
    class Broken(StageBackend):
        def infer(self, prepared):
            result = super().infer(prepared)
            result.cpu_handoff_complete = False
            return result
    model, backend = pipeline(Broken())
    backend.gate("gpu", "first", blocked=True)
    first = submit(model, "first")
    assert backend.started["gpu", "first"].wait(2)
    second = submit(model, "second")
    eventually(lambda: any(a["attempt_id"] == second["attempt_id"] and a["status"] == "gpu_wait"
                           for a in model.status()["attempts"]))
    backend.release_all()
    assert terminal(model, first["attempt_id"])["status"] == "failed"
    assert terminal(model, second["attempt_id"])["status"] == "cancelled"
    assert model.join(3) and not model.status()["ready"]
    assert ("gpu", "second") not in backend.started


def test_resource_budget_has_no_job_count_cap_and_distinguishes_impossible(tmp_path):
    budget = PipelineBudget(tmp_path, limits={"ram_bytes": 1000, "disk_bytes": 1000, "cpu_threads": 2})
    for i in range(8):
        budget.reserve(str(i), {"ram_bytes": 100, "disk_bytes": 20, "cpu_threads": 1})
    assert budget.snapshot()["admitted_attempts"] == 8
    with pytest.raises(BudgetUnavailable) as impossible:
        budget.reserve("large", {"ram_bytes": 1001, "disk_bytes": 20, "cpu_threads": 1})
    assert impossible.value.permanent
    with pytest.raises(BudgetUnavailable) as busy:
        budget.reserve("busy", {"ram_bytes": 300, "disk_bytes": 20, "cpu_threads": 1})
    assert not busy.value.permanent


def test_cpu_stages_obey_shared_thread_budget(pipeline):
    model, backend = pipeline(cpu_threads=1)
    backend.gate("prepare", "A", blocked=True)
    backend.gate("prepare", "B", blocked=True)
    first = submit(model, "A")
    assert backend.started["prepare", "A"].wait(2)
    second = submit(model, "B")
    eventually(lambda: len(model._stage_workers) == 2)
    assert not backend.started["prepare", "B"].is_set()
    assert model.status()["budget"]["cpu_threads_in_use"] == 1
    backend.gates["prepare", "A"].set()
    assert backend.started["prepare", "B"].wait(2)
    backend.release_all()
    assert terminal(model, first["attempt_id"])["status"] == "succeeded"
    assert terminal(model, second["attempt_id"])["status"] == "succeeded"


def test_failed_cleanup_retains_reservation_and_revokes_accepting(pipeline):
    class Unclean(StageBackend):
        def cleanup(self, prepared=None, result=None):
            raise OSError("CPU workspace cleanup unavailable")
    model, backend = pipeline(Unclean())
    payload = submit(model, "unclean")
    result = terminal(model, payload["attempt_id"])
    assert result["status"] == "failed" and result["cleanup_complete"] is False
    assert model.status()["budget"]["admitted_attempts"] == 1
    assert not model.status()["accepting"]
    assert model.join(3)


def test_live_host_check_reserves_future_model_peak(tmp_path, monkeypatch):
    import pipeline_scheduler
    mib = 1024 ** 2
    monkeypatch.setenv("H3_PIPELINE_RAM_MIB", "8")
    monkeypatch.setenv("H3_PIPELINE_OS_RESERVE_MIB", "1")
    monkeypatch.setattr(pipeline_scheduler.psutil, "Process", lambda: SimpleNamespace(
        memory_info=lambda: SimpleNamespace(rss=10 * mib)))
    monkeypatch.setattr(pipeline_scheduler.psutil, "virtual_memory", lambda: SimpleNamespace(available=20 * mib))
    budget = PipelineBudget(tmp_path, model_peak_ram_bytes=29 * mib)
    with pytest.raises(BudgetUnavailable) as pressure:
        budget.reserve("job", {"ram_bytes": mib, "disk_bytes": mib, "cpu_threads": 1})
    assert str(pressure.value) == "live_host_memory_pressure"
    assert budget.snapshot()["admitted_attempts"] == 0


def test_cgroup_v2_ancestor_and_affinity_limit_cpu_configuration(tmp_path, monkeypatch):
    import pipeline_scheduler
    cgroup = tmp_path / 'cgroup'
    leaf = cgroup / 'parent' / 'child'
    leaf.mkdir(parents=True)
    (cgroup / 'cpu.max').write_text('max 100000')
    (leaf.parent / 'cpu.max').write_text('150000 100000')
    (leaf / 'cpu.max').write_text('400000 100000')
    membership = tmp_path / 'membership'
    membership.write_text('0::/parent/child\n')
    quotas = pipeline_scheduler.cgroup_cpu_quotas(cgroup, membership)
    assert sorted(item['cores'] for item in quotas) == [1.5, 4]
    monkeypatch.setattr(pipeline_scheduler.os, 'cpu_count', lambda: 32)
    monkeypatch.setattr(pipeline_scheduler.os, 'sched_getaffinity', lambda *_: set(range(8)), raising=False)
    monkeypatch.setattr(pipeline_scheduler.sys, 'platform', 'linux')
    monkeypatch.setattr(pipeline_scheduler, 'cgroup_cpu_quotas', lambda: quotas)
    monkeypatch.setenv('H3_PIPELINE_CPU_THREADS', '64')
    monkeypatch.setenv('H3_PIPELINE_GPU_CPU_THREADS', '8')
    budget = PipelineBudget(tmp_path)
    state = budget.snapshot()
    assert state['cpu_capacity']['effective_cpu_cores'] == 1.5
    assert state['limits']['cpu_threads'] == state['gpu_host_cpu_threads'] == 1
    assert state['requested_cpu_threads'] == 64
    assert state['io_budget']['scope'] == 'disk_capacity_only'
    assert state['io_budget']['bytes_per_second_limit'] is None
    assert state['io_budget']['bandwidth_calibrated'] is False


def test_cgroup_v1_quota_includes_parent_and_ignores_unlimited(tmp_path):
    from pipeline_scheduler import cgroup_cpu_quotas
    cgroup = tmp_path / 'cgroup'
    parent = cgroup / 'cpu'
    leaf = parent / 'session'
    leaf.mkdir(parents=True)
    for path, quota in ((parent, '100000'), (leaf, '200000')):
        (path / 'cpu.cfs_quota_us').write_text(quota)
        (path / 'cpu.cfs_period_us').write_text('100000')
    (cgroup / 'cpu.max').write_text('max 100000')
    membership = tmp_path / 'membership'
    membership.write_text('2:cpu,cpuacct:/session\n')
    assert sorted(item['cores'] for item in cgroup_cpu_quotas(cgroup, membership)) == [1, 2]


def test_affinity_is_default_when_no_cgroup_limit_is_visible(tmp_path, monkeypatch):
    import pipeline_scheduler
    monkeypatch.delenv('H3_PIPELINE_CPU_THREADS', raising=False)
    monkeypatch.setattr(pipeline_scheduler.os, 'cpu_count', lambda: 16)
    monkeypatch.setattr(pipeline_scheduler.os, 'sched_getaffinity', lambda *_: {1, 3}, raising=False)
    monkeypatch.setattr(pipeline_scheduler, 'cgroup_cpu_quotas', lambda: [])
    assert PipelineBudget(tmp_path).snapshot()['limits']['cpu_threads'] == 2


def test_gpu_host_and_encoding_share_two_cpu_lanes(tmp_path, monkeypatch):
    monkeypatch.setenv('H3_PIPELINE_GPU_CPU_THREADS', '1')
    budget = PipelineBudget(tmp_path, limits={'cpu_threads': 2})
    acquired, release = threading.Event(), threading.Event()
    def third_lane():
        with budget.stage('prepare-B', 'cpu') as held:
            assert held
            acquired.set()
            assert release.wait(3)
    worker = threading.Thread(target=third_lane)
    try:
        with budget.stage('GPU-B', 'gpu') as held:
            assert held and budget.snapshot()['cpu_threads_in_use'] == 1
            with budget.stage('encode-A', 'cpu') as held:
                assert held and budget.snapshot()['cpu_threads_in_use'] == 2
                worker.start()
                assert not acquired.wait(.05)
            assert acquired.wait(2)
            assert budget.snapshot()['gpu_attempt_id'] == 'GPU-B'
            assert budget.snapshot()['cpu_threads_in_use'] == 2
            release.set()
            worker.join(2)
            assert not worker.is_alive()
        assert budget.snapshot()['cpu_threads_in_use'] == 0
    finally:
        release.set()
        if worker.ident is not None:
            worker.join(3)


def test_native_cpu_configuration_occurs_before_model_load(pipeline, monkeypatch):
    monkeypatch.setenv('H3_PIPELINE_GPU_CPU_THREADS', '1')
    order = []
    class Configured(StageBackend):
        def configure_cpu_budget(self, threads):
            order.append(('configure', threads))
            return {'configured': True, 'torch_num_threads': threads, 'scope': 'test hook'}
        def load(self):
            assert order == [('configure', 1)]
            order.append(('load', None))
            super().load()
        def cpu_budget_snapshot(self):
            assert order == [('configure', 1), ('load', None)]
            order.append(('readback', None))
            return {'configured': True, 'torch_num_threads': 1, 'observed_after_load': True}
    model, backend = pipeline(Configured())
    assert order == [('configure', 1), ('load', None), ('readback', None)]
    assert model.status()['budget']['native_cpu']['configured'] is True
    assert model.status()['budget']['native_cpu']['observed_after_load'] is True
