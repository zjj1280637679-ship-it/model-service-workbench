import itertools
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest

import resource_scheduler as rs


def test_optimizer_matches_brute_force_and_respects_bandwidth():
    costs = [4.0, 3.0, 2.0, 2.0, 1.0]
    for cap in (1, 2, 3):
        allocation = rs.optimal_allocation(costs, [0] * 5, cpu_slots=cap,
                                           memory_slots=4, bandwidth_mib_s=100)
        best = float("inf")
        for workers in range(1, cap + 1):
            for assignments in itertools.product(range(workers), repeat=len(costs)):
                loads = [sum(c for c, assigned in zip(costs, assignments) if assigned == worker)
                         for worker in range(workers)]
                best = min(best, max(loads) + .002 * workers)
        assert allocation.predicted_seconds == pytest.approx(best)
        assert sorted(i for lane in allocation.lanes for i in lane) == list(range(5))
    constrained = rs.optimal_allocation(costs, [100] * 5, cpu_slots=4,
                                        memory_slots=4, bandwidth_mib_s=1)
    assert len(constrained.lanes) == 1  # More lanes cannot raise shared bandwidth.


def test_memory_and_cpu_caps_and_negative_feedback():
    allocation = rs.optimal_allocation([2.0] * 8, [0] * 8, cpu_slots=3,
                                       memory_slots=2, bandwidth_mib_s=10)
    assert len(allocation.lanes) == 2
    controller = rs.MediaController()
    controller.observe(allocation, elapsed=20, available_slots=2, succeeded=True)
    assert controller.ceiling == 2
    controller.observe(allocation, elapsed=20, available_slots=0, succeeded=False)
    assert controller.ceiling == 1
    for _ in range(3):
        controller.observe(allocation, elapsed=1, available_slots=4, succeeded=True)
    assert controller.ceiling == 2


def test_missing_calibration_is_serial(tmp_path, monkeypatch):
    monkeypatch.delenv("H3_MEDIA_PROFILE", raising=False)
    monkeypatch.setenv("H3_MEDIA_WORKERS", "4")
    allocation, _ = rs.MediaController().plan([{"kind": "image", "path": tmp_path / "a"}] * 4, "prepare")
    assert len(allocation.lanes) == 1


def test_probe_barrier_prevents_prepare_on_invalid_aggregate(tmp_path):
    def probe(path, kind):
        return 8
    def forbidden(path):
        pytest.fail("No preparation before aggregate validation")
    with pytest.raises(ValueError, match="Total reference duration"):
        rs.prepare_references([{"kind": "video", "path": tmp_path / "a.mp4"},
                               {"kind": "video", "path": tmp_path / "b.mp4"}], {}, probe, forbidden)


def test_children_join_before_failure_and_keep_semantic_order(tmp_path, monkeypatch):
    allocation = rs.Allocation(((0,), (1,)), 1.0, "test-calibrated")
    monkeypatch.setattr(rs.CONTROLLER, "plan", lambda *args: (allocation, 4))
    finished = threading.Event()
    entered = threading.Barrier(2)
    def failing(item):
        entered.wait(timeout=2)
        if item == 0:
            raise ValueError("child failed")
        time.sleep(.03)
        finished.set()
        return item
    with pytest.raises(ValueError, match="child failed"):
        rs._children([0, 1], failing, "prepare", None)
    assert finished.is_set()
    assert rs._children(["first", "second"], lambda x: x, "prepare", None) == ["first", "second"]


def test_partial_normalization_is_cleaned(tmp_path):
    path = tmp_path / "reference.mp4"
    def normalize(source):
        source.with_name(source.stem + "_24fps.mp4").write_bytes(b"partial")
        raise RuntimeError("ffmpeg failed")
    with pytest.raises(RuntimeError, match="ffmpeg failed"):
        rs.prepare_references([{"kind": "video", "path": path}], {}, lambda *args: 3, normalize)
    assert not (tmp_path / "reference_24fps.mp4").exists()


def test_feedback_failure_preserves_child_exception_and_success(monkeypatch):
    import psutil
    allocation = rs.Allocation(((0,),), 0, "fixture")
    monkeypatch.setattr(rs.CONTROLLER, "plan", lambda *args: (allocation, 1))
    monkeypatch.setattr(psutil, "virtual_memory", lambda: (_ for _ in ()).throw(OSError("counter denied")))
    with pytest.raises(ValueError, match="original failure"):
        rs._children([0], lambda item: (_ for _ in ()).throw(ValueError("original failure")), "prepare", None)
    assert rs._children([7], lambda item: item, "prepare", None) == [7]


def test_empty_children_skip_resource_planning(monkeypatch):
    monkeypatch.setattr(rs.CONTROLLER, "plan", lambda *args: pytest.fail("nothing to plan"))
    assert rs._children([], lambda item: item, "prepare", None) == []


def test_profile_requires_thread_and_memory_calibration(tmp_path, monkeypatch):
    monkeypatch.setattr(rs, "_cpu_slots", lambda: 8)
    profile = {"schema_version": 1, "cpu_slots": 8,
               "seconds_per_mib": {"prepare": {"image": 1.0}}, "bandwidth_mib_s": 1000}
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    monkeypatch.setenv("H3_MEDIA_PROFILE", str(path))
    allocation, _ = rs.MediaController().plan([{"kind": "image", "path": "not-opened"}] * 4, "prepare")
    assert len(allocation.lanes) == 1
    assert "thread/memory" in allocation.reason


def test_estimated_budget_counts_native_threads_and_retained_results(tmp_path, monkeypatch):
    import psutil
    monkeypatch.setattr(rs, "_cpu_slots", lambda: 8)
    monkeypatch.setenv("H3_MEDIA_WORKERS", "auto")
    profile = {"schema_version": 1, "cpu_slots": 8, "threads_per_worker": 4,
               "task_memory_mib": {"prepare": {"image": 512}},
               "seconds_per_mib": {"prepare": {"image": 1.0}}, "bandwidth_mib_s": 1000}
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    media = tmp_path / "reference.jpg"
    media.write_bytes(b"fixture")
    items = [{"kind": "image", "path": media}] * 4
    monkeypatch.setenv("H3_MEDIA_PROFILE", str(path))
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=10 * 1024 ** 3))
    allocation, _ = rs.MediaController().plan(items, "prepare")
    assert len(allocation.lanes) == 2  # Eight logical cores / four estimated threads.
    assert len(allocation.profile_metadata["sha256"]) == 64
    assert allocation.profile_metadata["cpu_budget_slots"] == 2
    assert str(path) not in json.dumps(allocation.profile_metadata)
    profile["threads_per_worker"] = 1
    path.write_text(json.dumps(profile))
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=5 * 1024 ** 3))
    allocation, slots = rs.MediaController().plan(items, "prepare")
    assert len(allocation.lanes) == slots == 2  # 5GiB - 2GiB reserve - 2GiB retained.
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=4 * 1024 ** 3))
    with pytest.raises(MemoryError, match="working sets"):
        rs.MediaController().plan(items, "prepare")


def test_allocator_matches_small_random_exhaustive_schedules():
    import random
    rng = random.Random(4107)
    for _ in range(10):
        costs = [rng.randint(1, 15) / 3 for _ in range(6)]
        demands = [rng.randint(0, 20) for _ in costs]
        io_floor = sum(demands) / 10
        allocation = rs.optimal_allocation(costs, demands, cpu_slots=3, memory_slots=3,
                                           bandwidth_mib_s=10)
        best = float("inf")
        for workers in range(1, 4):
            for assignments in itertools.product(range(workers), repeat=len(costs)):
                loads = [sum(cost for cost, assigned in zip(costs, assignments) if assigned == worker)
                         for worker in range(workers)]
                best = min(best, max(max(loads), io_floor) + .002 * workers)
        assert allocation.predicted_seconds == pytest.approx(best)


def test_schedule_attribution_has_input_sizes_without_paths(tmp_path, monkeypatch):
    from runtime_options import RuntimeMetrics, RuntimeOptions
    media = tmp_path / "private-name.jpg"
    media.write_bytes(b"fixture")
    allocation = rs.Allocation(((0,),), 0, "fixture")
    monkeypatch.setattr(rs.CONTROLLER, "plan", lambda *args: (allocation, 1))
    metrics = RuntimeMetrics("generation", "fixture", RuntimeOptions())
    assert rs._children([{"kind": "image", "path": media}], lambda item: "decoded", "prepare", metrics) == ["decoded"]
    events = metrics.data["scheduling"]
    assert [item["event"] for item in events] == ["H3_SCHEDULE", "H3_SCHEDULE_FEEDBACK"]
    assert events[0]["input_bytes"] == 7
    assert events[0]["children"] == [{"index": 0, "kind": "image", "input_bytes": 7}]
    assert "private-name" not in json.dumps(metrics.data)
    assert str(tmp_path) not in json.dumps(metrics.data)
