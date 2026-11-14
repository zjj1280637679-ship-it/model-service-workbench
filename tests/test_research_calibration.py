"""Counterexamples for empirical units, asynchronous timing and honest GPU absence."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location("research_calibration", Path(__file__).resolve().parents[1] / "research/calibrate_hardware.py")
calibration = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(calibration)


def arguments(**changes):
    base = dict(gpu=False, transfers=False, buffer_mib=64, matrix_size=4096,
                samples=3, iterations=1, disk_mib=16, disk_dir=None)
    return SimpleNamespace(**(base | changes))


def test_empty_and_single_sample_variation_do_not_become_false_zero():
    assert calibration.summary([])["median"] is None
    assert calibration.summary([])["sample_stdev"] is None
    assert calibration.summary([3.0])["sample_stdev"] is None
    stats = calibration.summary([1, 2, 9])
    assert stats["median"] == 2
    assert stats["mean"] == 4
    assert stats["sample_stdev"] > 4


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf"), True, "1"])
def test_invalid_duration_cannot_produce_rate(bad):
    with pytest.raises(ValueError):
        calibration.measured_rates([{"iterations": 1, "wall_seconds": bad}], 1024, "byte")


def test_rates_preserve_iteration_count_units_and_unequal_durations():
    result = calibration.measured_rates([{"iterations": 3, "wall_seconds": 2},
                                        {"iterations": 2, "wall_seconds": 4}], 1e9, "byte")
    assert result["unit"] == "byte/s"
    assert result["samples"] == [1.5e9, 0.5e9]
    assert result["summary"]["median"] == 1e9
    assert result["physical_peak"] is None and result["model_capacity"] is None
    with pytest.raises(ValueError):
        calibration.measured_rates([], 1, "GB/s")
    with pytest.raises(ValueError):
        calibration.measured_rates([{"iterations": True, "wall_seconds": 1}], 1, "byte")


def test_cuda_samples_wait_for_completion_and_convert_ms_to_seconds():
    events, calls = [], []

    class Event:
        def __init__(self, enable_timing):
            assert enable_timing
            self.done = False
            events.append(self)

        def record(self):
            calls.append("record")

        def synchronize(self):
            calls.append("event_sync")
            self.done = True

        def elapsed_time(self, other):
            assert other.done, "Asynchronous work was read before completion"
            return 2.5

    torch = SimpleNamespace(cuda=SimpleNamespace(Event=Event, synchronize=lambda: calls.append("warmup_sync")))
    rows = calibration.cuda_samples(torch, lambda: calls.append("op"), samples=3, iterations=4, warmup=2)
    assert calls[:3] == ["op", "op", "warmup_sync"]
    assert len(events) == 6 and calls.count("op") == 14
    assert len(rows) == 3 and all(row["cuda_event_seconds"] == .0025 for row in rows)
    result = calibration.measured_rates(rows, 2e9, "FLOP", "cuda_event_seconds")
    assert result["samples"] == [3.2e12] * 3


def test_cpu_only_run_cannot_claim_gpu_or_import_torch(monkeypatch):
    monkeypatch.setattr(calibration, "cpu_cases", lambda *args: {"cpu": {"status": "measured"}})

    def forbidden():
        raise AssertionError("A CPU-only observation must never load torch")

    report = calibration.run(arguments(), torch_loader=forbidden)
    assert report["status"] == "completed"
    assert report["gpu"] == {"status": "unknown", "reason": "not_requested"}
    assert report["limits"]["explicit_device_tensors_bytes"] == 0
    assert report["physical_peak_reference"] is None and report["theoretical_model_seconds"] is None


def test_requested_but_unavailable_cuda_remains_unknown(monkeypatch):
    monkeypatch.setattr(calibration, "cpu_cases", lambda *args: {})
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False, is_initialized=lambda: False))
    report = calibration.run(arguments(gpu=True), torch_loader=lambda: torch)
    assert report["status"] == "refused_or_failed"
    assert report["gpu"]["status"] == "unknown"
    assert "bf16_gemm" not in report["benchmarks"]


@pytest.mark.parametrize("changes", [{"buffer_mib": 4096}, {"matrix_size": 65536}, {"samples": 1000},
                                      {"iterations": True}, {"disk_mib": 100000}, {"transfers": True}])
def test_direct_calls_cannot_bypass_cli_resource_bounds(changes, monkeypatch):
    monkeypatch.setattr(calibration, "cpu_cases", lambda *args: pytest.fail("allocation started before budget validation"))
    with pytest.raises(ValueError):
        calibration.run(arguments(**changes))


def test_disk_records_warm_cache_limit_and_removes_only_owned_file(tmp_path):
    existing = tmp_path / "existing.bin"
    existing.write_bytes(b"preserve me")
    report = calibration.disk_cases(tmp_path, 65536, samples=3)
    assert report["status"] == "measured" and report["verified"] is True
    assert report["cache_state"] == "uncontrolled_likely_warm"
    assert report["total_bytes_written"] == 3 * 65536
    assert report["owned_file_removed"] is True
    assert len(report["read_after_write"]["samples"]) == 3
    assert existing.read_bytes() == b"preserve me"
    assert list(tmp_path.iterdir()) == [existing]


def test_disk_error_still_cleans_owned_file(tmp_path, monkeypatch):
    def fail(_):
        raise OSError("injected fsync failure")

    monkeypatch.setattr(calibration.os, "fsync", fail)
    with pytest.raises(OSError):
        calibration.disk_cases(tmp_path, 1024, samples=3)
    assert list(tmp_path.iterdir()) == []


def test_gpu_preflight_rejects_mig_before_any_benchmark(monkeypatch):
    props = SimpleNamespace(name="NVIDIA A100-SXM4-80GB MIG", multi_processor_count=14, major=8, minor=0)
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, set_device=lambda _: None,
                                                get_device_properties=lambda _: props))
    with pytest.raises(RuntimeError, match="full A100"):
        calibration.gpu_preflight(torch)


def test_gpu_preflight_rejects_competing_model_process(monkeypatch):
    props = SimpleNamespace(name="NVIDIA A100-SXM4-80GB", multi_processor_count=108, major=8, minor=0)
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, set_device=lambda _: None,
        get_device_properties=lambda _: props, is_bf16_supported=lambda **kwargs: True))
    monkeypatch.setattr(calibration, "other_gpu_processes", lambda: [999999])
    with pytest.raises(RuntimeError, match="other GPU compute processes"):
        calibration.gpu_preflight(torch)
