"""CPU-only regression coverage; no model downloads or CUDA initialization."""
from concurrent.futures import ThreadPoolExecutor
import json
from types import SimpleNamespace

import pytest

from runtime_options import RuntimeMetrics, RuntimeOptions, RuntimeSupport, measured_generation, measured_load


def test_default_keeps_accepted_offload_and_attention():
    options = RuntimeOptions()
    kwargs = options.group_offload_kwargs(SimpleNamespace(device=lambda value: value))
    assert kwargs == {
        "onload_device": "cuda", "offload_device": "cpu",
        "offload_type": "block_level", "num_blocks_per_group": 1,
        "use_stream": True, "low_cpu_mem_usage": True,
    }
    options.configure_attention(object())  # auto must not call a model setter.


@pytest.mark.parametrize("values", [
    {"denoiser_offload": "prepin"}, {"attention_backend": "sage"},
    {"attention_backend": "_flash_3"}, {"attention_backend": "fp8"},
])
def test_unsupported_or_quantized_options_rejected(values):
    with pytest.raises(ValueError):
        RuntimeOptions(**values)


def test_sync_is_explicit_and_has_no_pinning_or_stream():
    options = RuntimeOptions(denoiser_offload="sync")
    kwargs = options.group_offload_kwargs(SimpleNamespace(device=lambda value: value))
    assert not kwargs["use_stream"]
    assert not kwargs["low_cpu_mem_usage"]


def test_forced_attention_failure_propagates():
    def reject(backend):
        assert backend == "_native_flash"
        raise RuntimeError("unsupported kernel")
    with pytest.raises(RuntimeError, match="unsupported kernel"):
        RuntimeOptions(attention_backend="_native_flash").configure_attention(
            SimpleNamespace(set_attention_backend=reject)
        )


class SmallBackend(RuntimeSupport):
    def __init__(self):
        self._init_runtime(RuntimeOptions())
        self.load_count = 0

    @measured_load
    def load(self, fail=False):
        self.load_count += 1
        if fail:
            raise RuntimeError("load failed")

    @measured_generation
    def generate(self, output_path, fail=False):
        assert self.last_metrics is None
        with self._stage("fake_inference"):
            assert self.current_phase == "fake_inference"
            if fail:
                raise RuntimeError("generation failed")
        output_path.write_bytes(b"test fixture")


def test_load_is_idempotent_only_after_success():
    backend = SmallBackend()
    with pytest.raises(RuntimeError):
        backend.load(fail=True)
    assert not backend._loaded
    assert backend.startup_metrics["status"] == "failed"
    backend.load()
    backend.load()
    assert backend.load_count == 2
    assert backend.startup_metrics["status"] == "succeeded"
    assert backend.current_phase is None


def test_job_failure_replaces_previous_metrics_and_writes_sidecar(tmp_path):
    backend = SmallBackend()
    backend.generate(tmp_path / "first.mp4")
    assert backend.last_metrics["status"] == "succeeded"
    with pytest.raises(RuntimeError, match="generation failed"):
        backend.generate(tmp_path / "result.mp4", fail=True)
    saved = json.loads((tmp_path / "result.metrics.json").read_text(encoding="utf-8"))
    assert saved == backend.last_metrics
    assert saved["status"] == "failed"
    assert saved["error_type"] == "RuntimeError"
    assert saved["stages"][0]["phase"] == "fake_inference"
    assert saved["stages"][0]["status"] == "failed"
    assert backend.phase_snapshot() is None
    assert not (tmp_path / "result.metrics.json.tmp").exists()


def test_child_stages_do_not_replace_visible_main_phase():
    metrics = RuntimeMetrics("generation", "test", RuntimeOptions())

    def child():
        with metrics.stage("child.decode"):
            assert metrics.phase_snapshot()["phase"] == "references.prepare"

    with metrics.stage("references.prepare"):
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lambda _: child(), range(2)))
    assert [item["scope"] for item in metrics.data["stages"]] == ["child", "child", "main"]
    assert metrics.phase_snapshot() is None


def test_serial_child_keeps_parent_phase_visible():
    metrics = RuntimeMetrics("generation", "test", RuntimeOptions())
    with metrics.stage("references.prepare"):
        with metrics.stage("child.probe.0"):
            assert metrics.phase_snapshot()["phase"] == "references.prepare"
    assert metrics.data["stages"][0]["scope"] == "child"


def test_sidecar_write_error_does_not_mask_inference_error(tmp_path, monkeypatch):
    from pathlib import Path
    backend = SmallBackend()

    def denied(*args, **kwargs):
        raise PermissionError("fixture")

    monkeypatch.setattr(Path, "write_text", denied)
    with pytest.raises(RuntimeError, match="generation failed"):
        backend.generate(tmp_path / "result.mp4", fail=True)
    assert backend.last_metrics["error_type"] == "RuntimeError"
    assert backend.last_metrics["metrics_write_error"] == "PermissionError"


@pytest.mark.parametrize("module_name,class_name", [
    ("model_backend", "MiniMaxBackend"), ("staged_backend", "StagedMiniMaxBackend"),
])
def test_backend_adapter_validation_happens_before_gpu_work(module_name, class_name, tmp_path, monkeypatch):
    import importlib
    monkeypatch.setenv("H3_ACCELERATION_LORA", "lightx2v-ref2va-4step")
    monkeypatch.setenv("H3_STYLE_LORA", "none")
    backend = getattr(importlib.import_module(module_name), class_name)(RuntimeOptions())
    backend.validate_request(steps=5, acceleration_lora="lightx2v-ref2va-4step")
    backend.validate_request(steps=50)  # Preloading never changes request defaults.
    with pytest.raises(ValueError):
        backend.generate(
            prompt="private prompt", references=[], output_path=tmp_path / "result.mp4",
            seconds=5, width=512, height=288, steps=8, seed=42,
            acceleration_lora="lightx2v-ref2va-4step",
        )
    assert backend.last_metrics["status"] == "failed"
    assert backend.last_metrics["adapters"]["required_steps"] == 5
    assert "private prompt" not in json.dumps(backend.last_metrics)
    assert backend.pipe is None
