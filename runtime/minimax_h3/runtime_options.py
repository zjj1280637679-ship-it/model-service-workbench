"""BF16 runtime controls and lightweight host-wall-time diagnostics.

The accepted stream/on-demand-pinning path remains the default. ``sync`` and
explicit attention choices are diagnostic alternatives, not GPU-validated
speedups. No option changes weight precision or downloads attention kernels.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import wraps
import inspect
import json
import os
from pathlib import Path
import sys
import threading
import time


@dataclass(frozen=True)
class RuntimeOptions:
    denoiser_offload: str = "stream"
    attention_backend: str = "auto"

    def __post_init__(self):
        if self.denoiser_offload not in {"stream", "sync"}:
            raise ValueError("H3_DENOISER_OFFLOAD must be stream or sync (experimental).")
        if self.attention_backend not in {"auto", "native", "_native_flash"}:
            raise ValueError("H3_ATTENTION_BACKEND must be auto, native, or _native_flash.")

    @classmethod
    def from_env(cls):
        return cls(
            denoiser_offload=os.environ.get("H3_DENOISER_OFFLOAD", "stream"),
            attention_backend=os.environ.get("H3_ATTENTION_BACKEND", "auto"),
        )

    def group_offload_kwargs(self, torch):
        return {
            "onload_device": torch.device("cuda"),
            "offload_device": torch.device("cpu"),
            "offload_type": "block_level",
            "num_blocks_per_group": 1,
            "use_stream": self.denoiser_offload == "stream",
            "low_cpu_mem_usage": self.denoiser_offload == "stream",
        }

    def configure_attention(self, transformer):
        if self.attention_backend != "auto":
            # Official Diffusers v0.43 API. _native_flash forces PyTorch SDPA
            # FLASH_ATTENTION; errors must propagate, with no silent fallback.
            transformer.set_attention_backend(self.attention_backend)


def configure_qwen_offload(text_encoder):
    """Preserve the A100-tested placement of Qwen's directly accessed weights."""
    import torch
    from diffusers.hooks import apply_group_offloading

    text_encoder.requires_grad_(False)
    encoder = text_encoder.model
    language = encoder.language_model
    # Qwen reads visual.pos_embed.weight.device outside forward and calls
    # embed_tokens before language.forward. Rotary uses its buffer directly.
    encoder.visual.to("cuda")
    language.embed_tokens.to("cuda")
    language.norm.to("cuda")
    language.rotary_emb.to("cuda")
    for layer in language.layers:
        apply_group_offloading(
            layer, onload_device=torch.device("cuda"), offload_device=torch.device("cpu"),
            offload_type="block_level", num_blocks_per_group=1, use_stream=False,
        )


def resource_snapshot():
    """Read counters only: never initialize CUDA, synchronize, or scan weights."""
    result = {}
    try:
        import psutil
        process = psutil.Process()
        memory = process.memory_info()
        cpu = process.cpu_times()
        host = psutil.virtual_memory()
        result.update(
            rss_bytes=memory.rss, cpu_seconds=cpu.user + cpu.system,
            host_available_bytes=host.available, host_free_bytes=host.free,
        )
        for source, key in ((memory, "shared"), (host, "cached")):
            if hasattr(source, key):
                result[("process_" if source is memory else "host_") + key + "_bytes"] = getattr(source, key)
        try:
            result["read_bytes"] = process.io_counters().read_bytes
        except (AttributeError, OSError, psutil.Error):
            pass
    except Exception:
        # Counter access can be denied by a container or disappear during exit.
        # Diagnostics must not abort inference.
        pass
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if torch.cuda.is_initialized():
                result["cuda_allocated_bytes"] = torch.cuda.memory_allocated()
                result["cuda_reserved_bytes"] = torch.cuda.memory_reserved()
        except (AttributeError, RuntimeError):
            pass
    return result


def emit_metric(event):
    # A leading newline separates our JSON from tqdm's carriage-return line.
    try:
        print("\nH3_METRIC " + json.dumps(event, ensure_ascii=False, sort_keys=True), flush=True)
    except (OSError, ValueError):
        pass


class RuntimeMetrics:
    def __init__(self, kind, backend, options):
        self.started = time.perf_counter()
        self._owner_thread = threading.get_ident()
        self._phase_stack = []
        self._lock = threading.RLock()
        self.data = {
            "schema_version": 1, "kind": kind, "backend": backend,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "runtime_options": asdict(options),
            "timing": "host_wall_seconds_without_cuda_synchronize",
            "resource_counter_scope": "process; overlapping stage deltas are not additive",
            "resources_before": resource_snapshot(), "stages": [],
            "status": "running",
        }

    @contextmanager
    def stage(self, phase):
        started = time.perf_counter()
        # A job may move from its prepare executor to the GPU and encode workers.
        # Child stages identify themselves explicitly; thread identity is not job identity.
        is_owner = not phase.startswith("child.")
        if is_owner:
            with self._lock:
                self._phase_stack.append((phase, started))
        before = resource_snapshot()
        emit_metric({"event": "stage_start", "phase": phase, "kind": self.data["kind"]})
        status, error_type = "succeeded", None
        try:
            yield
        except BaseException as error:
            status, error_type = "failed", type(error).__name__
            raise
        finally:
            after = resource_snapshot()
            record = {
                "phase": phase, "status": status,
                "wall_seconds": round(time.perf_counter() - started, 6),
                "scope": "main" if is_owner else "child",
                "resources_before": before, "resources_after": after,
            }
            for key in ("read_bytes", "cpu_seconds"):
                if key in before and key in after:
                    record[key + "_delta"] = max(0, after[key] - before[key])
            if error_type:
                record["error_type"] = error_type
            with self._lock:
                self.data["stages"].append(record)
                if is_owner:
                    self._phase_stack.pop()
            emit_metric({"event": "stage_end", "kind": self.data["kind"], **record})

    def phase_snapshot(self):
        with self._lock:
            if not self._phase_stack:
                return None
            phase, started = self._phase_stack[-1]
        return {"phase": phase, "elapsed_seconds": round(time.perf_counter() - started, 3), "kind": self.data["kind"]}

    def record_schedule(self, record):
        """Append path-free scheduling attribution under the metrics lock."""
        with self._lock:
            self.data.setdefault("scheduling", []).append(dict(record))

    def finish(self, error=None):
        self.data.update(
            status="succeeded" if error is None else "failed",
            finished_at=datetime.now(timezone.utc).isoformat(),
            wall_seconds=round(time.perf_counter() - self.started, 6),
            resources_after=resource_snapshot(),
        )
        if error is not None:
            self.data["error_type"] = type(error).__name__
        return self.data


class RuntimeSupport:
    def _init_runtime(self, options=None):
        self.runtime_options = options or RuntimeOptions.from_env()
        self._loaded = False
        self._metrics_context = ContextVar(f"h3_metrics_{id(self)}", default=None)
        self._last_metrics_context = ContextVar(f"h3_last_metrics_{id(self)}", default=None)
        self._startup_active_metrics = None
        self.startup_metrics = None
        self.last_metrics = None

    @property
    def _active_metrics(self):
        return self._metrics_context.get()

    @_active_metrics.setter
    def _active_metrics(self, value):
        self._metrics_context.set(value)

    @property
    def last_metrics(self):
        return self._last_metrics_context.get()

    @last_metrics.setter
    def last_metrics(self, value):
        self._last_metrics_context.set(value)

    @contextmanager
    def metrics_context(self, metrics):
        token = self._metrics_context.set(metrics)
        try:
            yield metrics
        finally:
            self._metrics_context.reset(token)

    @property
    def current_phase(self):
        snapshot = self.phase_snapshot()
        return snapshot["phase"] if snapshot else None

    def phase_snapshot(self, context=None):
        metrics = getattr(context, "metrics", None) if context is not None else self._active_metrics or self._startup_active_metrics
        return metrics.phase_snapshot() if metrics is not None else None

    def _new_metrics(self, kind):
        metrics = RuntimeMetrics(kind, type(self).__name__, self.runtime_options)
        settings = getattr(self, "adapter_settings", None)
        if settings is not None:
            metrics.data["adapters"] = settings.as_dict()
            metrics.data["adapters_scope"] = "preloaded_deployment; per-job applied choices are in generation_options"
        return metrics

    @contextmanager
    def _stage(self, phase):
        if self._active_metrics is None:
            yield
        else:
            with self._active_metrics.stage(phase):
                yield


def measured_load(function):
    @wraps(function)
    def wrapper(self, *args, **kwargs):
        if self._loaded:
            return None
        metrics = self._new_metrics("startup")
        self._active_metrics = metrics
        self._startup_active_metrics = metrics
        error = None
        try:
            with metrics.stage("load.total"):
                result = function(self, *args, **kwargs)
            self._loaded = True
            return result
        except BaseException as caught:
            error = caught
            raise
        finally:
            self.startup_metrics = metrics.finish(error)
            self._active_metrics = None
            self._startup_active_metrics = None
    return wrapper


def measured_generation(function):
    signature = inspect.signature(function)

    @wraps(function)
    def wrapper(self, *args, **kwargs):
        self.last_metrics = None  # A failed job must never expose the previous job's metrics.
        arguments = signature.bind(self, *args, **kwargs).arguments
        output_path = Path(arguments["output_path"])
        metrics = self._new_metrics("generation")
        self._active_metrics = metrics
        error = None
        try:
            with metrics.stage("generate.total"):
                return function(self, *args, **kwargs)
        except BaseException as caught:
            error = caught
            raise
        finally:
            self.last_metrics = metrics.finish(error)
            self._active_metrics = None
            sidecar = output_path.with_suffix(".metrics.json")
            temporary = sidecar.with_suffix(sidecar.suffix + ".tmp")
            try:
                sidecar.parent.mkdir(parents=True, exist_ok=True)
                temporary.write_text(json.dumps(self.last_metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                temporary.replace(sidecar)
            except OSError as write_error:
                # Observability must not turn a generated video into a failed job
                # or replace the original inference exception.
                self.last_metrics["metrics_write_error"] = type(write_error).__name__
                emit_metric({"event": "metrics_write_failed", "phase": "metrics", "error_type": type(write_error).__name__})
    return wrapper
