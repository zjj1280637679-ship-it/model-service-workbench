"""CPU prepare -> exclusive GPU inference -> CPU encoding, with per-job ownership.

Reference/codec behavior follows Diffusers v0.43 references.py/export_utils.py.
Its encode_video has no thread-budget argument; this local PyAV writer sets
codec thread_count explicitly and avoids stacking every output frame again.
Resource numbers are conservative admission estimates, not calibrated optima.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from fractions import Fraction
import json
import gc
import math
import os
from pathlib import Path

from runtime_options import RuntimeSupport, measured_generation
from generation_options import requested_options, validate_generation_options, effective_options
from adapter_config import select_reference_setup, select_loaded_adapters

MIB, GIB = 1024**2, 1024**3


@dataclass
class Prepared:
    parameters: dict
    references: list
    normalized: list
    metrics: object
    cpu_threads: int = 1
    cleaned: bool = False


@dataclass
class CPUResult:
    video: list
    audio: object
    sampling_rate: int
    metrics: object
    cpu_threads: int
    cpu_handoff_complete: bool = False
    cleaned: bool = False


def _audio_from_container(av, container, stream):
    import numpy as np
    import torch
    stream.codec_context.thread_count = 1
    rate = int(stream.codec_context.sample_rate)
    if not 0 < rate <= 96000 or len(stream.layout.channels) > 2:
        raise ValueError("Reference soundtrack exceeds 96 kHz / stereo")
    resampler = av.AudioResampler(format="fltp", layout=stream.layout, rate=rate)
    chunks, samples = [], 0
    def append(frames):
        nonlocal samples
        for item in frames:
            value = item.to_ndarray()
            samples += value.shape[-1]
            if samples > math.ceil(15.1 * rate):
                raise ValueError("Decoded reference soundtrack exceeds 15 seconds")
            chunks.append(value)
    for frame in container.decode(stream):
        append(resampler.resample(frame))
    append(resampler.resample(None))
    if not chunks:
        raise ValueError("Reference soundtrack is empty")
    return torch.from_numpy(np.concatenate(chunks, axis=-1).astype(np.float32, copy=False)), rate


def decode_reference(kind, path, classes):
    """CPU-only decoding with one native codec thread and decoded-size guards."""
    if kind == "image":
        return classes[kind].from_file(str(path))
    import av
    import numpy as np
    with av.open(str(path)) as container:
        if kind == "audio":
            audio, rate = _audio_from_container(av, container, container.streams.audio[0])
            return classes[kind](audio=audio, sample_rate=rate)
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        frames = []
        for frame in container.decode(stream):
            if len(frames) >= 362 or frame.width * frame.height > 960 * 544:
                raise ValueError("Normalized reference exceeds bounded decoded geometry")
            frames.append(frame.to_ndarray(format="rgb24"))
        if not frames:
            raise ValueError("Reference video is empty")
        fps = float(stream.average_rate or stream.guessed_rate)
        audio, rate = None, None
        if container.streams.audio:
            container.seek(0)
            audio, rate = _audio_from_container(av, container, container.streams.audio[0])
        return classes[kind](frames=np.stack(frames), fps=fps, audio=audio, sample_rate=rate)


def cpu_media(video, audio, sampling_rate, torch):
    """Copy outputs into CPU-owned numpy buffers; never retain PipelineState."""
    import numpy as np
    def array(value):
        if isinstance(value, torch.Tensor):
            value = value.detach().to(device="cpu", copy=True)
            if value.is_floating_point():
                value = value.float()
            value = value.numpy()
        return np.array(value, copy=True)
    frames = []
    for frame in video:
        pixels = array(frame)
        if pixels.ndim != 3 or pixels.shape[-1] != 3:
            raise ValueError("Expected HWC RGB video output")
        if pixels.dtype != np.uint8:
            if not np.isfinite(pixels).all() or pixels.min() < 0 or pixels.max() > 1:
                raise ValueError("Float video output must be finite and normalized")
            pixels = (pixels * 255).round().astype(np.uint8)
        frames.append(pixels)
    waveform = array(audio).astype(np.float32, copy=False)
    if waveform.ndim != 2 or waveform.shape[0] != 2 or not np.isfinite(waveform).all():
        raise ValueError("Expected finite stereo audio output")
    if not frames or not 0 < int(sampling_rate) <= 96000:
        raise ValueError("Invalid output media")
    return frames, waveform, int(sampling_rate)


def encode_cpu_media(result, output_path):
    """Same H.264/AAC formats as official exporter, explicitly bounded threads."""
    import av
    import numpy as np
    if not result.cpu_handoff_complete or result.cleaned:
        raise RuntimeError("Encoding requires a completed CPU ownership handoff")
    with av.open(str(output_path), mode="w") as container:
        stream = container.add_stream("libx264", rate=24)
        stream.width, stream.height = result.video[0].shape[1], result.video[0].shape[0]
        stream.pix_fmt = "yuv420p"
        stream.codec_context.thread_count = result.cpu_threads
        audio_stream = container.add_stream("aac", rate=result.sampling_rate)
        cc = audio_stream.codec_context
        cc.sample_rate, cc.layout = result.sampling_rate, "stereo"
        cc.time_base, cc.thread_count = Fraction(1, result.sampling_rate), 1
        for pixels in result.video:
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        samples = (np.clip(result.audio.T, -1, 1) * 32767).astype(np.int16)
        audio_frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(samples).reshape(1, -1), format="s16", layout="stereo")
        audio_frame.sample_rate = result.sampling_rate
        resampler = av.AudioResampler(format=cc.format or "fltp", layout=cc.layout or "stereo", rate=cc.sample_rate)
        next_pts = 0
        for frame in resampler.resample(audio_frame) + resampler.resample(None):
            if frame.pts is None:
                frame.pts = next_pts
            next_pts += frame.samples
            frame.sample_rate = result.sampling_rate
            for packet in audio_stream.encode(frame):
                container.mux(packet)
        for packet in audio_stream.encode():
            container.mux(packet)
    if not Path(output_path).is_file() or not Path(output_path).stat().st_size:
        raise RuntimeError("Encoder produced no video")


class PipelineBackendSupport(RuntimeSupport):
    pipeline_model_peak_ram_bytes = 144 * GIB
    resource_residency = "resident"

    def validate_generation_request(self, *, steps, attention_backend=None, acceleration_lora=None):
        values = {"steps": steps}
        if attention_backend is not None:
            values["attention_backend"] = attention_backend
        if acceleration_lora is not None:
            values["acceleration_lora"] = acceleration_lora
        return validate_generation_options(requested_options(values), self.adapter_settings)

    def _apply_generation_transformer(self, effective, *, audit=True):
        transformer = getattr(self.pipe, "transformer_ref", None)
        if transformer is None:
            raise RuntimeError("The generation transformer is unavailable")
        # Always set native explicitly, so forced Flash never leaks into auto.
        transformer.set_attention_backend(effective["attention_backend"])
        select_loaded_adapters(self.pipe, self.adapter_settings, effective["acceleration_lora"])
        if audit and self._active_metrics is not None:
            self._active_metrics.data["generation_options"]["effective"] = dict(effective)

    @contextmanager
    def generation_context(self, prepared):
        """Mutate and restore shared model state only inside infer's GPU lease.

        CPU preparation can overlap other jobs; it must not change adapters,
        attention processors, reference setup, or scheduler state. Staged
        residency applies the same choices after its transformer is loaded.
        """
        if getattr(self, "_active_generation_options", None) is not None:
            raise RuntimeError("Generation configuration already owns the GPU lease")
        requested = requested_options(prepared.parameters)
        validate_generation_options(requested, self.adapter_settings)
        effective = effective_options(requested, self.adapter_settings, self.runtime_options)
        prepared.metrics.data.setdefault("generation_options", {"requested": dict(requested), "effective": None})
        original = self.pipe.blocks.sub_blocks["before_encode"]
        self._active_generation_options = effective
        try:
            select_reference_setup(self.pipe, requested["acceleration_lora"])
            if getattr(self.pipe, "transformer_ref", None) is not None:
                self._apply_generation_transformer(effective)
            yield
        finally:
            try:
                self.pipe.blocks.sub_blocks["before_encode"] = original
                if getattr(self.pipe, "transformer_ref", None) is not None:
                    defaults = effective_options({"attention_backend": "auto",
                        "acceleration_lora": self.adapter_settings.acceleration, "steps": requested["steps"]},
                        self.adapter_settings, self.runtime_options)
                    self._apply_generation_transformer(defaults, audit=False)
            finally:
                self._active_generation_options = None

    def configure_cpu_budget(self, gpu_threads):
        """Configure native CPU pools exactly once, before any model is loaded.

        GPU work also occupies host CPU capacity. CPU preparation/encoding use
        bounded codecs and numpy copies, not multithreaded torch/BLAS operators.
        Environment defaults cover libraries imported later; the caller can
        audit again after load without changing any process-wide setting.
        """
        if isinstance(gpu_threads, bool) or not isinstance(gpu_threads, int) or gpu_threads < 1:
            raise ValueError("GPU host CPU reservation must be a positive integer")
        previous = getattr(self, "_native_cpu_requested", None)
        if previous is not None:
            if previous != gpu_threads:
                raise RuntimeError("Native CPU pools cannot be reconfigured during this model lifetime")
            return self.cpu_budget_snapshot()
        if self._loaded or self.pipe is not None:
            raise RuntimeError("Native CPU pools must be configured before model loading")
        # Keep the hardware gate before torch imports/downloads, including the
        # real Windows CPU integration's explicit Linux/NVIDIA rejection.
        from preflight import check_resources
        check_resources(residency=self.resource_residency)
        environment = {name: str(gpu_threads) for name in (
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")}
        environment.update(OMP_MAX_ACTIVE_LEVELS="1", OMP_DYNAMIC="FALSE", MKL_DYNAMIC="FALSE",
                           TOKENIZERS_PARALLELISM="false", RAYON_NUM_THREADS=str(gpu_threads))
        os.environ.update(environment)
        import numpy  # Load the CPU pools we use before applying/readback.
        import torch
        from threadpoolctl import threadpool_limits
        torch.set_num_threads(gpu_threads)
        interop_note = "set_before_load"
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            if torch.get_num_interop_threads() != 1:
                raise RuntimeError("Cannot enforce one torch interop thread before model loading") from None
            interop_note = "already_initialized_at_one"
        self._native_cpu_limiter = threadpool_limits(limits=gpu_threads)
        self._native_cpu_requested = gpu_threads
        self._native_cpu_environment = environment
        self._native_cpu_interop_note = interop_note
        try:
            return self.cpu_budget_snapshot()
        except BaseException:
            self._native_cpu_requested = None
            raise

    def cpu_budget_snapshot(self):
        """Read and validate real pool limits; never change them in running jobs."""
        requested = getattr(self, "_native_cpu_requested", None)
        if requested is None:
            raise RuntimeError("Native CPU budget has not been configured")
        import torch
        from threadpoolctl import threadpool_info
        pools = [{key: pool.get(key) for key in ("user_api", "internal_api", "prefix", "version", "num_threads")}
                 for pool in threadpool_info()]
        actual = torch.get_num_threads()
        interop = torch.get_num_interop_threads()
        if actual != requested or interop != 1:
            raise RuntimeError(f"Torch CPU budget differs from reservation: intra={actual}, interop={interop}, reserved={requested}")
        for pool in pools:
            threads = pool["num_threads"]
            if isinstance(threads, bool) or not isinstance(threads, int) or not 1 <= threads <= requested:
                raise RuntimeError(f"Native pool CPU limit is not confirmed: {pool['internal_api']}={threads}, reserved={requested}")
        return {"configured": True, "configured_before_load": True, "gpu_host_cpu_threads": requested,
                "torch_num_threads": actual, "torch_num_interop_threads": interop,
                "interop_configuration": self._native_cpu_interop_note,
                "native_pools": pools, "environment_defaults": dict(self._native_cpu_environment),
                "scope": "observed native CPU pools; codec lanes have separate explicit thread budgets"}

    @property
    def pipeline_model_headroom_bytes(self):
        import psutil
        return max(0, self.pipeline_model_peak_ram_bytes - psutil.Process().memory_info().rss)

    def estimate_resources(self, parameters):
        """Bounds retained media + decode/transfer/encode copies; excludes model."""
        from PIL import Image
        references = parameters["references"]
        durations, retained, source_bytes, normalized_bytes, pipeline_media = [], 0, 0, 0, 0
        frames = self._aligned_frames(parameters["seconds"])
        for item in references:
            path, kind = Path(item["path"]), item["kind"]
            duration = self._reference_probe(path, kind)
            durations.append((kind, duration))
            source_bytes += path.stat().st_size
            if kind == "image":
                with Image.open(path) as image:
                    retained += image.width * image.height * 4
                    # Official conditioning upscales even tiny images to a 2048
                    # short edge. Reserve resized RGB plus float/temporary copies.
                    ratio = max(image.width, image.height) / min(image.width, image.height)
                    pipeline_media += 2048 * (math.ceil(2048 * ratio / 32) * 32) * 20
            elif kind == "video":
                size = (math.ceil(duration * 24) + 2) * 960 * 544 * 3 + math.ceil(duration * 32000) * 2 * 4
                retained += size
                normalized_bytes += size + MIB
                # The GPU workflow can resize reference video again to the
                # official 1344*768 canvas; it is not a CPU preparation stage.
                pipeline_media += min(frames, math.ceil(duration * 24) + 2) * 1344 * 768 * 3 * 8
            elif kind == "audio":
                retained += math.ceil(duration * 96000) * 2 * 4
            else:
                raise ValueError("Unknown reference modality")
        for kind in ("video", "audio"):
            if sum(d for k, d in durations if k == kind) > 15:
                raise ValueError("Total reference duration per modality must be <=15 seconds")
        output = frames * parameters["width"] * parameters["height"] * 3 + math.ceil(frames / 24 * 96000) * 2 * 4
        return {"ram_bytes": 3 * retained + 4 * output + pipeline_media + 256 * MIB,
                "cpu_threads": 1, "disk_bytes": source_bytes + normalized_bytes + output + 4 * MIB,
                "reference_retained_bytes": retained, "output_retained_bytes": output,
                "pipeline_reference_workspace_bytes": pipeline_media,
                "estimate_kind": "conservative decoded-media admission estimate; not calibrated physical optimum"}

    def prepare(self, *, prompt, references, seconds, width, height, steps, seed, cpu_threads=1,
                attention_backend=None, acceleration_lora=None):
        options = {}
        if attention_backend is not None:
            options["attention_backend"] = attention_backend
        if acceleration_lora is not None:
            options["acceleration_lora"] = acceleration_lora
        self.validate_request(steps=steps, **options)
        if self.pipe is None:
            raise RuntimeError("The model has not loaded")
        if isinstance(cpu_threads, bool) or not isinstance(cpu_threads, int) or cpu_threads < 1:
            raise ValueError("cpu_threads must be a positive reserved CPU budget")
        from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3ImageReference, MiniMaxH3VideoReference, MiniMaxH3AudioReference
        classes = {"image": MiniMaxH3ImageReference, "video": MiniMaxH3VideoReference, "audio": MiniMaxH3AudioReference}
        parameters = dict(prompt=prompt, seconds=seconds, width=width, height=height, steps=steps, seed=seed, **options)
        metrics = self._active_metrics or self._new_metrics("generation")
        metrics.data["timing"] = "host_wall_seconds; inference synchronizes at CPU handoff"
        metrics.data["pipeline_stage_contract"] = "cpu_prepare / exclusive_gpu_infer / cpu_encode_validate"
        metrics.data["generation_options"] = {"requested": requested_options(parameters), "effective": None}
        prepared = Prepared(parameters, [], [], metrics, cpu_threads)
        try:
            with self.metrics_context(metrics), self._stage("references.prepare"):
                durations = [self._reference_probe(Path(item["path"]), item["kind"]) for item in references]
                for kind in ("video", "audio"):
                    if sum(d for item, d in zip(references, durations) if item["kind"] == kind) > 15:
                        raise ValueError("Total reference duration per modality must be <=15 seconds")
                normalized_targets = {}
                for index, item in enumerate(references):
                    if item["kind"] == "video":
                        path = Path(item["path"])
                        target = path.with_name(path.stem + f"_ref{index}_24fps.mp4")
                        prepared.normalized.append(target)
                        normalized_targets[index] = target
                def decode(index):
                    item = references[index]
                    with metrics.stage(f"child.prepare.{index}"):
                        path = Path(item["path"])
                        if item["kind"] == "video":
                            path = self._normalize_reference(path, target=normalized_targets[index])
                        return decode_reference(item["kind"], path, classes)
                prepared.references = [None] * len(references)
                lanes = min(cpu_threads, len(references))
                metrics.record_schedule({"phase": "prepare", "workers": lanes, "child_count": len(references),
                                         "reason": "explicit CPU budget; each native decoder/normalizer uses one thread"})
                if lanes <= 1:
                    for index in range(len(references)):
                        prepared.references[index] = decode(index)
                else:
                    with ThreadPoolExecutor(max_workers=lanes, thread_name_prefix="h3-reference") as pool:
                        futures = {pool.submit(decode, i): i for i in range(len(references))}
                        for future in as_completed(futures):
                            prepared.references[futures[future]] = future.result()
            return prepared
        except BaseException as error:
            metrics.finish(error)
            self.cleanup(prepared=prepared)
            raise

    def infer(self, prepared):
        import torch
        if prepared.cleaned:
            raise RuntimeError("Prepared references have already been released")
        raw = None
        try:
            with self.metrics_context(prepared.metrics), self._stage("pipeline"), torch.inference_mode():
                configuration = self.generation_context(prepared) if hasattr(self, "adapter_settings") else nullcontext()
                with configuration:
                    try:
                        raw = self._infer_raw(prepared)
                        video, audio, rate = cpu_media(raw["videos"][0], raw["audio"][0], raw["sampling_rate"], torch)
                        result = CPUResult(video, audio, rate, prepared.metrics, prepared.cpu_threads)
                    finally:
                        raw = None
                        prepared.references.clear()
                        # Synchronize before restoring adapters and attention.
                        # CPU encoding never changes model or CUDA state.
                        gc.collect()
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                result.cpu_handoff_complete = True
                return result
        except BaseException as error:
            prepared.metrics.finish(error)
            raise
        finally:
            raw = None

    def encode(self, result, output_path):
        path = Path(output_path)
        error = None
        try:
            with self.metrics_context(result.metrics), self._stage("encode_mp4"):
                encode_cpu_media(result, path)
        except BaseException as caught:
            error = caught
            raise
        finally:
            metrics = result.metrics.finish(error)
            self.last_metrics = metrics
            sidecar = path.with_suffix(".metrics.json")
            temporary = sidecar.with_suffix(sidecar.suffix + ".tmp")
            try:
                temporary.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                temporary.replace(sidecar)
            except OSError as write_error:
                metrics["metrics_write_error"] = type(write_error).__name__
        return metrics

    def cleanup(self, prepared=None, result=None):
        """CPU ownership only: never collect globally or touch CUDA from this lane."""
        if prepared is not None and not prepared.cleaned:
            prepared.references.clear()
            for path in prepared.normalized:
                Path(path).unlink(missing_ok=True)
            prepared.normalized.clear()
            prepared.cleaned = True
        if result is not None and not result.cleaned:
            result.video.clear()
            result.audio = None
            result.cleaned = True

    @measured_generation
    def generate(self, prompt, references, output_path, seconds, width, height, steps, seed,
                 attention_backend=None, acceleration_lora=None):
        """Standalone compatibility: the same stages, called sequentially."""
        prepared = result = None
        try:
            prepared = self.prepare(prompt=prompt, references=references, seconds=seconds,
                                    width=width, height=height, steps=steps, seed=seed,
                                    attention_backend=attention_backend, acceleration_lora=acceleration_lora)
            result = self.infer(prepared)
            return self.encode(result, output_path)
        finally:
            self.cleanup(prepared=prepared, result=result)
