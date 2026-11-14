"""Real MiniMax-H3 ref2va inference. No synthetic output or mock fallback."""
from __future__ import annotations

import gc
import json
import math
import os
from pathlib import Path
import subprocess
from runtime_options import configure_qwen_offload, measured_load
from pipeline_backend import PipelineBackendSupport
from adapter_config import load_adapter_settings, configure_pipeline, apply_adapters, validate_steps
from download_progress import tracked_model_load

MODEL_ID = "MiniMaxAI/MiniMax-H3"
MODEL_REVISION = "0919fb684102ef115126612f3fbb84d1027e7317"


def _require_loaded_components(pipe) -> None:
    """Diffusers may log component-load failures without propagating them."""
    required = [spec.name for spec in pipe.blocks.expected_components]
    if not required:
        raise RuntimeError("Ref2VA workflow declared no required components.")
    missing = [name for name in required if getattr(pipe, name, None) is None]
    if missing:
        raise RuntimeError(
            "MiniMax-H3 component loading was incomplete: " + ", ".join(missing)
            + ". Inspect the component loading errors above; API readiness is blocked."
        )


def _parameter_kept_in_fp32(parameter_name, keep_fp32) -> bool:
    """Mirror Diffusers 0.43 load_model_dict_into_meta's exact path segments.

    Its dtype/device-map helpers use looser substring checks, but they do not
    define the dtype assigned by the actual weight loader. In particular,
    ``quant_conv`` must not match ``post_quant_conv``.
    """
    if isinstance(keep_fp32, str):
        keep_fp32 = [keep_fp32]
    return any(part in parameter_name.split(".") for part in (keep_fp32 or []))


def _audit_parameter_precision(pipe) -> dict:
    """Inspect loaded parameter metadata, without copying weights or buffers.

    Each official component owns its FP32 exceptions. In particular, MiniMax's
    audio VAE keeps its weights in FP32 even when the pipeline requests BF16.
    Match the actual loader's complete dotted-path segments, not substrings.
    """
    from collections import Counter
    import torch

    components = {}
    for spec in pipe.blocks.expected_components:
        name = spec.name
        module = getattr(pipe, name, None)
        if not callable(getattr(module, "named_parameters", None)):
            continue  # tokenizers, schedulers and processors have no weights
        if (any(getattr(module, flag, False) for flag in
                ("is_quantized", "is_loaded_in_4bit", "is_loaded_in_8bit"))
                or getattr(module, "hf_quantizer", None) is not None):
            raise RuntimeError(f"Quantized component is forbidden: {name}")
        keep_fp32 = getattr(module, "_keep_in_fp32_modules", None) or []
        if isinstance(keep_fp32, str):
            keep_fp32 = [keep_fp32]
        counts, tensors = Counter(), Counter()
        for parameter_name, parameter in module.named_parameters():
            full_name = f"{name}.{parameter_name}"
            # Quantized tensor subclasses may report their compute dtype as
            # BF16. The native loader uses ordinary nn.Parameter objects.
            if type(parameter) is not torch.nn.Parameter or getattr(parameter, "is_quantized", False):
                raise RuntimeError(f"Non-native or quantized parameter is forbidden: {full_name}")
            if getattr(parameter, "is_meta", False):
                raise RuntimeError(f"Parameter was not materialized: {full_name}")
            expected = torch.float32 if _parameter_kept_in_fp32(parameter_name, keep_fp32) else torch.bfloat16
            if parameter.dtype != expected:
                raise RuntimeError(
                    f"Unexpected precision in {full_name}: {parameter.dtype}, expected {expected}"
                )
            dtype = str(parameter.dtype)
            counts[dtype] += parameter.numel()
            tensors[dtype] += 1
        if not tensors:
            raise RuntimeError(f"No loaded parameters available for precision audit: {name}")
        record = {"component": name, "status": "passed", "parameter_counts": dict(counts),
                  "tensor_counts": dict(tensors), "keep_in_fp32_modules": list(keep_fp32)}
        components[name] = record
        print("H3_PRECISION_AUDIT " + json.dumps(record, sort_keys=True), flush=True)
    if not components:
        raise RuntimeError("No model components available for precision audit")
    report = {"status": "passed", "scope": "loaded_parameters_only", "components": components}
    print("H3_PRECISION_AUDIT " + json.dumps(
        {"status": "passed", "scope": report["scope"], "components": list(components)}, sort_keys=True
    ), flush=True)
    return report


def aligned_frames(seconds: float) -> int:
    frames = 5 + 17 * math.ceil((round(seconds * 24) - 5) / 17)
    if not 5 <= seconds <= 15 or not 120 <= frames <= 360:
        raise ValueError("Duration must align to 17*n+5 frames within 5–15s; maximum request is 14.375s.")
    return frames


def _probe(path: Path, kind: str) -> float:
    """Reject oversized media before loading all decoded frames into host RAM."""
    if kind == "image":
        from PIL import Image
        with Image.open(path) as im:
            if im.width * im.height > 16_000_000 or getattr(im, "n_frames", 1) != 1:
                raise ValueError("Reference images must be static and at most 16 megapixels.")
            if not .25 <= im.width / im.height <= 4:
                raise ValueError("Reference image aspect ratio must be between 1:4 and 4:1.")
            im.verify()
        return 0
    process = subprocess.run(
        ["ffprobe", "-v", "error", "-threads", "1", "-show_format", "-show_streams", "-of", "json", str(path)],
        capture_output=True, text=True, timeout=30, check=True,
    )
    info = json.loads(process.stdout)
    streams = info.get("streams", [])
    wanted = [s for s in streams if s.get("codec_type") == kind]
    if not wanted:
        raise ValueError(f"No {kind} stream in reference file.")
    stream = wanted[0]
    duration = float(stream.get("duration") or info.get("format", {}).get("duration") or 0)
    if not math.isfinite(duration) or not 2 <= duration <= 15:
        raise ValueError(f"Reference {kind} duration must be known and between 2 and 15s.")
    if kind == "video":
        from fractions import Fraction
        rate = float(Fraction(stream.get("avg_frame_rate", "0")))
        if not 0 < rate <= 60 or stream.get("width", 0) * stream.get("height", 0) > 1920 * 1080:
            raise ValueError("Reference video must be at most 1080p / 60 fps.")
    for audio in (s for s in streams if s.get("codec_type") == "audio"):
        if int(audio.get("sample_rate", 0)) > 96000 or int(audio.get("channels", 0)) > 2:
            raise ValueError("Reference audio must be mono/stereo and at most 96 kHz.")
    return duration


def _normalize_video(path: Path, target: Path | None = None) -> Path:
    # Bound decoded memory before MiniMaxH3VideoReference.from_file stacks all frames.
    target = target or path.with_name(path.stem + "_24fps.mp4")
    subprocess.run([
        "ffmpeg", "-nostdin", "-y", "-v", "error", "-threads", "1", "-filter_threads", "1", "-i", str(path),
        "-map", "0:v:0", "-map", "0:a:0?", "-t", "15",
        "-vf", "scale=960:544:force_original_aspect_ratio=decrease:force_divisible_by=2,fps=24",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-ac", "2", "-ar", "32000", "-threads", "1", str(target),
    ], check=True, capture_output=True, timeout=180)
    return target


class MiniMaxBackend(PipelineBackendSupport):
    _reference_probe = staticmethod(_probe)
    _normalize_reference = staticmethod(_normalize_video)
    _aligned_frames = staticmethod(aligned_frames)

    def __init__(self, runtime_options=None):
        self._init_runtime(runtime_options)
        self.adapter_settings = load_adapter_settings()
        self.pipe = None
        self.precision_audit = None

    @measured_load
    @tracked_model_load
    def load(self) -> None:
        self.precision_audit = None
        from preflight import check_resources
        resources = check_resources()
        import torch
        from diffusers import ComponentsManager, ModularPipeline

        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("A CUDA GPU with native BF16 support is required.")
        revision = os.environ.get("H3_MODEL_REVISION", MODEL_REVISION)
        torch.backends.cuda.matmul.allow_tf32 = True
        strategy = resources["strategy"]
        print(f"A100 inference: native BF16; memory strategy={strategy}", flush=True)
        if strategy == "component_cpu_offload":
            if self.runtime_options.denoiser_offload != "stream":
                raise ValueError("H3_DENOISER_OFFLOAD=sync applies only to group offload; this profile uses component offload.")
            manager = ComponentsManager()
            pipe = ModularPipeline.from_pretrained(
                MODEL_ID, revision=revision, workflow="ref2va", components_manager=manager,
            )
            configure_pipeline(pipe, self.adapter_settings)
            with self._stage("load.components"):
                pipe.load_components(dtype=torch.bfloat16, revision=revision)
            _require_loaded_components(pipe)
            with self._stage("configure.adapters"):
                apply_adapters(pipe, self.adapter_settings)
            with self._stage("audit.parameter_precision"):
                self.precision_audit = _audit_parameter_precision(pipe)
            self.runtime_options.configure_attention(pipe.transformer_ref)
            manager.enable_auto_cpu_offload(device="cuda", memory_reserve_margin="12GB")
        elif strategy == "group_cpu_offload":
            pipe = ModularPipeline.from_pretrained(MODEL_ID, revision=revision, workflow="ref2va")
            configure_pipeline(pipe, self.adapter_settings)
            with self._stage("load.components"):
                pipe.load_components(dtype=torch.bfloat16, revision=revision)
            _require_loaded_components(pipe)
            pipe.transformer_ref.requires_grad_(False)
            with self._stage("configure.adapters"):
                apply_adapters(pipe, self.adapter_settings)
            with self._stage("audit.parameter_precision"):
                self.precision_audit = _audit_parameter_precision(pipe)
            self.runtime_options.configure_attention(pipe.transformer_ref)
            pipe.transformer_ref.enable_group_offload(**self.runtime_options.group_offload_kwargs(torch))
            configure_qwen_offload(pipe.text_encoder)
            # A100 40GB cannot hold the 61.7GB BF16 denoiser as one component.
            # Stream BF16 blocks without quantization; PCIe bandwidth affects speed.
            pipe.vae.to("cuda")
            pipe.audio_vae.to("cuda")
        else:
            raise ValueError(f"Unsupported A100 memory strategy: {strategy}")
        self.pipe = pipe

    def validate_request(self, *, steps, attention_backend=None, acceleration_lora=None):
        return self.validate_generation_request(steps=steps, attention_backend=attention_backend,
                                                acceleration_lora=acceleration_lora)

    def _infer_raw(self, prepared):
        """The original BF16 workflow stays wholly inside the exclusive GPU stage."""
        if self.pipe is None:
            raise RuntimeError("The model has not loaded.")
        import torch
        p = prepared.parameters
        return self.pipe(
            prompt=p["prompt"], references=prepared.references, num_frames=aligned_frames(p["seconds"]),
            height=p["height"], width=p["width"], num_inference_steps=p["steps"],
            generator=torch.Generator(device="cpu").manual_seed(p["seed"]),
            output_type="pil", output=["videos", "audio", "sampling_rate"],
        )
