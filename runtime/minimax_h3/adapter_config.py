"""Opt-in, pinned MiniMax-H3 Ref2VA adapters. No GPU acceptance is claimed.

The baseline is unchanged with both settings ``none``. LoRAs are loaded into
each freshly loaded transformer_ref BEFORE group-offload hooks are installed.
Only reviewed safetensors assets are accepted; no arbitrary remote code, URLs,
pickle files, quantized weights, or silent cross-workflow routing.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import re
from download_progress import adapter_progress, current_progress, tracked_adapter_application


@dataclass(frozen=True)
class AdapterSpec:
    key: str
    repository: str
    filename: str
    revision: str
    sha256: str
    size: int
    source_tensors: int


TURBO = AdapterSpec(
    "lightx2v-ref2va-4step", "lightx2v/Minimax-h3-Turbo",
    "minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors",
    "1a4a150fc6707329409eb52e5d75a0e987ac1928",
    "9e642fc8749c74f8da5e2382877ab5c7aa37b9a73b7fd0d6d457bd1b3cb1ae99",
    1383677768, 624,
)
REALISM = AdapterSpec(
    "fal-realism-people", "fal/MiniMax-H3-Realism-People-LoRA",
    "h3-realism-people-t2v-i2v-r2v.safetensors",
    "039cc8579d7aa357a882d7f4111b25da4f72dccc",
    "acc529601d2da117fb81179e76c56e488a3beab1171659d305f04fa3655b787e",
    131229656, 208,
)


@dataclass(frozen=True)
class AdapterSettings:
    acceleration: str = "none"
    style: str = "none"
    style_scale: float = 1.0

    def __post_init__(self):
        if self.acceleration not in {"none", TURBO.key}:
            raise ValueError("H3_ACCELERATION_LORA must be none or lightx2v-ref2va-4step")
        if self.style not in {"none", REALISM.key}:
            raise ValueError("H3_STYLE_LORA must be none or fal-realism-people; arbitrary/FL2VA adapters are not accepted")
        if not math.isfinite(self.style_scale) or not 0.0 < self.style_scale <= 1.0:
            raise ValueError("H3_STYLE_LORA_SCALE must be finite and in (0, 1]")

    @property
    def turbo(self):
        return self.acceleration != "none"

    def as_dict(self):
        return {
            "acceleration": self.acceleration, "style": self.style,
            "style_scale": self.style_scale,
            "reference_image_resize": "match" if self.turbo else "official-2048-short-edge",
            "required_steps": 5 if self.turbo else None,
            "denoiser_evaluations": 4 if self.turbo else None,
            "fused": False,
            "validation": "not GPU-tested" if self.turbo or self.style != "none" else "baseline",
        }


def load_adapter_settings(environ=None):
    env = os.environ if environ is None else environ
    return AdapterSettings(
        acceleration=env.get("H3_ACCELERATION_LORA", "none").strip(),
        style=env.get("H3_STYLE_LORA", "none").strip(),
        style_scale=float(env.get("H3_STYLE_LORA_SCALE", "1.0")),
    )


def validate_steps(settings, steps):
    if settings.turbo and steps != 5:
        raise ValueError("Ref2VA Turbo requires steps=5 sigma grid points, which gives exactly 4 denoiser evaluations")


def reference_match_size(width, height, target_width, target_height, multiple=32):
    """ModelTC training policy: cap reference area at canvas area, preserve ratio.

    Rounding follows the author's implementation. Small inputs are not scaled
    up before rounding; a final nearest-32 rounding can increase an axis.
    """
    if min(width, height, target_width, target_height, multiple) <= 0:
        raise ValueError("Image and target dimensions must be positive")
    if width > 4 * height or height > 4 * width:
        raise ValueError("Reference image aspect ratio must be between 1:4 and 4:1")
    scale = min(1.0, math.sqrt(target_width * target_height / (width * height)))
    return (max(multiple, round(width * scale / multiple) * multiple),
            max(multiple, round(height * scale / multiple) * multiple))


def configure_pipeline(pipe, settings):
    """Replace only this pipeline's resolved Ref2VA setup when Turbo is enabled.

    The original setup still owns validation and video/audio normalization. Its
    CPU image resize is replaced from the original pixels before any encoder
    runs. This intentionally retains a small CPU resize overhead instead of
    copying upstream's full setup implementation or patching process globals.
    """
    if not settings.turbo:
        return
    from diffusers.modular_pipelines.minimax_h3.before_encoder import MiniMaxH3Ref2VASetupStep
    from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3ImageReference
    from PIL import Image

    original = pipe.blocks.sub_blocks.get("before_encode")
    if getattr(original, "_h3_turbo_match", False):
        return
    if not isinstance(original, MiniMaxH3Ref2VASetupStep):
        raise RuntimeError("Unexpected Ref2VA setup block; refusing to change reference resizing")

    class MatchReferenceSetup(MiniMaxH3Ref2VASetupStep):
        _h3_turbo_match = True

        def __call__(self, components, state):
            _, state = original(components, state)
            normalized = list(state.get("normalized_references"))
            references = state.get("references")
            if len(normalized) != len(references):
                raise RuntimeError("Reference normalization changed the reference count")
            for index, reference in enumerate(references):
                if reference.kind != "image":
                    continue
                # This API decodes uploaded images with from_file(), which
                # returns PIL. Reject a future unsupported path explicitly.
                if not isinstance(reference.image, Image.Image):
                    raise TypeError("Turbo API requires decoded PIL image references")
                image = reference.image.convert("RGB")
                width, height = reference_match_size(
                    *image.size, state.get("width"), state.get("height"), components.canvas_multiple,
                )
                if image.size != (width, height):
                    image = components.image_processor.resize(image, height=height, width=width)
                normalized[index] = MiniMaxH3ImageReference(image=image)
            state.set("normalized_references", normalized)
            return components, state

    pipe._h3_original_before_encode = original
    pipe._h3_turbo_before_encode = MatchReferenceSetup()
    pipe.blocks.sub_blocks["before_encode"] = pipe._h3_turbo_before_encode


def select_reference_setup(pipe, acceleration):
    """Select an already constructed setup block inside the exclusive GPU lease."""
    turbo = acceleration == TURBO.key
    selected = getattr(pipe, "_h3_turbo_before_encode" if turbo else "_h3_original_before_encode", None)
    if turbo and selected is None:
        raise RuntimeError("Turbo reference setup was not prepared during model loading")
    if selected is not None:
        pipe.blocks.sub_blocks["before_encode"] = selected


def select_loaded_adapters(pipe, settings, acceleration):
    """Toggle only preloaded, unfused adapters; preserve independent style LoRA.

    This never downloads, loads, fuses, deletes, casts, or changes base weights.
    Call only while the model execution owns the GPU lease, including restore.
    """
    transformer = getattr(pipe, "transformer_ref", None)
    if transformer is None:
        raise RuntimeError("Cannot select adapters without transformer_ref")
    if not settings.turbo and settings.style == "none":
        if acceleration != "none":
            raise RuntimeError("Requested acceleration adapter is not loaded")
        return
    if getattr(transformer, "_h3_adapter_settings", None) != settings:
        raise RuntimeError("Loaded adapter identity does not match deployment configuration")
    names, scales = [], []
    if acceleration != "none":
        if acceleration != settings.acceleration:
            raise RuntimeError("Requested acceleration adapter is not loaded")
        for name, expected in (("scheduler", 12.0), ("audio_scheduler", 3.0)):
            if float(getattr(pipe, name).config.shift) != expected:
                raise RuntimeError(f"Ref2VA Turbo requires {name} shift={expected}")
        names.append("h3_acceleration")
        scales.append(1.0)
    if settings.style != "none":
        names.append("h3_style")
        scales.append(settings.style_scale)
    if names:
        transformer.set_adapters(names, weights=scales)
        transformer.enable_lora()
    else:
        transformer.disable_lora()
    # PEFT activation can turn LoRA factors trainable again. Preserve the
    # inference-only lifetime after enabling, disabling, and restoring.
    transformer.requires_grad_(False)
    transformer.eval()
    if hasattr(transformer, "_reset_stateful_cache"):
        transformer._reset_stateful_cache()


_VERIFIED_FILES = set()


def _download_verified(spec):
    from huggingface_hub import hf_hub_download
    progress = current_progress()
    category = 'acceleration' if spec == TURBO else 'style'
    cached = None
    try:
        from huggingface_hub import try_to_load_from_cache
        cached = try_to_load_from_cache(spec.repository, spec.filename, revision=spec.revision)
        cached = isinstance(cached, str) and Path(cached).is_file()
    except (ImportError, OSError, ValueError):
        cached = None
    if progress:
        progress.phase(category, 'verifying' if cached else 'downloading',
                       '命中缓存，检查文件完整性' if cached else '下载所选 LoRA 文件', total=spec.size, cache_hit=cached)
    path = Path(hf_hub_download(spec.repository, spec.filename, revision=spec.revision))
    stat = path.stat()
    if progress:
        progress.phase(category, 'verifying', '校验文件大小与 SHA-256', total=spec.size, cache_hit=cached)
    signature = (str(path.resolve()), stat.st_size, stat.st_mtime_ns, spec.sha256)
    if signature not in _VERIFIED_FILES:
        if stat.st_size != spec.size:
            raise RuntimeError(f"Unexpected adapter file size for {spec.key}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != spec.sha256:
            raise RuntimeError(f"SHA-256 mismatch for {spec.key}")
        _VERIFIED_FILES.add(signature)
    return path


def _validate_converted(state_dict, transformer, spec):
    """Verify every converted tensor has a paired, dimension-compatible target."""
    pairs = {}
    pattern = re.compile(r"^transformer\.(.+)\.lora_([AB])\.weight$")
    for key, tensor in state_dict.items():
        match = pattern.fullmatch(key)
        if not match or len(tensor.shape) != 2:
            raise ValueError(f"Unsupported converted LoRA tensor: {key}")
        name, side = match.groups()
        if spec == REALISM and not re.fullmatch(r"(?:transformer_blocks|token_refiner\.refiner_blocks)\.\d+\.attn\.(to_[qkv]|to_out\.0)", name):
            raise ValueError(f"Realism adapter must target only shared attention projections: {name}")
        pairs.setdefault(name, {})[side] = tensor
    if not pairs:
        raise ValueError("Converted adapter has no target tensors")
    for name, pair in pairs.items():
        if set(pair) != {"A", "B"}:
            raise ValueError(f"Unpaired LoRA target: {name}")
        module = transformer.get_submodule(name)
        base = module.get_base_layer() if hasattr(module, "get_base_layer") else module
        a, b = pair["A"], pair["B"]
        if a.shape[0] != b.shape[1] or tuple(base.weight.shape) != (b.shape[0], a.shape[1]):
            raise ValueError(f"LoRA shape mismatch for {name}; pruned/other-model adapters are unsupported")
    return pairs


@tracked_adapter_application
def apply_adapters(pipe, settings):
    """Load reviewed adapters into transformer_ref without fusing base weights.

    Both named adapters can be composed, but that combination still needs GPU
    quality/latency acceptance. Only LoRA factors are cast; keep-FP32 base layers
    are never changed. Cache state is cleared when supported.
    """
    specs = []
    if settings.turbo:
        specs.append((TURBO, "h3_acceleration", 1.0))
    if settings.style != "none":
        specs.append((REALISM, "h3_style", settings.style_scale))
    if not specs:
        return
    import importlib.util
    if importlib.util.find_spec("peft") is None:
        raise RuntimeError("LoRA is enabled but PEFT is missing; install requirements-lora.txt using the existing torch constraints")
    import torch
    from safetensors import safe_open
    transformer = getattr(pipe, "transformer_ref", None)
    if transformer is None or getattr(pipe, "transformer", None) is not None:
        raise RuntimeError("Adapters require the ref2va-only pipeline with transformer_ref loaded")
    if getattr(transformer, "_h3_adapter_settings", None) is not None:
        if transformer._h3_adapter_settings != settings:
            raise RuntimeError("Changing adapters requires a freshly loaded transformer_ref")
        return
    if settings.turbo:
        for name, expected in (("scheduler", 12.0), ("audio_scheduler", 3.0)):
            if float(getattr(pipe, name).config.shift) != expected:
                raise RuntimeError(f"Ref2VA Turbo requires {name} shift={expected}")
    base_dtypes = {id(p): p.dtype for p in transformer.parameters()}
    names, scales = [], []
    for spec, adapter_name, scale in specs:
        category = "acceleration" if spec == TURBO else "style"
        with adapter_progress(category, spec) as progress:
            path = _download_verified(spec)
            if progress:
                progress.phase(category, "loading", "加载并校验 LoRA 张量；不改变主干精度")
            with safe_open(path, framework="pt", device="cpu") as handle:
                keys = list(handle.keys())
                if len(keys) != spec.source_tensors:
                    raise ValueError(f"Unexpected tensor count for {spec.key}")
                if any(handle.get_slice(key).get_dtype() != "BF16" for key in keys):
                    raise ValueError(f"The reviewed {spec.key} asset must contain only BF16 factors")
                if spec == TURBO and (handle.metadata() or {}).get("alpha") != "8":
                    raise ValueError("Ref2VA Turbo training alpha must be 8")
            state_dict, metadata = pipe.lora_state_dict(
                str(path.parent), weight_name=path.name, use_safetensors=True,
                local_files_only=True, return_lora_metadata=True,
            )
            pairs = _validate_converted(state_dict, transformer, spec)
            # Use the official ref-specific loader so the file metadata's alpha/rank
            # scaling survives conversion. Generic load_lora_weights(dict) would
            # lose the original file's alpha=8 metadata.
            pipe.load_lora_into_transformer_ref(
                state_dict, transformer_ref=transformer, prefix="transformer",
                adapter_name=adapter_name, metadata=metadata, _pipeline=pipe,
                low_cpu_mem_usage=True,
            )
            del state_dict
            for name in pairs:
                layer = transformer.get_submodule(name)
                if adapter_name not in layer.lora_A or adapter_name not in layer.lora_B:
                    raise RuntimeError(f"Adapter failed to load into {name}")
                expected_scale = 8.0 / 128 if spec == TURBO else 1.0
                if not math.isclose(float(layer.scaling[adapter_name]), expected_scale, rel_tol=1e-6):
                    raise RuntimeError(f"Unexpected training alpha/rank scale for {spec.key} at {name}")
                dtype = layer.get_base_layer().weight.dtype
                if dtype not in {torch.bfloat16, torch.float32}:
                    raise RuntimeError(f"Unexpected base precision at {name}: {dtype}")
                layer.lora_A[adapter_name].to(dtype=dtype)
                layer.lora_B[adapter_name].to(dtype=dtype)
            del pairs
            names.append(adapter_name)
            scales.append(scale)
            if progress:
                progress.phase(category, "loaded", "LoRA 已加载并校验", total=spec.size)
    transformer.set_adapters(names, weights=scales)
    for parameter in transformer.parameters():
        if id(parameter) in base_dtypes and parameter.dtype != base_dtypes[id(parameter)]:
            raise RuntimeError("LoRA loading changed the base-model precision")
    transformer.requires_grad_(False)
    transformer.eval()
    if hasattr(transformer, "_reset_stateful_cache"):
        transformer._reset_stateful_cache()
    transformer._h3_adapter_settings = settings
