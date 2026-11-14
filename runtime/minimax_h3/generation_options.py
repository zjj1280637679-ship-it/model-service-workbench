"""Auditable per-job BF16 choices; no arbitrary kernels or adapter downloads.

Configuration permits queue admission before readiness. Only the model's GPU
lease may apply these choices; CPU preparation only validates immutable data.
"""
from __future__ import annotations

from adapter_config import AdapterSettings, TURBO, load_adapter_settings, validate_steps
from runtime_options import RuntimeOptions

ATTENTION_BACKENDS = ("auto", "native", "_native_flash")
ACCELERATION_LORAS = ("none", TURBO.key)


def requested_options(parameters):
    """The same defaults apply at public API, queue, and model boundaries."""
    return {
        "attention_backend": parameters.get("attention_backend", "auto"),
        "acceleration_lora": parameters.get("acceleration_lora", "none"),
        "steps": parameters["steps"],
    }


def validate_generation_options(requested, settings):
    if requested["attention_backend"] not in ATTENTION_BACKENDS:
        raise ValueError("attention_backend must be auto, native or _native_flash; quantized/external attention kernels are unsupported")
    selected = requested["acceleration_lora"]
    if selected not in ACCELERATION_LORAS:
        raise ValueError("acceleration_lora must be none or lightx2v-ref2va-4step; arbitrary adapters are unsupported")
    steps = requested["steps"]
    if isinstance(steps, bool) or not isinstance(steps, int) or not 2 <= steps <= 50:
        raise ValueError("steps must be an integer between 2 and 50")
    if selected != "none" and selected != settings.acceleration:
        raise ValueError("Requested acceleration LoRA is not configured for preloading; restart with H3_ACCELERATION_LORA=lightx2v-ref2va-4step")
    validate_steps(AdapterSettings(acceleration=selected), steps)
    return requested


def effective_options(requested, settings, runtime_options):
    attention = requested["attention_backend"]
    if attention == "auto":
        attention = runtime_options.attention_backend
    # Explicitly restore PyTorch dispatch after a previous forced-Flash job.
    if attention == "auto":
        attention = "native"
    return {**requested, "attention_backend": attention, "attention_scope": "transformer_ref",
            "precision": "bf16", "denoiser_evaluations": requested["steps"] - 1,
            "style_lora": settings.style, "style_scale": settings.style_scale}


def generation_capabilities(*, model_ready=False, settings=None, runtime_options=None):
    settings = settings or load_adapter_settings()
    runtime_options = runtime_options or RuntimeOptions.from_env()
    default_attention = runtime_options.attention_backend
    attention = []
    for identity, label in (("auto", "自动 · PyTorch SDPA"), ("native", "PyTorch SDPA"),
                            ("_native_flash", "PyTorch Flash SDPA")):
        value = {"id": identity, "label": label, "supported": True, "configured": True,
                 "ready": bool(model_ready), "scope": "transformer_ref", "precision": "bf16"}
        if identity == "auto":
            value["resolves_to"] = "native" if default_attention == "auto" else default_attention
        if identity == "_native_flash":
            value["reason"] = "强制 PyTorch Flash SDPA；内核不兼容时明确失败，不静默回退"
        attention.append(value)
    configured = settings.acceleration == TURBO.key
    return {"schema_version": 1, "workflow": "ref2va", "precision": "bf16",
            "model_ready": bool(model_ready),
            "defaults": {"attention_backend": "auto", "acceleration_lora": "none", "steps": 50},
            "options": {"attention_backends": attention, "acceleration_loras": [
                {"id": "none", "label": "关闭", "supported": True, "configured": True,
                 "ready": bool(model_ready), "steps": {"min": 2, "max": 50, "default": 50}},
                {"id": TURBO.key, "label": "LightX2V Ref2VA Turbo · 4 次评估", "supported": True,
                 "configured": configured, "ready": bool(configured and model_ready),
                 "steps": {"min": 5, "max": 5, "required": 5, "denoiser_evaluations": 4},
                 "reason": "已配置预加载；模型就绪前可排队" if configured else "未在启动配置中预加载",
                 "repository": TURBO.repository, "revision": TURBO.revision, "sha256": TURBO.sha256,
                 "precision": "bf16", "validation": "GPU acceptance is reported separately; configuration is not a speed claim"},
            ]},
            "queue_semantics": "supported && configured permits admission before model_ready; readiness never means completion",
            "step_semantics": "steps counts sigma grid points; actual denoiser evaluations = steps - 1",
            "style_lora": {"id": settings.style, "scale": settings.style_scale, "control": "deployment"},
            "quantization": False}
