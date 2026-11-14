"""Sequential component residency: single-image A100 40GB acceptance passed.

Verified on Colab A100 40GB with about 83.47 GiB host RAM: public submission,
polling, generation, download, and full audiovisual decode. The 512x288 test
used one image, a 5s request/124 output frames, and 8 steps (7 denoising rounds).
The backbone stays BF16 while official _keep_in_fp32_modules layers remain FP32.
No quantization or fake output exists. This is an integration test, not a
quality benchmark or acceptance of video/audio reference combinations.
Optional runtime modes, adapters, and media scheduling added afterward have not
been GPU-retested; their defaults preserve the accepted inference choices.

Tested with Python 3.13.16, PyTorch 2.14.1+cu130, Diffusers 0.43.0, and
Transformers 5.7.0. The separately pinned PyTorch 2.10 install has not been
GPU-tested. A100 80GB resident acceptance is recorded separately in README.md;
it does not establish acceptance of this refactored staged pipeline.
"""
from __future__ import annotations

import gc
import os
from pathlib import Path
import time

from model_backend import MODEL_ID, MODEL_REVISION, _normalize_video, _probe, aligned_frames, _parameter_kept_in_fp32, _audit_parameter_precision
from runtime_options import configure_qwen_offload, measured_load
from pipeline_backend import PipelineBackendSupport, GIB
from adapter_config import load_adapter_settings, configure_pipeline, apply_adapters, validate_steps
from download_progress import tracked_model_load


class StagedMiniMaxBackend(PipelineBackendSupport):
    """Exclusive GPU inference; Qwen and transformer_ref never co-reside."""
    pipeline_model_peak_ram_bytes = 76 * GIB
    resource_residency = "staged"
    _reference_probe = staticmethod(_probe)
    _normalize_reference = staticmethod(_normalize_video)
    _aligned_frames = staticmethod(aligned_frames)

    def __init__(self, runtime_options=None):
        self._init_runtime(runtime_options)
        self.adapter_settings = load_adapter_settings()
        self.pipe = None
        self.revision = os.environ.get("H3_MODEL_REVISION", MODEL_REVISION)
        self._phase = None
        self.precision_audit = None

    @staticmethod
    def _log(message):
        print(f"[{time.strftime('%H:%M:%S')}] BF16 staged: {message}", flush=True)

    def _load_required(self, names):
        import torch
        # Diffusers 0.43 load_components logs some failures instead of raising.
        # Load individually to avoid its temporary dict retaining multiple newly
        # loaded components before registering them.
        for name in names:
            if getattr(self.pipe, name, None) is None:
                self._log(f"loading {name}")
                with self._stage(f"load.{name}"):
                    self.pipe.load_components(names=name, dtype=torch.bfloat16, revision=self.revision)
            if getattr(self.pipe, name, None) is None:
                raise RuntimeError(f"Required MiniMax-H3 component failed to load: {name}")
            if name in {"text_encoder", "transformer_ref"}:
                from collections import Counter
                module = getattr(self.pipe, name)
                keep_fp32 = getattr(module, "_keep_in_fp32_modules", []) or []
                counts = Counter()
                for parameter_name, parameter in module.named_parameters():
                    counts[str(parameter.dtype)] += parameter.numel()
                    expected = torch.float32 if _parameter_kept_in_fp32(parameter_name, keep_fp32) else torch.bfloat16
                    if parameter.dtype != expected:
                        raise RuntimeError(f"Unexpected precision in {name}.{parameter_name}: {parameter.dtype}, expected {expected}")
                self._log(f"{name} original precision verified, parameter counts={dict(counts)}")

    def _switch(self, target):
        if self._phase != target:
            with self._stage(f"switch.{target}"):
                self._switch_impl(target)

    def _switch_impl(self, target):
        import torch
        other = "text_encoder" if target == "denoise" else "transformer_ref"
        self._phase = None  # a failed load must never leave a stale ready phase
        self._log(f"unloading {other} before {target}")
        self.pipe.unload_components(other)
        gc.collect()
        torch.cuda.empty_cache()
        if target == "encode":
            self._load_required(["text_encoder"])
            configure_qwen_offload(self.pipe.text_encoder)
            self._log("Qwen vision/embeddings/rotary on CUDA; decoder layers offloaded as whole BF16 blocks")
        else:
            self._load_required(["transformer_ref"])
            self.pipe.transformer_ref.requires_grad_(False)
            with self._stage("configure.adapters"):
                apply_adapters(self.pipe, self.adapter_settings)
            self.precision_audit = _audit_parameter_precision(self.pipe)
            self.runtime_options.configure_attention(self.pipe.transformer_ref)
            selected = getattr(self, "_active_generation_options", None)
            if selected is not None:
                self._apply_generation_transformer(selected)
            self.pipe.transformer_ref.enable_group_offload(**self.runtime_options.group_offload_kwargs(torch))
            self._log(f"denoiser offload={self.runtime_options.denoiser_offload}; attention={self.runtime_options.attention_backend}")
        self._phase = target
        self._log(f"{target} stage loaded; other large component released")

    @measured_load
    @tracked_model_load
    def load(self):
        import torch
        from diffusers import ModularPipeline
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Native-BF16 CUDA device is required")
        if "A100" not in torch.cuda.get_device_name(0).upper():
            raise RuntimeError("This acceptance build targets an NVIDIA A100")
        import psutil, shutil
        if psutil.virtual_memory().available < 70 * 1024**3:
            raise RuntimeError("Staged BF16 requires at least 70 GiB available host RAM before loading")
        if torch.cuda.get_device_properties(0).total_memory < 38 * 1024**3:
            raise RuntimeError("Staged BF16 requires an A100 40 GB or larger")
        cache = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
        cache.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(cache).free < 140 * 1024**3 and os.environ.get("H3_SKIP_DISK_CHECK") != "1":
            raise RuntimeError("Staged BF16 needs 140 GiB free cache disk on the first download")
        self.pipe = ModularPipeline.from_pretrained(
            MODEL_ID, revision=self.revision, workflow="ref2va",
        )
        configure_pipeline(self.pipe, self.adapter_settings)
        names = list(self.pipe.blocks.sub_blocks)
        if names[:3] != ["before_encode", "text_encoder", "vae_encoder"]:
            raise RuntimeError(f"Unexpected Ref2VA block topology: {names}")
        if not any(name.startswith("denoise.") for name in names) or not any(name.startswith("decode.") for name in names):
            raise RuntimeError(f"Unexpected Ref2VA block topology: {names}")
        self._load_required(["tokenizer", "processor", "scheduler", "audio_scheduler", "vae", "audio_vae"])
        self.pipe.vae.requires_grad_(False).to("cuda")
        self.pipe.audio_vae.requires_grad_(False).to("cuda")
        # Actually load both large components during startup, one at a time. This
        # also downloads all required weights before the API advertises readiness.
        self._switch("denoise")
        self._switch("encode")
        self._log("both large components loaded successfully in turn; encoder stage ready")

    @staticmethod
    def _cpu(value):
        import torch
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        if isinstance(value, list):
            return [StagedMiniMaxBackend._cpu(item) for item in value]
        if isinstance(value, tuple):
            return tuple(StagedMiniMaxBackend._cpu(item) for item in value)
        if isinstance(value, dict):
            return {key: StagedMiniMaxBackend._cpu(item) for key, item in value.items()}
        return value

    def _run(self, state, stage):
        import torch
        # The resolved workflow has flat leaf keys: before_encode, text_encoder,
        # vae_encoder, denoise.prepare_layout ... denoise.after_denoise,
        # decode.video, decode.audio. Calling the documented block contract
        # (components, PipelineState) avoids a deepcopy of large reference media.
        with torch.inference_mode():
            for name, block in self.pipe.blocks.sub_blocks.items():
                selected = name in {"before_encode", "text_encoder", "vae_encoder"} if stage == "encode" else name.startswith(stage + ".")
                if selected:
                    self._log(f"running {name}")
                    with self._stage(f"block.{name}"):
                        _, state = block(self.pipe, state)
        return state

    def validate_request(self, *, steps, attention_backend=None, acceleration_lora=None):
        return self.validate_generation_request(steps=steps, attention_backend=attention_backend,
                                                acceleration_lora=acceleration_lora)

    def _infer_raw(self, prepared):
        """All staged model swapping, text/VAE encode, denoise and decode use GPU lease."""
        import torch
        from diffusers.modular_pipelines.modular_pipeline import PipelineState
        if self.pipe is None:
            raise RuntimeError("Backend has not loaded")
        p = prepared.parameters
        state = None
        try:
            self._switch("encode")
            state = PipelineState(values={
                "prompt": p["prompt"], "references": prepared.references, "num_frames": aligned_frames(p["seconds"]),
                "height": p["height"], "width": p["width"], "num_inference_steps": p["steps"],
                "generator": torch.Generator(device="cpu").manual_seed(p["seed"]),
                "output_type": "pil",
            })
            state = self._run(state, "encode")
            # prompt_embeds is the one GPU tensor emitted by the encoder; VAE
            # conditioning latents already live on CPU. Do not retain CUDA state
            # while swapping 62GB components; preserve generator state unchanged.
            state.values = self._cpu(state.values)
            self._switch("denoise")
            # The denoiser passes prompt_embeds straight through, unlike layout/
            # condition latents, which their preparation blocks move to device.
            state.set("prompt_embeds", state.get("prompt_embeds").to("cuda"))
            state = self._run(state, "denoise")
            # Free the large denoiser before VAE decode; no longer needed for this
            # job. Keep the final latents on CUDA for the official decoder blocks.
            with self._stage("unload.transformer_ref"):
                self.pipe.unload_components("transformer_ref")
                self._phase = None
            state = self._run(state, "decode")
            return {"videos": state.get("videos"), "audio": state.get("audio"), "sampling_rate": state.get("sampling_rate")}
        finally:
            state = None
