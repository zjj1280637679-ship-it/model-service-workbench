"""Fresh-process LoRA dependency check before model downloads or GPU loading.

PEFT 0.21.2 visits its optional torchao dispatcher even for ordinary BF16
Linear layers. An installed torchao < 0.16 raises before the normal dispatcher.
The dedicated Colab runtime may explicitly remove that unused optional package;
this helper never installs or upgrades packages, or changes the PyTorch stack.
"""
from __future__ import annotations

import argparse
from importlib import metadata
import importlib.util
import json
import subprocess
import sys

from packaging.version import Version


STACK_PACKAGES = ("torch", "torchvision", "torchaudio", "psutil")


def distribution_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def prepare_optional_torchao(*, repair=False):
    """Run before importing torch/PEFT; only the explicit old optional package changes."""
    before = {name: distribution_version(name) for name in STACK_PACKAGES}
    old_version = distribution_version("torchao")
    removed = None
    if old_version is not None and Version(old_version) < Version("0.16.0"):
        if not repair:
            raise RuntimeError(
                f"Unused torchao {old_version} blocks PEFT BF16 LoRA dispatch. "
                "In this dedicated runtime run lora_preflight.py --repair-optional-torchao; "
                "do not upgrade the existing PyTorch stack."
            )
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
                       check=True, capture_output=True, text=True)
        importlib.invalidate_caches()
        removed = old_version
    after = {name: distribution_version(name) for name in STACK_PACKAGES}
    if before != after:
        raise RuntimeError("LoRA optional dependency preparation changed the protected PyTorch/psutil stack")
    current = distribution_version("torchao")
    if removed is not None and (current is not None or importlib.util.find_spec("torchao") is not None):
        raise RuntimeError("torchao remains importable after uninstall; use a clean model process before loading LoRA")
    return {"optional_torchao": {"before": old_version, "after": current, "removed_version": removed},
            "stack_before": before, "stack_after": after, "stack_unchanged": True}


def bf16_lora_smoke():
    """Exercise PEFT's actual dispatch and injection, which import-only checks miss.

    Tiny CPU tensors avoid GPU initialization, large model downloads, and changes
    to the model container. Native BF16 GPU support is checked by normal preflight.
    """
    import torch
    import peft
    from peft import LoraConfig, get_peft_model
    from peft.import_utils import is_torchao_available

    base = torch.nn.Sequential(torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16))
    model = get_peft_model(base, LoraConfig(r=2, lora_alpha=2, target_modules=["0"]),
                           autocast_adapter_dtype=False)
    model.requires_grad_(False)
    model.eval()
    value = torch.ones((1, 4), dtype=torch.bfloat16)
    with torch.no_grad():
        enabled = model(value)
        with model.disable_adapter():
            disabled = model(value)
        restored = model(value)
    if any(parameter.dtype != torch.bfloat16 for parameter in model.parameters()):
        raise RuntimeError("Tiny LoRA injection changed BF16 parameter precision")
    if any(output.dtype != torch.bfloat16 or not torch.isfinite(output).all().item()
           for output in (enabled, disabled, restored)):
        raise RuntimeError("Tiny BF16 LoRA enable/disable/restore forward check failed")
    if not any("lora_A" in name for name, _ in model.named_parameters()):
        raise RuntimeError("Tiny LoRA test did not inject an adapter")
    return {"peft": peft.__version__, "torch": torch.__version__,
            "torchao_available": is_torchao_available(), "device": "cpu", "precision": "bf16",
            "injected": True, "enable_disable_restore_forward": True,
            "parameter_elements": sum(parameter.numel() for parameter in model.parameters()),
            "real_model_loaded": False, "gpu_generation_verified": False}


def run(*, repair=False):
    report = prepare_optional_torchao(repair=repair)
    report["smoke"] = bf16_lora_smoke()
    report["status"] = "passed"
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repair-optional-torchao", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(repair=args.repair_optional_torchao), ensure_ascii=False))
