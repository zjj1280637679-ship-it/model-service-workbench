"""Conservative project resource checks; no downloads and no model imports."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess

GIB = 1024 ** 3
FIRST_DOWNLOAD_BYTES = 140 * GIB
MODEL_REVISION = "0919fb684102ef115126612f3fbb84d1027e7317"
_WEIGHT_COMPONENTS = {"text_encoder": "model", "transformer_ref": "diffusion_pytorch_model",
                      "vae": "diffusion_pytorch_model", "audio_vae": "diffusion_pytorch_model"}
_METADATA_LIMIT = 16 * 1024 ** 2


def _hub_cache_path() -> Path:
    default_home = Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")) / "huggingface"
    home = Path(os.environ.get("HF_HOME", str(default_home))).expanduser()
    return Path(os.environ.get("HF_HUB_CACHE", os.environ.get("HUGGINGFACE_HUB_CACHE", str(home / "hub")))).expanduser()


def _cache_file(path, snapshot, blobs):
    """Accept snapshot files and completed repo/shared content-addressed blobs."""
    if path.parent.resolve(strict=True) != path.parent:
        raise ValueError("untrusted cache component directory redirection")
    resolved = path.resolve(strict=True)
    # Newer Hub caches share immutable SHA-256 blobs across repositories. Only
    # an exact snapshot reference may reach this pool; never scan/credit it.
    shared_pool = blobs.parent.parent / "blobs"
    shared_bucket = shared_pool / resolved.name[:2]
    shared_blob = (re.fullmatch(r"[0-9a-f]{64}", resolved.name) is not None
                   and resolved.parent == shared_bucket
                   and shared_pool.resolve(strict=True) == shared_pool
                   and shared_bucket.resolve(strict=True) == shared_bucket)
    if not (resolved.is_relative_to(snapshot) or
            (resolved.parent == blobs and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", resolved.name))
            or shared_blob):
        raise ValueError("cache file escapes its snapshot/repository blobs")
    info = resolved.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_size <= 0 or ".incomplete" in resolved.name:
        raise ValueError("cache file is not a completed regular file")
    return resolved, info


def _small_json(path, snapshot, blobs):
    resolved, info = _cache_file(path, snapshot, blobs)
    if info.st_size > _METADATA_LIMIT:
        raise ValueError("cache metadata exceeds inspection limit")
    return json.loads(resolved.read_bytes())


def _weight_metadata(path, snapshot, blobs):
    """Check only the bounded safetensors header and file size, not weight data."""
    resolved, before = _cache_file(path, snapshot, blobs)
    with resolved.open("rb") as source:
        prefix = source.read(8)
        if len(prefix) != 8:
            raise ValueError("incomplete safetensors header")
        length = int.from_bytes(prefix, "little")
        if not 2 <= length <= _METADATA_LIMIT or length + 8 > before.st_size:
            raise ValueError("invalid safetensors header size")
        header = json.loads(source.read(length))
    if not isinstance(header, dict):
        raise ValueError("invalid safetensors header")
    ranges, names = [], set()
    sizes = {"BOOL": 1, "U8": 1, "I8": 1, "U16": 2, "I16": 2, "F16": 2, "BF16": 2,
             "U32": 4, "I32": 4, "F32": 4, "U64": 8, "I64": 8, "F64": 8,
             "F8_E4M3": 1, "F8_E5M2": 1}
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        start, end = entry["data_offsets"]
        shape, element_bytes = entry["shape"], sizes[entry["dtype"]]
        if not isinstance(shape, list) or len(shape) > 16 or any(type(n) is not int or n < 0 for n in shape):
            raise ValueError("invalid tensor shape")
        count = 1
        for dimension in shape:
            count *= dimension
        if type(start) is not int or type(end) is not int or start < 0 or end - start != count * element_bytes:
            raise ValueError("invalid tensor data range")
        ranges.append((start, end))
        names.add(name)
    cursor = 0
    for start, end in sorted(ranges):
        if start != cursor:
            raise ValueError("tensor data ranges are incomplete or overlap")
        cursor = end
    after = resolved.stat()
    if not names or cursor <= 0 or 8 + length + cursor != before.st_size or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("weight file is truncated, changing, or has trailing data")
    identity = (after.st_dev, after.st_ino) if after.st_ino else str(resolved)
    return names, cursor, after.st_size, identity


def cached_weight_budget(cache: Path) -> dict:
    """Credit complete weight components, not readiness or model integrity."""
    revision = os.environ.get("H3_MODEL_REVISION", MODEL_REVISION)
    result = {"model_revision": revision, "reusable_bytes": 0, "verified_components": [],
              "observation": "no_verified_weights", "scope": "disk_budget_only_not_model_readiness"}
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        result["observation"] = "revision_not_an_exact_commit"
        return result
    try:
        root = cache.resolve(strict=True)
        repo = root / "models--MiniMaxAI--MiniMax-H3"
        snapshot = repo / "snapshots" / revision
        blobs = repo / "blobs"
        if not snapshot.is_dir():
            return result
        # Parent directory symlinks may otherwise redirect an exact snapshot
        # name into a different revision/repository or an unrelated filesystem.
        for path in (repo, snapshot.parent, snapshot):
            if path.resolve(strict=True) != path:
                raise ValueError("untrusted cache directory redirection")
        credited = {}
        for component, stem in _WEIGHT_COMPONENTS.items():
            directory = snapshot / component
            single = directory / (stem + ".safetensors")
            index = directory / (stem + ".safetensors.index.json")
            if single.exists():
                files, weight_map, total_size = [single.name], None, None
            elif index.exists():
                data = _small_json(index, snapshot, blobs)
                weight_map = data["weight_map"]
                if not isinstance(weight_map, dict) or not weight_map:
                    raise ValueError("invalid weight index")
                files = sorted(set(weight_map.values()))
                total_size = data.get("metadata", {}).get("total_size")
                if total_size is not None and (type(total_size) is not int or total_size <= 0):
                    raise ValueError("invalid indexed weight size")
            else:
                continue
            if any(not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+\.safetensors", name) for name in files):
                raise ValueError("untrusted shard filename")
            if any(not (directory / name).exists() for name in files):
                continue  # partially downloaded components receive no credit
            component_files, data_bytes = {}, 0
            for filename in files:
                names, size, disk_bytes, identity = _weight_metadata(directory / filename, snapshot, blobs)
                if weight_map is not None and names != {key for key, value in weight_map.items() if value == filename}:
                    raise ValueError("weight index does not match shard tensors")
                component_files[identity] = disk_bytes
                data_bytes += size
            if total_size is not None and data_bytes != total_size:
                raise ValueError("indexed weight bytes are incomplete")
            credited.update(component_files)
            result["verified_components"].append(component)
        result["reusable_bytes"] = min(sum(credited.values()), FIRST_DOWNLOAD_BYTES)
        if credited:
            result["observation"] = "completed_weight_metadata_verified"
    except FileNotFoundError:
        # Cache creation/deletion races and missing roots never relax the gate.
        result.update(reusable_bytes=0, verified_components=[], observation="cache_missing_or_changed")
    except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError, RuntimeError) as error:
        result.update(reusable_bytes=0, verified_components=[], observation="cache_unverified",
                      error_type=type(error).__name__)
    return result


def check_resources(mode: str = "bf16", residency: str = "resident") -> dict:
    if mode != "bf16":
        raise ValueError("This project requires native BF16 on an NVIDIA A100; reduced-precision modes are disabled.")
    if residency not in {"resident", "staged"}:
        raise ValueError("residency must be resident or staged")
    if not Path("/proc/meminfo").is_file():
        raise RuntimeError("Deploy this project on a Linux NVIDIA GPU machine.")
    memory = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, value = line.split(":", 1)
        memory[name] = int(value.strip().split()[0]) * 1024
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            text=True, capture_output=True, check=True, timeout=20,
        )
        first = result.stdout.strip().splitlines()[0]
        gpu, memory_mib = first.rsplit(",", 1)
        vram_gib = float(memory_mib.strip()) / 1024
    except (OSError, ValueError, IndexError, subprocess.SubprocessError) as exc:
        raise RuntimeError("No working NVIDIA GPU detected. Select a GPU runtime first.") from exc
    # The first visible CUDA GPU is used. Report conservatively when device masks are present.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and visible not in {"0", "0,"}:
        raise RuntimeError("Preflight expects the first NVIDIA GPU. Unset CUDA_VISIBLE_DEVICES or set it to 0.")
    cache = _hub_cache_path()
    disk_path = cache
    while not disk_path.exists():
        disk_path = disk_path.parent
    free_disk_bytes = shutil.disk_usage(disk_path).free
    free_disk = free_disk_bytes / GIB
    cache_budget = cached_weight_budget(cache)
    remaining_disk_bytes = max(0, FIRST_DOWNLOAD_BYTES - cache_budget["reusable_bytes"])
    available_ram = memory["MemAvailable"] / GIB
    strategy = ("staged_component_residency" if residency == "staged" else
                "component_cpu_offload" if vram_gib >= 75 else "group_cpu_offload")
    required_ram_gib = 70 if residency == "staged" else 140
    report = {
        "mode": mode, "gpu": gpu.strip(), "vram_gib": round(vram_gib, 1),
        "available_ram_gib": round(available_ram, 1),
        "total_ram_gib": round(memory["MemTotal"] / GIB, 1),
        "cache_free_disk_gib": round(free_disk, 1), "cache_path": str(cache),
        "cache_reusable_weight_gib": round(cache_budget["reusable_bytes"] / GIB, 2),
        "cache_remaining_required_gib": round(remaining_disk_bytes / GIB, 2),
        "cache_weight_observation": cache_budget,
        "disk_check_skipped": os.environ.get("H3_SKIP_DISK_CHECK") == "1",
        "a100_variant": "80GB" if vram_gib >= 75 else "40GB",
        "strategy": strategy,
        "residency": residency, "required_available_ram_gib": required_ram_gib,
    }
    errors = []
    if "A100" not in gpu.upper():
        errors.append(f"GPU must be an NVIDIA A100, detected {gpu.strip()}")
    if available_ram < required_ram_gib:
        errors.append(f"available host RAM {available_ram:.1f} GiB < {required_ram_gib} GiB ({residency})")
    if vram_gib < 38:
        errors.append(f"GPU memory {vram_gib:.1f} GiB < 38 GiB (A100 40GB class required)")
    if free_disk_bytes < remaining_disk_bytes and not report["disk_check_skipped"]:
        errors.append(
            f"cache disk free {free_disk:.1f} GiB < remaining required {remaining_disk_bytes / GIB:.1f} GiB "
            f"(140 GiB first-download budget minus {cache_budget['reusable_bytes'] / GIB:.1f} GiB "
            "of completed reusable weights; unverified/partial files receive no credit)"
        )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    if errors:
        raise RuntimeError(
            "Resource preflight failed before model download: " + "; ".join(errors)
            + ". These are conservative project gates, not a guarantee against peak-memory OOM."
        )
    if residency == "staged":
        print(
            "Staged A100 BF16: the text encoder and transformer_ref are loaded one at a time, "
            "with the previous large component unloaded first. Disk reloads and CPU/GPU transfers add latency. "
            "The 70 GiB host-RAM gate is a project threshold. The original staged A100 40GB single-image baseline passed; this optimized build still needs GPU acceptance.",
            flush=True,
        )
    elif strategy == "group_cpu_offload":
        print(
            "A100 40GB: native BF16 weights are preserved. The 61.7 GB transformer cannot fit in GPU memory, "
            "so blocks move between CPU RAM and GPU. Throughput depends on PCIe and host memory bandwidth and may be slow.",
            flush=True,
        )
    else:
        print("A100 80GB: native BF16 with components alternated between CPU RAM and GPU.", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", action="store_true", help="Explicit sequential BF16 residency profile: 70 GiB available host RAM")
    args = parser.parse_args()
    check_resources(os.environ.get("H3_MODE", "bf16"), residency="staged" if args.staged else "resident")
