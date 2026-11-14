"""Cache-budget metadata checks with tiny weights; no downloads/model imports."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import preflight


def weights(path, names=("weight",), truncate=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = {name: {"dtype": "BF16", "shape": [4], "data_offsets": [i * 8, (i + 1) * 8]}
              for i, name in enumerate(names)}
    raw = json.dumps(header).encode()
    content = len(raw).to_bytes(8, "little") + raw + bytes(8 * len(names))
    path.write_bytes(content[:len(content) - truncate] if truncate else content)
    return path.stat().st_size


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.delenv("H3_MODEL_REVISION", raising=False)
    root = tmp_path / "hub"
    snapshot = root / "models--MiniMaxAI--MiniMax-H3" / "snapshots" / preflight.MODEL_REVISION
    snapshot.mkdir(parents=True)
    return root, snapshot


def sharded(snapshot, component="transformer_ref", missing=False):
    directory = snapshot / component
    directory.mkdir(parents=True, exist_ok=True)
    stem = preflight._WEIGHT_COMPONENTS[component]
    names = [stem + "-00001-of-00002.safetensors", stem + "-00002-of-00002.safetensors"]
    size = weights(directory / names[0], ("first",))
    if not missing:
        size += weights(directory / names[1], ("second",))
    (directory / (stem + ".safetensors.index.json")).write_text(json.dumps({
        "metadata": {"total_size": 16}, "weight_map": {"first": names[0], "second": names[1]}}))
    return size


def test_only_complete_required_components_reduce_first_download_budget(cache):
    root, snapshot = cache
    expected = sharded(snapshot)
    expected += weights(snapshot / "audio_vae" / "diffusion_pytorch_model.safetensors")
    weights(snapshot / "unrelated" / "diffusion_pytorch_model.safetensors")
    result = preflight.cached_weight_budget(root)
    assert result["reusable_bytes"] == expected
    assert result["verified_components"] == ["transformer_ref", "audio_vae"]
    assert result["scope"] == "disk_budget_only_not_model_readiness"


def test_partial_component_and_incomplete_files_get_no_credit(cache):
    root, snapshot = cache
    sharded(snapshot, missing=True)
    weights(snapshot / "audio_vae" / "diffusion_pytorch_model.safetensors.incomplete")
    assert preflight.cached_weight_budget(root)["reusable_bytes"] == 0


def test_absent_cache_and_empty_snapshot_do_not_reduce_budget(cache):
    root, snapshot = cache
    assert preflight.cached_weight_budget(root)["reusable_bytes"] == 0
    assert preflight.cached_weight_budget(root / "missing")["reusable_bytes"] == 0


def test_metadata_probe_reads_only_header_not_weight_payload(cache, monkeypatch):
    root, snapshot = cache
    path = snapshot / "audio_vae" / "diffusion_pytorch_model.safetensors"
    size = weights(path)
    original_open = Path.open
    reads = []
    class HeaderOnly:
        def __enter__(self):
            self.stream = original_open(path, "rb")
            return self
        def read(self, n):
            reads.append(n)
            assert n >= 0 and len(reads) <= 2
            return self.stream.read(n)
        def __exit__(self, *args):
            self.stream.close()
    monkeypatch.setattr(Path, "open", lambda p, *a, **kw: HeaderOnly() if p == path else original_open(p, *a, **kw))
    assert preflight.cached_weight_budget(root)["reusable_bytes"] == size
    assert len(reads) == 2 and reads[0] == 8 and sum(reads) == size - 8


@pytest.mark.parametrize("bad", ["truncated", "index_tensor", "index_size", "unsafe_name", "invalid_header"])
def test_bad_weight_metadata_never_relaxes_gate_even_after_other_valid_component(cache, bad):
    root, snapshot = cache
    weights(snapshot / "text_encoder" / "model.safetensors")
    sharded(snapshot)
    path = snapshot / "transformer_ref" / "diffusion_pytorch_model-00001-of-00002.safetensors"
    index = path.parent / "diffusion_pytorch_model.safetensors.index.json"
    if bad == "truncated":
        weights(path, ("first",), truncate=1)
    elif bad == "invalid_header":
        path.write_bytes((preflight._METADATA_LIMIT + 1).to_bytes(8, "little"))
    else:
        data = json.loads(index.read_text())
        if bad == "index_tensor":
            data["weight_map"]["wrong"] = data["weight_map"].pop("first")
        elif bad == "index_size":
            data["metadata"]["total_size"] += 1
        else:
            data["weight_map"]["first"] = "../../other.safetensors"
        index.write_text(json.dumps(data))
    result = preflight.cached_weight_budget(root)
    assert result["reusable_bytes"] == 0 and result["verified_components"] == []
    assert result["observation"] == "cache_unverified"


def test_other_repositories_and_revisions_are_ignored_and_custom_commit_is_exact(cache, monkeypatch):
    root, snapshot = cache
    other_revision = snapshot.parent / ("a" * 40)
    size = weights(other_revision / "audio_vae" / "diffusion_pytorch_model.safetensors")
    weights(root / "models--other--repo" / "snapshots" / preflight.MODEL_REVISION / "audio_vae" / "diffusion_pytorch_model.safetensors")
    assert preflight.cached_weight_budget(root)["reusable_bytes"] == 0
    monkeypatch.setenv("H3_MODEL_REVISION", "a" * 40)
    assert preflight.cached_weight_budget(root)["reusable_bytes"] == size
    monkeypatch.setenv("H3_MODEL_REVISION", "main")
    assert preflight.cached_weight_budget(root)["reusable_bytes"] == 0


def symlink(path, target, directory=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.symlink_to(target, target_is_directory=directory)
    except OSError as error:
        pytest.skip(f"Symlinks unavailable on this test host: {error}")


def test_snapshot_blob_symlinks_are_supported_and_duplicate_blob_is_credited_once(cache):
    root, snapshot = cache
    blob = snapshot.parent.parent / "blobs" / ("b" * 64)
    expected = weights(blob)
    symlink(snapshot / "vae" / "diffusion_pytorch_model.safetensors", blob)
    symlink(snapshot / "audio_vae" / "diffusion_pytorch_model.safetensors", blob)
    result = preflight.cached_weight_budget(root)
    assert result["reusable_bytes"] == expected
    assert result["verified_components"] == ["vae", "audio_vae"]


def test_shared_content_addressed_blob_pool_requires_exact_prefix_and_hub_root(cache):
    root, snapshot = cache
    repo_blobs = snapshot.parent.parent / "blobs"
    digest = "be" + "1" * 62
    target = root / "blobs" / "be" / digest
    weights(target)
    assert preflight._cache_file(target, snapshot, repo_blobs)[0] == target
    # Boundary cases run without symlinks, including on Windows hosts that do
    # not grant symlink creation. The caller only supplies exact-snapshot files.
    for wrong in (root / "blobs" / "aa" / digest,
                  root.parent / "other_hub" / "blobs" / "be" / digest,
                  root / "blobs" / "be" / (digest + ".incomplete"),
                  root / "blobs" / "be" / ("be" + "1" * 38)):
        weights(wrong)
        with pytest.raises(ValueError, match="escapes"):
            preflight._cache_file(wrong, snapshot, repo_blobs)


def test_exact_snapshot_can_reference_shared_pool_without_crediting_unreferenced_blobs(cache):
    root, snapshot = cache
    digest = "be" + "1" * 62
    target = root / "blobs" / "be" / digest
    expected = weights(target)
    weights(root / "blobs" / "aa" / ("aa" + "2" * 62))
    symlink(snapshot / "audio_vae" / "diffusion_pytorch_model.safetensors", target)
    assert preflight.cached_weight_budget(root)["reusable_bytes"] == expected


def test_shared_blob_bucket_redirect_is_rejected(cache):
    root, snapshot = cache
    digest = "be" + "1" * 62
    target = root.parent / "elsewhere" / digest
    weights(target)
    symlink(root / "blobs" / "be", target.parent, directory=True)
    symlink(snapshot / "audio_vae" / "diffusion_pytorch_model.safetensors", root / "blobs" / "be" / digest)
    assert preflight.cached_weight_budget(root)["reusable_bytes"] == 0


@pytest.mark.parametrize("kind", ["other_repo", "other_revision", "incomplete", "component_redirect"])
def test_unsafe_or_incomplete_symlink_targets_are_not_credited(cache, kind):
    root, snapshot = cache
    if kind == "other_repo":
        target = root / "models--other--repo" / "blobs" / ("c" * 64)
    elif kind == "other_revision":
        target = snapshot.parent / ("d" * 40) / "audio_vae" / "diffusion_pytorch_model.safetensors"
    elif kind == "incomplete":
        target = snapshot.parent.parent / "blobs" / ("c" * 64 + ".incomplete")
    else:
        target = snapshot.parent / ("d" * 40) / "audio_vae" / "diffusion_pytorch_model.safetensors"
    weights(target)
    if kind == "component_redirect":
        symlink(snapshot / "audio_vae", target.parent, directory=True)
    else:
        symlink(snapshot / "audio_vae" / "diffusion_pytorch_model.safetensors", target)
    assert preflight.cached_weight_budget(root)["reusable_bytes"] == 0


def test_hub_cache_environment_selection(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    assert preflight._hub_cache_path() == tmp_path / "home" / "hub"
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path / "legacy"))
    assert preflight._hub_cache_path() == tmp_path / "legacy"
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "current"))
    assert preflight._hub_cache_path() == tmp_path / "current"
    for key in ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert preflight._hub_cache_path() == tmp_path / "xdg" / "huggingface" / "hub"


def test_preflight_requires_remaining_disk_budget_and_unchanged_ram_gate(cache, monkeypatch):
    root, snapshot = cache
    monkeypatch.setenv("HF_HUB_CACHE", str(root))
    monkeypatch.delenv("H3_SKIP_DISK_CHECK", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    original_is_file, original_read = Path.is_file, Path.read_text
    mem_path = Path("/proc/meminfo")
    ram_kib = [160 * 1024 ** 2]
    monkeypatch.setattr(Path, "is_file", lambda p: True if p == mem_path else original_is_file(p))
    monkeypatch.setattr(Path, "read_text", lambda p, *a, **kw: (
        f"MemAvailable: {ram_kib[0]} kB\nMemTotal: {170 * 1024 ** 2} kB\n" if p == mem_path else original_read(p, *a, **kw)))
    monkeypatch.setattr(preflight.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout="NVIDIA A100-SXM4-80GB, 81920"))
    free = [preflight.FIRST_DOWNLOAD_BYTES - 10]
    monkeypatch.setattr(preflight.shutil, "disk_usage", lambda path: SimpleNamespace(free=free[0]))
    with pytest.raises(RuntimeError, match="remaining required"):
        preflight.check_resources()
    credit = weights(snapshot / "audio_vae" / "diffusion_pytorch_model.safetensors")
    result = preflight.check_resources()
    assert result["cache_weight_observation"]["reusable_bytes"] == credit
    assert result["required_available_ram_gib"] == 140 and result["mode"] == "bf16"
    free[0] = preflight.FIRST_DOWNLOAD_BYTES - credit - 1
    with pytest.raises(RuntimeError, match="remaining required"):
        preflight.check_resources()
    free[0] = preflight.FIRST_DOWNLOAD_BYTES
    ram_kib[0] = 139 * 1024 ** 2
    with pytest.raises(RuntimeError, match="host RAM"):
        preflight.check_resources()
