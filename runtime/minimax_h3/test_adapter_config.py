"""CPU-only contract tests; these do not establish GPU inference acceptance."""
import sys
from types import SimpleNamespace, ModuleType
from unittest.mock import Mock, patch
from pathlib import Path

import pytest
from PIL import Image

import adapter_config as adapters


def test_disabled_adapters_are_lazy_and_do_not_require_torch_or_network():
    settings = adapters.load_adapter_settings({})
    assert settings.as_dict()["validation"] == "baseline"
    adapters.configure_pipeline(None, settings)
    adapters.apply_adapters(None, settings)
    adapters.validate_steps(settings, 8)


@pytest.mark.parametrize("env", [
    {"H3_ACCELERATION_LORA": "fl2va"},
    {"H3_STYLE_LORA": "https://unreviewed.example/lora.safetensors"},
    {"H3_STYLE_LORA": "/content/arbitrary.safetensors"},
    {"H3_STYLE_LORA_SCALE": "nan"},
    {"H3_STYLE_LORA_SCALE": "inf"},
    {"H3_STYLE_LORA_SCALE": "0"},
    {"H3_STYLE_LORA_SCALE": "1.1"},
])
def test_unknown_sources_and_invalid_scales_fail_early(env):
    with pytest.raises(ValueError):
        adapters.load_adapter_settings(env)


def test_turbo_counts_terminal_sigma_point():
    settings = adapters.AdapterSettings(acceleration=adapters.TURBO.key)
    adapters.validate_steps(settings, 5)
    with pytest.raises(ValueError, match="4 denoiser"):
        adapters.validate_steps(settings, 4)
    with pytest.raises(ValueError):
        adapters.validate_steps(settings, 8)


def test_reference_match_preserves_ratio_area_policy_and_small_inputs():
    assert adapters.reference_match_size(2048, 2048, 960, 544) == (736, 736)
    assert adapters.reference_match_size(128, 64, 960, 544) == (128, 64)
    assert adapters.reference_match_size(1920, 1080, 512, 288) == (512, 288)
    with pytest.raises(ValueError):
        adapters.reference_match_size(0, 128, 512, 288)
    with pytest.raises(ValueError):
        adapters.reference_match_size(1000, 100, 512, 288)


def tensor(shape):
    return SimpleNamespace(shape=shape)


def test_converted_weights_are_paired_and_exactly_fit_base():
    model = SimpleNamespace(get_submodule=lambda name: SimpleNamespace(weight=tensor((8, 6))))
    key = "transformer.transformer_blocks.0.attn.to_q"
    valid = {key + ".lora_A.weight": tensor((2, 6)), key + ".lora_B.weight": tensor((8, 2))}
    assert len(adapters._validate_converted(valid, model, adapters.REALISM)) == 1
    invalid = dict(valid)
    invalid[key + ".lora_B.weight"] = tensor((4, 2))
    with pytest.raises(ValueError, match="shape mismatch"):
        adapters._validate_converted(invalid, model, adapters.REALISM)
    with pytest.raises(ValueError, match="Unpaired"):
        adapters._validate_converted({key + ".lora_A.weight": tensor((2, 6))}, model, adapters.REALISM)
    with pytest.raises(ValueError, match="shared attention"):
        adapters._validate_converted({k.replace("attn.to_q", "ff.net.2"): v for k, v in valid.items()}, model, adapters.REALISM)


def test_turbo_resize_is_scoped_and_preserves_video_audio_and_original_pixels():
    class Setup:
        def __call__(self, components, state):
            # Simulate upstream's validation/normalization, with a deliberately
            # different intermediate image to catch resizing the wrong pixels.
            refs = state.get("references")
            state.set("normalized_references", [
                SimpleNamespace(kind="image", image=Image.new("RGB", (2048, 2048), "blue")),
                refs[1], refs[2],
            ])
            return components, state
    class State:
        def __init__(self, values): self.values = values
        def get(self, key): return self.values[key]
        def set(self, key, value): self.values[key] = value
    before = ModuleType("diffusers.modular_pipelines.minimax_h3.before_encoder")
    before.MiniMaxH3Ref2VASetupStep = Setup
    module = ModuleType("diffusers.modular_pipelines.minimax_h3")
    module.MiniMaxH3ImageReference = lambda image: SimpleNamespace(kind="image", image=image)
    original = Setup()
    pipe = SimpleNamespace(blocks=SimpleNamespace(sub_blocks={"before_encode": original}), canvas_multiple=32,
        image_processor=SimpleNamespace(resize=lambda image, height, width: image.resize((width, height))))
    refs = [SimpleNamespace(kind="image", image=Image.new("RGB", (1024, 1024), "red")),
            SimpleNamespace(kind="video"), SimpleNamespace(kind="audio")]
    state = State({"references": refs, "width": 512, "height": 288})
    with patch.dict(sys.modules, {before.__name__: before, module.__name__: module}):
        adapters.configure_pipeline(pipe, adapters.AdapterSettings(acceleration=adapters.TURBO.key))
        replacement = pipe.blocks.sub_blocks["before_encode"]
        assert replacement is not original
        replacement(pipe, state)
        assert state.get("normalized_references")[0].image.size == (384, 384)
        assert state.get("normalized_references")[0].image.getpixel((0, 0)) == (255, 0, 0)
        assert state.get("normalized_references")[1:] == refs[1:]
        assert original.__class__ is Setup
        adapters.configure_pipeline(pipe, adapters.AdapterSettings(acceleration=adapters.TURBO.key))
        assert pipe.blocks.sub_blocks["before_encode"] is replacement


def test_file_checksum_failure_is_not_accepted(tmp_path):
    path = tmp_path / "test.safetensors"
    path.write_bytes(b"bad")
    module = ModuleType("huggingface_hub")
    module.hf_hub_download = Mock(return_value=str(path))
    spec = adapters.AdapterSpec("test", "repo/test", path.name, "a" * 40, "0" * 64, 3, 1)
    with patch.dict(sys.modules, {module.__name__: module}):
        with pytest.raises(RuntimeError, match="SHA-256"):
            adapters._download_verified(spec)
    module.hf_hub_download.assert_called_once_with("repo/test", path.name, revision="a" * 40)


def test_apply_preserves_metadata_routes_ref_partition_and_only_casts_factors():
    class Factor:
        def __init__(self): self.dtype = "float32"
        def to(self, dtype): self.dtype = dtype
    base = SimpleNamespace(weight=SimpleNamespace(shape=(8, 6), dtype="bfloat16"))
    protected = SimpleNamespace(dtype="float32")
    layer = SimpleNamespace(get_base_layer=lambda: base, lora_A={}, lora_B={}, scaling={})
    model = SimpleNamespace(
        get_submodule=lambda name: layer,
        parameters=lambda: iter([base.weight, protected]),
        set_adapters=Mock(), requires_grad_=Mock(), eval=Mock(), _reset_stateful_cache=Mock(),
    )
    key = "transformer.transformer_blocks.0.attn.to_q"
    state = {key + ".lora_A.weight": tensor((2, 6)), key + ".lora_B.weight": tensor((8, 2))}
    loaded = []
    def load(state_dict, **kwargs):
        loaded.append(kwargs)
        name = kwargs["adapter_name"]
        layer.lora_A[name], layer.lora_B[name] = Factor(), Factor()
        layer.scaling[name] = 8 / 128 if name == "h3_acceleration" else 1.0
    def state_dict(path, **kwargs):
        return dict(state), {"preserved_metadata": kwargs["weight_name"]}
    pipe = SimpleNamespace(
        transformer_ref=model, transformer=None,
        scheduler=SimpleNamespace(config=SimpleNamespace(shift=12)),
        audio_scheduler=SimpleNamespace(config=SimpleNamespace(shift=3)),
        lora_state_dict=Mock(side_effect=state_dict), load_lora_into_transformer_ref=load,
    )
    torch = ModuleType("torch")
    torch.bfloat16, torch.float32 = "bfloat16", "float32"
    safetensors = ModuleType("safetensors")
    class Handle:
        def __init__(self, path, **kwargs): self.path = path
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def keys(self): return range(624 if self.path.name == adapters.TURBO.filename else 208)
        def get_slice(self, key): return SimpleNamespace(get_dtype=lambda: "BF16")
        def metadata(self): return {"alpha": "8"}
    safetensors.safe_open = Handle
    settings = adapters.AdapterSettings(acceleration=adapters.TURBO.key, style=adapters.REALISM.key, style_scale=0.7)
    with patch.dict(sys.modules, {"torch": torch, "safetensors": safetensors}), \
         patch("importlib.util.find_spec", return_value=True), \
         patch.object(adapters, "_download_verified", side_effect=lambda spec: Path(spec.filename)):
        adapters.apply_adapters(pipe, settings)
        # Re-applying to the same instance must not duplicate adapters.
        adapters.apply_adapters(pipe, settings)
    assert len(loaded) == 2
    for spec, kwargs in zip((adapters.TURBO, adapters.REALISM), loaded):
        assert kwargs["transformer_ref"] is model
        assert kwargs["prefix"] == "transformer"
        assert kwargs["metadata"] == {"preserved_metadata": spec.filename}
    model.set_adapters.assert_called_once_with(["h3_acceleration", "h3_style"], weights=[1.0, 0.7])
    assert all(f.dtype == "bfloat16" for f in layer.lora_A.values())
    assert all(f.dtype == "bfloat16" for f in layer.lora_B.values())
    assert protected.dtype == "float32" and base.weight.dtype == "bfloat16"
    model._reset_stateful_cache.assert_called_once()
