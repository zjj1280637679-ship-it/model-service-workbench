"""Metadata-only precision audit regressions; no model downloads or CUDA."""
from contextlib import nullcontext
import inspect
import json
import sys
from types import SimpleNamespace

import pytest

import model_backend as backend_module
from model_backend import MiniMaxBackend, _audit_parameter_precision


class Parameter:
    is_quantized = False
    is_meta = False

    def __init__(self, dtype, count=8):
        self.dtype, self.count = dtype, count

    def numel(self):
        return self.count

    def to(self, *args, **kwargs):
        raise AssertionError("Precision audit must not copy or cast weights")


class Module:
    def __init__(self, parameters, keep_fp32=None):
        self.parameters = parameters
        self._keep_in_fp32_modules = keep_fp32

    def named_parameters(self):
        yield from self.parameters.items()

    def named_buffers(self):
        raise AssertionError("Integer/FP32 buffers are outside parameter audit")


@pytest.fixture
def torch(monkeypatch):
    value = SimpleNamespace(bfloat16="torch.bfloat16", float32="torch.float32",
                            nn=SimpleNamespace(Parameter=Parameter))
    monkeypatch.setitem(sys.modules, "torch", value)
    return value


def pipeline(**components):
    return SimpleNamespace(blocks=SimpleNamespace(expected_components=[
        SimpleNamespace(name=name) for name in components]), **components)


def test_precision_audit_uses_each_official_component_declaration(torch, capsys):
    pipe = pipeline(
        transformer_ref=Module({"blocks.0.weight": Parameter(torch.bfloat16, 20),
                                "audio_proj_out.weight": Parameter(torch.float32, 2)}, ["proj_out", "audio_proj_out"]),
        text_encoder=Module({"model.weight": Parameter(torch.bfloat16, 10)}),
        vae=Module({"encoder.conv.weight": Parameter(torch.float32, 6),
                    "decoder.blocks.0.weight": Parameter(torch.bfloat16, 7)}, ["encoder"]),
        audio_vae=Module({"encoder.weight": Parameter(torch.float32, 3),
                          "decoder.weight": Parameter(torch.float32, 4)}, ["encoder", "decoder"]),
        tokenizer=object(),
    )
    report = _audit_parameter_precision(pipe)
    assert report["components"]["transformer_ref"]["parameter_counts"] == {
        "torch.bfloat16": 20, "torch.float32": 2}
    assert report["components"]["audio_vae"]["parameter_counts"] == {"torch.float32": 7}
    events = [json.loads(line.removeprefix("H3_PRECISION_AUDIT "))
              for line in capsys.readouterr().out.splitlines()]
    assert len(events) == 5 and events[-1]["status"] == "passed"
    assert set(events[-1]["components"]) == {"transformer_ref", "text_encoder", "vae", "audio_vae"}


def test_vae_actual_quant_conv_and_post_quant_conv_have_distinct_precision(torch):
    # Diffusers 0.43 MiniMax VAE: the loader keeps quant_conv, but the observed
    # post_quant_conv tensor is BF16. A substring check incorrectly rejects it.
    module = Module({
        "quant_conv.weight": Parameter(torch.float32, 3),
        "quant_conv.bias": Parameter(torch.float32, 1),
        "post_quant_conv.weight": Parameter(torch.bfloat16, 5),
        "encoder.down_blocks.0.conv.weight": Parameter(torch.float32, 7),
        "decoder.transformer_blocks.0.attn.to_q.weight": Parameter(torch.bfloat16, 11),
        "decoder.transformer_blocks.0.norm1.weight": Parameter(torch.float32, 2),
    }, ["encoder", "quant_conv", "norm1", "norm2", "norm_out", "scale1", "scale2"])
    result = _audit_parameter_precision(pipeline(vae=module))
    assert result["components"]["vae"]["parameter_counts"] == {"torch.float32": 13, "torch.bfloat16": 16}
    # Fixing the false rejection must not grant FP32 to the wrong component.
    module.parameters["post_quant_conv.weight"].dtype = torch.float32
    with pytest.raises(RuntimeError, match=r"vae.post_quant_conv.weight.*expected torch.bfloat16"):
        _audit_parameter_precision(pipeline(vae=module))


@pytest.mark.parametrize("name,keep_fp32,expected_fp32", [
    ("encoder.conv.weight", ["encoder"], True),
    ("nested.encoder.conv.weight", ["encoder"], True),
    ("encoder_extra.conv.weight", ["encoder"], False),
    ("nested.pre_encoder.conv.weight", ["encoder"], False),
    ("nested.quant_conv.weight", ["quant_conv"], True),
    ("nested.post_quant_conv.weight", ["quant_conv"], False),
    ("audio_proj_out.weight", ["proj_out"], False),
    ("audio_proj_out.weight", ["audio_proj_out"], True),
    # The loader checks an entry against individual path segments: an entire
    # dotted path or a regex is not an alternative matching syntax.
    ("nested.encoder.conv.weight", ["nested.encoder"], False),
    ("encoder.conv.weight", ["encoder.*"], False),
])
def test_precision_audit_matches_actual_loader_path_boundaries(torch, name, keep_fp32, expected_fp32):
    dtype = torch.float32 if expected_fp32 else torch.bfloat16
    _audit_parameter_precision(pipeline(vae=Module({name: Parameter(dtype)}, keep_fp32)))


def test_staged_precision_audit_uses_same_loader_boundaries(torch):
    from staged_backend import StagedMiniMaxBackend
    backend = StagedMiniMaxBackend()
    backend.pipe = pipeline(transformer_ref=Module({
        "proj_out.weight": Parameter(torch.float32),
        "audio_proj_out.weight": Parameter(torch.bfloat16),
    }, ["proj_out"]))
    backend._load_required(["transformer_ref"])


@pytest.mark.parametrize("name,dtype,exceptions", [
    ("blocks.0.weight", "torch.float32", []),
    ("proj_out.weight", "torch.bfloat16", ["proj_out"]),
    ("blocks.0.weight", "torch.float16", []),
    ("blocks.0.weight", "torch.float8_e4m3fn", []),
    ("blocks.0.weight", "torch.int8", []),
])
def test_precision_audit_rejects_incorrect_backbone_and_fp32_exception(torch, name, dtype, exceptions):
    with pytest.raises(RuntimeError, match="Unexpected precision"):
        _audit_parameter_precision(pipeline(transformer_ref=Module({name: Parameter(dtype)}, exceptions)))


@pytest.mark.parametrize("flag", ["is_quantized", "is_loaded_in_4bit", "is_loaded_in_8bit", "hf_quantizer"])
def test_precision_audit_rejects_quantization_even_if_dtype_reports_bf16(torch, flag):
    module = Module({"weight": Parameter(torch.bfloat16)})
    setattr(module, flag, True)
    with pytest.raises(RuntimeError, match="Quantized component"):
        _audit_parameter_precision(pipeline(transformer_ref=module))


def test_precision_audit_rejects_quantized_parameter_subclass(torch):
    class QuantizedParameter(Parameter):
        pass
    module = Module({"weight": QuantizedParameter(torch.bfloat16)})
    with pytest.raises(RuntimeError, match="Non-native or quantized parameter"):
        _audit_parameter_precision(pipeline(transformer_ref=module))


def test_precision_audit_rejects_unloaded_or_empty_weights(torch):
    parameter = Parameter(torch.bfloat16)
    parameter.is_meta = True
    with pytest.raises(RuntimeError, match="not materialized"):
        _audit_parameter_precision(pipeline(transformer_ref=Module({"weight": parameter})))
    with pytest.raises(RuntimeError, match="No loaded parameters"):
        _audit_parameter_precision(pipeline(transformer_ref=Module({})))


@pytest.mark.parametrize("strategy", ["component_cpu_offload", "group_cpu_offload"])
@pytest.mark.parametrize("valid", [True, False])
def test_load_audits_after_adapters_before_offload_and_readiness(monkeypatch, torch, strategy, valid):
    import preflight
    calls = []
    module = Module({"weight": Parameter(torch.bfloat16)})
    module.requires_grad_ = lambda value: None
    module.enable_group_offload = lambda **kw: calls.append("offload")
    pipe = pipeline(transformer_ref=module)
    pipe.load_components = lambda **kw: calls.append("load")
    pipe.text_encoder = object()
    pipe.vae = pipe.audio_vae = SimpleNamespace(to=lambda device: None)
    torch.cuda = SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: True)
    torch.backends = SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace()))
    monkeypatch.setattr(preflight, "check_resources", lambda: {"strategy": strategy})
    monkeypatch.setitem(sys.modules, "diffusers", SimpleNamespace(
        ComponentsManager=lambda: SimpleNamespace(enable_auto_cpu_offload=lambda **kw: calls.append("offload")),
        ModularPipeline=SimpleNamespace(from_pretrained=lambda *a, **kw: pipe)))
    monkeypatch.setattr(backend_module, "configure_pipeline", lambda *args: None)
    monkeypatch.setattr(backend_module, "configure_qwen_offload", lambda *args: None)
    def adapters(*args):
        calls.append("adapters")
        if not valid:
            module.parameters["weight"].dtype = "torch.int8"
    monkeypatch.setattr(backend_module, "apply_adapters", adapters)
    backend = MiniMaxBackend()
    backend.runtime_options = SimpleNamespace(denoiser_offload="stream", configure_attention=lambda m: None,
                                             group_offload_kwargs=lambda t: {})
    backend._stage = lambda name: nullcontext()
    load = inspect.unwrap(MiniMaxBackend.load)
    if valid:
        load(backend)
        assert backend.pipe is pipe and backend.precision_audit["status"] == "passed"
        assert calls == ["load", "adapters", "offload"]
    else:
        with pytest.raises(RuntimeError, match="Unexpected precision"):
            load(backend)
        assert backend.pipe is None and "offload" not in calls
