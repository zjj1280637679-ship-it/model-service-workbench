"""CPU acceptance of API semantics and exclusive model-configuration ownership.

These fakes cannot establish GPU compatibility, quality, or performance gains.
"""
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from adapter_config import AdapterSettings, TURBO, REALISM
from generation_options import generation_capabilities, requested_options, validate_generation_options
from model_backend import MiniMaxBackend
from pipeline_backend import Prepared
from runtime_options import RuntimeOptions
from test_server import api, submit, HEADERS, INSTANCE, KEY, TestController
import server


def test_capability_configuration_is_independent_of_model_readiness():
    settings = AdapterSettings(acceleration=TURBO.key)
    cold = generation_capabilities(settings=settings, runtime_options=RuntimeOptions())
    turbo = cold["options"]["acceleration_loras"][1]
    assert turbo["configured"] and not turbo["ready"] and not cold["model_ready"]
    assert turbo["steps"] == {"min": 5, "max": 5, "required": 5, "denoiser_evaluations": 4}
    assert cold["defaults"] == {"attention_backend": "auto", "acceleration_lora": "none", "steps": 50}
    assert cold["quantization"] is False and cold["precision"] == "bf16"
    warm = generation_capabilities(model_ready=True, settings=settings, runtime_options=RuntimeOptions())
    assert all(item["ready"] for item in warm["options"]["attention_backends"])
    assert warm["options"]["acceleration_loras"][1]["ready"]


def test_capabilities_requires_auth_and_a_matching_runtime_instance(api):
    client, owner, _ = api
    assert client.get("/v1/capabilities").status_code == 401
    assert client.get("/v1/capabilities", headers=HEADERS).json()["model_ready"] is False
    owner.state.update(model_ready=True)
    assert client.get("/v1/capabilities", headers=HEADERS).json()["model_ready"] is True
    owner.state["instance_id"] = "b" * 32
    assert client.get("/v1/capabilities", headers=HEADERS).json()["model_ready"] is False


@pytest.mark.parametrize("parameters", [
    {"attention_backend": "sage"}, {"attention_backend": "_sage_attention"},
    {"attention_backend": "flash3"}, {"acceleration_lora": "https://arbitrary.invalid/lora"},
    {"acceleration_lora": TURBO.key, "steps": 5},
])
def test_unavailable_options_are_rejected_before_admission_or_disk_ownership(api, parameters):
    client, owner, root = api
    reply = submit(client, **parameters)
    assert reply.status_code == 422
    assert not owner.records and not any(operation == "submit" for operation, _ in owner.calls)
    assert not [p for p in root.iterdir() if p.is_dir()]


def turbo_api(tmp_path, monkeypatch):
    monkeypatch.setenv("H3_INSTANCE_ID", INSTANCE)
    monkeypatch.setenv("H3_ACCELERATION_LORA", TURBO.key)
    monkeypatch.delenv("H3_PARENT_PID", raising=False)
    owner = TestController(tmp_path)
    return TestClient(server.create_app(controller=owner, api_key=KEY, output_dir=tmp_path)), owner


def test_preloading_turbo_does_not_enable_it_for_unspecified_public_requests(tmp_path, monkeypatch):
    client, owner = turbo_api(tmp_path, monkeypatch)
    with client:
        reply = submit(client)
        assert reply.status_code == 202
        parameters = owner.records[reply.json()["id"]]["input"]
        assert parameters["acceleration_lora"] == "none" and parameters["steps"] == 50
        assert parameters["attention_backend"] == "auto"
        capabilities = client.get("/v1/capabilities", headers=HEADERS).json()
        turbo = capabilities["options"]["acceleration_loras"][1]
        assert turbo["configured"] and not turbo["ready"]
        for steps in (4, 8, 50):
            assert submit(client, acceleration_lora=TURBO.key, steps=steps).status_code == 422
        assert submit(client, acceleration_lora=TURBO.key, steps=5).status_code == 202


def test_idempotency_includes_both_acceleration_controls(tmp_path, monkeypatch):
    client, _ = turbo_api(tmp_path, monkeypatch)
    headers = {**HEADERS, "Idempotency-Key": "same-generation"}
    with client:
        first = submit(client, headers=headers, steps=5)
        assert first.status_code == 202
        assert submit(client, headers=headers, steps=5).json()["id"] == first.json()["id"]
        assert submit(client, headers=headers, steps=5, attention_backend="_native_flash").status_code == 409
        assert submit(client, headers=headers, steps=5, acceleration_lora=TURBO.key).status_code == 409


class Transformer:
    def __init__(self, settings):
        self._h3_adapter_settings = settings
        self.attention = "native"
        self.names, self.weights = ["h3_acceleration", "h3_style"], [1.0, settings.style_scale]
        self.enabled = True
        self.trainable = False
        self.fail_attention_once = False

    def set_attention_backend(self, value):
        if self.fail_attention_once:
            self.fail_attention_once = False
            raise RuntimeError("kernel unsupported")
        self.attention = value

    def set_adapters(self, names, weights):
        self.names, self.weights = list(names), list(weights)
        self.trainable = True

    def enable_lora(self): self.enabled = True; self.trainable = True
    def disable_lora(self): self.enabled = False
    def requires_grad_(self, value): self.trainable = value
    def eval(self): pass
    def _reset_stateful_cache(self): pass


def configured_backend(*, style=REALISM.key):
    backend = MiniMaxBackend(RuntimeOptions())
    settings = backend.adapter_settings = AdapterSettings(acceleration=TURBO.key, style=style, style_scale=0.6)
    original, turbo = object(), object()
    backend.pipe = SimpleNamespace(transformer_ref=Transformer(settings),
        blocks=SimpleNamespace(sub_blocks={"before_encode": turbo}),
        _h3_original_before_encode=original, _h3_turbo_before_encode=turbo,
        scheduler=SimpleNamespace(config=SimpleNamespace(shift=12)),
        audio_scheduler=SimpleNamespace(config=SimpleNamespace(shift=3)))
    return backend, original, turbo


def prepared(backend, acceleration="none", attention="auto", steps=5):
    return Prepared({"steps": steps, "acceleration_lora": acceleration, "attention_backend": attention}, [], [],
                    backend._new_metrics("generation"))


def test_each_job_switches_and_restores_attention_resize_and_lora_without_losing_style():
    backend, original, turbo = configured_backend()
    transformer = backend.pipe.transformer_ref
    first = prepared(backend, attention="_native_flash", steps=8)
    with backend.metrics_context(first.metrics), backend.generation_context(first):
        assert transformer.attention == "_native_flash"
        assert transformer.enabled and transformer.names == ["h3_style"] and transformer.weights == [0.6]
        assert transformer.trainable is False
        assert backend.pipe.blocks.sub_blocks["before_encode"] is original
        assert first.metrics.data["generation_options"]["effective"]["denoiser_evaluations"] == 7
        # Concurrent CPU validation must not change the active GPU configuration.
        backend.validate_request(steps=5, acceleration_lora=TURBO.key, attention_backend="native")
        assert transformer.attention == "_native_flash" and transformer.names == ["h3_style"]
    assert transformer.attention == "native" and transformer.names == ["h3_acceleration", "h3_style"]
    assert transformer.trainable is False
    assert backend.pipe.blocks.sub_blocks["before_encode"] is turbo
    second = prepared(backend, acceleration=TURBO.key)
    with backend.metrics_context(second.metrics), backend.generation_context(second):
        assert transformer.attention == "native" and transformer.names == ["h3_acceleration", "h3_style"]
        assert backend.pipe.blocks.sub_blocks["before_encode"] is turbo
        receipt = second.metrics.data["generation_options"]
        assert receipt["requested"]["attention_backend"] == "auto"
        assert receipt["effective"]["precision"] == "bf16" and receipt["effective"]["steps"] == 5
    assert backend._active_generation_options is None


def test_none_really_disables_preloaded_acceleration_without_style():
    backend, original, _ = configured_backend(style="none")
    item = prepared(backend)
    with backend.metrics_context(item.metrics), backend.generation_context(item):
        assert backend.pipe.transformer_ref.enabled is False
        assert backend.pipe.blocks.sub_blocks["before_encode"] is original
    assert backend.pipe.transformer_ref.enabled is True


def test_failed_job_and_failed_activation_restore_before_the_next_lease():
    backend, _, turbo = configured_backend()
    item = prepared(backend, attention="_native_flash", steps=8)
    with pytest.raises(RuntimeError, match="inference failed"):
        with backend.metrics_context(item.metrics), backend.generation_context(item):
            raise RuntimeError("inference failed")
    transformer = backend.pipe.transformer_ref
    assert transformer.attention == "native" and transformer.names == ["h3_acceleration", "h3_style"]
    assert backend.pipe.blocks.sub_blocks["before_encode"] is turbo
    assert backend._active_generation_options is None
    transformer.fail_attention_once = True
    failed = prepared(backend, attention="_native_flash")
    with pytest.raises(RuntimeError, match="kernel unsupported"):
        with backend.metrics_context(failed.metrics), backend.generation_context(failed):
            pytest.fail("Failed explicit backend must not silently continue")
    assert failed.metrics.data["generation_options"]["effective"] is None
    assert transformer.attention == "native" and backend._active_generation_options is None


def test_turbo_scheduler_identity_is_checked_at_activation_without_mutating_it():
    backend, _, _ = configured_backend()
    backend.pipe.scheduler.config.shift = 9
    item = prepared(backend, acceleration=TURBO.key)
    with pytest.raises(RuntimeError, match="shift=12"):
        with backend.metrics_context(item.metrics), backend.generation_context(item):
            pytest.fail("Wrong scheduler must be rejected")
    assert backend.pipe.scheduler.config.shift == 9
