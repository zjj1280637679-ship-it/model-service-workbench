import subprocess

import pytest

import lora_preflight as preflight


def environment(monkeypatch, torchao="0.10.0"):
    versions = {"torch": "2.14.1+cu130", "torchvision": "0.29.1+cu130",
                "torchaudio": "2.14.1+cu130", "psutil": "7.0.0", "torchao": torchao}
    commands = []
    monkeypatch.setattr(preflight, "distribution_version", versions.get)
    monkeypatch.setattr(preflight.importlib.util, "find_spec", lambda name: object() if versions.get(name) else None)

    def uninstall(command, **kwargs):
        commands.append((command, kwargs))
        versions["torchao"] = None

    monkeypatch.setattr(preflight.subprocess, "run", uninstall)
    return versions, commands


def test_old_optional_package_requires_explicit_repair_before_any_mutation(monkeypatch):
    _, commands = environment(monkeypatch)
    with pytest.raises(RuntimeError, match="blocks PEFT"):
        preflight.prepare_optional_torchao()
    assert commands == []


def test_repair_removes_only_old_torchao_and_records_unchanged_stack(monkeypatch):
    _, commands = environment(monkeypatch)
    report = preflight.prepare_optional_torchao(repair=True)
    assert commands == [([preflight.sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
                         {"check": True, "capture_output": True, "text": True})]
    assert report["optional_torchao"] == {"before": "0.10.0", "after": None, "removed_version": "0.10.0"}
    assert report["stack_before"] == report["stack_after"]
    assert report["stack_after"]["torch"] == "2.14.1+cu130"


@pytest.mark.parametrize("version", [None, "0.16.0", "0.18.0"])
def test_absent_or_compatible_torchao_is_not_changed(monkeypatch, version):
    _, commands = environment(monkeypatch, version)
    report = preflight.prepare_optional_torchao(repair=True)
    assert commands == []
    assert report["optional_torchao"]["removed_version"] is None


def test_uninstall_failure_stops_before_smoke(monkeypatch):
    environment(monkeypatch)
    def failed(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0])
    monkeypatch.setattr(preflight.subprocess, "run", failed)
    monkeypatch.setattr(preflight, "bf16_lora_smoke", lambda: pytest.fail("must not continue"))
    with pytest.raises(subprocess.CalledProcessError):
        preflight.run(repair=True)


def test_stack_drift_and_leftover_import_block_loading(monkeypatch):
    versions, _ = environment(monkeypatch)
    def changed(*args, **kwargs):
        versions["torch"] = "different"
        versions["torchao"] = None
    monkeypatch.setattr(preflight.subprocess, "run", changed)
    with pytest.raises(RuntimeError, match="protected"):
        preflight.prepare_optional_torchao(repair=True)
    environment(monkeypatch)
    monkeypatch.setattr(preflight.importlib.util, "find_spec", lambda name: object())
    with pytest.raises(RuntimeError, match="remains importable"):
        preflight.prepare_optional_torchao(repair=True)


def test_run_records_actual_smoke_result_after_preparation(monkeypatch):
    environment(monkeypatch)
    observed = []
    def smoke():
        assert preflight.distribution_version("torchao") is None
        observed.append(True)
        return {"injected": True, "precision": "bf16", "gpu_generation_verified": False}
    monkeypatch.setattr(preflight, "bf16_lora_smoke", smoke)
    report = preflight.run(repair=True)
    assert observed == [True]
    assert report["status"] == "passed"
    assert report["smoke"]["gpu_generation_verified"] is False
