"""Execute the assembled install control flow with no pip, network or CUDA."""
import copy
import hashlib
import json
from pathlib import Path
import runpy
import shutil
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture(scope="module")
def assembled():
    builder = Path(__file__).resolve().parents[2] / "tools" / "build_notebook.py"
    before = sys.argv
    try:
        sys.argv = [str(builder), "--check-only", "--preload-acceleration", "lightx2v-ref2va-4step"]
        return runpy.run_path(str(builder))
    finally:
        sys.argv = before


def execute_install(assembled, tmp_path, monkeypatch, *, after_psutil="5.9.5"):
    monkeypatch.setenv("H3_ACCELERATION_LORA", "lightx2v-ref2va-4step")
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/ffprobe")
    before = {"pytorch_stack": {"torch": "2.14.1+cu130", "torchvision": "0.25.1+cu130"},
              "host_runtime": {"psutil": "5.9.5"}}
    after = copy.deepcopy(before)
    after["host_runtime"]["psutil"] = after_psutil
    reports = iter([before, {"status": "passed", "stack_unchanged": True}, after,
                    {"python": "fixture", "torch": "fixture", "diffusers": "fixture"}])
    commands = []
    def logged(command, *args, **kwargs):
        constraints = Path(command[command.index("-c") + 1])
        commands.append((command, constraints.read_text(encoding="utf-8")))
    scope = {"PREFLIGHT_OK": True, "PROJECT_DIR": tmp_path, "sys": sys,
             "json_probe": lambda *a, **kw: next(reports), "run_logged": logged,
             "StatusCard": lambda *a, **kw: SimpleNamespace(update=lambda *a, **kw: None)}
    return scope, commands


def test_both_base_and_lora_pip_preserve_psutil_separately_from_torch(assembled, tmp_path, monkeypatch):
    scope, commands = execute_install(assembled, tmp_path, monkeypatch)
    exec(compile(assembled["install"], "assembled-install", "exec"), scope)
    assert scope["INSTALL_OK"] is True and len(commands) == 2
    for command, constraints in commands:
        assert "psutil==5.9.5\n" in constraints and "torch==2.14.1+cu130\n" in constraints
        assert command[command.index("-c") + 1].endswith("colab-runtime-constraints.txt")
    assert "psutil" not in scope["stack_before"]["pytorch_stack"]
    assert scope["stack_before"]["host_runtime"] == {"psutil": "5.9.5"}
    assert "psutil>=5.9.5,<8" in (tmp_path / "requirements-colab.txt").read_text(encoding="utf-8")
    assert "generation_options.py" in assembled["FILES"]
    assert "lora_preflight.py" in assembled["FILES"]
    assert scope["DEPENDENCY_REPORT"]["lora_preflight"]["status"] == "passed"
    assert assembled["args"].preload_acceleration == "lightx2v-ref2va-4step"


def test_changed_host_psutil_cannot_produce_install_success(assembled, tmp_path, monkeypatch):
    scope, commands = execute_install(assembled, tmp_path, monkeypatch, after_psutil="7.2.2")
    with pytest.raises(RuntimeError, match="psutil 发生变化"):
        exec(compile(assembled["install"], "assembled-install", "exec"), scope)
    assert scope["INSTALL_OK"] is False and len(commands) == 2


def test_preload_configuration_participates_in_console_reuse_identity(assembled):
    def digest(options):
        return hashlib.sha256(json.dumps({"files": assembled["FILES"], "programs": assembled["programs"],
            "bootstrap_options": options}, ensure_ascii=False, sort_keys=True,
            separators=(",", ":")).encode("utf-8")).hexdigest()
    assert assembled["bootstrap_options"] == {"public": True, "acceleration": "lightx2v-ref2va-4step"}
    assert assembled["console_bundle_hash"] == digest(assembled["bootstrap_options"])
    assert assembled["console_bundle_hash"] != digest({"public": True, "acceleration": "none"})
