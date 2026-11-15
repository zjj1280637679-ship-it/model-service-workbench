from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("source_bindings", ROOT / "research/source_bindings.py")
bindings = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bindings)


@pytest.fixture
def fixture(tmp_path):
    directory = tmp_path / "frozen"
    directory.mkdir()
    hashes = {}
    for name in bindings.GATE_FILES:
        (directory / name).write_text("# historical " + name, encoding="utf-8")
        hashes[name] = hashlib.sha256((directory / name).read_bytes()).hexdigest()
    trace = {"manager_instance_id": "manager", "model_instance_id": "model",
             "runtime_source_identity": {"files": {name: {"sha256": digest} for name, digest in hashes.items()}}}
    binding = {"schema_version": 1, "experiment_id": "recorded-run", "source_commit": "a" * 40,
               "manager_instance_id": "manager", "model_instance_id": "model",
               "trace": "trace.json", "source_directory": "frozen", "files": hashes}
    (tmp_path / "trace.json").write_text(json.dumps(trace), encoding="utf-8")
    (tmp_path / "binding.json").write_text(json.dumps(binding), encoding="utf-8")
    return tmp_path, binding, trace


def test_new_current_sources_do_not_rewrite_historical_scope(fixture):
    root, _, _ = fixture
    current = root / "runtime"
    current.mkdir()
    for name in bindings.GATE_FILES:
        (current / name).write_text("# different valid current implementation", encoding="utf-8")
    assert bindings.bound_trace("binding.json", root=root)[1] == root / "frozen"


def test_corrupt_history_is_not_rescued_by_current_sources(fixture):
    root, _, _ = fixture
    (root / "frozen/condition_guard.py").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="bytes differ"):
        bindings.bound_trace("binding.json", root=root)


@pytest.mark.parametrize("case", ["missing", "wrong_hash", "wrong_instance", "incomplete_set", "escape", "missing_commit"])
def test_missing_and_conflicting_history_never_falls_back(fixture, case):
    root, binding, trace = fixture
    if case == "missing":
        (root / "frozen/condition_guard.py").unlink()
    elif case == "wrong_hash":
        trace["runtime_source_identity"]["files"]["condition_guard.py"]["sha256"] = "b" * 64
    elif case == "wrong_instance":
        trace["model_instance_id"] = "different-model"
    elif case == "incomplete_set":
        binding["files"].pop("model_container.py")
    elif case == "escape":
        binding["source_directory"] = "../outside"
    else:
        binding.pop("source_commit")
    (root / "binding.json").write_text(json.dumps(binding), encoding="utf-8")
    with pytest.raises(ValueError):
        bindings.bound_source_root("binding.json", root=root, trace=trace)


def test_published_binding_matches_actual_old_runtime_bytes():
    trace, directory = bindings.bound_trace("research/source-bindings/runtime-conditions-v040.json")
    assert directory.name == "minimax-h3-v0.4.0"
    assert trace["manager_instance_id"] == "f5d32604aad44ecbaed925602acd4886"
