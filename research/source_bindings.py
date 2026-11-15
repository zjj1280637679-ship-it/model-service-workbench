"""Resolve explicitly frozen historical sources without executing or importing them."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PureWindowsPath
import re

ROOT = Path(__file__).resolve().parents[1]
GATE_FILES = {"condition_guard.py", "global_manager.py", "model_container.py", "pipeline_scheduler.py"}


def _owned(root, relative, *, directory=False):
    if (not isinstance(relative, str) or not relative or PureWindowsPath(relative).drive
            or "\\" in relative or Path(relative).is_absolute() or ".." in Path(relative).parts):
        raise ValueError("Binding path must be repository-relative")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not (path.is_dir() if directory else path.is_file()):
        raise ValueError("Binding source missing or outside repository")
    return path


def read_binding(binding_path, *, root=ROOT):
    root = Path(root).resolve()
    path = Path(binding_path)
    if not path.is_absolute():
        path = root / path
    if not path.resolve().is_relative_to(root):
        raise ValueError("Binding descriptor must belong to repository")
    value = json.loads(path.read_text(encoding="utf-8"))
    if (type(value.get("schema_version")) is not int or value["schema_version"] != 1
            or not re.fullmatch(r"[a-f0-9]{40}", value.get("source_commit", ""))
            or not isinstance(value.get("experiment_id"), str) or not value["experiment_id"]):
        raise ValueError("Historical source binding identity is incomplete")
    if not isinstance(value.get("files"), dict) or set(value["files"]) != GATE_FILES:
        raise ValueError("Historical gate source set is incomplete")
    for digest in value["files"].values():
        if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ValueError("Historical source digest is invalid")
    return value


def bound_source_root(binding_path, *, root=ROOT, trace=None):
    root = Path(root).resolve()
    binding = read_binding(binding_path, root=root)
    if trace is None:
        trace = json.loads(_owned(root, binding["trace"]).read_text(encoding="utf-8"))
    if (trace.get("manager_instance_id") != binding.get("manager_instance_id")
            or trace.get("model_instance_id") != binding.get("model_instance_id")
            or not binding.get("manager_instance_id") or not binding.get("model_instance_id")):
        raise ValueError("Historical trace instance does not match source binding")
    directory = _owned(root, binding["source_directory"], directory=True)
    recorded = trace.get("runtime_source_identity", {}).get("files", {})
    for name, expected in binding["files"].items():
        path = directory / name
        if not path.resolve().is_relative_to(directory) or not path.is_file():
            raise ValueError("Frozen source is missing: " + name)
        if recorded.get(name, {}).get("sha256") != expected:
            raise ValueError("Trace and binding source digests conflict: " + name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Frozen historical source bytes differ: " + name)
    return directory


def bound_trace(binding_path, *, root=ROOT):
    root = Path(root).resolve()
    binding = read_binding(binding_path, root=root)
    trace = json.loads(_owned(root, binding["trace"]).read_text(encoding="utf-8"))
    return trace, bound_source_root(binding_path, root=root, trace=trace)
