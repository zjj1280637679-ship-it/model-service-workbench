"""Check knowledge structure and source identity, not philosophical adequacy or GPU success.

Optional metric observations preserve an explicit measured/unknown distinction.
This checker does not read prose to infer facts, invent thresholds, or promote evidence.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import re

MODULES = {"model-container", "interaction-panel", "event-orchestrator", "api-factory"}
PERSPECTIVES = {"core", "interaction", "recursion", "expression"}
METRIC_FIELDS = {"id", "name", "definition", "unit", "scope", "unknown_rule", "threshold_policy"}


class KnowledgeError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise KnowledgeError(message)


def text(value, location):
    require(isinstance(value, str) and bool(value.strip()), f"{location}: nonempty string required")


def strings(value, location):
    require(isinstance(value, list) and bool(value), f"{location}: nonempty string list required")
    for index, item in enumerate(value):
        text(item, f"{location}[{index}]")


def local_file(root: Path, relative: str) -> Path:
    text(relative, "path")
    # Reject Windows drives/UNC on POSIX too; resolve() handles symlink escapes.
    posix = PurePosixPath(relative)
    windows = PureWindowsPath(relative)
    require(not windows.drive and not posix.is_absolute() and "\\" not in relative,
            f"unsafe repository path: {relative}")
    require(".." not in posix.parts and relative not in {".", ""}, f"unsafe repository path: {relative}")
    candidate = (root / relative).resolve()
    require(candidate.is_relative_to(root.resolve()), f"path leaves repository: {relative}")
    require(candidate.is_file(), f"missing source: {relative}")
    return candidate


def json_file(root, relative):
    try:
        return json.loads(local_file(root, relative).read_text(encoding="utf-8-sig"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise KnowledgeError(f"invalid JSON: {relative}: {error}") from error


def validate_observation(observation, location):
    """Validate recorded measurements without coercing missing evidence to zero.

    Units and definitions belong to the metric. Each observation records its own
    sampling window and evidence path. The checker verifies bookkeeping only.
    """
    require(isinstance(observation, dict), f"{location}: observation must be an object")
    status = observation.get("status")
    require(status in {"measured", "unknown"}, f"{location}: status must be measured or unknown")
    require("value" in observation, f"{location}: value must be explicit (null for unknown)")
    if status == "unknown":
        require(observation["value"] is None, f"{location}: unknown must retain null, never zero")
        text(observation.get("reason"), f"{location}.reason")
    else:
        value = observation["value"]
        require(type(value) in {int, float} and math.isfinite(value),
                f"{location}: measured value must be a finite number")
        text(observation.get("window"), f"{location}.window")
        text(observation.get("evidence"), f"{location}.evidence")
    if "denominator" in observation:
        denominator = observation["denominator"]
        if status == "measured":
            require(type(denominator) in {int, float} and math.isfinite(denominator) and denominator > 0,
                    f"{location}: measured ratio denominator must be positive")
        else:
            require(denominator is None or (type(denominator) in {int, float}
                    and math.isfinite(denominator) and denominator >= 0),
                    f"{location}: unknown denominator must be null or nonnegative")


def check(root: Path):
    root = root.resolve()
    index = json_file(root, "knowledge/index.json")
    require(isinstance(index, dict) and index.get("schema_version") == 1, "index schema_version must be 1")
    require(isinstance(index.get("taxonomy"), dict) and bool(index["taxonomy"]), "index taxonomy is required")
    entries = index.get("modules")
    require(isinstance(entries, list) and len(entries) == len(MODULES), "index must list exactly four modules")
    identifiers, module_paths, metric_ids = set(), set(), set()
    references = set()
    for entry in entries:
        require(isinstance(entry, dict), "module entry must be an object")
        identifier, relative = entry.get("id"), entry.get("path")
        require(isinstance(identifier, str) and identifier in MODULES and identifier not in identifiers,
                f"unknown or duplicate module: {identifier}")
        text(relative, f"{identifier}.path")
        require(relative not in module_paths, f"duplicate module path: {relative}")
        identifiers.add(identifier)
        module_paths.add(relative)
        module = json_file(root, relative)
        require(isinstance(module, dict) and module.get("id") == identifier, f"module identity mismatch: {relative}")
        text(module.get("name"), f"{identifier}.name")
        require(isinstance(module.get("contract_version"), str)
                and re.fullmatch(r"\d+\.\d+\.\d+", module["contract_version"]), f"{identifier}: invalid contract version")
        require(module.get("implementation_status") == "specialized_reference",
                f"{identifier}: current implementation must remain a specialized reference")
        perspectives = module.get("perspectives")
        require(isinstance(perspectives, dict) and set(perspectives) == PERSPECTIVES,
                f"{identifier}: exactly core/interaction/recursion/expression required")
        for perspective, description in perspectives.items():
            location = f"{identifier}.{perspective}"
            require(isinstance(description, dict), f"{location}: perspective must be an object")
            for category in ("requirements", "conditions", "strategies"):
                strings(description.get(category), f"{location}.{category}")
            metrics = description.get("metrics")
            require(isinstance(metrics, list) and bool(metrics), f"{location}: at least one optional metric definition required")
            for metric in metrics:
                require(isinstance(metric, dict), f"{location}: metric must be an object")
                for field in METRIC_FIELDS:
                    text(metric.get(field), f"{location}.metric.{field}")
                metric_id = metric["id"]
                require(metric_id not in metric_ids, f"duplicate metric id: {metric_id}")
                metric_ids.add(metric_id)
                observations = metric.get("observations", [])
                require(isinstance(observations, list), f"{metric_id}.observations must be a list")
                for number, observation in enumerate(observations):
                    validate_observation(observation, f"{metric_id}.observations[{number}]")
                    if observation["status"] == "measured":
                        local_file(root, observation["evidence"])
        strings(module.get("sources"), f"{identifier}.sources")
        for reference in module["sources"]:
            local_file(root, reference)
            references.add(reference)
        strings(module.get("gaps"), f"{identifier}.gaps")

    require(identifiers == MODULES, "required module set is incomplete")
    manifest = json_file(root, "evidence/source-manifest.json")
    require(isinstance(manifest, dict) and manifest.get("schema_version") == 1,
            "source manifest schema_version must be 1")
    files = manifest.get("files")
    require(isinstance(files, dict) and bool(files), "source manifest must record files")
    for relative, digest in files.items():
        require(isinstance(digest, str) and re.fullmatch(r"[a-f0-9]{64}", digest), f"invalid SHA-256: {relative}")
        actual = sha256(local_file(root, relative).read_bytes()).hexdigest()
        require(actual == digest, f"source fingerprint drift: {relative}")
    return {"status": "PASS", "modules": len(identifiers), "perspectives": len(identifiers) * len(PERSPECTIVES),
            "metrics": len(metric_ids), "references": len(references), "fingerprints": len(files),
            "scope": "Structure, safe source paths, explicit unknown measurements and recorded byte identities only; not semantic completeness, mathematical optimality, fresh GPU validation or a universal adapter implementation."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        result = check(args.root)
    except (KnowledgeError, OSError) as error:
        print(json.dumps({"status": "FAIL", "error": str(error)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
