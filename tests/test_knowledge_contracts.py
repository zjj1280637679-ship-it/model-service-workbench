"""Counterexamples for source identity, safe composition and unknown measurements."""
from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


knowledge = load("check_knowledge")
publication = load("check_publication")


def save(root, relative, value):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


@pytest.fixture
def repository(tmp_path):
    source = tmp_path / "runtime.py"
    source.write_text("# Reference implementation\n", encoding="utf-8")
    entries = []
    for identifier in sorted(knowledge.MODULES):
        perspectives = {}
        for perspective in sorted(knowledge.PERSPECTIVES):
            perspectives[perspective] = {
                "requirements": ["Declared desired outcome, not an observation."],
                "conditions": ["Inspect runtime.py in this repository."],
                "strategies": ["A bounded alternative may be selected for this context."],
                "metrics": [{"id": f"{identifier}.{perspective}.test", "name": "Observed defects",
                             "definition": "Count defects in the selected scenario set.", "unit": "count",
                             "scope": "One declared test run, not every possible execution.",
                             "unknown_rule": "Missing observations remain unknown.",
                             "threshold_policy": "The scenario owner chooses a threshold before acceptance."}],
            }
        module = {"id": identifier, "name": identifier, "contract_version": "0.1.0",
                  "implementation_status": "specialized_reference", "perspectives": perspectives,
                  "sources": ["runtime.py"], "gaps": ["General implementations not validated."]}
        relative = f"knowledge/modules/{identifier}.json"
        save(tmp_path, relative, module)
        entries.append({"id": identifier, "path": relative})
    save(tmp_path, "knowledge/index.json", {"schema_version": 1, "taxonomy": {"scope": "declared types"}, "modules": entries})
    save(tmp_path, "evidence/source-manifest.json", {"schema_version": 1, "files": {"runtime.py": sha256(source.read_bytes()).hexdigest()}})
    return tmp_path


def change_module(root, mutate):
    relative = "knowledge/modules/model-container.json"
    value = json.loads((root / relative).read_text(encoding="utf-8"))
    mutate(value)
    save(root, relative, value)


def test_complete_record_passes_without_claiming_semantic_truth(repository):
    result = knowledge.check(repository)
    assert result["status"] == "PASS" and result["metrics"] == 16
    assert "not semantic completeness" in result["scope"]


@pytest.mark.parametrize("category", ["requirements", "conditions", "strategies", "metrics"])
def test_no_quadrant_can_be_silently_omitted(repository, category):
    change_module(repository, lambda value: value["perspectives"]["core"].pop(category))
    with pytest.raises(knowledge.KnowledgeError):
        knowledge.check(repository)


def test_no_perspective_can_be_silently_omitted(repository):
    change_module(repository, lambda value: value["perspectives"].pop("recursion"))
    with pytest.raises(knowledge.KnowledgeError, match="exactly core"):
        knowledge.check(repository)


@pytest.mark.parametrize("field", sorted(knowledge.METRIC_FIELDS))
def test_metric_needs_operational_definition_and_unknown_rule(repository, field):
    change_module(repository, lambda value: value["perspectives"]["core"]["metrics"][0].pop(field))
    with pytest.raises(knowledge.KnowledgeError):
        knowledge.check(repository)


def test_duplicate_metric_identity_is_rejected(repository):
    def duplicate(value):
        value["perspectives"]["recursion"]["metrics"][0]["id"] = value["perspectives"]["core"]["metrics"][0]["id"]
    change_module(repository, duplicate)
    with pytest.raises(knowledge.KnowledgeError, match="duplicate metric"):
        knowledge.check(repository)


def test_duplicate_module_is_not_counted_as_another_module(repository):
    index = json.loads((repository / "knowledge/index.json").read_text())
    index["modules"][-1] = deepcopy(index["modules"][0])
    save(repository, "knowledge/index.json", index)
    with pytest.raises(knowledge.KnowledgeError, match="duplicate module"):
        knowledge.check(repository)


@pytest.mark.parametrize("path", ["../outside.py", "/outside.py", "C:/Users/person/secret.py", "C:relative.py", "\\\\host\\share\\secret.py"])
def test_sources_cannot_escape_repository(repository, path):
    change_module(repository, lambda value: value.update(sources=[path]))
    with pytest.raises(knowledge.KnowledgeError, match="unsafe repository path"):
        knowledge.check(repository)


def test_missing_source_and_drift_are_distinct_failures(repository):
    (repository / "runtime.py").write_text("changed\n", encoding="utf-8")
    with pytest.raises(knowledge.KnowledgeError, match="fingerprint drift"):
        knowledge.check(repository)
    (repository / "runtime.py").unlink()
    with pytest.raises(knowledge.KnowledgeError, match="missing source"):
        knowledge.check(repository)


def test_unknown_measurement_is_not_coerced_to_zero():
    knowledge.validate_observation({"status": "unknown", "value": None, "reason": "No sample"}, "metric")
    with pytest.raises(knowledge.KnowledgeError, match="never zero"):
        knowledge.validate_observation({"status": "unknown", "value": 0, "reason": "No sample"}, "metric")


def test_real_measured_zero_is_allowed_but_zero_denominator_is_not():
    measured = {"status": "measured", "value": 0, "window": "One run", "evidence": "observation.json"}
    knowledge.validate_observation(measured, "metric")
    with pytest.raises(knowledge.KnowledgeError, match="denominator must be positive"):
        knowledge.validate_observation({**measured, "denominator": 0}, "metric")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), None, True])
def test_measured_values_require_finite_numeric_evidence(value):
    with pytest.raises(knowledge.KnowledgeError, match="finite number"):
        knowledge.validate_observation({"status": "measured", "value": value, "window": "run", "evidence": "evidence.json"}, "metric")


def test_publication_flags_actual_secret_without_echoing_it(tmp_path):
    secret = "fixture-secret-value-" + "x" * 24
    (tmp_path / "exposed.json").write_text(json.dumps({"key": secret}), encoding="utf-8")
    result = publication.check(tmp_path, [secret])
    assert {"path": "exposed.json", "rule": "known_secret"} in result["issues"]
    assert secret not in json.dumps(result)


@pytest.mark.parametrize("name,rule", [("api-connection.private.json", "private_runtime_or_credentials"),
                                       ("control.sqlite3", "private_runtime_or_credentials"),
                                       ("weights.safetensors", "model_weights")])
def test_publication_rejects_private_runtime_and_weights(tmp_path, name, rule):
    (tmp_path / name).write_bytes(b"fixture")
    assert {"path": name, "rule": rule} in publication.check(tmp_path)["issues"]


def test_test_fixture_home_path_is_allowed_but_document_leak_is_not(tmp_path):
    sample = "C:/Users/person/project/file.py"
    (tmp_path / "test_paths.py").write_text(repr(sample), encoding="utf-8")
    assert publication.check(tmp_path)["status"] == "PASS"
    (tmp_path / "README.md").write_text(sample, encoding="utf-8")
    assert {"path": "README.md", "rule": "personal_absolute_path"} in publication.check(tmp_path)["issues"]
