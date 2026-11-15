"""A new fingerprint must not silently approve an old strategy/metric review."""
from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("condition_review_checker", ROOT / "tools/check_condition_reviews.py")
reviewer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reviewer)


def save(root, relative, value):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def update(root, relative, change):
    value = json.loads((root / relative).read_text(encoding="utf-8"))
    change(value)
    save(root, relative, value)


@pytest.fixture
def repository(tmp_path):
    (tmp_path / "runtime.py").write_text("# checked execution conditions\n", encoding="utf-8")
    modules, reviews = [], []
    for module in ("model-container", "event-orchestrator", "api-factory", "interaction-panel"):
        body = {"id": module, "sources": ["runtime.py"], "perspectives": {}}
        for perspective in sorted(reviewer.PERSPECTIVES):
            quadrant = {"requirements": ["Preserve the user's goal."], "conditions": ["Current owner and capacity evidence."],
                        "strategies": ["Chosen bounded strategy."], "metrics": [{"id": module + "." + perspective, "scope": "complete declared pool"}]}
            body["perspectives"][perspective] = quadrant
            identity = module + "/" + perspective
            reviews.append({"scope": identity, "requirements_baseline_sha256": reviewer.digest(quadrant["requirements"]),
                            "reviewed_hashes": {k: reviewer.digest(v) for k, v in quadrant.items()},
                            "disposition": "retained_with_reason", "rationale": "Review is current; the open debt still requires work.",
                            "debt_ids": ["TD-001"] if identity == "model-container/core" else []})
        relative = "knowledge/modules/" + module + ".json"
        save(tmp_path, relative, body)
        modules.append({"id": module, "path": relative})
    save(tmp_path, "knowledge/index.json", {"modules": modules})
    save(tmp_path, "knowledge/condition-reviews.json", {"schema_version": 1, "review_id": "fixture", "baseline_version": "v0.4.0", "baseline_commit": "a" * 40,
         "reviews": reviews, "reviewed_sources": {"runtime.py": sha256((tmp_path / "runtime.py").read_bytes()).hexdigest()}})
    save(tmp_path, "knowledge/strategy-metric-debt.json", {"schema_version": 1, "review_id": "fixture", "baseline_version": "v0.4.0", "debts": [
        {"id": "TD-001", "status": "open", "priority": "P1", "scope_refs": ["model-container/core"],
         "metric_ids": ["model-container.core"], "title": "Missing measurement", "unchanged_requirement": "Control stays usable.",
         "changed_condition": "Larger audit payload.", "lagging_decision": "Old polling policy.",
         "acceptance": "Measure with complete error counts.", "disposition": "Retained as open.", "evidence": ["runtime.py"]}]})
    dependencies = ["runtime.py", "knowledge/index.json", "knowledge/strategy-metric-debt.json", *(e["path"] for e in modules)]
    update(tmp_path, "knowledge/condition-reviews.json", lambda d: d.update(reviewed_sources={p: sha256((tmp_path / p).read_bytes()).hexdigest() for p in dependencies}))
    return tmp_path


def test_fresh_review_never_claims_open_debt_is_closed(repository):
    result = reviewer.check(repository)
    assert result["review_status"] == "current"
    assert result["debt_counts"] == {"open": 1, "resolved": 0}
    assert result["open_debts"] == ["TD-001"]
    assert "does not mean all debt resolved" in result["scope"]


@pytest.mark.parametrize("category", sorted(reviewer.CATEGORIES))
def test_changed_decision_requires_new_review(repository, category):
    update(repository, "knowledge/modules/model-container.json",
           lambda d: d["perspectives"]["core"][category].append({"id": "new.metric"} if category == "metrics" else "changed"))
    result = reviewer.check(repository)
    assert result["review_status"] == "review_required"
    assert {"scope": "model-container/core", "changed": category} in result["drift"]


def test_rehashing_requirements_review_cannot_silently_change_frozen_goal(repository):
    update(repository, "knowledge/modules/model-container.json", lambda d: d["perspectives"]["core"].update(requirements=["Different goal"]))
    update(repository, "knowledge/condition-reviews.json", lambda d: next(r for r in d["reviews"] if r["scope"] == "model-container/core")["reviewed_hashes"].update(requirements=reviewer.digest(["Different goal"])))
    assert {"scope": "model-container/core", "changed": "requirements_baseline"} in reviewer.check(repository)["drift"]


def test_source_manifest_refresh_does_not_refresh_strategy_review(repository):
    (repository / "runtime.py").write_text("# new condition ownership\n", encoding="utf-8")
    save(repository, "evidence/source-manifest.json", {"files": {"runtime.py": sha256((repository / "runtime.py").read_bytes()).hexdigest()}})
    assert reviewer.check(repository)["review_status"] == "review_required"


def test_new_measurement_scope_invalidates_old_review(repository):
    update(repository, "knowledge/modules/model-container.json", lambda d: d["perspectives"]["core"]["metrics"][0].update(scope="only CPU; not whole pool"))
    assert reviewer.check(repository)["review_status"] == "review_required"


@pytest.mark.parametrize("change", [
    lambda d: d["reviews"].pop(),
    lambda d: d["reviews"].append(deepcopy(d["reviews"][0])),
    lambda d: d["reviews"][0]["reviewed_hashes"].pop("strategies"),
    lambda d: d.update(reviewed_sources={}),
    lambda d: d.update(review_id="different"),
])
def test_incomplete_or_conflicting_review_is_not_current(repository, change):
    update(repository, "knowledge/condition-reviews.json", change)
    with pytest.raises(reviewer.ReviewError):
        reviewer.check(repository)


def test_no_closure_from_status_word_without_evidence(repository):
    update(repository, "knowledge/strategy-metric-debt.json", lambda d: d["debts"][0].update(status="resolved"))
    with pytest.raises(reviewer.ReviewError, match="closure requires evidence"):
        reviewer.check(repository)


@pytest.mark.parametrize("target", ["../outside.py", "C:/elsewhere.py", "missing.py"])
def test_review_sources_must_exist_inside_repository(repository, target):
    update(repository, "knowledge/condition-reviews.json", lambda d: d.update(reviewed_sources={target: "a" * 64}))
    with pytest.raises(reviewer.ReviewError):
        reviewer.check(repository)


def test_no_orphan_debt_or_unknown_metric_reference(repository):
    update(repository, "knowledge/strategy-metric-debt.json", lambda d: d["debts"][0].update(metric_ids=["invented.metric"]))
    with pytest.raises(reviewer.ReviewError, match="metric identity"):
        reviewer.check(repository)


def test_referenced_source_cannot_disappear_from_review_inventory(repository):
    update(repository, "knowledge/condition-reviews.json", lambda d: d["reviewed_sources"].pop("runtime.py"))
    with pytest.raises(reviewer.ReviewError, match="declared dependency absent"):
        reviewer.check(repository)


@pytest.mark.parametrize("version", [None, "v9.9.9"])
def test_missing_or_conflicting_baseline_version_is_rejected(repository, version):
    update(repository, "knowledge/condition-reviews.json", lambda d: d.update(baseline_version=version))
    with pytest.raises(reviewer.ReviewError, match="baseline version mismatch"):
        reviewer.check(repository)


def test_one_remaining_link_cannot_cover_another_missing_scope_review(repository):
    update(repository, "knowledge/strategy-metric-debt.json", lambda d: d["debts"][0]["scope_refs"].append("model-container/recursion"))
    with pytest.raises(reviewer.ReviewError, match="debt scope missing"):
        reviewer.check(repository)


def test_v040_whole_resource_metric_does_not_inherit_partial_pool_zero():
    module = json.loads((ROOT / "knowledge/modules/event-orchestrator.json").read_text(encoding="utf-8"))
    metrics = {m["id"]: m for m in module["perspectives"]["recursion"]["metrics"]}
    experiment = "3732565998754d26919f0706b838ebff"
    aggregate = next(o for o in metrics["EO-REC-OVERCOMMIT"]["observations"] if o["experiment_id"] == experiment)
    assert aggregate["status"] == "unknown" and aggregate["value"] is None
    partial = next(o for o in metrics["EO-REC-CPU-GPU-OVERCOMMIT"]["observations"] if o["experiment_id"] == experiment)
    assert partial["status"] == "measured" and partial["value"] == 0
    assert partial["resource_scope"] == ["declared_cpu_threads", "exclusive_gpu_lease"]
