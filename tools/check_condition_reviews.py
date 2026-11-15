"""Detect drift in an explicitly reviewed condition/strategy/metric dependency set.

Freshness is not debt closure or semantic correctness. This checker has no refresh
mode: rebuilding source-manifest.json cannot silently approve changed decisions.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path, PureWindowsPath
import re

PERSPECTIVES = {"core", "interaction", "recursion", "expression"}
CATEGORIES = {"requirements", "conditions", "strategies", "metrics"}
SHA = re.compile(r"[a-f0-9]{64}\Z")


class ReviewError(ValueError):
    pass


def require(value, message):
    if not value:
        raise ReviewError(message)


def source(root, relative):
    require(isinstance(relative, str) and relative, "source path required")
    path = Path(relative)
    require(not path.is_absolute() and not PureWindowsPath(relative).drive
            and "\\" not in relative and ".." not in path.parts, "unsafe source path")
    resolved = (root / path).resolve()
    require(resolved.is_relative_to(root.resolve()) and resolved.is_file(), "source missing or outside repository: " + relative)
    return resolved


def read(root, relative):
    return json.loads(source(root, relative).read_text(encoding="utf-8"))


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                             separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def check(root):
    root = Path(root)
    index = read(root, "knowledge/index.json")
    review = read(root, "knowledge/condition-reviews.json")
    ledger = read(root, "knowledge/strategy-metric-debt.json")
    require(review.get("schema_version") == 1 and ledger.get("schema_version") == 1, "unsupported schema")
    require(review.get("review_id") == ledger.get("review_id") and bool(review.get("review_id")), "review identity mismatch")
    require(isinstance(review.get("baseline_version"), str) and bool(review["baseline_version"].strip())
            and review["baseline_version"] == ledger.get("baseline_version"), "baseline version mismatch")
    require(isinstance(review.get("baseline_commit"), str) and re.fullmatch(r"[a-f0-9]{40}", review["baseline_commit"]), "baseline commit required")
    modules = {}
    metric_ids = set()
    dependencies = {"knowledge/index.json", "knowledge/strategy-metric-debt.json"}
    for entry in index["modules"]:
        require(entry["id"] not in modules, "duplicate module")
        module = read(root, entry["path"])
        dependencies.add(entry["path"])
        require(isinstance(module.get("sources"), list) and bool(module["sources"]), "module sources required")
        dependencies.update(module["sources"])
        require(module.get("id") == entry["id"], "module identity mismatch")
        modules[entry["id"]] = module
        for perspective in PERSPECTIVES:
            metric_ids.update(m["id"] for m in module["perspectives"][perspective]["metrics"])
    expected = {f"{module}/{perspective}" for module in modules for perspective in PERSPECTIVES}
    debts = {}
    for item in ledger.get("debts", []):
        identity = item.get("id")
        require(isinstance(identity, str) and re.fullmatch(r"TD-\d{3}", identity) and identity not in debts, "invalid or duplicate debt identity")
        require(item.get("status") in {"open", "resolved"}, identity + ": invalid closure state")
        require(item.get("priority") in {"P1", "P2", "P3"}, identity + ": invalid priority")
        require(isinstance(item.get("scope_refs"), list) and bool(item["scope_refs"])
                and set(item["scope_refs"]) <= expected, identity + ": invalid requirement scope")
        require(isinstance(item.get("metric_ids"), list) and set(item["metric_ids"]) <= metric_ids, identity + ": invalid metric identity")
        for key in ("title", "unchanged_requirement", "changed_condition", "lagging_decision", "acceptance", "disposition"):
            require(isinstance(item.get(key), str) and bool(item[key].strip()), identity + ": missing " + key)
        require(isinstance(item.get("evidence"), list) and bool(item["evidence"]), identity + ": missing evidence")
        for path in item["evidence"]:
            source(root, path)
            dependencies.add(path)
        if item["status"] == "resolved":
            require(isinstance(item.get("resolution_evidence"), list) and bool(item["resolution_evidence"]), identity + ": closure requires evidence")
            for path in item["resolution_evidence"]:
                source(root, path)
                dependencies.add(path)
        debts[identity] = item
    require(bool(debts), "explicit debt inventory required")
    seen, linked, drift = set(), set(), []
    for row in review.get("reviews", []):
        identity = row.get("scope")
        require(identity in expected and identity not in seen, "invalid or duplicate review scope")
        seen.add(identity)
        module, perspective = identity.split("/")
        body = modules[module]["perspectives"][perspective]
        require(set(row.get("reviewed_hashes", {})) == CATEGORIES, identity + ": incomplete decision review")
        for category in CATEGORIES:
            previous = row["reviewed_hashes"][category]
            require(isinstance(previous, str) and SHA.fullmatch(previous), identity + ": invalid digest")
            if digest(body[category]) != previous:
                drift.append({"scope": identity, "changed": category})
        baseline = row.get("requirements_baseline_sha256")
        require(isinstance(baseline, str) and SHA.fullmatch(baseline), identity + ": requirement baseline required")
        if digest(body["requirements"]) != baseline:
            drift.append({"scope": identity, "changed": "requirements_baseline"})
        require(row.get("disposition") in {"updated", "retained_with_reason", "no_change_needed"}, identity + ": review disposition required")
        require(isinstance(row.get("rationale"), str) and bool(row["rationale"].strip()), identity + ": rationale required")
        require(isinstance(row.get("debt_ids"), list) and set(row["debt_ids"]) <= debts.keys(), identity + ": invalid debt references")
        for debt_id in row["debt_ids"]:
            require(identity in debts[debt_id]["scope_refs"], identity + ": debt scope mismatch")
            linked.add((identity, debt_id))
    require(seen == expected, "every module and perspective needs an explicit review")
    expected_links = {(scope, identity) for identity, item in debts.items() for scope in item["scope_refs"]}
    require(linked == expected_links, "debt scope missing from decision reviews")
    inputs = review.get("reviewed_sources", {})
    require(isinstance(inputs, dict) and bool(inputs), "reviewed source set required")
    # A review cannot hash itself; its contents are structurally cross-checked.
    dependencies.discard("knowledge/condition-reviews.json")
    require(dependencies <= inputs.keys(), "declared dependency absent from reviewed sources: " + ", ".join(sorted(dependencies - inputs.keys())))
    for relative, previous in inputs.items():
        require(isinstance(previous, str) and SHA.fullmatch(previous), "invalid reviewed source digest")
        if sha256(source(root, relative).read_bytes()).hexdigest() != previous:
            drift.append({"source": relative, "changed": "reviewed_source"})
    counts = Counter(item["status"] for item in debts.values())
    return {"review_status": "review_required" if drift else "current", "drift": drift,
            "perspectives": len(seen), "reviewed_sources": len(inputs),
            "debt_counts": {key: counts[key] for key in ("open", "resolved")},
            "open_debts": sorted(key for key, item in debts.items() if item["status"] == "open"),
            "scope": "Freshness of this declared review only; current does not mean all debt resolved, complete condition discovery, runtime acceptance or semantic proof. No automatic refresh is provided."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        result = check(args.root)
    except (ReviewError, OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({"review_status": "invalid", "error": str(error)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["review_status"] == "current" else 1


if __name__ == "__main__":
    raise SystemExit(main())
