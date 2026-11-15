"""Reproduce bounded searches and recheck historical evidence, without GPU/network."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from condition_sets import audit_evidence, search, verify

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    catalog_path = ROOT / "research/condition-models.json"
    catalog = read(catalog_path)
    models = {model["id"]: model for model in catalog["models"]}
    searches = [search(model) for model in models.values()]
    # Deliberately omit one real predecessor while keeping plausible substitutes.
    gpu = models["gpu_stage_eligibility"]
    weak_gpu = [p for p in gpu["candidate_literals"] if p["atom"] not in {"model_ready", "model_loaded"}]
    public = models["public_api_transport"]
    weak_public = [p for p in public["candidate_literals"] if p["atom"] != "probe_same_instance"]
    closure = models["closed_lifetime_review"]
    weak_release = [p for p in closure["candidate_literals"] if p["atom"] != "platform_release_observed"]
    counterexamples = {
        "downloaded_is_not_model_ready": verify(gpu, weak_gpu),
        "tunnel_exists_is_not_verified_public_api": verify(public, weak_public),
        "service_exit_is_not_platform_release": verify(closure, weak_release),
    }
    base = ROOT / "evidence/research-2026-11-15"
    trace, analysis = read(base / "trace.json"), read(base / "analysis.json")
    from analyze_trace import analyze
    if analyze(trace) != analysis:
        raise ValueError("Stored historical analysis is not reproducible")
    bindings = {key: trace[key] for key in ("manager_instance_id", "model_instance_id")}
    # Historical review window, not a TTL policy for a live service.
    cutoff = trace["platform_release"]["observed_at"]
    observations = []

    def observe(atom, filename, pointer, value_map, method="derived"):
        path = base / filename
        observations.append({"atom": atom, "bindings": bindings, "valid_from": cutoff,
            "valid_until": cutoff + 1, "validity_basis": "Retrospective immutable lifetime review window; no present readiness claim.",
            "method": method, "source": {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path)},
            "binding_selectors": {key: "/" + key for key in bindings}, "pointer": pointer, "value_map": value_map})

    pass_map = [{"equals": "pass", "state": "true"}, {"equals": "fail", "state": "false"}, {"equals": "unknown", "state": "unknown"}]
    for atom, check in {
        "terminal_receipts_complete": "terminal_receipt_coverage", "task_cleanup_recorded": "terminal_job_cleanup",
        "service_exit_recorded": "service_process_cleanup", "platform_release_observed": "platform_release",
        "gpu_stage_nonoverlap_observed": "gpu_stage_exclusivity", "ready_gate_observed": "model_ready_gate",
    }.items():
        observe(atom, "analysis.json", "/checks/" + check + "/status", pass_map)
    snapshot_index = next(i for i, s in enumerate(trace["system_snapshots"])
                          if isinstance(s.get("modules", {}).get("model", {}).get("budget", {}).get("io_budget"), dict))
    observe("pipeline_bandwidth_calibrated_in_selected_snapshot", "trace.json",
            f"/system_snapshots/{snapshot_index}/modules/model/budget/io_budget/bandwidth_calibrated",
            [{"equals": True, "state": "true"}, {"equals": False, "state": "false"}], "observed")
    missing = ["complete_gpu_lease_history", "whole_model_workload_known", "live_current_model_ready", "panel_semantics_retested"]
    atoms = [row["atom"] for row in observations] + missing
    audit = audit_evidence(atoms, observations, bindings, cutoff + .5, ROOT)
    closure_facts = [{"atom": atom, "value": True} for atom in closure["atoms"] if audit["states"][atom] == "true"]
    closure_result = verify(closure, closure_facts)
    return {"schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
        "new_gpu_run": False, "network_used": False, "catalog_sha256": sha(catalog_path),
        "implementation_sha256": sha(ROOT / "research/condition_sets.py"),
        "searches": searches, "counterexamples": counterexamples, "evidence_observations": observations,
        "historical_evidence_audit": audit, "historical_closure_review": closure_result,
        "coverage": {"declared_models": len(models), "completed_model_searches": sum(x["status"] == "complete" for x in searches),
                     "runtime_all_conditions_covered": None, "physical_world_exhausted": False,
                     "source_audit": catalog["source_audit"]},
        "limits": ["The declared models are not all real-world conditions.",
                   "Definitions/assumptions are versioned policy abstractions, not measured discoveries.",
                   "Source hashes cannot authenticate physical reality or a platform observer.",
                   "Unknown facts do not become false; conflict facts are not premises.",
                   "Historical evidence never authorizes a new instance or replaces atomic reservation."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8", newline="\n")
    ok = (all(x["status"] == "complete" for x in result["searches"])
          and all(x["status"] == "refuted" for x in result["counterexamples"].values())
          and result["historical_closure_review"]["status"] == "sufficient"
          and result["historical_evidence_audit"]["states"]["pipeline_bandwidth_calibrated_in_selected_snapshot"] == "false"
          and all(result["historical_evidence_audit"]["states"][key] == "unknown" for key in
                  ("complete_gpu_lease_history", "whole_model_workload_known", "live_current_model_ready", "panel_semantics_retested"))
          and not result["historical_evidence_audit"]["excluded"])
    print(json.dumps({"verification_passed": ok, "models": len(result["searches"]),
                      "counterexamples": len(result["counterexamples"]),
                      "fact_states": result["historical_evidence_audit"]["states"]}, ensure_ascii=False))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
