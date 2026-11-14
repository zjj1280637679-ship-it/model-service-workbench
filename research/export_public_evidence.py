"""Create an explicit public projection of this experiment's private evidence.

Run only after reviewing the destination and experimental content. Credentials,
raw snapshots, source paths, process logs and the platform screenshot stay local.
Exact byte identities describe observations; they are not signed attestations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fields(value, names):
    return {key: value[key] for key in names.split() if key in value}


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    source, target = args.source.resolve(), args.destination.resolve()
    if target == source or source.is_relative_to(target):
        raise SystemExit("Destination must not contain the private source")
    target.mkdir(parents=True, exist_ok=True)
    trace = read(source / "trace-complete.private.json")
    public = fields(trace, "schema_version manager_instance_id model_instance_id collection queued_cancel_job_ids")
    event_keys = "event_id sequence instance_id job_id attempt_id type monotonic_ns observed_at stage succeeded accepting phase worker_cleanup_complete resources generation_options"
    for stream in ("manager", "model"):
        public[stream + "_events"] = []
        for row in trace[stream + "_events"]:
            item = fields(row, event_keys)
            if "data" in row:
                item["data"] = fields(row["data"], "module pid model_instance_id api_ready model_ready public_api_ready ready phase cleanup_complete model_alive server_alive tunnel_alive command_id action")
            public[stream + "_events"].append(item)
    public["jobs"] = []
    for job in trace["jobs"]:
        item = fields(job, "job_id status created_at started_at finished_at attempt_id model_instance_id instance_id generation_options cleanup_complete model_lease_released dispatch_state lifecycle")
        if job.get("artifact"):
            item["artifact"] = fields(job["artifact"], "bytes sha256 media")
        if job.get("metrics"):
            item["metrics"] = fields(job["metrics"], "schema_version kind backend started_at finished_at status wall_seconds timing runtime_options resource_counter_scope resources_before resources_after stages pipeline_stage_contract generation_options")
        public["jobs"].append(item)
    public["system_snapshots"] = []
    for snapshot in trace["system_snapshots"]:
        item = fields(snapshot, "instance_id model_instance_id phase observed_at observation_age_seconds observation_stale api_ready model_ready public_api_ready")
        model = (snapshot.get("modules") or {}).get("model") or {}
        item["modules"] = {"model": fields(model, "instance_id observation_unknown")}
        if "budget" in model:
            item["modules"]["model"]["budget"] = fields(model["budget"], "limits reserved cpu_threads_in_use gpu_attempt_id model_peak_ram_bytes os_reserve_bytes io_budget")
        public["system_snapshots"].append(item)
    cleanup = trace["cleanup_receipt"]
    public["cleanup_receipt"] = {
        "instance_id": cleanup["instance_id"],
        "state": fields(cleanup["state"], "instance_id cleanup_complete shutdown_complete manager_alive server_alive model_alive tunnel_alive manager_launcher_pid"),
        "owner_observation": fields(cleanup["owner_observation"], "alive cleanup_complete pids identities errors observed_at"),
    }
    public["platform_release"] = fields(trace["platform_release"], "released manager_instance_id observed_at source evidence_sha256")
    public["publication_projection"] = "Explicit fields, with private addresses, paths, prompts and nested execution results omitted. Analysis equivalence checked against private trace. Screenshot retained locally."
    from analyze_trace import analyze
    analysis = analyze(public)
    if analysis != analyze(trace):
        raise SystemExit("Projection changed an analysis result; do not publish")
    write(target / "trace.json", public)
    write(target / "analysis.json", analysis)
    state = read(source / "client/state.json")
    client = fields(state, "schema_version run_id instance_id manifest_sha256 state started_at updated_at finished_at reason parameters cold_queue_precondition")
    client["jobs"] = [fields(job, "name disposition state job_id reference_sha256 pre_submit_observation submission_http_status admission_state attempt_id cleanup_complete download cancel_intent cancel_http_status") for job in state["jobs"]]
    write(target / "client-receipt.json", client)
    for name, origin in {
        "calibration.json": "cloud/calibration.json", "finite-model.json": "finite-model.json",
        "runtime-source-manifest.json": "cloud/collection/runtime-source-manifest.json",
        "gpu-after-cleanup.json": "cloud/gpu-after-cleanup.json",
        "timeline.json": "timeline.json", "timeline.png": "timeline.png", "timeline.svg": "timeline.svg",
    }.items():
        shutil.copyfile(source / origin, target / name)
        if name.endswith(".svg"):
            svg = target / name
            svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf8").splitlines()) + "\n", encoding="utf8", newline="\n")
    # These files contain only bounded telemetry, with no raw server logs.
    gpu = [json.loads(line) for line in (source / "cloud/collection/gpu_samples.jsonl").read_text(encoding="utf8").splitlines()]
    write(target / "gpu-samples.json", gpu)
    manifest = read(source / "content/manifest.json")
    # The generated references and output videos are explicitly selected assets.
    assets = target / "assets"
    assets.mkdir(exist_ok=True)
    for name in ("coral-cable-car.png", "brass-dragonfly.png", "amber-lighthouse.png"):
        shutil.copyfile(source / "content" / name, assets / name)
    for job in state["jobs"]:
        if job["state"] == "succeeded":
            video = source / "client" / (job["job_id"] + ".mp4")
            if digest(video) != job["download"]["sha256"]:
                raise SystemExit("Artifact changed since client validation")
            shutil.copyfile(video, assets / (job["name"] + ".mp4"))
    # Publish exact prompts/options, but replace local input paths with asset names.
    for job in manifest["jobs"]:
        for key in list(job):
            if "path" in key or key in ("reference_file", "reference"):
                if isinstance(job[key], str):
                    job[key] = "assets/" + Path(job[key]).name
    write(target / "content-manifest.json", manifest)
    runtime = read(source / "cloud/collection/runtime-source-manifest.json")
    root = Path(__file__).resolve().parents[1]
    matches = {name: digest(root / "runtime/minimax_h3" / name) == row["sha256"] for name, row in runtime["files"].items()}
    provenance = {
        "schema_version": 1, "baseline_commit": "db7d10051f7259ad8ce1e8bf0d155a33fa329ff5",
        "runtime_source_match": matches,
        "runtime_changes": "Added monotonic_ns to manager and model audit events; no scheduling, precision or console-layout change.",
        "private_evidence_zip_sha256": digest(source / "h3-research-20261115-evidence.zip"),
        "private_collected_trace_sha256": digest(source / "cloud/collection/trace.json"),
        "platform_observation_sha256": digest(source / "platform-release-observation.json"),
        "restored_notebook_sha256": digest(source / "notebook-restored-readback.ipynb"),
        "restored_notebook_exact_match": (source / "notebook-restored-readback.ipynb").read_bytes() == (root / "notebooks/minimax-h3-ref2va.ipynb").read_bytes(),
        "research_source_sha256": {name: digest(root / "research" / name) for name in ("calibrate_hardware.py", "collect_runtime.py", "run_experiment.py")},
        "scope": "Export-time runtime files and retained experiment records. No cryptographic attestation of executed memory or independent witness; platform screenshot is local only.",
    }
    if not all(matches.values()) or not provenance["restored_notebook_exact_match"]:
        raise SystemExit("Runtime or restored notebook source mismatch")
    write(target / "provenance.json", provenance)
    print(json.dumps({"analysis": analysis["counts"], "runtime_source_matches": len(matches), "public_files": len(list(target.rglob("*")))}))


if __name__ == "__main__":
    main()
