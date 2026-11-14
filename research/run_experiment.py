"""Auditable public-API experiment client. Never starts/stops a service or GPU.

run --connection PRIVATE.json --manifest INPUTS.json --output PRIVATE_DIR
status --output PRIVATE_DIR
cancel --output PRIVATE_DIR --job-id ID

Resume only reconciles persisted job IDs; it never replays submissions. Exit 0
means this finite client experiment completed and downloaded retained artifacts;
1 means a definite failure; 2 means unknown/incomplete and requires reconciliation.
Raw local evidence is private and is not suitable for committing without review.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from fractions import Fraction
import hashlib
import json
import math
import mimetypes
import os
from pathlib import Path
import re
import subprocess
import time
from urllib.parse import urlsplit
import uuid

import requests


TERMINAL = {"succeeded", "failed", "cancelled"}
IDENTITY = re.compile(r"[A-Za-z0-9._:-]{1,128}\Z")


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(value, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def append_json(path, value):
    with Path(path).open("a", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        output.flush()
        os.fsync(output.fileno())


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def exclusive_run(path):
    """A process-owned lock is released by the OS on a crash, on either host OS."""
    lock = Path(path).open("a+b")
    try:
        if os.name == "nt":
            import msvcrt
            if lock.tell() == 0:
                lock.write(b"0")
                lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        lock.close()


def safe_status(state):
    return {"state": state.get("state"), "instance_id": state.get("instance_id"),
            "run_id": state.get("run_id"), "reason": state.get("reason"),
            "jobs": [{"name": job.get("name"), "job_id": job.get("job_id"), "state": job.get("state"),
                      "disposition": job.get("disposition"), "download_verified": bool(job.get("download", {}).get("verified"))}
                     for job in state.get("jobs", [])]}


def validate_manifest(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 3 or sorted(job.get("disposition") for job in jobs) != ["cancel", "retain", "retain"]:
        raise ValueError("Expected exactly two retained and one queued-cancel job")
    if len({job.get("name") for job in jobs}) != 3:
        raise ValueError("Experiment job names must be unique")
    for job in jobs:
        if not isinstance(job.get("name"), str) or not IDENTITY.fullmatch(job["name"]):
            raise ValueError("Invalid job name")
        if not isinstance(job.get("prompt"), str) or not job["prompt"].strip():
            raise ValueError("A new prompt is required")
        if type(job.get("seed")) is not int or not 0 <= job["seed"] <= 2147483647:
            raise ValueError("Invalid seed")
        reference = Path(job["reference_file"])
        if not reference.is_absolute():
            reference = path.parent / reference
        reference = reference.resolve()
        if not reference.is_file() or reference.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            raise ValueError("Expected an existing reference image")
        job["reference_file"] = str(reference)
    parameters = manifest.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("Missing generation parameters")
    allowed = {"seconds", "width", "height", "steps", "acceleration_lora", "attention_backend"}
    if set(parameters) != allowed:
        raise ValueError("Explicit generation parameters are required")
    return manifest, digest_file(path)


class ExperimentClient:
    def __init__(self, connection, output, *, session=None, timeout_seconds=2700, poll_seconds=5,
                 ffmpeg="ffmpeg", ffprobe="ffprobe"):
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.base = connection["url"].rstrip("/")
        parsed = urlsplit(self.base)
        if parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in {"", "/"}:
            raise ValueError("Connection must name an API origin")
        if parsed.scheme != "https" and not (parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost"}):
            raise ValueError("Public API requires HTTPS")
        self.instance = connection["instance_id"]
        if not isinstance(self.instance, str) or not IDENTITY.fullmatch(self.instance):
            raise ValueError("Missing instance identity")
        self.headers = {"Authorization": "Bearer " + connection["api_key"]}
        self.session = session or requests.Session()
        # Requests' default adapters do not retry POST. Redirects are also
        # refused so authorization and upload contents stay at this origin.
        self.timeout_seconds, self.poll_seconds = timeout_seconds, poll_seconds
        self.ffmpeg, self.ffprobe = ffmpeg, ffprobe
        self.state = None

    def save(self):
        self.state["updated_at"] = time.time()
        atomic_json(self.output / "state.json", self.state)

    def request(self, method, route, *, kind, **kwargs):
        record = {"request_id": uuid.uuid4().hex, "method": method, "route": route,
                  "kind": kind, "started_monotonic_ns": time.monotonic_ns(), "started_at": time.time()}
        append_json(self.output / "request-audit.jsonl", {**record, "event": "request.started"})
        try:
            response = self.session.request(method, self.base + route, headers={**self.headers, **kwargs.pop("headers", {})},
                timeout=kwargs.pop("timeout", (10, 30)), allow_redirects=False, **kwargs)
            record["status_code"] = response.status_code
            return response
        except requests.RequestException as error:
            record["error_type"] = type(error).__name__
            return None
        finally:
            record["finished_monotonic_ns"] = time.monotonic_ns()
            append_json(self.output / "request-audit.jsonl", {**record, "event": "request.finished"})

    @staticmethod
    def body(response):
        try:
            value = response.json() if response is not None else None
            return value if isinstance(value, dict) else None
        except (ValueError, requests.RequestException):
            return None

    def system(self):
        response = self.request("GET", "/v1/system", kind="system")
        value = self.body(response)
        if response is None or response.status_code != 200 or value is None:
            return None
        if value.get("instance_id") != self.instance:
            raise ValueError("API instance identity changed")
        append_json(self.output / "system-snapshots.jsonl", value)
        return value

    def record_job(self, row, value):
        if value.get("instance_id") != self.instance or value.get("id") != row.get("job_id"):
            raise ValueError("Job identity mismatch")
        if value.get("status") not in TERMINAL | {"queued", "running", "cancel_requested"}:
            raise ValueError("Unknown job state")
        atomic_json(self.output / ("job-" + row["job_id"] + ".json"), value)
        previous = row.get("state")
        row["state"] = value["status"]
        row["attempt_id"] = value.get("attempt_id")
        row["cleanup_complete"] = value.get("cleanup_complete")
        if previous != row["state"]:
            append_json(self.output / "client-events.jsonl", {"event": "job.observed", "job_id": row["job_id"],
                "state": row["state"], "observed_at": time.time(), "monotonic_ns": time.monotonic_ns()})
            print(json.dumps({"state": row["state"], "job_id": row["job_id"]}), flush=True)
        self.save()
        return value

    def query_job(self, row):
        response = self.request("GET", "/v1/jobs/" + row["job_id"], kind="job.query")
        value = self.body(response)
        if response is not None and response.status_code == 200 and value:
            return self.record_job(row, value)
        return None

    def submit(self, row, job, parameters):
        # Persist the mutation intent before crossing the network boundary.
        row["state"] = "submission_started"
        self.save()
        path = Path(job["reference_file"])
        with path.open("rb") as reference:
            response = self.request("POST", "/v1/jobs", kind="job.submit", timeout=(10, 120),
                headers={"Idempotency-Key": row["idempotency_key"]},
                files=[("files", (path.name, reference, mimetypes.guess_type(path.name)[0] or "application/octet-stream"))],
                data={**parameters, "kinds": '["image"]', "prompt": job["prompt"], "seed": job["seed"]})
        value = self.body(response)
        detail = value.get("detail", {}) if value else {}
        job_id = value.get("id") if value else None
        if job_id is None and isinstance(detail, dict):
            job_id = detail.get("job_id")
        if isinstance(job_id, str) and IDENTITY.fullmatch(job_id):
            row["job_id"] = job_id
        row["submission_http_status"] = response.status_code if response is not None else None
        if response is not None and response.status_code == 202 and value and row.get("job_id"):
            row["admission_state"] = value.get("status")
            self.record_job(row, value)
        elif response is not None and 400 <= response.status_code < 500:
            row["state"] = "submission_rejected"
        else:
            row["state"] = "submission_outcome_unknown"
        self.save()
        return row.get("job_id")

    def cancel(self, row, *, source):
        if not row.get("job_id") or row.get("state") in TERMINAL:
            return
        row["cancel_intent"] = {"source": source, "observed_at": time.time()}
        self.save()
        response = self.request("POST", "/v1/jobs/" + row["job_id"] + "/cancel", kind="job.cancel")
        value = self.body(response)
        row["cancel_http_status"] = response.status_code if response is not None else None
        if response is not None and response.status_code in {200, 202} and value and value.get("id"):
            self.record_job(row, value)
        else:
            row["cancel_outcome"] = "unknown_reconcile_by_job_id"
            self.save()

    def drain_control(self):
        directory = self.output / "commands"
        directory.mkdir(exist_ok=True)
        for path in sorted(directory.glob("*.json")):
            command = json.loads(path.read_text(encoding="utf-8"))
            command_id = command.get("command_id")
            if command_id in self.state["handled_commands"]:
                continue
            row = next((row for row in self.state["jobs"] if row.get("job_id") == command.get("job_id")), None)
            if command.get("action") != "cancel" or row is None:
                outcome = "rejected_unknown_action_or_job"
            else:
                # Mark before POST: a crash is reconciled by job ID, not replay.
                self.state["handled_commands"][command_id] = "started"
                self.save()
                self.cancel(row, source="command_file")
                outcome = "submitted_query_job_for_effect"
            self.state["handled_commands"][command_id] = outcome
            self.save()

    def collect_events(self):
        # Bound each poll to four pages, retaining the cursor for the next pass.
        for _ in range(4):
            cursor = self.state.get("event_cursor", 0)
            response = self.request("GET", f"/v1/events?after={cursor}&limit=1000", kind="events")
            value = self.body(response)
            if response is None or response.status_code != 200 or not value:
                return
            events = value.get("events", [])
            if not isinstance(events, list) or any(event.get("instance_id") != self.instance for event in events):
                raise ValueError("Event instance mismatch")
            for event in events:
                append_json(self.output / "manager-events.jsonl", event)
            next_cursor = value.get("next_cursor")
            if type(next_cursor) is not int or next_cursor < cursor:
                raise ValueError("Invalid event cursor")
            self.state["event_cursor"] = next_cursor
            self.save()
            if next_cursor == cursor or len(events) < 1000:
                return

    def download(self, row, job, parameters):
        final = self.output / (row["job_id"] + ".mp4")
        partial = final.with_suffix(".mp4.part")
        artifact = job.get("artifact", {})
        expected = artifact.get("sha256")
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            row["download"] = {"verified": False, "state": "missing_expected_hash"}
            self.save()
            return False
        if not final.exists():
            response = self.request("GET", "/v1/jobs/" + row["job_id"] + "/file", kind="artifact.download", timeout=(10, 120), stream=True)
            if response is None or response.status_code != 200:
                if response is not None:
                    response.close()
                return False
            try:
                with response, partial.open("wb") as output:
                    for block in response.iter_content(1024 * 1024):
                        if block:
                            output.write(block)
                partial.replace(final)
            except requests.RequestException:
                return False
        observed = digest_file(final)
        row["download"] = {"verified": False, "sha256": observed, "expected_sha256": expected, "bytes": final.stat().st_size}
        if observed != expected or (artifact.get("bytes") is not None and final.stat().st_size != artifact["bytes"]):
            row["download"]["state"] = "hash_or_size_mismatch"
            self.save()
            return False
        try:
            probe = subprocess.run([self.ffprobe, "-v", "error", "-threads", "1", "-count_frames", "-show_streams", "-show_format", "-of", "json", str(final)],
                capture_output=True, text=True, check=True, timeout=120)
            media = json.loads(probe.stdout)
            video = next(item for item in media.get("streams", []) if item.get("codec_type") == "video")
            audio = next(item for item in media.get("streams", []) if item.get("codec_type") == "audio")
            frames = int(video.get("nb_read_frames") or video.get("nb_frames") or 0)
            fps = float(Fraction(video.get("avg_frame_rate", "0")))
            duration = float(video.get("duration") or media.get("format", {}).get("duration") or 0)
            expected_frames = 5 + 17 * math.ceil((round(parameters["seconds"] * 24) - 5) / 17)
            if (video.get("width"), video.get("height")) != (parameters["width"], parameters["height"]) or frames != expected_frames or not math.isfinite(fps) or abs(fps - 24) > .01 or video.get("codec_name") != "h264" or not math.isfinite(duration) or abs(duration - expected_frames / 24) > .1 or int(audio.get("channels", 0)) != 2:
                raise ValueError("Media contract mismatch")
            subprocess.run([self.ffmpeg, "-nostdin", "-v", "error", "-xerror", "-threads", "1", "-filter_threads", "1", "-filter_complex_threads", "1", "-i", str(final), "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-"],
                capture_output=True, check=True, timeout=300)
            row["download"].update(verified=True, state="verified", media={"frames": frames, "fps": fps,
                "width": video["width"], "height": video["height"], "audio_channels": audio["channels"], "full_decode_passed": True})
        except (OSError, subprocess.SubprocessError, ValueError, StopIteration, KeyError) as error:
            row["download"].update(state="validation_failed", error_type=type(error).__name__)
        self.save()
        return row["download"]["verified"]

    def run(self, manifest_path, *, resume=False):
        manifest, manifest_hash = validate_manifest(manifest_path)
        state_path = self.output / "state.json"
        if resume:
            self.state = json.loads(state_path.read_text(encoding="utf-8"))
            if self.state.get("instance_id") != self.instance or self.state.get("manifest_sha256") != manifest_hash:
                raise ValueError("Resume identity or manifest changed")
        else:
            if state_path.exists():
                raise ValueError("Existing experiment requires --resume")
            self.state = {"schema_version": 1, "run_id": uuid.uuid4().hex, "instance_id": self.instance,
                "manifest_sha256": manifest_hash, "state": "preflight", "started_at": time.time(), "event_cursor": 0,
                "handled_commands": {}, "parameters": manifest["parameters"], "jobs": [
                    {"name": job["name"], "disposition": job["disposition"], "state": "not_submitted", "job_id": None,
                     "idempotency_key": "research-" + uuid.uuid4().hex, "reference_sha256": digest_file(job["reference_file"])}
                    for job in manifest["jobs"]]}
            self.save()
        deadline = time.monotonic() + self.timeout_seconds
        observed = None
        while observed is None and time.monotonic() < deadline:
            observed = self.system()
            if observed is None:
                time.sleep(self.poll_seconds)
        if observed is None:
            return self.finish("incomplete", "system_not_observed", 2)
        if not resume:
            if observed.get("model_ready") is not False or observed.get("public_api_ready") is not True:
                return self.finish("incomplete", "cold_public_queue_precondition_not_observed", 2)
            self.state["cold_queue_precondition"] = {"model_ready": False, "public_api_ready": True,
                                                     "observed_at": observed.get("observed_at")}
            self.state["state"] = "submitting"
            self.save()
            for row, job in zip(self.state["jobs"], manifest["jobs"]):
                # Recheck readiness for every mutation; a changed condition is
                # reported, never overwritten as if all uploads occurred cold.
                observed = self.system()
                if observed is None or observed.get("model_ready") is not False:
                    return self.finish("incomplete", "cold_queue_window_closed_or_unknown", 2)
                row["pre_submit_observation"] = {"model_ready": False, "observed_at": observed.get("observed_at")}
                if not self.submit(row, job, manifest["parameters"]):
                    return self.finish("incomplete", "submission_has_no_reconcilable_id", 2)
                if row["disposition"] == "cancel":
                    self.cancel(row, source="planned_queued_cancellation")
        self.state["state"] = "reconciling" if resume else "observing"
        self.state["reason"] = None
        self.save()
        while time.monotonic() < deadline:
            self.system()
            self.drain_control()
            for row in self.state["jobs"]:
                if not row.get("job_id"):
                    continue
                job = self.query_job(row)
                if job and job.get("status") == "succeeded" and row["disposition"] == "retain" and not row.get("download", {}).get("verified"):
                    self.download(row, job, manifest["parameters"])
            self.collect_events()
            rows = self.state["jobs"]
            if all(row.get("state") in TERMINAL for row in rows):
                expected = all(row["state"] == ("cancelled" if row["disposition"] == "cancel" else "succeeded") for row in rows)
                clean = all(row.get("cleanup_complete") is True for row in rows)
                downloaded = all(row.get("download", {}).get("verified") for row in rows if row["disposition"] == "retain")
                if not expected:
                    return self.finish("failed", "unexpected_terminal_state", 1)
                if any(row.get("download", {}).get("state") in {"hash_or_size_mismatch", "missing_expected_hash", "validation_failed"} for row in rows):
                    return self.finish("failed", "artifact_identity_or_media_validation_failed", 1)
                if clean and downloaded:
                    return self.finish("completed", "retained_artifacts_verified_and_cancelled_job_terminal", 0)
            if all(not row.get("job_id") or row.get("state") in TERMINAL for row in rows) and any(not row.get("job_id") for row in rows):
                return self.finish("incomplete", "unsubmitted_or_unknown_jobs_not_replayed", 2)
            time.sleep(min(self.poll_seconds, max(0, deadline - time.monotonic())))
        return self.finish("incomplete", "deadline_reached_service_left_untouched", 2)

    def finish(self, state, reason, code):
        self.state.update(state=state, reason=reason, finished_at=time.time())
        self.save()
        print(json.dumps(safe_status(self.state)), flush=True)
        return code


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    run = sub.add_parser("run")
    run.add_argument("--connection", type=Path, required=True)
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--timeout", type=float, default=2700)
    run.add_argument("--poll", type=float, default=5)
    run.add_argument("--ffmpeg", default="ffmpeg")
    run.add_argument("--ffprobe", default="ffprobe")
    status = sub.add_parser("status")
    status.add_argument("--output", type=Path, required=True)
    cancel = sub.add_parser("cancel")
    cancel.add_argument("--output", type=Path, required=True)
    cancel.add_argument("--job-id", required=True)
    args = parser.parse_args()
    try:
        if args.action == "status":
            print(json.dumps(safe_status(json.loads((args.output / "state.json").read_text(encoding="utf-8")))) )
            return 0
        if args.action == "cancel":
            if not IDENTITY.fullmatch(args.job_id):
                raise ValueError("Invalid job identity")
            state = json.loads((args.output / "state.json").read_text(encoding="utf-8"))
            if not any(row.get("job_id") == args.job_id for row in state.get("jobs", [])):
                raise ValueError("Job is not owned by this experiment")
            command = {"command_id": uuid.uuid4().hex, "action": "cancel", "job_id": args.job_id, "created_at": time.time()}
            (args.output / "commands").mkdir(exist_ok=True)
            atomic_json(args.output / "commands" / (command["command_id"] + ".json"), command)
            print(json.dumps({"state": "cancel_request_recorded", "job_id": args.job_id}))
            return 0
        if not 0 < args.timeout <= 2700 or not 0 < args.poll <= 30:
            raise ValueError("Timeout must be 0-2700s and poll 0-30s")
        args.output.mkdir(parents=True, exist_ok=True)
        with exclusive_run(args.output / ".run.lock"):
            connection = json.loads(args.connection.read_text(encoding="utf-8"))
            client = ExperimentClient(connection, args.output, timeout_seconds=args.timeout, poll_seconds=args.poll,
                                      ffmpeg=args.ffmpeg, ffprobe=args.ffprobe)
            try:
                return client.run(args.manifest, resume=args.resume)
            except BaseException as error:
                if client.state is not None:
                    client.state.update(state="interrupted", reason=type(error).__name__)
                    client.save()
                raise
    except (OSError, ValueError, KeyError, requests.RequestException) as error:
        # Never print request exception text: it can contain a private origin.
        print(json.dumps({"state": "client_error", "error_type": type(error).__name__}), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
