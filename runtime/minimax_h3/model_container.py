"""Private, independently supervised MiniMax-H3 model runtime.

The HTTP controller never imports torch or loads weights. The supervised owner
loads after binding its socket. Budgeted CPU stages overlap across requests and
share one exclusive GPU lease. Production backends retain native BF16; injected
backends are only a test seam.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
from fractions import Fraction
import gc
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import queue
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import uuid

from control_protocol import ControlError, make_server
from condition_guard import evaluate_gate, negate_observation, observation_state
from download_progress import read_progress
from pipeline_scheduler import BudgetAuditUnavailable, BudgetUnavailable, PipelineBudget

LOG = logging.getLogger("h3.model_container")
IDENTIFIER = re.compile(r"[a-f0-9]{32}\Z")
TERMINAL = {"succeeded", "failed", "cancelled"}
SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
EXECUTE_CONDITIONS = ("instance_matches", "request_validated", "model_loaded", "worker_alive",
    "capability_available", "not_draining", "not_stopping", "attempt_available", "job_unowned",
    "artifact_absent", "generation_options_valid", "resources_reserved")
STAGE_CONDITIONS = ("instance_matches", "attempt_owned", "attempt_live", "model_loaded", "worker_alive",
    "capability_available", "admission_confirmed", "predecessor_completed", "stage_not_started",
    "cancel_flag_clear", "cancel_event_clear", "reservation_owned", "budget_grant_held")


class ExecutionCancelled(Exception):
    pass


class ConditionGateBlocked(RuntimeError):
    """Missing or contradictory evidence is a failed execution, not cancellation."""
    pass


def _all_observed(values):
    """Conjoin distinct obligations; a missing obligation is not a spare witness."""
    states = [observation_state(value) for value in values]
    if "conflict" in states:
        return "conflict"
    if "unknown" in states:
        return None
    return all(state == "true" for state in states)


def parent_identity(parent_pid):
    """Bind ownership to one OS process generation, including on Windows."""
    import psutil
    try:
        process = psutil.Process(parent_pid)
        # A Windows venv redirector may be the OS parent. The authenticated
        # launcher specifies the actual owner; bind to that process generation.
        if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
            raise RuntimeError("The requested model owner is not alive")
        return process.create_time()
    except psutil.Error as error:
        raise RuntimeError("Cannot establish the model parent process identity") from error


def parent_is_alive(parent_pid, created_at):
    import psutil
    try:
        process = psutil.Process(parent_pid)
        return (process.is_running() and process.create_time() == created_at
                and process.status() != psutil.STATUS_ZOMBIE)
    except psutil.Error:
        # Unverified ownership must not retain a costly unmanaged model worker.
        return False


def verify_artifact(path: Path, parameters: dict, *, cpu_threads=1) -> dict:
    """Validate the entire audiovisual artifact before publishing success."""
    if path.is_symlink() or not path.is_file() or not path.stat().st_size:
        raise ValueError("Missing model artifact")
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-threads", str(cpu_threads), "-count_frames", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True, text=True, timeout=120, check=True,
    )
    data = json.loads(probe.stdout)
    streams = data.get("streams", [])
    video = next((x for x in streams if x.get("codec_type") == "video"), None)
    audio = next((x for x in streams if x.get("codec_type") == "audio"), None)
    if video is None or audio is None:
        raise ValueError("Artifact must contain video and audio")
    frames = int(video.get("nb_read_frames") or video.get("nb_frames") or 0)
    fps = float(Fraction(video.get("avg_frame_rate", "0")))
    duration = float(video.get("duration") or data.get("format", {}).get("duration") or 0)
    expected_frames = 5 + 17 * math.ceil((round(parameters["seconds"] * 24) - 5) / 17)
    if ((video.get("width"), video.get("height")) != (parameters["width"], parameters["height"])
            or video.get("codec_name") != "h264" or frames != expected_frames
            or abs(fps - 24) > .01 or not math.isfinite(duration)
            or abs(duration - expected_frames / 24) > .1 or int(audio.get("channels", 0)) != 2):
        raise ValueError("Artifact media does not match the admitted request")
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-threads", str(cpu_threads),
         "-filter_threads", "1", "-filter_complex_threads", "1", "-i", str(path),
         "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-"],
        capture_output=True, text=True, timeout=300, check=True,
    )
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest(),
            "media": {"video_codec": video["codec_name"], "width": video["width"], "height": video["height"],
                      "decoded_frames": frames, "fps": fps, "duration_seconds": duration,
                      "audio_codec": audio.get("codec_name"), "audio_channels": audio["channels"],
                      "audio_sample_rate": audio.get("sample_rate"), "full_decode_passed": True}}


class ModelContainer:
    def __init__(self, root, instance_id, *, backend=None, staged=False,
                 artifact_validator=verify_artifact, resource_limits=None, progress_path=None):
        if not isinstance(instance_id, str) or not IDENTIFIER.fullmatch(instance_id):
            raise ValueError("Model instance must be a 32-character hexadecimal ID")
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.instance_id, self.staged = instance_id, bool(staged)
        self.backend = backend
        self._injected_backend = backend is not None
        self._resource_limits = resource_limits
        self._pipeline = False
        self._budget = None
        self._stage_workers = {}
        self._results = {}
        self.artifact_validator = artifact_validator
        self.event_directory = self.root / "_model_runtime" / instance_id
        if self.event_directory.resolve() != self.event_directory:
            raise ValueError("Model audit directory must not traverse a symlink")
        self.event_directory.mkdir(parents=True, exist_ok=True)
        self.events_path = self.event_directory / "events.jsonl"
        self.progress_path = Path(progress_path) if progress_path else self.event_directory / "downloads.json"
        self._lock = threading.RLock()
        self._queue = queue.Queue()
        self._worker = None
        self._done = threading.Event()
        self._stop_requested = False
        self._draining = False
        self._loaded = False
        self._phase = "loading"
        self._error = None
        self._active = None
        self._last_result = None
        self._attempts = {}
        self._sequence = 0
        self._consecutive_failures = 0
        self._worker_cleanup_complete = False
        self._last_condition_gate = None
        self._ledger_path = self.event_directory / "attempts.sqlite3"
        ledger = sqlite3.connect(self._ledger_path)
        try:
            with ledger:
                ledger.execute("CREATE TABLE IF NOT EXISTS acknowledged (attempt_id TEXT PRIMARY KEY, record TEXT NOT NULL)")
        finally:
            ledger.close()

    def _previous_attempt(self, attempt_id):
        if attempt_id in self._attempts:
            return self._attempts[attempt_id]
        # Open briefly so acknowledgement does not retain a lifetime DB handle.
        ledger = sqlite3.connect(self._ledger_path)
        try:
            row = ledger.execute("SELECT record FROM acknowledged WHERE attempt_id=?", (attempt_id,)).fetchone()
            return json.loads(row[0]) if row else None
        finally:
            ledger.close()

    def _live_attempts(self):
        return [value for value in self._attempts.values() if value["status"] not in TERMINAL]

    def _event(self, kind, **data):
        with self._lock:
            self._sequence += 1
            event = {"sequence": self._sequence, "event_id": uuid.uuid4().hex,
                     "instance_id": self.instance_id, "type": kind, "observed_at": time.time(),
                     "monotonic_ns": time.monotonic_ns(), **data}
            try:
                with self.events_path.open("a", encoding="utf-8") as output:
                    output.write(json.dumps(event, ensure_ascii=False) + "\n")
                    output.flush()
                    os.fsync(output.fileno())
            except OSError as error:
                self._loaded = False
                self._phase = "failed"
                self._error = "model_audit_unavailable"
                raise ControlError("Model audit storage unavailable", status=503) from error

    def _safe_event(self, kind, **data):
        try:
            self._event(kind, **data)
        except BaseException:
            LOG.exception("Could not persist model event %s", kind)

    def _gate(self, gate_id, required, observations, *, job_id, attempt_id, attempt=None, **context):
        """Caller owns _lock; persist the decision before permitting its effect."""
        receipt = evaluate_gate(gate_id, required, observations, {
            "instance_id": self.instance_id, "job_id": job_id, "attempt_id": attempt_id,
            "source_file": "model_container.py", "source_sha256": SOURCE_SHA256, **context})
        self._event("condition.gate", job_id=job_id, attempt_id=attempt_id,
                    gate_id=gate_id, receipt=receipt)
        self._last_condition_gate = receipt
        if attempt is not None:
            attempt.setdefault("condition_gates", {})[gate_id] = receipt
        return receipt

    def _capability_observation(self, *, admitted=False):
        if type(self._phase) is not str:
            return None
        allowed = {"ready", "running", "draining"} if admitted else {"ready", "running"}
        if self._phase in allowed:
            return True
        if self._phase in {"loading", "starting", "failed", "stopped", "draining"}:
            return False
        return None

    def start(self):
        with self._lock:
            if self._worker is not None:
                raise ControlError("Model container has already started", status=409)
            self._event("model.loading")
            self._worker = threading.Thread(target=self._work, name="h3-model-worker", daemon=False)
            try:
                self._worker.start()
            except BaseException:
                self._phase, self._error = "failed", "worker_start_failed"
                self._loaded = False
                self._done.set()
                self._safe_event("model.failed", error=self._error)
                raise
        return self

    def status(self):
        with self._lock:
            alive = self._worker is not None and self._worker.is_alive()
            if self._worker is not None and not alive and self._phase not in {"failed", "stopped"}:
                self._phase, self._error, self._loaded = "failed", "worker_exited", False
            ready = self._loaded is True and alive and self._capability_observation() is True
            downloads = read_progress(self.progress_path, expected_instance_id=self.instance_id)
            downloads.update(model_ready=ready, lifecycle=self._phase,
                             observation_stale=bool(downloads.get("observation_stale") or not alive))
            phase = None
            snapshot = getattr(self.backend, "phase_snapshot", None)
            if snapshot:
                try:
                    phase = snapshot()
                except Exception:
                    phase = None
            active = self._active
            live = self._live_attempts()
            # Keep each reply below the private protocol's bounded body. Results
            # remain owned until ack; additional pages surface after that ack.
            results, result_bytes = [], 0
            for result in self._results.values():
                size = len(json.dumps(result).encode())
                if results and result_bytes + size > 1024 * 1024:
                    break
                results.append(result)
                result_bytes += size
            # A current capacity snapshot has fixed-size audit metadata. Read it
            # once, then attach the independent copy after copying live state.
            budget = self._budget.snapshot(include_history=False) if self._budget else None
            response = copy.deepcopy({
                "instance_id": self.instance_id, "phase": self._phase, "ready": ready,
                "accepting": ready and self._draining is False and self._stop_requested is False and (
                    (budget is not None and budget['lease_journal']['complete'] is True
                     and all(value > 0 for value in budget['free'].values())) if self._pipeline else active is None),
                "pipeline": self._pipeline,
                "downloads": downloads,
                "attempts": [{**self._accepted(item), "phase": item.get("stage", item["status"]),
                              "generation_options": item.get("generation_options"),
                              "condition_gates": item.get("condition_gates", {}),
                              "stage_completions": item.get("stage_completions", {}),
                              "cancel_requested": item["cancel_requested"]} for item in live],
                "results": results, "pending_results": len(self._results),
                "active_job_id": active["job_id"] if active else None,
                "active_attempt_id": active["attempt_id"] if active else None,
                "cancel_requested": bool(active and active["cancel_requested"]),
                "active_phase": phase, "last_result": self._last_result,
                "worker_alive": alive, "cleanup_complete": False,
                "worker_cleanup_complete": self._worker_cleanup_complete,
                "cleanup_scope": "Process exit must be confirmed by the owner before GPU ownership is released",
                "error": self._error, "observed_at": time.time(), "event_sequence": self._sequence,
                "last_condition_gate": self._last_condition_gate,
            })
            response['budget'] = budget
            return response

    def budget_audit(self, payload):
        self._check_instance(payload, required=True)
        with self._lock:
            budget = self._budget
            if budget is None or payload.get('budget_instance_id') != budget.instance_id:
                raise ControlError('Budget audit instance does not match', status=409)
        # Read-only disk pages never wait under the model state lock.
        try:
            page = budget.audit_page(after_sequence=payload.get('after_sequence', 0),
                through_sequence=payload.get('through_sequence'), limit=payload.get('limit', 128))
        except ValueError as error:
            raise ControlError(str(error), status=422) from error
        except (BudgetAuditUnavailable, OSError, sqlite3.Error, KeyError, TypeError) as error:
            raise ControlError('Budget audit is unavailable', status=503) from error
        return {'model_instance_id': self.instance_id, **page}

    def _check_instance(self, payload, required=False):
        instance = payload.get("instance_id")
        if instance != self.instance_id and (required or instance is not None):
            raise ControlError("Model instance does not match this execution generation", status=409)

    def _request(self, payload):
        self._check_instance(payload, required=True)
        job_id, attempt_id = payload.get("job_id"), payload.get("attempt_id")
        if any(not isinstance(x, str) or not IDENTIFIER.fullmatch(x) for x in (job_id, attempt_id)):
            raise ControlError("Invalid job or attempt identity", status=422)
        parameters = payload.get("parameters")
        names = {"prompt", "references", "seconds", "width", "height", "steps", "seed"}
        optional = {"attention_backend", "acceleration_lora"}
        if not isinstance(parameters, dict) or not names <= set(parameters) or set(parameters) - names - optional:
            raise ControlError("Invalid generation parameters", status=422)
        p = copy.deepcopy(parameters)
        try:
            from adapter_config import load_adapter_settings
            from generation_options import requested_options, validate_generation_options
            validate_generation_options(requested_options(p), load_adapter_settings())
        except (TypeError, ValueError) as error:
            raise ControlError(str(error), status=422) from None
        if not isinstance(p["prompt"], str) or not p["prompt"].strip() or len(p["prompt"]) > 4000:
            raise ControlError("Invalid prompt", status=422)
        if any(isinstance(p[k], bool) or not isinstance(p[k], int) for k in ("width", "height", "steps", "seed")):
            raise ControlError("Generation dimensions, steps and seed must be integers", status=422)
        if (not 256 <= p["width"] <= 1344 or not 256 <= p["height"] <= 1344
                or p["width"] % 32 or p["height"] % 32 or p["width"] * p["height"] > 1344 * 768
                or not 2 <= p["steps"] <= 50 or not 0 <= p["seed"] <= 2147483647):
            raise ControlError("Generation parameters outside resource limits", status=422)
        seconds = p["seconds"]
        if (isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds)
                or not 5 <= seconds <= 15 or 5 + 17 * math.ceil((round(seconds * 24) - 5) / 17) > 360):
            raise ControlError("Invalid generation duration", status=422)
        refs = p["references"]
        if not isinstance(refs, list) or not 1 <= len(refs) <= 12:
            raise ControlError("Supply 1-12 references", status=422)
        directory = self.root / job_id
        inputs = directory / "inputs"
        if directory.resolve() != directory or inputs.resolve() != inputs:
            raise ControlError("Reference directory must stay within its assigned job", status=422)
        kinds = []
        total = 0
        for ref in refs:
            if not isinstance(ref, dict) or set(ref) != {"kind", "path"} or ref["kind"] not in {"image", "video", "audio"}:
                raise ControlError("Invalid reference metadata", status=422)
            try:
                path = Path(ref["path"])
                resolved = path.resolve(strict=True)
                size = resolved.stat().st_size
                valid = (path.is_absolute() and resolved.is_relative_to(inputs) and path == resolved
                         and resolved.is_file() and not path.is_symlink() and 0 < size <= 32 * 1024 * 1024)
            except (OSError, TypeError, ValueError):
                valid = False
            if not valid:
                raise ControlError("Reference path is missing, oversized or outside its assigned inputs", status=422)
            total += size
            ref["path"] = str(resolved)
            kinds.append(ref["kind"])
        if (total > 64 * 1024 * 1024 or kinds.count("image") > 9 or kinds.count("video") > 3
                or kinds.count("audio") > 3 or all(k == "audio" for k in kinds)):
            raise ControlError("Reference combination outside resource limits", status=422)
        fingerprint = hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()
        return job_id, attempt_id, p, fingerprint

    def execute(self, payload):
        # Idempotent replies must remain possible after inputs have been cleaned.
        self._check_instance(payload, required=True)
        attempt_id = payload.get("attempt_id")
        with self._lock:
            old = self._previous_attempt(attempt_id) if isinstance(attempt_id, str) else None
            if old:
                candidate = hashlib.sha256(json.dumps(payload.get("parameters"), sort_keys=True).encode()).hexdigest()
                if old["job_id"] != payload.get("job_id") or old["fingerprint"] != candidate:
                    raise ControlError("Attempt identity was reused with different parameters", status=409)
                return self._accepted(old)
        try:
            job_id, attempt_id, parameters, fingerprint = self._request(payload)
        except ControlError as error:
            if error.status == 422:
                raise ControlError(json.dumps({"guarantees_not_accepted": True,
                                               "reason": "invalid_request", "detail": str(error)}), 422) from None
            raise
        # Header probes can perform bounded I/O. Do not hold the controller's
        # state lock while estimating an otherwise unadmitted request.
        resources = None
        with self._lock:
            estimator = self.backend if self._pipeline and self._loaded is True and self._draining is False else None
        if estimator is not None:
            try:
                resources = estimator.estimate_resources(parameters)
            except ValueError:
                raise ControlError(json.dumps({"guarantees_not_accepted": True,
                                               "reason": "invalid_reference_media"}), status=422) from None
        with self._lock:
            old = self._previous_attempt(attempt_id)
            if old:
                if old["job_id"] != job_id or old["fingerprint"] != fingerprint:
                    raise ControlError("Attempt identity conflict", status=409)
                return self._accepted(old)
            output = self.root / job_id / "result.mp4"
            validator = getattr(self.backend, "validate_request", None)
            options_valid = True
            if validator:
                try:
                    validator(steps=parameters["steps"], **{name: parameters[name]
                        for name in ("attention_backend", "acceleration_lora") if name in parameters})
                except ValueError:
                    options_valid = False
            observations = {
                "instance_matches": payload.get("instance_id") == self.instance_id,
                "request_validated": True,
                "model_loaded": self._loaded,
                "worker_alive": self._worker.is_alive() if self._worker is not None else False,
                "capability_available": self._capability_observation(),
                "not_draining": negate_observation(self._draining),
                "not_stopping": negate_observation(self._stop_requested),
                "attempt_available": old is None,
                "job_unowned": not any(item["job_id"] == job_id for item in self._live_attempts()),
                "artifact_absent": not (output.exists() or output.is_symlink()),
                "generation_options_valid": options_valid,
                "resources_reserved": None,
            }
            context = {"pipeline": self._pipeline, "budget_scope": "stage_budget" if self._pipeline else "legacy_single_active"}
            if any(observation_state(value) != "true" for key, value in observations.items() if key != "resources_reserved"):
                receipt = self._gate("model.execute", EXECUTE_CONDITIONS, observations,
                    job_id=job_id, attempt_id=attempt_id, **context)
                raise ControlError(json.dumps({"guarantees_not_accepted": True, "reason": "model_conditions_blocked",
                    "decision_id": receipt["decision_id"], "blocked": receipt["blocked"]}), 409 if options_valid else 422)
            attempt = {"job_id": job_id, "attempt_id": attempt_id, "instance_id": self.instance_id,
                       "status": "accepted", "cancel_requested": False, "parameters": parameters,
                       "fingerprint": fingerprint, "accepted_at": time.time(), "condition_gates": {},
                       "stage_started": {"prepare": False, "gpu": False, "encode_validate": False},
                       "stage_completions": {"prepare": None, "gpu": None, "encode_validate": None}}
            reserved = False
            if self._pipeline:
                if resources is None:
                    receipt = self._gate("model.execute", EXECUTE_CONDITIONS, observations,
                        job_id=job_id, attempt_id=attempt_id, **context)
                    raise ControlError(json.dumps({"guarantees_not_accepted": True,
                                                   "reason": "model_not_accepting", "decision_id": receipt["decision_id"]}), status=409)
                try:
                    self._budget.reserve(attempt_id, resources)
                except BudgetUnavailable as error:
                    self._event("reservation.blocked", job_id=job_id, attempt_id=attempt_id,
                                reservation_witness=self._budget.reservation_witness(attempt_id),
                                reservation_receipt=self._budget.gate_receipt(attempt_id, "reserve"))
                    observations["resources_reserved"] = False
                    receipt = self._gate("model.execute", EXECUTE_CONDITIONS, observations,
                        job_id=job_id, attempt_id=attempt_id, budget_reason=str(error), **context)
                    raise ControlError(json.dumps({"guarantees_not_accepted": True, "reason": str(error),
                                                   "decision_id": receipt["decision_id"]}),
                                       status=507 if error.permanent else 429) from None
                reserved = True
                attempt["resources"] = resources
                attempt["_cancel_event"] = threading.Event()
            try:
                witness = self._budget.reservation_witness(attempt_id) if self._pipeline else None
                if self._pipeline:
                    self._event("reservation.acquired", job_id=job_id, attempt_id=attempt_id,
                                reservation_witness=witness,
                                reservation_receipt=self._budget.gate_receipt(attempt_id, "reserve"))
                observations["resources_reserved"] = (_all_observed([
                    witness.get("owned"), witness.get("attempt_id") == attempt_id,
                    witness.get("budget_instance_id") == self._budget.instance_id,
                    type(witness.get("reservation_id")) is str and bool(witness["reservation_id"])
                ]) if witness is not None else self._active is None if not self._pipeline else None)
                receipt = self._gate("model.execute", EXECUTE_CONDITIONS, observations,
                    job_id=job_id, attempt_id=attempt_id, attempt=attempt,
                    budget_instance_id=witness.get("budget_instance_id") if witness else None,
                    reservation_id=witness.get("reservation_id") if witness else None, **context)
                if receipt["allowed"] is not True:
                    raise ControlError(json.dumps({"guarantees_not_accepted": True, "reason": "model_conditions_blocked",
                        "decision_id": receipt["decision_id"], "blocked": receipt["blocked"]}), 409)
            except BaseException:
                if reserved:
                    self._budget.finish(attempt_id)
                raise
            self._attempts[attempt_id] = attempt
            if not self._pipeline:
                self._active = attempt
            self._phase = "running"
            try:
                self._event("execution.accepted", job_id=job_id, attempt_id=attempt_id,
                            resources=attempt.get("resources"), condition_decision_id=receipt["decision_id"])
                self._queue.put_nowait(attempt)
            except BaseException:
                attempt.update(status="failed", error="dispatch_failed")
                self._active = None
                if self._budget:
                    self._budget.finish(attempt_id)
                self._loaded, self._phase, self._error = False, "failed", "dispatch_failed"
                raise
            return self._accepted(attempt)

    def _accepted(self, attempt):
        return copy.deepcopy({key: attempt[key] for key in ("job_id", "attempt_id", "instance_id", "status")}
                             | {"accepted": True, "result": attempt.get("result")})

    def cancel(self, payload):
        self._check_instance(payload, required=True)
        if not isinstance(payload.get("attempt_id"), str):
            raise ControlError("Invalid execution attempt", status=422)
        with self._lock:
            attempt = self._previous_attempt(payload.get("attempt_id"))
            if attempt is None or attempt["job_id"] != payload.get("job_id"):
                raise ControlError("Execution attempt not found", status=404)
            if attempt["status"] in TERMINAL:
                return self._accepted(attempt)
            if not attempt["cancel_requested"]:
                self._event("execution.cancel_requested", job_id=attempt["job_id"], attempt_id=attempt["attempt_id"])
                attempt.update(cancel_requested=True, status="cancel_requested")
                if "_cancel_event" in attempt:
                    attempt["_cancel_event"].set()
                    self._budget.set_cancelled(attempt["attempt_id"], True)
            return self._accepted(attempt)

    def drain(self, payload=None):
        self._check_instance(payload or {})
        with self._lock:
            if not self._draining:
                self._event("model.drain_requested")
                self._draining = True
                if self._phase not in {"failed", "stopped"}:
                    self._phase = "draining"
        return self.status()

    def stop(self, payload=None):
        self._check_instance(payload or {})
        with self._lock:
            if not self._stop_requested:
                self._event("model.stop_requested")
                self._stop_requested, self._draining = True, True
                if self._phase not in {"failed", "stopped"}:
                    self._phase = "draining"
        return self.status()

    def dispatch(self, operation, payload=None):
        payload = payload or {}
        if operation == "status":
            return self.status()
        if operation == "budget_audit":
            return self.budget_audit(payload)
        if operation == "execute":
            return self.execute(payload)
        if operation == "cancel":
            return self.cancel(payload)
        if operation == "drain":
            return self.drain(payload)
        if operation == "stop":
            return self.stop(payload)
        if operation == "ack_results":
            return self.ack_results(payload)
        raise ControlError("Unknown model operation", status=404)

    def ack_results(self, payload):
        self._check_instance(payload, required=True)
        identities = payload.get("attempt_ids")
        if not isinstance(identities, list) or any(not isinstance(x, str) or not IDENTIFIER.fullmatch(x) for x in identities):
            raise ControlError("Invalid result acknowledgement", status=422)
        with self._lock:
            acknowledged = []
            for identity in identities:
                attempt = self._attempts.get(identity)
                if attempt and attempt["status"] not in TERMINAL:
                    raise ControlError("Cannot acknowledge an unfinished execution", status=409)
                if identity not in self._results:
                    if self._previous_attempt(identity):
                        acknowledged.append(identity)
                    continue
                record = {key: value for key, value in attempt.items() if not key.startswith("_")}
                ledger = sqlite3.connect(self._ledger_path)
                try:
                    with ledger:
                        ledger.execute("INSERT OR REPLACE INTO acknowledged VALUES (?,?)", (identity, json.dumps(record)))
                finally:
                    ledger.close()
                self._results.pop(identity)
                self._attempts.pop(identity)
                if self._budget:
                    self._budget.acknowledge(identity)
                acknowledged.append(identity)
            return {"instance_id": self.instance_id, "acknowledged": acknowledged}

    def _revoke_pipeline(self, error, *, current=None):
        with self._lock:
            self._loaded, self._phase, self._error = False, "failed", error
            for attempt in self._live_attempts():
                if attempt is not current:
                    attempt["cancel_requested"] = True
                    attempt["_cancel_event"].set()
                    self._budget.set_cancelled(attempt["attempt_id"], True)

    @contextmanager
    def _pipeline_stage(self, attempt, name, lease):
        if name not in {"prepare", "gpu", "encode_validate"} or lease != ("gpu" if name == "gpu" else "cpu"):
            raise ValueError("Unknown pipeline stage or resource lease")
        cancellation = attempt["_cancel_event"]
        identity, job_id = attempt["attempt_id"], attempt["job_id"]
        witness = None
        # Never hold the model lock while waiting for a budget grant. Budget
        # callbacks read Event only and never acquire this model's lock.
        try:
            with self._budget.stage(identity, lease, cpu_threads=attempt["resources"]["cpu_threads"],
                                    cancellation_event=cancellation,
                                    context={"model_instance_id": self.instance_id, "job_id": job_id,
                                             "stage": name}) as acquired:
                with self._lock:
                    witness = self._budget.lease_witness(identity, lease)
                    reservation = self._budget.reservation_witness(identity)
                    if witness is not None:
                        self._event("lease.acquired", job_id=job_id, attempt_id=identity, stage=name,
                                    lease_id=witness.get("lease_id"), budget_instance_id=witness.get("budget_instance_id"),
                                    lease_witness=witness)
                    admission = attempt.get("condition_gates", {}).get("model.execute", {})
                    admission_context = admission.get("context", {})
                    previous = {"prepare": None, "gpu": "prepare", "encode_validate": "gpu"}[name]
                    live_state = attempt.get("status")
                    live = (None if type(live_state) is not str else False if live_state in TERMINAL else True if live_state in {
                        "accepted", "preparing", "gpu_wait", "gpu_running", "encoding", "validating", "cancel_requested"} else None)
                    observations = {
                        "instance_matches": attempt.get("instance_id") == self.instance_id,
                        "attempt_owned": (self._attempts.get(identity) is attempt
                            and admission_context.get("instance_id") == self.instance_id
                            and admission_context.get("job_id") == job_id and admission_context.get("attempt_id") == identity),
                        "attempt_live": live, "model_loaded": self._loaded,
                        "worker_alive": self._worker.is_alive() if self._worker is not None else False,
                        "capability_available": self._capability_observation(admitted=True),
                        "admission_confirmed": admission.get("allowed"),
                        "predecessor_completed": admission.get("allowed") if previous is None else attempt.get("stage_completions", {}).get(previous),
                        "stage_not_started": negate_observation(attempt.get("stage_started", {}).get(name)),
                        "cancel_flag_clear": negate_observation(attempt.get("cancel_requested")),
                        "cancel_event_clear": negate_observation(cancellation.is_set()),
                        "reservation_owned": _all_observed([
                            reservation.get("owned"), reservation.get("attempt_id") == identity,
                            reservation.get("budget_instance_id") == self._budget.instance_id,
                            type(reservation.get("reservation_id")) is str and bool(reservation["reservation_id"])
                        ]) if reservation else None,
                        "budget_grant_held": False if acquired is False else _all_observed([acquired,
                            witness.get("owned") if witness else (False if acquired is False else None),
                            witness.get("attempt_id") == identity if witness else None,
                            witness.get("kind") == lease if witness else None,
                            witness.get("budget_instance_id") == self._budget.instance_id if witness else None,
                            witness.get("reservation_id") == reservation.get("reservation_id") if witness and reservation else None]),
                    }
                    receipt = self._gate("model.stage." + name, STAGE_CONDITIONS, observations,
                        job_id=job_id, attempt_id=identity, attempt=attempt, stage=name,
                        lease_id=witness.get("lease_id") if witness else None,
                        budget_instance_id=witness.get("budget_instance_id") if witness else None,
                        budget_decision_id=witness.get("decision_id") if witness else None)
                    if receipt["allowed"] is not True:
                        # A confirmed cancellation is distinct from unknown or
                        # contradictory ownership/capability/precedence evidence.
                        non_cancel_unknown = any(state in {"unknown", "conflict"} for state in receipt["conditions"].values())
                        if (not non_cancel_unknown and attempt.get("cancel_requested") is True
                                and cancellation.is_set() is True):
                            raise ExecutionCancelled()
                        raise ConditionGateBlocked("Pipeline stage condition evidence is blocked")
                    attempt["stage"] = name
                    attempt["stage_started"][name] = True
                    attempt["status"] = {"prepare": "preparing", "gpu": "gpu_running", "encode_validate": "encoding"}[name]
                    stage_resources = dict(attempt["resources"], cpu_threads=(
                        self._budget.gpu_host_cpu_threads if lease == "gpu" else attempt["resources"]["cpu_threads"]))
                    event_context = {"condition_decision_id": receipt["decision_id"], "lease_id": witness["lease_id"],
                                     "budget_instance_id": witness["budget_instance_id"], "lease_witness": witness}
                    self._event("stage.started", job_id=job_id, attempt_id=identity, stage=name,
                                resources=stage_resources, **event_context)
                succeeded = False
                try:
                    yield
                    succeeded = True
                except BaseException:
                    if lease == "gpu":
                        self._revoke_pipeline("gpu_execution_failed", current=attempt)
                    raise
                finally:
                    with self._lock:
                        # A successfully persisted finish precedes the completion
                        # witness. An audit failure cannot unlock the next stage.
                        self._event("stage.finished", job_id=job_id, attempt_id=identity,
                                    stage=name, succeeded=succeeded, resources=stage_resources, **event_context)
                        attempt["stage_completions"][name] = succeeded
        finally:
            if witness is not None:
                with self._lock:
                    released = self._budget.release_witness(witness["lease_id"])
                    if released is None:
                        self._event("lease.release_unconfirmed", job_id=job_id, attempt_id=identity, stage=name,
                                    lease_id=witness["lease_id"], budget_instance_id=witness["budget_instance_id"])
                        raise ConditionGateBlocked("Lease release evidence is unavailable")
                    self._event("lease.released", job_id=job_id, attempt_id=identity, stage=name,
                                lease_id=witness["lease_id"], budget_instance_id=witness["budget_instance_id"],
                                lease_witness=released)

    def _run_pipeline_attempt(self, attempt):
        identity = attempt["attempt_id"]
        path = self.root / attempt["job_id"] / "result.mp4"
        result = {"job_id": attempt["job_id"], "attempt_id": identity, "instance_id": self.instance_id,
                  "status": "failed", "error": None, "artifact": None, "metrics": None,
                  "started_at": time.time(), "cleanup_complete": False}
        prepared = cpu_result = None
        try:
            parameters = copy.deepcopy(attempt["parameters"])
            for reference in parameters["references"]:
                reference["path"] = Path(reference["path"])
            with self._pipeline_stage(attempt, "prepare", "cpu"):
                prepared = self.backend.prepare(cpu_threads=attempt["resources"]["cpu_threads"], **parameters)
                data = getattr(getattr(prepared, "metrics", None), "data", {})
                if isinstance(data.get("generation_options"), dict):
                    with self._lock:
                        attempt["generation_options"] = data["generation_options"]
            with self._lock:
                if not attempt["cancel_requested"]:
                    attempt["status"], attempt["stage"] = "gpu_wait", "gpu_wait"
            with self._pipeline_stage(attempt, "gpu", "gpu"):
                cpu_result = self.backend.infer(prepared)
                if getattr(cpu_result, "cpu_handoff_complete", False) is not True:
                    raise RuntimeError("Backend did not confirm CPU handoff and CUDA synchronization")
                if isinstance(attempt.get("generation_options"), dict):
                    self._event("generation.configured", job_id=attempt["job_id"], attempt_id=identity,
                                generation_options=copy.deepcopy(attempt["generation_options"]))
            # CPU encoding and full-file decoding consume their CPU budget, but
            # no GPU lease. Another admitted job may now perform its inference.
            with self._pipeline_stage(attempt, "encode_validate", "cpu"):
                metrics = self.backend.encode(cpu_result, output_path=path)
                if isinstance(metrics, dict):
                    result["metrics"] = metrics
                with self._lock:
                    if attempt["cancel_requested"]:
                        raise ExecutionCancelled()
                    attempt["status"] = "validating"
                if self.artifact_validator is verify_artifact:
                    result["artifact"] = self.artifact_validator(path, attempt["parameters"],
                        cpu_threads=attempt["resources"]["cpu_threads"])
                else:
                    result["artifact"] = self.artifact_validator(path, attempt["parameters"])
            result["status"] = "succeeded"
            with self._lock:
                self._consecutive_failures = 0
        except ExecutionCancelled:
            result["status"] = "cancelled"
        except BaseException as error:
            LOG.exception("Pipeline attempt failed: %s/%s", attempt["job_id"], identity)
            result["error"] = ("condition_gate_blocked" if isinstance(error, ConditionGateBlocked)
                               else "generation_failed; inspect model log")
            with self._lock:
                if not isinstance(error, ValueError):
                    self._consecutive_failures += 1
                fatal = not isinstance(error, Exception) or self._consecutive_failures >= 2
            if fatal:
                self._revoke_pipeline("model_execution_unavailable", current=attempt)
        finally:
            try:
                if isinstance(attempt.get("generation_options"), dict):
                    result["generation_options"] = copy.deepcopy(attempt["generation_options"])
                result["condition_gates"] = copy.deepcopy(attempt.get("condition_gates", {}))
                result["stage_completions"] = copy.deepcopy(attempt.get("stage_completions", {}))
                # This operation is strictly CPU cleanup; it must not empty the
                # CUDA cache while the next attempt owns the GPU lease.
                self.backend.cleanup(prepared=prepared, result=cpu_result)
                prepared = cpu_result = None
                if result["metrics"] is not None:
                    try:
                        encoded = json.dumps(result["metrics"], allow_nan=False)
                        result["metrics"] = json.loads(encoded) if len(encoded.encode()) <= 128 * 1024 else None
                    except (TypeError, ValueError):
                        result["metrics"] = None
                        LOG.warning("Pipeline metrics were not finite JSON; omitting diagnostics")
                inputs = self.root / attempt["job_id"] / "inputs"
                if inputs.is_symlink() or inputs.resolve() != inputs:
                    raise ValueError("Input ownership changed during pipeline execution")
                if inputs.exists():
                    shutil.rmtree(inputs)
                result["cleanup_complete"] = True
            except BaseException:
                LOG.exception("Pipeline attempt cleanup failed")
                result.update(status="failed", artifact=None, cleanup_complete=False,
                              error="execution_cleanup_failed; inspect model log")
                self._revoke_pipeline("execution_cleanup_failed", current=attempt)
            with self._lock:
                if attempt["cancel_requested"] and result["status"] == "succeeded":
                    result["status"] = "cancelled"
                if result["status"] != "succeeded":
                    result["artifact"] = None
                    try:
                        path.unlink(missing_ok=True)
                        path.with_suffix(".metrics.json").unlink(missing_ok=True)
                        path.with_suffix(".metrics.json.tmp").unlink(missing_ok=True)
                    except OSError:
                        result.update(status="failed", cleanup_complete=False, error="cancel_cleanup_failed")
                        self._revoke_pipeline("cancel_cleanup_failed", current=attempt)
                result["finished_at"] = time.time()
                attempt.update(status=result["status"], result=result)
                attempt.pop("parameters", None)
                self._results[identity] = self._last_result = result
                if result["cleanup_complete"]:
                    try:
                        self._budget.finish(identity, len(json.dumps(result).encode()))
                    except BudgetUnavailable:
                        result.update(status='failed', artifact=None, cleanup_complete=False,
                                      error='lease_release_unconfirmed')
                        attempt['status'] = 'failed'
                        self._revoke_pipeline('lease_release_unconfirmed', current=attempt)
                if self._phase != "failed":
                    self._phase = "draining" if self._draining else ("running" if self._live_attempts() else "ready")
                self._safe_event("execution." + result["status"], job_id=attempt["job_id"],
                                 attempt_id=identity, result=result)

    def _run_attempt(self, attempt):
        path = self.root / attempt["job_id"] / "result.mp4"
        result = {"job_id": attempt["job_id"], "attempt_id": attempt["attempt_id"],
                  "instance_id": self.instance_id, "status": "failed", "error": None,
                  "artifact": None, "metrics": None, "started_at": time.time(),
                  "cleanup_complete": False}
        fatal = None
        generated = False
        try:
            with self._lock:
                if not attempt["cancel_requested"]:
                    attempt["status"] = "running"
                self._event("execution.running", job_id=attempt["job_id"], attempt_id=attempt["attempt_id"])
            if not attempt["cancel_requested"]:
                params = copy.deepcopy(attempt["parameters"])
                for reference in params["references"]:
                    reference["path"] = Path(reference["path"])
                self.backend.generate(output_path=path, **params)
                generated = True
                if not attempt["cancel_requested"]:
                    result["artifact"] = self.artifact_validator(path, attempt["parameters"])
            with self._lock:
                result["status"] = "cancelled" if attempt["cancel_requested"] else "succeeded"
            self._consecutive_failures = 0
        except BaseException as error:
            LOG.exception("Model attempt failed: %s/%s", attempt["job_id"], attempt["attempt_id"])
            result["error"] = "generation_failed; inspect model log"
            # Bad input media may be a request error. Invalid generated output
            # is an execution failure even when the validator raises ValueError.
            if generated or not isinstance(error, ValueError):
                self._consecutive_failures += 1
            if not isinstance(error, Exception) or self._consecutive_failures >= 2:
                fatal = error
                with self._lock:
                    self._loaded, self._phase, self._error = False, "failed", "model_execution_unavailable"
        finally:
            try:
                # Keep the lease until execution and its local cleanup have ended.
                metrics = getattr(self.backend, "last_metrics", None)
                if isinstance(metrics, dict):
                    try:
                        encoded = json.dumps(metrics, allow_nan=False)
                        if len(encoded.encode("utf-8")) <= 128 * 1024:
                            result["metrics"] = json.loads(encoded)
                    except (TypeError, ValueError):
                        LOG.warning("Model metrics were not finite JSON; omitting diagnostics")
                if result["status"] != "succeeded":
                    result["artifact"] = None
                    path.unlink(missing_ok=True)
                inputs = self.root / attempt["job_id"] / "inputs"
                if inputs.is_symlink() or inputs.resolve() != inputs:
                    raise ValueError("Input directory ownership changed during execution")
                if inputs.exists():
                    shutil.rmtree(inputs)
                # Attempt-local files are reclaimed; this does not release the
                # model process's persistent CPU/GPU resource ownership.
                result["cleanup_complete"] = True
            except BaseException as error:
                LOG.exception("Attempt cleanup failed")
                result.update(status="failed", artifact=None, cleanup_complete=False,
                              error="execution_cleanup_failed; inspect model log")
                fatal = error
                with self._lock:
                    self._loaded, self._phase, self._error = False, "failed", "execution_cleanup_failed"
            with self._lock:
                # A cancel racing with validation/cleanup still wins before commit.
                if attempt["cancel_requested"] and result["status"] == "succeeded":
                    result.update(status="cancelled", artifact=None)
                    try:
                        path.unlink(missing_ok=True)
                    except OSError as error:
                        result.update(status="failed", error="cancel_cleanup_failed", cleanup_complete=False)
                        fatal = error
                        self._loaded, self._phase, self._error = False, "failed", "cancel_cleanup_failed"
                result["finished_at"] = time.time()
                attempt.update(status=result["status"], result=result)
                attempt.pop("parameters", None)
                self._last_result = result
                self._results[attempt["attempt_id"]] = result
                self._active = None
                if self._phase != "failed":
                    self._phase = "draining" if self._draining else "ready"
                self._event("execution." + result["status"], job_id=attempt["job_id"],
                            attempt_id=attempt["attempt_id"], result=result)
        if fatal is not None:
            raise fatal

    def _work(self):
        try:
            if self.backend is None:
                if self.staged:
                    from staged_backend import StagedMiniMaxBackend
                    self.backend = StagedMiniMaxBackend()
                else:
                    from model_backend import MiniMaxBackend
                    self.backend = MiniMaxBackend()
            self._pipeline = all(callable(getattr(self.backend, name, None)) for name in
                                 ("prepare", "infer", "encode", "cleanup", "estimate_resources"))
            if not self._pipeline and not self._injected_backend:
                raise RuntimeError("Production backend must implement the staged pipeline contract")
            if self._pipeline:
                self._budget = PipelineBudget(self.root, limits=self._resource_limits,
                                              audit_directory=self.event_directory / 'budget-audit',
                    model_peak_ram_bytes=getattr(self.backend, "pipeline_model_peak_ram_bytes", 144 * 1024 ** 3))
                configure = getattr(self.backend, "configure_cpu_budget", None)
                if callable(configure):
                    self._budget.native_cpu_status = configure(self._budget.gpu_host_cpu_threads)
                elif not self._injected_backend:
                    raise RuntimeError("Production backend must configure its native CPU thread budget before loading")
                else:
                    self._budget.native_cpu_status = {"configured": False, "scope": "injected CPU test backend"}
            self.backend.load()
            if self._pipeline:
                native_snapshot = getattr(self.backend, "cpu_budget_snapshot", None)
                if callable(native_snapshot):
                    # Model loading can import additional BLAS/OpenMP runtimes.
                    # Read back their limits before publishing readiness; never
                    # reconfigure process-wide pools while jobs are running.
                    self._budget.native_cpu_status = native_snapshot()
                elif not self._injected_backend:
                    raise RuntimeError("Production backend must verify native CPU budgets after loading")
            with self._lock:
                self._loaded = True
                self._phase = "draining" if self._draining else "ready"
                self._event("model.ready", accepting=not self._draining)
            while True:
                with self._lock:
                    if self._pipeline:
                        for identity, worker in list(self._stage_workers.items()):
                            if not worker.is_alive():
                                worker.join()
                                del self._stage_workers[identity]
                        if self._phase == "failed":
                            self._revoke_pipeline(self._error or "pipeline_unavailable")
                        if (self._stop_requested or self._phase == "failed") and not self._live_attempts() and not self._stage_workers:
                            break
                    else:
                        if self._stop_requested and self._active is None:
                            break
                        if self._phase == "failed":
                            break
                try:
                    attempt = self._queue.get(timeout=.1)
                except queue.Empty:
                    continue
                try:
                    if self._pipeline:
                        worker = threading.Thread(target=self._run_pipeline_attempt, args=(attempt,),
                                                  name="h3-stage-" + attempt["attempt_id"][:8], daemon=False)
                        with self._lock:
                            self._stage_workers[attempt["attempt_id"]] = worker
                            try:
                                worker.start()
                            except BaseException:
                                self._stage_workers.pop(attempt["attempt_id"], None)
                                self._revoke_pipeline("pipeline_worker_start_failed")
                                self._run_pipeline_attempt(attempt)
                                raise
                    else:
                        self._run_attempt(attempt)
                finally:
                    self._queue.task_done()
        except BaseException as error:
            LOG.exception("Model worker exited")
            with self._lock:
                self._loaded, self._phase = False, "failed"
                self._error = self._error or "model_load_or_worker_failed"
                if self._active is not None:
                    attempt = self._active
                    result = {"job_id": attempt["job_id"], "attempt_id": attempt["attempt_id"],
                              "instance_id": self.instance_id, "status": "failed", "artifact": None,
                              "metrics": None, "error": "model_worker_exited", "finished_at": time.time(),
                              "cleanup_complete": False}
                    attempt.update(status="failed", result=result)
                    self._results[attempt["attempt_id"]] = result
                    self._last_result, self._active = result, None
                self._safe_event("model.failed", error=self._error, error_type=type(error).__name__)
        finally:
            with self._lock:
                self._loaded = False
            if self._pipeline:
                # Never unload model state while prepare, CUDA inference, CPU
                # encoding, validation or local cleanup still refers to it.
                for worker in list(self._stage_workers.values()):
                    if worker.ident is not None:
                        worker.join()
                for attempt in self._live_attempts():
                    attempt["cancel_requested"] = True
                    attempt["_cancel_event"].set()
                    self._budget.set_cancelled(attempt["attempt_id"], True)
                    self._run_pipeline_attempt(attempt)
            try:
                backend, self.backend = self.backend, None
                unload = getattr(backend, "unload", None)
                if unload:
                    unload()
                if hasattr(backend, "pipe"):
                    backend.pipe = None
                del backend
                gc.collect()
                torch = sys.modules.get("torch")
                if torch is not None and torch.cuda.is_initialized():
                    torch.cuda.empty_cache()
                final_budget = self._budget.snapshot(include_history=False) if self._budget is not None else None
                self._worker_cleanup_complete = (final_budget is None or
                    (final_budget['active_leases'] == [] and final_budget['lease_journal']['complete'] is True))
                if not self._worker_cleanup_complete:
                    self._phase, self._error = 'failed', 'lease_cleanup_unconfirmed'
            except BaseException:
                LOG.exception("Model worker cleanup failed")
                self._phase, self._error = "failed", "model_unload_failed"
            with self._lock:
                if self._phase != "failed":
                    self._phase = "stopped"
                self._safe_event("model.worker_exited", phase=self._phase,
                                 worker_cleanup_complete=self._worker_cleanup_complete,
                                 budget_snapshot=self._budget.snapshot() if self._budget is not None else None)
            self._done.set()

    def join(self, timeout=None):
        if self._worker is not None and self._worker.ident is not None:
            self._worker.join(timeout)
        return self._done.is_set()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--staged", action="store_true")
    parser.add_argument("--parent-pid", type=int)
    args = parser.parse_args()
    owner_created_at = parent_identity(args.parent_pid) if args.parent_pid else None
    key = os.environ.get("H3_CONTROL_KEY", "")
    if len(key) < 32:
        raise RuntimeError("A private H3_CONTROL_KEY is required")
    container = ModelContainer(args.root, args.instance_id, staged=args.staged,
                               progress_path=os.environ.get("H3_DOWNLOAD_PROGRESS_PATH"))
    server = make_server("127.0.0.1", args.port, key, container.dispatch)
    http = threading.Thread(target=server.serve_forever, name="h3-model-control", daemon=True)
    http.start()
    stop_pending = False
    stop_applied = False

    def request_stop(*_):
        # Python signal handlers may interrupt this thread while it holds an
        # RLock or is writing an event. Only record intent here; even Event.set
        # takes a lock and must not run in a reentrant signal callback.
        nonlocal stop_pending
        stop_pending = True

    def apply_stop():
        nonlocal stop_applied
        if stop_applied:
            return
        stop_applied = True
        try:
            container.stop()
        except BaseException:
            LOG.exception("Could not record model stop")
            with container._lock:
                container._stop_requested = container._draining = True
                container._loaded = False

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        container.start()
        while True:
            if args.parent_pid and not parent_is_alive(args.parent_pid, owner_created_at):
                # An ownerless CUDA worker must not outlive its resource lease.
                apply_stop()
                container.join(2)
                server.shutdown()
                server.server_close()
                os._exit(70)
            if stop_pending:
                apply_stop()
            if container._stop_requested:
                if container.join(.2):
                    break
            else:
                # Failed loading leaves the private controller available for diagnosis.
                time.sleep(.2)
    finally:
        apply_stop()
        container.join(2)
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
