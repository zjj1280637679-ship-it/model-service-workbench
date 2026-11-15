"""Auditable experiment sampler. It never starts/stops the model service.

RuntimeCollector(host, output_directory).start()/status()/stop() own sampling
threads only. export_snapshot() writes trace.json after sampling has stopped.
Local observations can contain paths/prompts/URLs: review before publication.
"""
from __future__ import annotations

import csv
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sqlite3
import subprocess
import threading
import time
import uuid

MAX_BYTES = 64 * 1024 * 1024
MAX_RECORDS = 100000
ID = re.compile(r"[0-9a-f]{32}\Z")
SOURCE_ALLOWLIST = (
    "condition_guard.py", "condition_projection.py", "queue_policy.py",
    "adapter_config.py", "client.py", "console.html", "control_protocol.py", "download_progress.py",
    "generation_options.py", "global_manager.py", "lora_preflight.py", "model_backend.py", "model_container.py",
    "notebook_console.py", "notebook_ui.py", "performance_model.py", "pipeline_backend.py", "pipeline_scheduler.py",
    "pipeline_smoke_test.py", "preflight.py", "process_owner.py", "requirements-colab.txt", "requirements-lora.txt",
    "requirements.txt", "resource_scheduler.py", "run.py", "runtime_options.py", "server.py", "smoke_test.py",
    "staged_backend.py", "staged_server.py", "startup.py", "tunnel.py",
)


def redact(value, secrets=()):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            normalized = re.sub(r"[^a-z0-9]", "", str(key).lower())
            sensitive = any(part in normalized for part in ("apikey", "controlkey", "authorization", "password", "secret", "accesstoken", "bearer"))
            result[str(key)] = "[REDACTED]" if sensitive else redact(item, secrets)
        return result
    if isinstance(value, (list, tuple)):
        return [redact(item, secrets) for item in value]
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[REDACTED]")
        return re.sub(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+", "Bearer [REDACTED]", value)
    return value


def atomic_json(path, document):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def owned_file(base, relative):
    base = Path(base).resolve()
    lexical = base / relative
    resolved = lexical.resolve()
    if resolved != lexical or not resolved.is_relative_to(base) or not resolved.is_file():
        raise ValueError("missing file or path escapes the declared owned directory")
    if resolved.stat().st_size > MAX_BYTES:
        raise ValueError("source exceeds bounded export size")
    return resolved


def read_jsonl(path):
    with path.open("rb") as stream:
        data = stream.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError("source exceeds bounded export size")
    records = []
    for line in data.decode("utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("JSONL observation must be an object")
        records.append(value)
        if len(records) > MAX_RECORDS:
            raise ValueError("record count exceeds export budget")
    return records


def source_identities(directory):
    files, unavailable = {}, []
    for name in SOURCE_ALLOWLIST:
        try:
            if directory is None:
                raise ValueError("source directory unknown")
            path = owned_file(directory, name)
            files[name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
        except (OSError, ValueError) as error:
            unavailable.append({"file": name, "error_type": type(error).__name__})
    return {"files": files, "unavailable": unavailable,
            "scope": "Exact allowlisted on-disk runtime code identities at export, not proof already-imported memory matches these bytes."}


def gpu_observation():
    result = subprocess.run(["nvidia-smi", "--query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu,power.draw",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True, timeout=2)
    if len(result.stdout) > 65536:
        raise ValueError("unexpectedly large GPU telemetry")
    rows = []
    for row in csv.reader(io.StringIO(result.stdout)):
        if len(row) != 7:
            raise ValueError("GPU telemetry has an unexpected field count")
        stamp, index, name, used, total, utilization, power = [item.strip() for item in row]

        def number(value):
            try:
                converted = float(value)
                return converted if math.isfinite(converted) else None
            except ValueError:
                return None

        rows.append({"device_timestamp": stamp, "index": index, "name": name,
                     "memory_used_mib": number(used), "memory_total_mib": number(total),
                     "utilization_gpu_percent": number(utilization), "power_watts": number(power)})
    return {"devices": rows, "source": "nvidia-smi", "synchronizes_cuda": False,
            "interpretation": "Periodic device telemetry; vendor sampling windows, not per-job occupancy or kernel timings."}


class RuntimeCollector:
    def __init__(self, host, output_directory, interval=1.0):
        if type(interval) not in (int, float) or not math.isfinite(interval) or not 0.1 <= interval <= 60:
            raise ValueError("interval must be between 0.1 and 60 seconds")
        if not isinstance(host.instance_id, str) or not ID.fullmatch(host.instance_id):
            raise ValueError("host requires its own 32-character manager identity")
        self.host, self.instance_id = host, host.instance_id
        self.collector_id = uuid.uuid4().hex
        self.directory = Path(output_directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / ".collector-owner").open("x", encoding="ascii") as owner:
            owner.write(self.collector_id)
        self.interval = float(interval)
        self._secrets = tuple(value for value in (getattr(host, "api_key", None), getattr(host, "_control_key", None)) if isinstance(value, str) and value)
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._threads = []
        self._audit_capture_lock = threading.Lock()
        self._audit_cursors = {}
        self._state = {"schema_version": 1, "collector_id": self.collector_id, "manager_instance_id": self.instance_id,
                       "phase": "created", "snapshot_samples": 0, "gpu_samples": 0, "probe_errors": 0,
                       "started_at": None, "stopped_at": None, "threads": {}, "last_error_type": None,
                       "budget_audit_pages": 0, "budget_audit_errors": 0,
                       "interval_seconds": self.interval, "gpu_interval_seconds": 2.0,
                       "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       "scope": "Observation only; owns sampler threads/files, never service termination or cloud release."}
        self._persist()
        self._audit("collector.created")

    def _persist(self):
        atomic_json(self.directory / "collector-state.json", redact(self._state, self._secrets))

    def _append(self, name, document):
        path = self.directory / name
        if path.exists() and path.stat().st_size >= MAX_BYTES:
            raise RuntimeError("collector output size budget reached")
        with path.open("a", encoding="utf-8", newline="\n") as output:
            output.write(json.dumps(redact(document, self._secrets), ensure_ascii=False, allow_nan=False) + "\n")

    def _audit(self, kind, **data):
        with self._lock:
            self._append("collector-events.jsonl", {"type": kind, "collector_id": self.collector_id,
                "manager_instance_id": self.instance_id, "monotonic_ns": time.monotonic_ns(),
                "observed_at": datetime.now(timezone.utc).isoformat(), **data})

    def _probe(self, kind, operation, destination, interval):
        try:
            while not self._stop.is_set():
                started = time.monotonic_ns()
                try:
                    observation = operation()
                    if not isinstance(observation, dict):
                        raise ValueError("probe must return an object")
                    if kind == "snapshot" and observation.get("instance_id") != self.instance_id:
                        raise ValueError("host snapshot belongs to another manager instance")
                    envelope = {"collector_id": self.collector_id, "manager_instance_id": self.instance_id,
                        "monotonic_ns": started, "finished_monotonic_ns": time.monotonic_ns(),
                        "observed_at": datetime.now(timezone.utc).isoformat(), "observation": observation}
                    envelope["probe_duration_seconds"] = (envelope["finished_monotonic_ns"] - started) / 1e9
                    with self._lock:
                        self._append(destination, envelope)
                        self._state[kind + "_samples"] += 1
                        self._persist()
                except Exception as error:
                    with self._lock:
                        self._state["probe_errors"] += 1
                        self._state["last_error_type"] = type(error).__name__
                        self._audit("probe.failed", probe=kind, error_type=type(error).__name__,
                                    probe_duration_seconds=(time.monotonic_ns() - started) / 1e9)
                        self._persist()
                elapsed = (time.monotonic_ns() - started) / 1e9
                self._stop.wait(max(0.05, interval - elapsed))
        except Exception as error:
            with self._lock:
                self._state["phase"] = "failed"
                self._state["last_error_type"] = type(error).__name__
                self._stop.set()
                try:
                    self._persist()
                except OSError:
                    pass
        finally:
            with self._lock:
                self._state["threads"][kind] = "exited"
                if all(value == "exited" for value in self._state["threads"].values()):
                    if self._state["phase"] != "failed":
                        self._state["phase"] = "stopped"
                    self._state["stopped_at"] = datetime.now(timezone.utc).isoformat()
                try:
                    self._audit("sampler.exited", probe=kind)
                    self._persist()
                except (OSError, RuntimeError):
                    pass

    def start(self):
        with self._lock:
            if self._state["phase"] == "running":
                return self
            if self._state["phase"] != "created":
                raise RuntimeError("A stopped collector cannot be restarted; use a new observation identity")
            self._state.update(phase="running", started_at=datetime.now(timezone.utc).isoformat(),
                               threads={"snapshot": "running", "gpu": "running"})
            probes = [
                ("snapshot", self.host.snapshot, "system_snapshots.jsonl", self.interval),
                ("gpu", gpu_observation, "gpu_samples.jsonl", 2.0),
            ]
            if callable(getattr(self.host, 'budget_audit', None)):
                self._state['threads']['budget'] = 'running'
                self._state['budget_samples'] = 0
                probes.append(('budget', lambda: self.capture_budget_audit(max_pages=4),
                               'budget_audit_samples.jsonl', self.interval))
            for kind, operation, filename, interval in probes:
                thread = threading.Thread(target=self._probe, args=(kind, operation, filename, interval),
                                          name="research-" + kind + "-" + self.collector_id[:8], daemon=True)
                self._threads.append(thread)
                thread.start()
            self._audit("collector.started")
            self._persist()
        return self

    def capture_budget_audit(self, snapshot=None, *, max_pages=1024):
        """Capture a fixed cutoff through the private owner-controlled interface.

        A separate sampler avoids serializing status/GPU probes with disk audit
        reads. Call explicitly after the final task releases and before shutdown.
        Advancing a cursor means the page was persisted, not that the lifetime is
        complete; the final worker receipt remains the completeness authority.
        """
        if type(max_pages) is not int or not 1 <= max_pages <= 1024:
            raise ValueError('audit capture page budget must be 1..1024')
        operation = getattr(self.host, 'budget_audit', None)
        if not callable(operation):
            return {'state': 'unavailable', 'reason': 'private_audit_interface_missing'}
        with self._audit_capture_lock:
            try:
                snapshot = self.host.snapshot() if snapshot is None else snapshot
                model = (snapshot.get('modules') or {}).get('model') or {}
                budget = model.get('budget') or {}
                journal = budget.get('lease_journal') or {}
                model_id, budget_id = model.get('instance_id'), budget.get('budget_instance_id')
                if not budget:
                    return {'state': 'waiting', 'reason': 'budget_not_created'}
                if (snapshot.get('instance_id') != self.instance_id or snapshot.get('model_instance_id') != model_id
                        or any(not isinstance(value, str) or not ID.fullmatch(value) for value in (model_id, budget_id))
                        or journal.get('budget_instance_id') != budget_id or journal.get('complete') is not True
                        or type(journal.get('count')) is not int or journal['count'] < 0):
                    raise ValueError('budget audit owner or journal is unconfirmed')
                key = model_id, budget_id
                cursor = self._audit_cursors.get(key, 0)
                cutoff = journal['count']
                if cursor > cutoff:
                    raise ValueError('budget audit durable count regressed')
                captured = 0
                while cursor < cutoff and captured < max_pages:
                    page = operation(model_instance_id=model_id, budget_instance_id=budget_id,
                                     after_sequence=cursor, through_sequence=cutoff, limit=128)
                    next_seq = page.get('next_sequence')
                    if (page.get('model_instance_id') != model_id or page.get('budget_instance_id') != budget_id
                            or page.get('after_sequence') != cursor or page.get('through_sequence') != cutoff
                            or page.get('page_complete') is not True or type(next_seq) is not int
                            or not cursor < next_seq <= min(cursor + 128, cutoff)
                            or page.get('tail_sha256') != journal.get('tail_sha256')
                            or page.get('more') is not (next_seq < cutoff)):
                        raise ValueError('budget audit response identity, range or cutoff conflicts')
                    with self._lock:
                        self._append('budget_audit_pages.jsonl', page)
                        self._audit_cursors[key] = cursor = next_seq
                        self._state['budget_audit_pages'] += 1
                        self._persist()
                    captured += 1
                return {'state': 'cutoff_captured' if cursor == cutoff else 'partial',
                        'model_instance_id': model_id, 'budget_instance_id': budget_id,
                        'after_sequence': cursor, 'through_sequence': cutoff, 'pages': captured}
            except Exception as error:
                with self._lock:
                    self._state['budget_audit_errors'] += 1
                    self._audit('budget_audit.failed', error_type=type(error).__name__)
                    self._persist()
                raise

    def status(self):
        with self._lock:
            return redact({**self._state, "threads_alive": sum(thread.is_alive() for thread in self._threads),
                           "stop_requested": self._stop.is_set()}, self._secrets)

    def stop(self, timeout=5.0):
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 <= timeout <= 30:
            raise ValueError("stop timeout must be between 0 and 30 seconds")
        with self._lock:
            if self._state["phase"] == "created":
                self._state.update(phase="stopped", stopped_at=datetime.now(timezone.utc).isoformat())
            elif self._state["phase"] == "running":
                self._state["phase"] = "stopping"
            self._stop.set()
            self._audit("collector.stop_requested")
            self._persist()
        deadline = time.monotonic() + timeout
        for thread in self._threads:
            thread.join(max(0, deadline - time.monotonic()))
        with self._lock:
            if any(thread.is_alive() for thread in self._threads) and self._state["phase"] != "failed":
                self._state["phase"] = "stop_pending"
                self._audit("collector.stop_pending")
            self._persist()
        return self.status()

    def export_snapshot(self, *, queued_cancel_job_ids=(), cleanup_receipt=None, platform_release=None):
        """Export only selected observations. Before service exit, streams remain partial."""
        if any(thread.is_alive() for thread in self._threads):
            raise RuntimeError("Stop collector threads before exporting a stable snapshot")
        targets = list(queued_cancel_job_ids)
        if any(not isinstance(value, str) or not ID.fullmatch(value) for value in targets):
            raise ValueError("queued cancellation targets must be explicit job identities")
        problems = []
        envelopes = read_jsonl(self.directory / "system_snapshots.jsonl") if (self.directory / "system_snapshots.jsonl").exists() else []
        snapshots = [value["observation"] for value in envelopes]
        try:
            current = self.host.snapshot()
            if current.get("instance_id") != self.instance_id:
                raise ValueError("final snapshot identity mismatch")
            snapshots.append(redact(current, self._secrets))
        except Exception as error:
            problems.append({"source": "final_snapshot", "error_type": type(error).__name__})
            current = snapshots[-1] if snapshots else {}
        models = list(dict.fromkeys(snap.get("model_instance_id") for snap in snapshots if snap.get("model_instance_id")))
        model_id = models[-1] if models else None
        if len(models) > 1:
            problems.append({"source": "model_identity", "reason": "multiple_model_lifetimes_export_selects_latest_only"})
        manager_events, jobs, model_events = [], [], []
        try:
            db = owned_file(self.host.root, "control.sqlite3")
            with closing(sqlite3.connect(db.as_uri() + "?mode=ro", uri=True, timeout=2)) as connection:
                connection.execute("PRAGMA query_only=ON")
                connection.execute("BEGIN")
                rows = connection.execute("SELECT sequence,record FROM events ORDER BY sequence LIMIT ?", (MAX_RECORDS + 1,)).fetchall()
                if len(rows) > MAX_RECORDS:
                    raise ValueError("manager event export exceeds row budget")
                for sequence, record in rows:
                    value = json.loads(record)
                    if value.get("instance_id") == self.instance_id:
                        manager_events.append({**value, "sequence": sequence})
                job_ids = {value.get("job_id") for value in manager_events if value.get("job_id")}
                rows = connection.execute("SELECT id,record FROM jobs LIMIT ?", (MAX_RECORDS + 1,)).fetchall()
                if len(rows) > MAX_RECORDS:
                    raise ValueError("job export exceeds row budget")
                jobs = [{**{key: value for key, value in json.loads(record).items() if key != "parameters"},
                         "instance_id": self.instance_id}
                        for job_id, record in rows if job_id in job_ids]
        except Exception as error:
            problems.append({"source": "manager_database", "error_type": type(error).__name__})
        if model_id and isinstance(model_id, str) and ID.fullmatch(model_id):
            try:
                model_events = read_jsonl(owned_file(self.host.root, "_model_runtime/" + model_id + "/events.jsonl"))
            except Exception as error:
                problems.append({"source": "model_events", "error_type": type(error).__name__})
        else:
            problems.append({"source": "model_identity", "reason": "unknown_or_invalid"})
        if cleanup_receipt is None:
            cleanup_receipt = {"instance_id": self.instance_id, "state": current,
                               "owner_observation": current.get("manager_process_tree", {})}
        receipt_state = cleanup_receipt.get("state", {})
        owner = cleanup_receipt.get("owner_observation", {})
        final_owner = (cleanup_receipt.get("instance_id") == self.instance_id
                       and receipt_state.get("instance_id") == self.instance_id
                       and receipt_state.get("shutdown_complete") is True
                       and receipt_state.get("cleanup_complete") is True
                       and all(receipt_state.get(key) is False for key in ("manager_alive", "server_alive", "model_alive", "tunnel_alive"))
                       and owner.get("alive") is False and owner.get("cleanup_complete") is True and owner.get("errors") == [])

        def complete(events, instance, last_type):
            return (final_owner and bool(events) and all(event.get("instance_id") == instance for event in events)
                    and [event.get("sequence") for event in events] == list(range(1, len(events) + 1))
                    and events[-1].get("type") == last_type)

        manager_complete = complete(manager_events, self.instance_id, "manager.shutdown_complete")
        model_complete = complete(model_events, model_id, "model.worker_exited") and model_events[-1].get("worker_cleanup_complete") is True
        trace = {"schema_version": 1, "manager_instance_id": self.instance_id, "model_instance_id": model_id,
                 "manager_events": manager_events, "model_events": model_events, "jobs": jobs,
                 "system_snapshots": snapshots, "queued_cancel_job_ids": targets, "cleanup_receipt": cleanup_receipt,
                 "runtime_source_identity": source_identities(getattr(self.host, "directory", None)),
                 "collection": {"manager_events_complete": manager_complete, "model_events_complete": model_complete,
                                "manager_event_sequence_final": manager_events[-1].get("sequence") if manager_complete else None,
                                "model_event_sequence_final": model_events[-1].get("sequence") if model_complete else None,
                                "basis": "Continuous declared lifetime, terminal audit event and independent matching process-owner exit observation.",
                                "problems": problems, "collector_id": self.collector_id}}
        page_path = self.directory / 'budget_audit_pages.jsonl'
        if page_path.exists():
            try:
                trace['budget_audit_pages'] = [page for page in read_jsonl(page_path)
                                               if page.get('model_instance_id') == model_id]
            except Exception as error:
                problems.append({'source': 'budget_audit_pages', 'error_type': type(error).__name__})
        if platform_release is not None:
            trace["platform_release"] = platform_release
        trace = redact(trace, self._secrets)
        atomic_json(self.directory / "runtime-source-manifest.json", trace["runtime_source_identity"])
        atomic_json(self.directory / "trace.json", trace)
        receipt = {"schema_version": 1, "collector_id": self.collector_id, "trace_sha256": hashlib.sha256((self.directory / "trace.json").read_bytes()).hexdigest(),
                   "counts": {"snapshots": len(snapshots), "jobs": len(jobs), "manager_events": len(manager_events), "model_events": len(model_events)},
                   "collection": trace["collection"], "public_safe": False,
                   "note": "Credential-redacted local observations: review prompts, URLs, paths and provenance before publication. No raw DB, model files or runtime logs copied."}
        atomic_json(self.directory / "export-receipt.json", receipt)
        self._audit("collector.exported", trace_sha256=receipt["trace_sha256"], counts=receipt["counts"])
        return receipt
