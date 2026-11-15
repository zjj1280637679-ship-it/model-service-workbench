"""Durable main-event owner for independent API and native-BF16 model processes."""
from __future__ import annotations

import argparse
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.request
import uuid

import psutil

from control_protocol import ControlError, make_server, rpc_call
from process_owner import ProcessOwner, stop_many
from generation_options import requested_options, validate_generation_options
from adapter_config import load_adapter_settings
from condition_guard import evaluate_gate, observation_state
from condition_projection import public_condition_fields, public_condition_summary, public_budget_page, public_execution_summary
from queue_policy import QueuePolicy, guaranteed_resource_deferral

LOG = logging.getLogger("h3_manager")
TERMINAL = {"succeeded", "failed", "cancelled"}
NONTERMINAL = {"queued", "running", "cancel_requested"}
IDENTIFIER = re.compile(r"[a-f0-9]{32}")
DRIVE_FILE_ID = re.compile(r"[A-Za-z0-9_-]{1,128}")
SHA256 = re.compile(r"[a-f0-9]{64}")
GATE_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
DISPATCH_CONDITIONS = ("manager_open", "job_queued", "attempt_unassigned", "model_identity_current",
                       "model_observation_fresh", "model_process_alive", "model_ready", "worker_alive",
                       "worker_accepting", "supervisor_alive")


class RootLock:
    """An OS-held lock, automatically released if the owner process dies."""

    def __init__(self, path):
        self.file = open(path, "a+b")
        try:
            if os.fstat(self.file.fileno()).st_size == 0:
                self.file.write(b"0")
                self.file.flush()
            self.file.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, IOError) as error:
            self.file.close()
            raise RuntimeError("Another manager owns this output directory") from error

    def close(self):
        if self.file.closed:
            return
        self.file.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(self.file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(self.file.fileno(), fcntl.LOCK_UN)
        self.file.close()


def probe_api(url, instance_id, timeout=2):
    try:
        req = urllib.request.Request(url.rstrip("/") + "/health/live",
                                     headers={"Cache-Control": "no-cache"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = json.loads(response.read(16385))
        return isinstance(body, dict) and body.get("instance_id") == instance_id and body.get("api_ready") is True
    except Exception:
        return False


class GlobalManager:
    def __init__(self, directory, *, root, instance_id, port, api_port, model_port,
                 staged=False, public=False, install_cloudflared=False, env=None,
                 poll_interval=1, probe_timeout=2, startup_timeout=3600, generation_timeout=5400, tunnel_timeout=180,
                 popen=None, rpc=None, probe=None, clock=None, owner_factory=None, start_model=True):
        if not IDENTIFIER.fullmatch(instance_id):
            raise ValueError("Invalid manager instance ID")
        self.directory, self.root = Path(directory).resolve(), Path(root).resolve()
        self.instance_id = instance_id
        self.port, self.api_port, self.model_port = int(port), int(api_port), int(model_port)
        if len({self.port, self.api_port, self.model_port}) != 3:
            raise ValueError("Manager, API and model ports must be distinct")
        self.staged, self.public = bool(staged), bool(public)
        self.install_cloudflared = bool(install_cloudflared)
        self.model_autostart = bool(start_model)
        self.env = dict(os.environ if env is None else env)
        self.queue_policy = QueuePolicy.from_environment(self.env)
        self.key = self.env.get("H3_CONTROL_KEY", "")
        if len(self.key) < 32 or len(self.env.get("H3_API_KEY", "")) < 32:
            raise ValueError("Separate private/public keys of at least 32 characters are required")
        if self.key == self.env["H3_API_KEY"]:
            raise ValueError("Private and public tokens must differ")
        self.poll_interval, self.probe_timeout = poll_interval, probe_timeout
        self.startup_timeout = startup_timeout
        self.generation_timeout = generation_timeout
        self.tunnel_timeout = tunnel_timeout
        self.disk_reserve_bytes = max(0, int(self.env.get("H3_DISK_RESERVE_BYTES", 64 * 1024 * 1024)))
        self._popen, self._rpc, self._probe = popen or subprocess.Popen, rpc or rpc_call, probe or probe_api
        self._owner_factory = owner_factory or ProcessOwner
        self._clock = clock or time.monotonic
        self._started = self._clock()
        self.run_directory = self.directory / "runtime" / instance_id
        self.run_directory.mkdir(parents=True, exist_ok=True)
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path, self.events_path = self.run_directory / "status.json", self.run_directory / "events.jsonl"
        self._owner = RootLock(self.root / "owner.lock")
        self._db = sqlite3.connect(self.root / "control.sqlite3", check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, status TEXT NOT NULL,
                idempotency_key TEXT UNIQUE, fingerprint TEXT, record TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS jobs_status ON jobs(status);
            CREATE TABLE IF NOT EXISTS events (sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT UNIQUE NOT NULL, record TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS commands (id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL,
                record TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS owned_processes (launch_id TEXT PRIMARY KEY, module TEXT NOT NULL,
                pid INTEGER NOT NULL, created REAL, exited INTEGER NOT NULL DEFAULT 0);
        """)
        self._lock = threading.RLock()
        self._control_lock = threading.RLock()
        self._refresh_lock = threading.Lock()
        self._cancel = threading.Event()
        self._thread = None
        self._control_threads = []
        self._control_admission_closed = False
        self._control_slots = threading.BoundedSemaphore(4)
        self.children = {"api": None, "model": None, "tunnel": None}
        self.owners = {}
        self._files = []
        self.model_instance_id = None
        self._model = {}
        self._api_ready = self._public_api_ready = self._connected = False
        self._api_expected = False
        self._api_expected_elapsed = 0.0
        self._api_expected_since = None
        self._api_intent_epoch = 0
        self._api_lifecycle = "not_started"
        self._public_url = None
        self._phase = "new"
        self._draining = False
        self._shutdown_started = False
        self._shutdown_complete = False
        self._exit_requested = False
        self._last_observation = None
        self._model_failure = None
        self._error = None
        self._cleanup_errors = []
        self._signature = None
        self._condition_gates = {}
        self._dispatch_gate_signature = None
        self._export_sequence = 0
        self._background = False
        self._ever_ready = False
        self._model_started = None
        self._model_unreachable = 0
        self._model_ever_ready = False
        self._public_ever_ready = False
        self._probe_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="independent-module-probe")
        self._closed = False
        self._recover()

    def _event(self, kind, *, job=None, data=None):
        event = {"event_id": uuid.uuid4().hex, "instance_id": self.instance_id,
                 "job_id": job.get("job_id") if job else None,
                 "attempt_id": job.get("attempt_id") if job else None,
                 "type": kind, "observed_at": time.time(),
                 "monotonic_ns": time.monotonic_ns(), "data": data or {}}
        cursor = self._db.execute("INSERT INTO events(event_id, record) VALUES (?, ?)",
                                  (event["event_id"], json.dumps(event)))
        event["sequence"] = cursor.lastrowid
        return event

    def _recover(self):
        # OS root ownership is held, but previous child fate may be uncertain.
        # Never replay an accepted expensive execution after manager replacement.
        with self._lock, self._db:
            for (encoded,) in self._db.execute("SELECT record FROM jobs WHERE status IN ('queued','running','cancel_requested')").fetchall():
                job = json.loads(encoded)
                self._transition(job, "failed", error="manager_restarted; execution not replayed",
                                 cleanup_complete=False, finished_at=time.time())
            for command_id, encoded in self._db.execute("SELECT id,record FROM commands").fetchall():
                command = json.loads(encoded)
                if command["status"] in {"accepted", "running"}:
                    command.update(status="failed", error="manager_restarted; command not replayed", finished_at=time.time())
                    self._db.execute("UPDATE commands SET record=? WHERE id=?", (json.dumps(command), command_id))
            self._event("manager.created", data={"recovery_policy": "no automatic execution replay"})

    def _get_job(self, job_id):
        row = self._db.execute("SELECT record FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not row:
            raise ControlError("Job not found", status=404)
        return json.loads(row[0])

    def _store_job(self, job):
        self._db.execute("UPDATE jobs SET status=?,record=? WHERE id=?",
                         (job["status"], json.dumps(job), job["job_id"]))

    def _transition(self, job, target, **updates):
        allowed = {"queued": {"running", "failed", "cancelled"},
                   "running": {"queued", "succeeded", "failed", "cancel_requested", "cancelled"},
                   "cancel_requested": {"cancelled", "failed"}}
        if target not in allowed.get(job["status"], set()):
            raise RuntimeError(f"Illegal main event transition {job['status']} -> {target}")
        job.update(status=target, **updates)
        job["lifecycle"].append({"state": target, "observed_at": time.time()})
        self._store_job(job)
        self._event("job." + target, job=job, data={"error": job.get("error")})

    def _active_jobs(self):
        rows = self._db.execute("SELECT record FROM jobs WHERE status IN ('running','cancel_requested') ORDER BY rowid").fetchall()
        return [json.loads(row[0]) for row in rows]

    def _queue_summary(self):
        counts = dict(self._db.execute("SELECT status,COUNT(*) FROM jobs GROUP BY status").fetchall())
        return {"queued": counts.get("queued", 0), "running": counts.get("running", 0),
                "cancel_requested": counts.get("cancel_requested", 0),
                "unfinished": sum(counts.get(s, 0) for s in NONTERMINAL),
                "job_count_limit": None, "order": ("durable_insertion_fifo" if self.queue_policy.mode == "fifo"
                    else "durable_fifo_with_bounded_resource_backfill"), "scheduling_policy": self.queue_policy.as_dict(),
                "admission_open": not (self._draining or self._shutdown_started or self._model_failure
                                        or self._phase in {"new", "failed", "stopped"}),
                "disk_reserve_bytes": self.disk_reserve_bytes}

    def _liveness(self):
        alive, errors = {}, {}
        for name, child in self.children.items():
            try:
                if name in self.owners:
                    observation = self.owners[name].observe()
                    alive[name] = observation["alive"]
                    if observation["errors"]:
                        errors[name] = "; ".join(observation["errors"])
                else:
                    alive[name] = child is not None and child.poll() is None
            except OSError as error:
                alive[name] = True  # Unknown exit retains ownership.
                errors[name] = type(error).__name__
        return alive, errors

    def _historical_ownership(self):
        current = {launch_id for child in self.children.values() if child is not None
                   for launch_id in getattr(child, "_h3_tree_records", {}).values()}
        unresolved = []
        rows = self._db.execute("SELECT launch_id,module,pid,created FROM owned_processes WHERE exited=0").fetchall()
        for launch_id, module, pid, created in rows:
            if launch_id in current:
                continue
            try:
                process = psutil.Process(pid)
                same = created is None or abs(process.create_time() - created) < .001
                if same and process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                    unresolved.append({"module": module, "pid": pid, "identity_known": created is not None})
            except psutil.NoSuchProcess:
                pass
            except psutil.Error:
                unresolved.append({"module": module, "pid": pid, "identity_known": False})
        return unresolved

    def snapshot(self):
        with self._lock:
            alive, process_errors = self._liveness()
            historical = self._historical_ownership()
            age = self._clock() - self._last_observation if self._last_observation is not None else None
            stale = age is None or age < 0 or age > 3 * self.probe_timeout + self.poll_interval + 2
            supervised = not self._background or (self._thread is not None and self._thread.is_alive())
            usable = not stale and supervised and not process_errors and not self._shutdown_started
            api = bool(usable and alive["api"] and self._api_ready)
            model = bool(usable and alive["model"] and self._model.get("ready") is True
                         and self._model.get("worker_alive") is True and not self._model_failure)
            public_api = bool(api and alive["tunnel"] and self._connected and self._public_api_ready)
            ready = api and model and not self._draining and (not self.public or public_api)
            phase = self._phase
            if phase not in {"new", "stopping", "stopped", "failed"}:
                phase = "draining" if self._draining else "ready" if ready else "degraded" if self._ever_ready or self._model_failure or self._error else "preparing"
            downloads = self._model.get("downloads")
            if (not isinstance(downloads, dict) or downloads.get("model_instance_id") != self.model_instance_id
                    or not isinstance(downloads.get("items"), dict)):
                downloads = {"schema_version": 1, "model_instance_id": self.model_instance_id,
                             "observed_at": None, "observation_stale": True, "items": {},
                             "error": "progress_not_observed_for_current_model"}
            else:
                downloads = deepcopy(downloads)
            download_lifecycle = (("stopping" if alive["model"] else "stopped") if self._shutdown_started else
                                  "failed" if self._model_failure and self._model_failure != "model_stopped" else
                                  "not_started" if self.model_instance_id is None else
                                  "stopped" if alive["model"] is False else self._model.get("phase", "unknown"))
            downloads.update(model_ready=model, observation_stale=bool(downloads.get("observation_stale")
                             or stale or not supervised or alive["model"] is not True or "model" in process_errors),
                             lifecycle=download_lifecycle)
            result = {"instance_id": self.instance_id, "phase": phase, "ready": ready,
                      "api_ready": api, "model_ready": model, "public_requested": self.public,
                      "public_api_ready": public_api, "public_ready": public_api and model and not self._draining,
                      "tunnel_connected": bool(alive["tunnel"] and self._connected), "public_url": self._public_url,
                      "server_alive": alive["api"], "model_alive": alive["model"], "tunnel_alive": alive["tunnel"],
                      "manager_alive": True, "manager_pid": os.getpid(), "api_pid": getattr(self.children["api"], "pid", None),
                      "server_pid": getattr(self.children["api"], "pid", None),
                      "model_pid": getattr(self.children["model"], "pid", None), "tunnel_pid": getattr(self.children["tunnel"], "pid", None),
                      "model_instance_id": self.model_instance_id,
                      "downloads": downloads,
                      "cleanup_complete": not any(alive.values()) and not historical and not process_errors,
                      "shutdown_complete": self._shutdown_complete,
                      "historical_ownership": historical, "cleanup_errors": list(self._cleanup_errors),
                      "error": self._error or self._model_failure, "elapsed_seconds": round(self._clock() - self._started, 3),
                      "observed_at": time.time(), "observation_age_seconds": round(age, 3) if age is not None else None,
                      "observation_stale": stale, "supervisor_alive": supervised,
                      "process_observation_errors": process_errors, "draining": self._draining,
                      "condition_gates": deepcopy(self._condition_gates),
                      "queue": self._queue_summary(),
                      "owned_processes": {name: owner.observe() for name, owner in self.owners.items()},
                      "state_path": str(self.state_path), "events_path": str(self.events_path),
                      "modules": {"api": {"alive": alive["api"], "ready": api,
                                          "expected_running": self._api_expected, "lifecycle": self._api_lifecycle,
                                          "intent_epoch": self._api_intent_epoch},
                                  "model": {**self._model, "alive": alive["model"], "ready": model, "downloads": downloads},
                                  "manager": {"alive": True, "supervisor_alive": supervised}}}
            row = self._db.execute("SELECT record FROM jobs WHERE status='queued' ORDER BY rowid LIMIT 1").fetchone()
            result["condition_summary"] = self._condition_summary(json.loads(row[0]) if row else None)
            active_row = self._db.execute("SELECT record FROM jobs WHERE status IN ('running','cancel_requested') ORDER BY rowid LIMIT 1").fetchone()
            result["execution_summary"] = (public_execution_summary(self._public_job(json.loads(active_row[0])), self.instance_id)
                                           if active_row else None)
            return result

    def _checkpoint(self):
        with self._lock:
            state = self.snapshot()
            signature = json.dumps({k: state[k] for k in ("phase", "ready", "api_ready", "model_ready", "public_api_ready", "server_alive", "model_alive", "tunnel_alive", "cleanup_complete", "error")}, sort_keys=True)
            if signature != self._signature:
                with self._db:
                    self._event("manager.state", data=json.loads(signature))
                self._signature = signature
            # SQLite remains authoritative if this readable export is interrupted.
            rows = self._db.execute("SELECT sequence,record FROM events WHERE sequence>? ORDER BY sequence", (self._export_sequence,)).fetchall()
            if rows:
                with self.events_path.open("a", encoding="utf-8") as output:
                    for sequence, record in rows:
                        event = json.loads(record)
                        event["sequence"] = sequence
                        output.write(json.dumps(event) + "\n")
                    if self._shutdown_complete:
                        output.flush()
                        os.fsync(output.fileno())
                self._export_sequence = rows[-1][0]
            # Publish the final receipt only after the matching audit export.
            temporary = self.state_path.with_suffix(".tmp")
            with temporary.open("w", encoding="utf-8") as output:
                output.write(json.dumps(state, indent=2))
                if self._shutdown_complete:
                    output.flush()
                    os.fsync(output.fileno())
            temporary.replace(self.state_path)

    def _spawn(self, name, command, env):
        output = (self.run_directory / (name + ".log")).open("a", encoding="utf-8")
        self._files.append(output)
        child = self._popen(command, cwd=self.directory, env=env, stdout=output, stderr=subprocess.STDOUT)
        self.children[name] = child
        self.owners[name] = self._owner_factory(child)
        with self._lock, self._db:
            try:
                created = psutil.Process(child.pid).create_time()
            except psutil.Error:
                created = None
            launch_id = uuid.uuid4().hex
            child._h3_launch_id = launch_id
            child._h3_tree_records = {(child.pid, created): launch_id}
            self._db.execute("INSERT INTO owned_processes(launch_id,module,pid,created) VALUES (?,?,?,?)",
                             (launch_id, name, child.pid, created))
            self._event("module.started", data={"module": name, "pid": child.pid,
                        "model_instance_id": self.model_instance_id if name == "model" else None})
        return child

    def _register_process_tree(self):
        with self._lock, self._db:
            for name, owner in self.owners.items():
                child = self.children[name]
                for identity in owner.observe()["identities"]:
                    identity_key = (identity["pid"], identity["created_at"])
                    if identity_key not in child._h3_tree_records:
                        launch_id = uuid.uuid4().hex
                        child._h3_tree_records[identity_key] = launch_id
                        self._db.execute("INSERT INTO owned_processes(launch_id,module,pid,created) VALUES (?,?,?,?)",
                                         (launch_id, name, *identity_key))

    def _child_env(self):
        return {**self.env, "H3_PARENT_PID": str(os.getpid()),
                "H3_PARENT_CREATED_AT": str(psutil.Process(os.getpid()).create_time()), "H3_INSTANCE_ID": self.instance_id,
                "H3_MANAGER_URL": f"http://127.0.0.1:{self.port}", "H3_OUTPUT_DIR": str(self.root)}

    def _set_api_expectation(self, expected):
        # Only time in which the API is requested to run consumes its first
        # public-delivery budget. An explicit stop pauses, rather than resets,
        # that budget, including when process exit remains unconfirmed.
        with self._lock:
            if expected != self._api_expected:
                now = self._clock()
                if self._api_expected_since is not None:
                    self._api_expected_elapsed += max(0.0, now - self._api_expected_since)
                self._api_expected_since = now if expected else None
                self._api_expected = expected

    def _launch_api(self):
        alive, errors = self._liveness()
        if alive["api"] or "api" in errors:
            raise ControlError("API ownership has not been released", status=409)
        with self._lock:
            self._api_ready = self._public_api_ready = False
            self._set_api_expectation(True)
            self._api_intent_epoch += 1
            self._api_lifecycle = "starting"
        self._spawn("api", [sys.executable, "-m", "uvicorn", "server:create_app", "--factory", "--host", "127.0.0.1",
                            "--port", str(self.api_port), "--workers", "1", "--limit-concurrency", "8"], self._child_env())

    def _launch_model(self):
        alive, errors = self._liveness()
        if alive["model"] or "model" in errors:
            raise ControlError("Old model ownership has not been released", status=409)
        with self._lock:
            old_processes = self._db.execute("SELECT launch_id,pid,created FROM owned_processes WHERE module='model' AND exited=0").fetchall()
        for launch_id, pid, created in old_processes:
            try:
                process = psutil.Process(pid)
                same = created is None or abs(process.create_time() - created) < .001
                if same and process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                    raise ControlError("Previous model process exit is unconfirmed", status=409)
            except psutil.NoSuchProcess:
                pass
            except psutil.Error:
                raise ControlError("Previous model process identity is inaccessible", status=409)
            with self._lock, self._db:
                self._db.execute("UPDATE owned_processes SET exited=1 WHERE launch_id=?", (launch_id,))
        self.model_instance_id = uuid.uuid4().hex
        self._model = {"phase": "loading", "instance_id": self.model_instance_id, "ready": False}
        self._model_failure = None
        self._model_started = self._clock()
        self._model_unreachable = 0
        self._model_ever_ready = False
        command = [sys.executable, "-u", str(self.directory / "model_container.py"), "--port", str(self.model_port),
                   "--instance-id", self.model_instance_id, "--root", str(self.root), "--parent-pid", str(os.getpid())]
        if self.staged:
            command.append("--staged")
        env = self._child_env()
        env.update(H3_MODEL_INSTANCE_ID=self.model_instance_id,
                   H3_DOWNLOAD_PROGRESS_PATH=str(self.run_directory / ("model-" + self.model_instance_id + ".downloads.json")))
        self._spawn("model", command, env)

    def start(self, *, background=True):
        with self._control_lock:
            if self._phase != "new":
                raise ControlError("Manager already started", status=409)
            self._phase, self._background = "preparing", bool(background)
            try:
                env = self._child_env()
                self._launch_api()
                if self.model_autostart:
                    try:
                        self._launch_model()
                    except ControlError:
                        # Recovery may observe a previous owner child still exiting.
                        # Keep the API/control plane available; never overlap models.
                        self._model_failure = "previous_model_exit_unconfirmed"
                else:
                    self._model = {"instance_id": None, "phase": "not_started", "ready": False,
                                   "worker_alive": False, "accepting": False}
                    with self._lock, self._db:
                        self._event("model.autostart_skipped", data={"reason": "explicit_configuration"})
                if self.public:
                    command = [sys.executable, "-u", str(self.directory / "tunnel.py"), "--port", str(self.api_port),
                               "--status-file", str(self.run_directory / "tunnel-status.json"), "--instance-id", self.instance_id]
                    if self.install_cloudflared:
                        command.append("--install")
                    self._spawn("tunnel", command, env)
                if background:
                    self._thread = threading.Thread(target=self._monitor, name="main-event-supervisor", daemon=True)
                    self._thread.start()
                self._checkpoint()
            except BaseException:
                self._error = "module_start_failed"
                self.shutdown()
                self._phase = "failed"
                raise
        return self

    def _transport(self):
        try:
            path = self.run_directory / "tunnel-status.json"
            if path.stat().st_size > 16384:
                return None, False
            body = json.loads(path.read_text(encoding="utf-8"))
            url = body.get("url") if isinstance(body, dict) else None
            if (body.get("instance_id") != self.instance_id or not isinstance(url, str)
                    or not re.fullmatch(r"https://[a-z0-9-]+\.trycloudflare\.com", url)):
                return None, False
            return url, body.get("connected") is True and body.get("phase") not in {"failed", "stopping", "stopped"}
        except (OSError, ValueError, TypeError, AttributeError):
            return None, False

    def _fail_jobs_after_exit(self, reason):
        with self._lock, self._db:
            for (encoded,) in self._db.execute("SELECT record FROM jobs WHERE status IN ('queued','running','cancel_requested')").fetchall():
                job = json.loads(encoded)
                target = "cancelled" if job["status"] == "cancel_requested" else "failed"
                cleaned, cleanup_errors = self._cleanup_job_files(job["job_id"], discard_result=True)
                self._transition(job, target, error=reason, finished_at=time.time(), cleanup_complete=cleaned,
                                 cleanup_errors=cleanup_errors, model_lease_released=True)

    def _cleanup_job_files(self, job_id, *, discard_result):
        errors = []
        job_root = self.root / job_id
        if not job_root.resolve().is_relative_to(self.root) or job_root.is_symlink():
            return False, ["job_directory_identity_invalid"]
        try:
            shutil.rmtree(job_root / "inputs")
        except FileNotFoundError:
            pass
        except OSError as error:
            errors.append("inputs: " + type(error).__name__)
        if discard_result:
            for name in ("result.mp4", "result.metrics.json"):
                try:
                    (job_root / name).unlink(missing_ok=True)
                except OSError as error:
                    errors.append(name + ": " + type(error).__name__)
        return not errors, errors

    def _model_probe(self, expected_instance_id):
        try:
            candidate = self._rpc(f"http://127.0.0.1:{self.model_port}", self.key, "status", {}, timeout=self.probe_timeout)
            if isinstance(candidate, dict) and candidate.get("instance_id") == expected_instance_id:
                return candidate
        except Exception:
            pass
        return None

    def refresh(self):
        with self._refresh_lock:
            if self._shutdown_started:
                return self.snapshot()
            with self._control_lock:
                self._register_process_tree()
                alive, errors = self._liveness()
                observed_model_id, observed_child = self.model_instance_id, self.children["model"]
                observed_api_child = self.children["api"]
                observed_api_epoch = self._api_intent_epoch
            api_future = (self._probe_pool.submit(self._probe, f"http://127.0.0.1:{self.api_port}", self.instance_id, self.probe_timeout)
                          if alive["api"] else None)
            model_future = (self._probe_pool.submit(self._model_probe, observed_model_id)
                            if alive["model"] and "model" not in errors else None)
            api_ready = bool(api_future.result()) if api_future else False
            model = model_future.result() if model_future else None
            url, connected = self._transport() if self.public and alive["tunnel"] else (None, False)
            public_api = bool(api_ready and connected and self._probe(url, self.instance_id, self.probe_timeout))
            with self._control_lock:
                if (self._shutdown_started or observed_model_id != self.model_instance_id
                        or observed_child is not self.children["model"] or observed_api_child is not self.children["api"]
                        or observed_api_epoch != self._api_intent_epoch):
                    return self.snapshot()  # Discard a previous model generation's observations.
                if alive["model"] and "model" not in errors:
                    self._model_unreachable = 0 if model else self._model_unreachable + 1
                with self._lock:
                    self._api_ready = bool(api_ready and self._api_expected)
                    self._public_api_ready = bool(public_api and self._api_expected)
                    self._connected = connected
                    if url:
                        self._public_url = url
                    previous_downloads = self._model.get("downloads")
                    self._model = model or {"instance_id": self.model_instance_id, "ready": False,
                                            "phase": "unreachable" if alive["model"] else
                                                     "not_started" if self.model_instance_id is None else "stopped"}
                    if not model and isinstance(previous_downloads, dict):
                        self._model["downloads"] = {**deepcopy(previous_downloads), "observation_stale": True}
                    self._last_observation = self._clock()
                    if model and model.get("ready") is True:
                        self._model_ever_ready = True
                    if self._public_api_ready:
                        self._public_ever_ready = True
                        if self._error and self._error.startswith("public_"):
                            self._error = None
                    if not alive["api"] and self._api_expected:
                        self._error = "api_process_exited; model and durable jobs retained"
                        self._api_lifecycle = "failed"
                    elif api_ready and self._error and self._error.startswith("api_"):
                        self._error = None
                    if self._api_expected and api_ready:
                        self._api_lifecycle = "ready"
                    if not alive["model"] and self.model_instance_id is not None:
                        self._model_failure = self._model_failure or "model_process_exited"
                    elif model and model.get("phase") == "failed":
                        self._model_failure = "model_load_or_worker_failed"
                if not alive["model"] and "model" not in errors and self.model_instance_id is not None:
                    self._fail_jobs_after_exit("model_process_exited")
                elif model:
                    self._consume_result(model)
                    with self._lock:
                        expired = any(time.time() - job["started_at"] > self.generation_timeout
                                      for job in self._active_jobs())
                    # A dead worker can no longer own a useful execution. Confirm
                    # process exit before releasing the main execution lease.
                    if expired:
                        self._stop_model(force=True, reason="generation_deadline_exceeded")
                    elif self._model_failure or model.get("phase") == "failed" or (model.get("worker_alive") is False and model.get("phase") not in {"starting", "loading"}):
                        with self._control_lock:
                            self._stop_model(force=True, reason="model_worker_failed")
                    else:
                        self._dispatch_next(model)
                if alive["model"] and self._model_unreachable >= 3:
                    with self._control_lock:
                        self._stop_model(force=True, reason="model_control_unreachable_for_three_cycles")
                with self._lock:
                    active = self._active_jobs()
                if any(time.time() - job["started_at"] > self.generation_timeout for job in active):
                    with self._control_lock:
                        self._stop_model(force=True, reason="generation_deadline_exceeded")
                if self.snapshot()["ready"]:
                    self._ever_ready = True
                if (self._model_started is not None and not self._model_ever_ready and self._clock() - self._model_started > self.startup_timeout
                        and not self._model.get("ready") and not active and alive["model"]):
                    with self._control_lock:
                        self._stop_model(force=True, reason="model_startup_deadline_exceeded")
                public_elapsed = self._api_expected_elapsed
                if self._api_expected_since is not None:
                    public_elapsed += max(0.0, self._clock() - self._api_expected_since)
                if (self.public and self._api_expected and not self._public_ever_ready
                        and public_elapsed > self.tunnel_timeout):
                    self._error = "public_preparation_deadline_exceeded; local API and model retained"
                self._checkpoint()
                return self.snapshot()

    def _consume_result(self, model):
        results = model.get("results")
        if not isinstance(results, list):
            results = [model.get("last_result")]  # Read old checkpoints without losing their result.
        acknowledgements = []
        for result in results:
            if not isinstance(result, dict) or result.get("status") not in TERMINAL:
                continue
            with self._lock, self._db:
                try:
                    job = self._get_job(result.get("job_id"))
                except ControlError:
                    continue
                if (result.get("attempt_id") != job["attempt_id"]
                        or model.get("instance_id") != job["model_instance_id"]
                        or result.get("instance_id", model["instance_id"]) != job["model_instance_id"]):
                    continue
                if job["status"] in TERMINAL:
                    acknowledgements.append(job["attempt_id"])
                    continue
                if job["status"] not in {"running", "cancel_requested"}:
                    continue
                if result.get("cleanup_complete") is not True:
                    self._model_failure = "model_attempt_cleanup_unconfirmed"
                    job.update(error=self._model_failure, cleanup_complete=False)
                    self._store_job(job)
                    self._event("job.cleanup_pending", job=job)
                    continue  # Retain ownership until process exit is confirmed.
                target, artifact = result["status"], result.get("artifact")
                cancellation_requested = job["status"] == "cancel_requested"
                if cancellation_requested and target != "failed":
                    target, artifact = "cancelled", None
                elif target == "succeeded":
                    artifact = self._verified_artifact(job["job_id"], artifact)
                    if artifact is None:
                        target = "failed"
                cleaned, cleanup_errors = self._cleanup_job_files(job["job_id"], discard_result=target != "succeeded")
                self._transition(job, target, artifact=artifact if target == "succeeded" else None,
                                 error=None if target == "succeeded" else result.get("error") or "result_not_committed",
                                 metrics=result.get("metrics"), finished_at=time.time(), cleanup_complete=cleaned,
                                 cancellation_requested=cancellation_requested,
                                 execution_condition_gates=deepcopy(result.get("condition_gates")),
                                 stage_completions=deepcopy(result.get("stage_completions")),
                                 generation_options=result.get("generation_options") or job.get("generation_options"),
                                 cleanup_errors=cleanup_errors, model_lease_released=True, dispatch_state="completed")
                acknowledgements.append(job["attempt_id"])
        # The durable transaction must commit before the model may discard a result.
        if acknowledgements and isinstance(model.get("results"), list):
            try:
                self._rpc(f"http://127.0.0.1:{self.model_port}", self.key, "ack_results",
                          {"instance_id": model["instance_id"], "attempt_ids": acknowledgements}, timeout=self.probe_timeout)
            except Exception:
                pass  # Next status repeats the durable results; acknowledge idempotently.

    def _verified_artifact(self, job_id, artifact):
        if not isinstance(artifact, dict):
            return None
        try:
            path = Path(artifact["path"]).resolve()
            lexical = self.root / job_id / "result.mp4"
            expected = lexical.resolve()
            if (path != expected or not path.is_relative_to(self.root) or not path.is_file()
                    or lexical.is_symlink() or not isinstance(artifact.get("media"), dict)
                    or artifact["media"].get("full_decode_passed") is not True):
                return None
            if path.stat().st_size != artifact["bytes"] or artifact["bytes"] <= 0:
                return None
            if not re.fullmatch(r"[a-f0-9]{64}", artifact.get("sha256", "")):
                return None
            with path.open("rb") as source:
                actual = hashlib.file_digest(source, "sha256").hexdigest()
            if actual != artifact["sha256"]:
                return None
            return artifact
        except (KeyError, OSError, TypeError, ValueError):
            return None

    def _dispatch_observations(self, model, job):
        """Read-only observations; callers still own atomic authorization."""
        model = model if isinstance(model, dict) else {}
        alive, errors = self._liveness()
        age = self._clock() - self._last_observation if self._last_observation is not None else None
        bound = 3 * self.probe_timeout + self.poll_interval + 2
        identity = self.model_instance_id
        identities = (identity, model.get("instance_id"), self._model.get("instance_id"))
        current_identity = (all(value == identity for value in identities)
                            if all(isinstance(value, str) and value for value in identities) else None)
        def capability(field):
            witnesses = [model.get(field), self._model.get(field)]
            # Missing or malformed fresh evidence cannot borrow a previous True.
            return witnesses if all(type(value) is bool for value in witnesses) else None
        return {
            # Draining closes new admission; the already-owned queue still drains.
            "manager_open": self._shutdown_started is False and self._model_failure is None,
            "job_queued": job.get("status") == "queued", "attempt_unassigned": job.get("attempt_id") is None,
            "model_identity_current": current_identity,
            "model_observation_fresh": None if age is None else 0 <= age <= bound,
            "model_process_alive": None if "model" in errors else alive.get("model"),
            "model_ready": capability("ready"),
            "worker_alive": capability("worker_alive"),
            "worker_accepting": capability("accepting"),
            "supervisor_alive": not self._background or (self._thread is not None and self._thread.is_alive()),
        }
    def _condition_summary(self, job=None):
        age = self._clock() - self._last_observation if self._last_observation is not None else None
        current = age is not None and 0 <= age <= 3 * self.probe_timeout + self.poll_interval + 2
        value = {"status": "current" if current else "stale", "manager_instance_id": self.instance_id,
                 "model_instance_id": self.model_instance_id, "observed_at": self._model.get("observed_at"),
                 "observation_age_seconds": age,
                 "last_decision": (job.get("condition_gate") if job else self._condition_gates.get("manager.dispatch"))}
        if job:
            value["job_id"] = job["job_id"]
            value["conditions"] = {k: observation_state(v) for k, v in self._dispatch_observations(self._model, job).items()}
        else:
            value["status"] = "historical"
        return public_condition_summary(value, self.instance_id)

    def _dispatch_gate(self, model, job, attempt_id):
        """An offer is authorized only under dispatch/cancel arbitration."""
        identity = self.model_instance_id
        receipt = evaluate_gate("manager.dispatch", DISPATCH_CONDITIONS, self._dispatch_observations(model, job),
            {"manager_instance_id": self.instance_id, "model_instance_id": identity, "job_id": job["job_id"],
             "attempt_id": attempt_id, "source_file": "global_manager.py", "source_sha256": GATE_SOURCE_SHA256})
        # Commit every positive authorization; coalesce unchanged blocked polls.
        signature = json.dumps({"job_id": job["job_id"], "model_instance_id": identity,
                                "conditions": receipt["conditions"]}, sort_keys=True)
        if receipt["allowed"] or signature != self._dispatch_gate_signature:
            self._dispatch_gate_signature = signature
            self._condition_gates["manager.dispatch"] = receipt
            job["condition_gate"] = receipt
            self._store_job(job)
            self._event("condition.checked", job=job, data={"receipt": receipt})
        return receipt

    def _record_queue_admission(self, job, bypassed_job_ids):
        # Caller holds the state transaction; count reconciled and direct accepts equally.
        policy_state = job.setdefault("queue_policy", {"mode": self.queue_policy.mode,
            "bypasses": 0, "deferrals": 0, "manager_instance_id": self.instance_id})
        now = self._clock()
        policy_state.update(admitted_at=now, clock="manager_monotonic")
        first = policy_state.get("first_deferred_at")
        policy_state["deferral_to_admission_seconds"] = now - first if first is not None and now >= first else None
        if bypassed_job_ids:
            counts = {}
            for predecessor_id in bypassed_job_ids:
                row = self._db.execute("SELECT record FROM jobs WHERE id=? AND status='queued'", (predecessor_id,)).fetchone()
                if row is None:
                    raise RuntimeError("Bypassed queue owner changed during admission arbitration")
                predecessor = json.loads(row[0])
                previous = predecessor.get("queue_policy", {}).get("bypasses")
                if type(previous) is not int or not 0 <= previous < self.queue_policy.max_bypasses:
                    raise RuntimeError("Bypassed queue owner has unconfirmed fairness budget")
                predecessor["queue_policy"]["bypasses"] = previous + 1
                counts[predecessor_id] = previous + 1
                self._store_job(predecessor)
            policy_state["bypassed_head_job_id"] = bypassed_job_ids[0]
            policy_state["bypassed_job_ids"] = list(bypassed_job_ids)
            self._event("queue.backfill_admitted", job=job, data={"bypassed_job_ids": list(bypassed_job_ids),
                "bypass_counts": counts, "policy": self.queue_policy.as_dict()})

    def _dispatch_next(self, model):
        with self._control_lock:
            deadline = time.monotonic() + max(.01, min(.25, self.poll_interval))
            deferred_ids = set()
            blocked_head_id = None
            while not self._shutdown_started and not self._model_failure:
                with self._lock, self._db:
                    bypassed_job_ids = []
                    if blocked_head_id is None:
                        row = self._db.execute("SELECT record FROM jobs WHERE status='queued' ORDER BY rowid LIMIT 1").fetchone()
                    else:
                        encoded_head = self._db.execute("SELECT record FROM jobs WHERE id=? AND status='queued'", (blocked_head_id,)).fetchone()
                        if not encoded_head:
                            return
                        candidates = self._db.execute("SELECT record FROM jobs WHERE status='queued' ORDER BY rowid LIMIT ?",
                                                      (self.queue_policy.lookahead,)).fetchall()
                        decoded = [json.loads(candidate[0]) for candidate in candidates]
                        index = next((i for i, candidate in enumerate(decoded) if candidate["job_id"] not in deferred_ids), None)
                        if index is None:
                            return
                        predecessors = decoded[:index]
                        if any(not isinstance(candidate.get("queue_policy"), dict)
                               or candidate["queue_policy"].get("manager_instance_id") != self.instance_id
                               for candidate in predecessors):
                            return
                        if not self.queue_policy.may_bypass_all([candidate.get("queue_policy") for candidate in predecessors], now=self._clock()):
                            return
                        bypassed_job_ids = [candidate["job_id"] for candidate in predecessors]
                        row = candidates[index]
                    if not row:
                        return
                    job = json.loads(row[0])
                    prospective_attempt = uuid.uuid4().hex
                    receipt = self._dispatch_gate(model, job, prospective_attempt)
                    if receipt["allowed"] is not True:
                        return
                    self._transition(job, "running", attempt_id=prospective_attempt,
                                     model_instance_id=self.model_instance_id, started_at=time.time(), cleanup_complete=False,
                                     model_lease_released=False, dispatch_state="offering", dispatch_reason=None)
                    self._event("job.dispatch_intent", job=job, data={"condition_decision_id": receipt["decision_id"]})
                # running means this attempt's ownership, not ownership of the GPU.
                # The model stage scheduler alone grants the one GPU lease.
                payload = {"job_id": job["job_id"], "attempt_id": job["attempt_id"],
                           "instance_id": self.model_instance_id, "parameters": job["parameters"]}
                try:
                    response = self._rpc(f"http://127.0.0.1:{self.model_port}", self.key, "execute", payload, timeout=self.probe_timeout)
                    if not isinstance(response, dict) or response.get("accepted") is not True:
                        raise ValueError("Invalid execution acknowledgement")
                    for field in ("job_id", "attempt_id", "instance_id"):
                        if response.get(field) != payload[field]:
                            raise ValueError("Execution acknowledgement identity mismatch")
                except Exception as error:
                    rejected = None
                    if isinstance(error, ControlError):
                        try:
                            rejected = json.loads(str(error))
                        except (ValueError, TypeError):
                            pass
                    if isinstance(rejected, dict) and rejected.get("guarantees_not_accepted") is True:
                        with self._lock, self._db:
                            if error.status in {409, 429}:
                                policy_state = job.setdefault("queue_policy", {"mode": self.queue_policy.mode,
                                    "bypasses": 0, "deferrals": 0, "manager_instance_id": self.instance_id})
                                policy_state["deferrals"] += 1
                                policy_state.setdefault("first_deferred_at", self._clock())
                                policy_state["last_deferred_at"] = self._clock()
                                policy_state["clock"] = "manager_monotonic"
                                self._transition(job, "queued", attempt_id=None, model_instance_id=None,
                                                 started_at=None, model_lease_released=True, dispatch_state="deferred",
                                                 dispatch_reason=rejected.get("reason"))
                                self._event("queue.deferred", job=job, data={"policy": self.queue_policy.as_dict(),
                                    "reason": rejected.get("reason"), "guarantees_not_accepted": True})
                                # Only an explicit resource rejection permits bounded lookahead.
                                # Unknown ownership/readiness never becomes a backfill opportunity.
                                if (not guaranteed_resource_deferral(error.status, rejected)
                                        or self.queue_policy.mode == "fifo" or time.monotonic() >= deadline):
                                    return
                                blocked_head_id = blocked_head_id or job["job_id"]
                                deferred_ids.add(job["job_id"])
                                continue
                            cleaned, cleanup_errors = self._cleanup_job_files(job["job_id"], discard_result=True)
                            self._transition(job, "failed", error=str(error), finished_at=time.time(),
                                             model_lease_released=True, cleanup_complete=cleaned,
                                             cleanup_errors=cleanup_errors, dispatch_state="rejected")
                    else:
                        # A missing acknowledgement is not evidence of rejection.
                        # Reconcile this exact attempt, never send execute twice.
                        observed = self._model_probe(self.model_instance_id)
                        records = ((observed or {}).get("attempts", []) + (observed or {}).get("results", []))
                        matched = any(isinstance(item, dict) and all(item.get(k) == payload[k]
                                      for k in ("job_id", "attempt_id", "instance_id")) for item in records)
                        if not matched:
                            self._stop_model(force=True, reason="execution_acknowledgement_uncertain")
                            return
                        with self._lock, self._db:
                            job.update(dispatch_state="reconciled")
                            self._record_queue_admission(job, bypassed_job_ids)
                            self._store_job(job)
                            self._event("job.dispatch_reconciled", job=job)
                        self._model = observed
                        self._consume_result(observed)
                        return
                else:
                    with self._lock, self._db:
                        job.update(dispatch_state="accepted")
                        self._record_queue_admission(job, bypassed_job_ids)
                        self._store_job(job)
                        self._event("job.dispatch_accepted", job=job)
                if time.monotonic() >= deadline:
                    return

    def _monitor(self):
        try:
            while not self._cancel.is_set():
                self.refresh()
                self._cancel.wait(self.poll_interval)
        except BaseException as error:
            LOG.exception("Main event supervision failed")
            with self._lock:
                self._error = "supervisor_failed: " + type(error).__name__
                self._phase, self._draining = "failed", True
            # Retain the control plane, but end uncertain model ownership.
            with self._control_lock:
                self._stop_model(force=True, reason="supervisor_failed")
            try:
                self._checkpoint()
            except Exception:
                LOG.exception("Could not export final supervisor failure")

    def submit(self, payload):
        job_id, parameters = payload.get("job_id"), payload.get("parameters")
        if not isinstance(job_id, str) or not IDENTIFIER.fullmatch(job_id) or not isinstance(parameters, dict):
            raise ControlError("Invalid job request", status=422)
        try:
            options = requested_options(parameters)
            validate_generation_options(options, load_adapter_settings())
        except (KeyError, ValueError, TypeError) as error:
            raise ControlError(f"Invalid generation options: {error}", status=422) from None
        refs = parameters.get("references")
        if not isinstance(refs, list) or not 1 <= len(refs) <= 12:
            raise ControlError("Invalid reference list", status=422)
        input_root = (self.root / job_id / "inputs").resolve()
        if not input_root.is_relative_to(self.root):
            raise ControlError("Input directory escapes output root", status=422)
        for ref in refs:
            try:
                path = Path(ref["path"]).resolve()
                if not path.is_relative_to(input_root) or not path.is_file() or Path(ref["path"]).is_symlink():
                    raise ValueError()
            except (KeyError, TypeError, OSError, ValueError):
                raise ControlError("References must be owned input files", status=422)
        idem, fingerprint = payload.get("idempotency_key"), payload.get("fingerprint")
        if idem is not None and (not isinstance(idem, str) or not 1 <= len(idem) <= 128 or not isinstance(fingerprint, str)):
            raise ControlError("Invalid idempotency key or fingerprint", status=422)
        with self._lock, self._db:
            if idem is not None:
                row = self._db.execute("SELECT fingerprint,record FROM jobs WHERE idempotency_key=?", (idem,)).fetchone()
                if row:
                    if row[0] != fingerprint:
                        raise ControlError("Idempotency key conflicts with request", status=409)
                    return self._public_job(json.loads(row[1]))
            if self._draining or self._shutdown_started or self._phase in {"new", "failed", "stopped"} or self._model_failure:
                raise ControlError("Generation admission is closed", status=409)
            if shutil.disk_usage(self.root).free < self.disk_reserve_bytes + 65536:
                raise ControlError("Insufficient disk space for durable queue metadata", status=507)
            if self._db.execute("SELECT 1 FROM jobs WHERE id=?", (job_id,)).fetchone():
                raise ControlError("Job ID already exists", status=409)
            now = time.time()
            job = {"job_id": job_id, "status": "queued", "created_at": now, "started_at": None,
                   "finished_at": None, "attempt_id": None, "model_instance_id": None,
                   "parameters": parameters, "error": None, "artifact": None, "metrics": None,
                   "generation_options": {"requested": options, "effective": None},
                    "cleanup_complete": False, "model_lease_released": True, "dispatch_state": "pending",
                   "lifecycle": [{"state": "queued", "observed_at": now}]}
            self._db.execute("INSERT INTO jobs VALUES (?,?,?,?,?)", (job_id, "queued", idem, fingerprint, json.dumps(job)))
            self._event("job.queued", job=job)
            return self._public_job(job)

    def _public_job(self, job):
        result = {k: v for k, v in job.items() if k != "parameters"}
        if "execution_condition_gates" in result:
            result["execution_condition_gates"] = self._bound_execution_gates(
                result["execution_condition_gates"], job, job.get("finished_at"))
        # Public task identity belongs to this durable manager/API lifetime;
        # model_instance_id separately identifies a replaceable GPU process.
        result["instance_id"] = self.instance_id
        result["queue_position"] = (self._db.execute("SELECT COUNT(*) FROM jobs WHERE status='queued' AND rowid <= (SELECT rowid FROM jobs WHERE id=?)",
                                                    (job["job_id"],)).fetchone()[0] if job["status"] == "queued" else None)
        result["waiting_for"] = None
        if job["status"] == "queued":
            result["condition_summary"] = self._condition_summary(job)
            # This is a current observation, separate from the last durable
            # dispatch rejection. Do not make clients infer it from a log line.
            if self._shutdown_started:
                result["waiting_for"] = "shutdown"
            elif self._model_failure:
                result["waiting_for"] = "model_recovery"
            elif self.model_instance_id is None:
                result["waiting_for"] = "model_ready"
            elif result["condition_summary"]["status"] != "current":
                result["waiting_for"] = "current_observation"
            elif any(value in {"unknown", "conflict"} for value in result["condition_summary"]["conditions"].values()):
                result["waiting_for"] = "condition_verification"
            elif result["condition_summary"]["conditions"].get("model_identity_current") != "true":
                result["waiting_for"] = "model_identity"
            elif self._model.get("ready") is not True or self._model.get("worker_alive") is not True:
                result["waiting_for"] = "model_ready"
            elif self._model.get("accepting") is not True:
                result["waiting_for"] = "model_capacity"
            elif result["queue_position"] > 1:
                result["waiting_for"] = "earlier_jobs"
            else:
                result["waiting_for"] = "scheduler"
        result["active_phase"] = None
        age = self._clock() - self._last_observation if self._last_observation is not None else None
        evidence = {"status": "historical" if job["status"] in TERMINAL else "unknown",
                    "manager_instance_id": self.instance_id, "model_instance_id": job.get("model_instance_id"),
                    "job_id": job["job_id"], "attempt_id": job.get("attempt_id"),
                    "observed_at": job.get("finished_at") if job["status"] in TERMINAL else self._model.get("observed_at")}
        if job["status"] in {"running", "cancel_requested"}:
            evidence["observation_age_seconds"] = age
            bound = job.get("model_instance_id") is not None and job.get("model_instance_id") == self.model_instance_id == self._model.get("instance_id")
            fresh = age is not None and 0 <= age <= 3 * self.probe_timeout + self.poll_interval + 2
            evidence["status"] = "identity_mismatch" if not bound else "stale" if not fresh else "unknown"
            for attempt in self._model.get("attempts", []) if bound and fresh else []:
                if (isinstance(attempt, dict) and attempt.get("instance_id") == self.model_instance_id
                        and attempt.get("attempt_id") == job["attempt_id"] and attempt.get("job_id") == job["job_id"]
                        and type(self._model.get("observed_at")) in {int, float}
                        and math.isfinite(self._model["observed_at"])
                        and self._model["observed_at"] >= (job.get("started_at") or 0)):
                    evidence["status"] = "current"
                    result["active_phase"] = attempt.get("active_phase") or {"phase": attempt.get("phase", attempt.get("status"))}
                    gates = self._bound_execution_gates(attempt.get("condition_gates"), job, self._model["observed_at"])
                    result.update(public_condition_fields({"execution_condition_gates": gates,
                        "stage_completions": attempt.get("stage_completions"), "cancellation_requested": attempt.get("cancel_requested")}))
                    if isinstance(attempt.get("generation_options"), dict):
                        result["generation_options"] = deepcopy(attempt["generation_options"])
                    break
        result.update(public_condition_fields({"execution_observation": evidence}))
        if job["status"] == "cancel_requested":
            result["cancellation_requested"] = True  # Durable intent, not remote acknowledgement.
        return result

    @staticmethod
    def _bound_execution_gates(raw, job, observed_at):
        gates = public_condition_fields({"execution_condition_gates": raw}).get("execution_condition_gates") or {}
        for gate_id, receipt in gates.items():
            context = receipt.get("context", {}) if receipt else {}
            if (not receipt or not receipt.get("decision_id") or type(receipt.get("observed_at")) not in {int, float}
                    or type(observed_at) not in {int, float} or not math.isfinite(observed_at)
                    or receipt["observed_at"] > observed_at
                    or any(not isinstance(job.get(key), str) or not IDENTIFIER.fullmatch(job[key])
                           for key in ("model_instance_id", "job_id", "attempt_id"))
                    or context.get("instance_id") != job.get("model_instance_id")
                    or context.get("job_id") != job.get("job_id") or context.get("attempt_id") != job.get("attempt_id")):
                gates[gate_id] = None
        return gates

    def budget_audit(self, payload):
        """Read an identity-bound page without retaining the main state lock."""
        with self._lock:
            model_id = self.model_instance_id
            if (payload.get("instance_id") != self.instance_id or not model_id
                    or payload.get("model_instance_id") != model_id):
                raise ControlError("Audit owner changed", status=409)
        request = {key: payload[key] for key in ("budget_instance_id", "after_sequence", "through_sequence", "limit") if key in payload}
        request["instance_id"] = model_id
        after, cutoff, limit = request.get("after_sequence", 0), request.get("through_sequence"), request.get("limit", 100)
        if (type(after) is not int or after < 0 or type(limit) is not int or not 1 <= limit <= 256
                or (cutoff is not None and (type(cutoff) is not int or cutoff < after))):
            raise ControlError("Invalid budget audit range", status=422)
        request.update(after_sequence=after, limit=limit)
        page = self._rpc(f"http://127.0.0.1:{self.model_port}", self.key, "budget_audit", request, timeout=self.probe_timeout)
        with self._lock:
            if (not isinstance(page, dict) or self.model_instance_id != model_id or page.get("model_instance_id") != model_id
                    or page.get("budget_instance_id") != payload.get("budget_instance_id")):
                raise ControlError("Audit owner changed", status=409)
        try:
            return public_budget_page(page, request)
        except ValueError:
            raise ControlError("Budget audit page is invalid", status=503) from None

    def archive(self, payload):
        """Record a client-confirmed Drive upload for the committed artifact."""
        job_id, file_id = payload.get("job_id"), payload.get("file_id")
        sha256, size = payload.get("sha256"), payload.get("bytes")
        if (not isinstance(job_id, str) or not IDENTIFIER.fullmatch(job_id)
                or not isinstance(file_id, str) or not DRIVE_FILE_ID.fullmatch(file_id)
                or not isinstance(sha256, str) or not SHA256.fullmatch(sha256)
                or type(size) is not int or size <= 0):
            raise ControlError("Invalid Google Drive archive receipt", status=422)
        receipt = {"provider": "google_drive", "file_id": file_id, "sha256": sha256, "bytes": size}
        with self._lock, self._db:
            job = self._get_job(job_id)
            if job["status"] != "succeeded":
                raise ControlError("Only a succeeded job can be archived", status=409)
            artifact = job.get("artifact")
            media = artifact.get("media") if isinstance(artifact, dict) else None
            if (not isinstance(media, dict) or media.get("full_decode_passed") is not True
                    or artifact.get("sha256") != sha256 or artifact.get("bytes") != size):
                raise ControlError("Archive receipt does not match the committed artifact", status=409)
            previous = job.get("archive")
            if previous is None:
                job["archive"] = receipt
                self._store_job(job)
                self._event("job.archived", job=job, data={"provider": "google_drive"})
            elif previous != receipt:
                raise ControlError("Job already has a different archive receipt", status=409)
            return self._public_job(job)

    def cancel(self, job_id):
        with self._control_lock:
            with self._lock, self._db:
                job = self._get_job(job_id)
                if job["status"] in TERMINAL:
                    return self._public_job(job)
                if job["status"] == "queued":
                    cleaned, cleanup_errors = self._cleanup_job_files(job_id, discard_result=True)
                    self._transition(job, "cancelled", finished_at=time.time(), cleanup_complete=cleaned,
                                     cleanup_errors=cleanup_errors, model_lease_released=True)
                    return self._public_job(job)
                if job["status"] == "running":
                    self._transition(job, "cancel_requested")
            try:
                self._rpc(f"http://127.0.0.1:{self.model_port}", self.key, "cancel",
                          {"job_id": job_id, "attempt_id": job["attempt_id"], "instance_id": job["model_instance_id"]},
                          timeout=self.probe_timeout)
            except Exception:
                pass  # Request remains pending; execution lease is not released.
            return self._public_job(job)

    def _terminate(self, name, timeout=15):
        child = self.children[name]
        if child is None:
            return True
        owner = self.owners.get(name)
        if owner:
            self._register_process_tree()
            return self._record_tree_exit(name, owner.terminate_all(timeout))
        try:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=3)
            exited = child.poll() is not None
            if exited and hasattr(child, "_h3_launch_id"):
                with self._lock, self._db:
                    self._db.execute("UPDATE owned_processes SET exited=1 WHERE launch_id=?", (child._h3_launch_id,))
            return exited
        except (OSError, subprocess.SubprocessError) as error:
            self._cleanup_errors.append(name + ": " + type(error).__name__)
            return False

    def _record_tree_exit(self, name, observed):
        if observed["cleanup_complete"]:
            with self._lock, self._db:
                for launch_id in self.children[name]._h3_tree_records.values():
                    self._db.execute("UPDATE owned_processes SET exited=1 WHERE launch_id=?", (launch_id,))
            return True
        self._cleanup_errors.append(name + ".tree: " + "; ".join(observed["errors"] or ["exit unconfirmed"]))
        return False

    def _stop_model(self, *, force=False, reason="model_stopped"):
        with self._lock:
            active = self._active_jobs()
            queued = self._db.execute("SELECT COUNT(*) FROM jobs WHERE status='queued'").fetchone()[0]
            if (active or queued) and not force:
                raise ControlError("Model has unfinished jobs; drain or explicitly force stop", status=409)
            self._model_failure = reason
            self._model["ready"] = False
        if self._terminate("model"):
            self._fail_jobs_after_exit(reason)
            downloads = self._model.get("downloads")
            self._model = {"instance_id": self.model_instance_id, "phase": "stopped", "ready": False,
                           "worker_alive": False, "cleanup_complete": True}
            if isinstance(downloads, dict):
                self._model["downloads"] = {**deepcopy(downloads), "observation_stale": True}
            with self._lock, self._db:
                self._event("model.stopped", data={"reason": reason, "cleanup_complete": True})
        else:
            raise ControlError("Model exit is unconfirmed; lease retained", status=503)

    def _control_action(self, action, payload):
        with self._control_lock:
            if action == "shutdown":
                state = self.shutdown()
                if (state.get("cleanup_complete") is not True or state.get("process_observation_errors")
                        or any(state.get(name) is not False for name in ("server_alive", "model_alive", "tunnel_alive"))):
                    raise ControlError("Shutdown cleanup is unconfirmed; manager remains available for retry", status=503)
            elif action == "stop_model":
                self._stop_model(force=payload.get("force") is True)
            elif action == "start_model":
                if self._shutdown_started or self._phase in {"failed", "stopped"}:
                    raise ControlError("Manager is stopping or failed", status=409)
                self._launch_model()
            elif action == "stop_api":
                with self._lock, self._db:
                    self._set_api_expectation(False)
                    self._api_intent_epoch += 1
                    self._api_lifecycle = "stopping"
                    self._api_ready = self._public_api_ready = False
                    if self._error and self._error.startswith("api_"):
                        self._error = None
                    self._event("api.stop_requested", data={"intent_epoch": self._api_intent_epoch})
                if not self._terminate("api"):
                    self._api_lifecycle = "stop_unconfirmed"
                    raise ControlError("API process exit is unconfirmed", status=503)
                self._api_lifecycle = "stopped"
                with self._lock, self._db:
                    self._event("api.stopped", data={"intent_epoch": self._api_intent_epoch, "cleanup_complete": True})
            elif action == "start_api":
                if self._shutdown_started or self._phase in {"failed", "stopped"}:
                    raise ControlError("Manager cannot start API", status=409)
                self._launch_api()
            elif action == "drain":
                self._draining = True  # Existing queue continues to drain.
            elif action == "resume":
                if self._shutdown_started or self._phase in {"failed", "stopped"}:
                    raise ControlError("Manager cannot resume", status=409)
                self._draining = False
            else:
                raise ControlError("Unknown control action", status=422)
            self._checkpoint()

    def control(self, payload):
        action = payload.get("action")
        if action not in {"drain", "resume", "start_model", "stop_model", "start_api", "stop_api", "shutdown"}:
            raise ControlError("Unknown control action", status=422)
        command_id = payload.get("command_id") or uuid.uuid4().hex
        if not isinstance(command_id, str) or not 1 <= len(command_id) <= 128:
            raise ControlError("Invalid command ID", status=422)
        fingerprint = json.dumps({"action": action, "force": payload.get("force") is True}, sort_keys=True)
        record = {"command_id": command_id, "action": action, "status": "accepted", "created_at": time.time(), "error": None}

        def run():
            try:
                self._control_action(action, payload)
                result = {**record, "status": "succeeded", "finished_at": time.time()}
            except BaseException as error:
                LOG.exception("Control operation failed")
                result = {**record, "status": "failed", "error": str(error) if isinstance(error, ControlError) else type(error).__name__,
                          "finished_at": time.time()}
            try:
                with self._lock, self._db:
                    self._db.execute("UPDATE commands SET record=? WHERE id=?", (json.dumps(result), command_id))
                    self._event("command." + result["status"], data={"command_id": command_id, "action": action, "error": result.get("error")})
                self._checkpoint()
                if action == "shutdown" and result["status"] == "succeeded":
                    self._exit_requested = True
            finally:
                self._control_slots.release()

        thread = threading.Thread(target=run, name="main-control-" + action, daemon=True)
        with self._lock:
            existing = self._db.execute("SELECT fingerprint,record FROM commands WHERE id=?", (command_id,)).fetchone()
            if existing:
                if existing[0] != fingerprint:
                    raise ControlError("Command ID conflicts", status=409)
                return json.loads(existing[1])
            if self._control_admission_closed or (self._shutdown_started and action != "shutdown"):
                raise ControlError("Manager is closing; new control operations are unavailable", status=409)
            if not self._control_slots.acquire(blocking=False):
                raise ControlError("Control operation queue is full", status=429)
            try:
                with self._db:
                    self._db.execute("INSERT INTO commands VALUES (?,?,?)", (command_id, fingerprint, json.dumps(record)))
                    self._event("command.accepted", data={"command_id": command_id, "action": action})
            except BaseException:
                self._control_slots.release()
                raise
            # Admission, registration and start share close()'s barrier. An
            # accepted command can never appear after its worker join snapshot.
            self._control_threads = [t for t in self._control_threads if t.ident is None or t.is_alive()]
            self._control_threads.append(thread)
            try:
                thread.start()
            except BaseException:
                self._control_slots.release()
                with self._db:
                    record.update(status="failed", error="command_worker_start_failed", finished_at=time.time())
                    self._db.execute("UPDATE commands SET record=? WHERE id=?", (json.dumps(record), command_id))
                    self._event("command.failed", data={"command_id": command_id, "error": record["error"]})
                raise
        return record

    def dispatch(self, operation, payload=None):
        payload = payload or {}
        if operation in {"submit", "cancel", "control", "archive"} and payload.get("expected_instance_id", self.instance_id) != self.instance_id:
            raise ControlError("Manager instance changed; command rejected", status=409)
        if operation == "status":
            return self.snapshot()
        if operation == "budget_audit":
            return self.budget_audit(payload)
        if operation == "submit":
            return self.submit(payload)
        if operation == "job":
            with self._lock:
                return self._public_job(self._get_job(payload.get("job_id")))
        if operation == "archive":
            return self.archive(payload)
        if operation == "cancel":
            return self.cancel(payload.get("job_id"))
        if operation == "control":
            return self.control(payload)
        if operation == "command":
            with self._lock:
                row = self._db.execute("SELECT record FROM commands WHERE id=?", (payload.get("command_id"),)).fetchone()
                if not row:
                    raise ControlError("Command not found", status=404)
                return json.loads(row[0])
        if operation == "events":
            try:
                after, limit = max(0, int(payload.get("after", 0))), min(500, max(1, int(payload.get("limit", 100))))
            except (TypeError, ValueError):
                raise ControlError("Invalid event cursor", status=422)
            with self._lock:
                rows = self._db.execute("SELECT sequence,record FROM events WHERE sequence>? ORDER BY sequence LIMIT ?", (after, limit)).fetchall()
            events = [{**json.loads(encoded), "sequence": seq} for seq, encoded in rows]
            return {"events": events, "next_cursor": events[-1]["sequence"] if events else after}
        raise ControlError("Unknown manager operation", status=404)

    def shutdown(self):
        with self._control_lock:
            self._shutdown_started, self._draining, self._phase = True, True, "stopping"
            with self._lock:
                self._set_api_expectation(False)
                self._api_intent_epoch += 1
                self._api_lifecycle = "stopping"
                self._api_ready = self._public_api_ready = False
            self._cancel.set()
            self._register_process_tree()
            names = [name for name in ("tunnel", "api", "model") if name in self.owners]
            results = stop_many([self.owners[name] for name in names], timeout=15)
            for name, observed in zip(names, results):
                self._record_tree_exit(name, observed)
            alive, errors = self._liveness()
            self._api_lifecycle = "stopped" if not alive["api"] and "api" not in errors else "stop_unconfirmed"
            if not alive["model"] and "model" not in errors:
                self._fail_jobs_after_exit("manager_stopped")
            with self._lock:
                historical = self._historical_ownership()
            self._phase = "failed" if any(alive.values()) or errors or historical else "stopped"
            if any(alive.values()) or errors or historical:
                self._error = "cleanup_incomplete; owned processes remain"
            elif self._error in {"cleanup_incomplete; owned processes remain", "audit_export_failed_during_shutdown"}:
                self._error = None
            try:
                self._checkpoint()
            except (OSError, sqlite3.Error):
                self._error = "audit_export_failed_during_shutdown"
                LOG.exception("Could not export shutdown audit")
            return self.snapshot()

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._control_admission_closed = True
        self.shutdown()
        if self._thread and self._thread is not threading.current_thread() and self._thread.ident is not None:
            self._thread.join(timeout=3 * self.probe_timeout + self.poll_interval + 2)
            if self._thread.is_alive():
                raise RuntimeError("Cannot finalize shutdown with a live supervisor")
        for thread in self._control_threads:
            if thread is not threading.current_thread() and thread.ident is not None:
                thread.join(timeout=20)
                if thread.is_alive():
                    raise RuntimeError("Cannot finalize shutdown with a live control operation")
        self._probe_pool.shutdown(wait=True, cancel_futures=True)
        alive, errors = self._liveness()
        if any(value is not False for value in alive.values()) or errors or self._historical_ownership():
            raise RuntimeError("Cannot release root ownership with live children")
        with self._lock:
            self._shutdown_complete = True
            try:
                with self._db:
                    self._event("manager.shutdown_complete")
                self._checkpoint()
            except BaseException:
                self._shutdown_complete = False
                raise
            self._db.close()
            self._owner.close()
            self._closed = True
        for output in self._files:
            output.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("port", "api-port", "model-port"):
        parser.add_argument("--" + argument, required=True, type=int)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--staged", action="store_true")
    parser.add_argument("--public", action="store_true")
    parser.add_argument("--install-cloudflared", action="store_true")
    parser.add_argument("--no-model-autostart", action="store_true",
                        help="Start API/public transport only; start_model control can launch the model later")
    parser.add_argument("--parent-pid", type=int)
    parser.add_argument("--startup-timeout", type=float, default=3600)
    parser.add_argument("--generation-timeout", type=float, default=5400)
    parser.add_argument("--poll-interval", type=float, default=1)
    parser.add_argument("--probe-timeout", type=float, default=2)
    parser.add_argument("--tunnel-timeout", type=float, default=180)
    args = parser.parse_args()
    manager = GlobalManager(args.directory, root=args.root, instance_id=args.instance_id,
                            port=args.port, api_port=args.api_port, model_port=args.model_port,
                            staged=args.staged, public=args.public, install_cloudflared=args.install_cloudflared,
                            start_model=not args.no_model_autostart,
                            startup_timeout=args.startup_timeout, generation_timeout=args.generation_timeout,
                            poll_interval=args.poll_interval, probe_timeout=args.probe_timeout, tunnel_timeout=args.tunnel_timeout)
    server = make_server("127.0.0.1", args.port, manager.key, manager.dispatch)
    serve_thread = threading.Thread(target=server.serve_forever, name="manager-control-http", daemon=True)
    stop_requested = False

    def request_stop(*_):
        # Python signal handlers can interrupt/re-enter Event.set while its
        # non-reentrant Condition lock is held. Only assign a flag here.
        nonlocal stop_requested
        stop_requested = True
    parent_identity = None
    if args.parent_pid:
        try:
            parent_identity = float(os.environ.get("H3_PARENT_CREATED_AT") or psutil.Process(args.parent_pid).create_time())
        except (psutil.Error, ValueError):
            stop_requested = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        serve_thread.start()
        if stop_requested:
            raise RuntimeError("Manager owner identity is unavailable")
        manager.start()
        while not stop_requested and not manager._exit_requested:
            time.sleep(.1)
            if args.parent_pid:
                try:
                    parent = psutil.Process(args.parent_pid)
                    if parent_identity is None or abs(parent.create_time() - parent_identity) >= .001 or not parent.is_running():
                        break
                except psutil.Error:
                    break
    finally:
        manager.close()
        server.shutdown()
        server.server_close()
        serve_thread.join(timeout=3)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
