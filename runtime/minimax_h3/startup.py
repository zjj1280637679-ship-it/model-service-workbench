"""Managed launcher/client: global_manager owns API, model and tunnel lifecycles."""
from __future__ import annotations
from copy import deepcopy
import json
import os
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from urllib.request import urlopen
from control_protocol import ControlError, rpc_call
from process_owner import ProcessOwner


def assert_port_available(port):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", int(port)))


def _free_port(excluded):
    while True:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        if port not in excluded:
            return port


def probe_ready(url, instance_id, timeout=3):
    try:
        with urlopen(url.rstrip("/") + "/health", timeout=timeout) as response:
            data = json.loads(response.read(16384))
        return isinstance(data, dict) and data.get("ready") is True and data.get("instance_id") == instance_id
    except Exception:
        return False


class StartupSession:
    def __init__(self, directory, *, port=8000, staged=False, public=False,
                 install_cloudflared=False, env=None, startup_timeout=3600,
                 tunnel_timeout=180, poll_interval=1, probe_timeout=3, clock=None,
                 start_model=True):
        self.directory = Path(directory).resolve()
        self.port, self.staged, self.public = int(port), bool(staged), bool(public)
        if not 1 <= self.port <= 65535 or min(startup_timeout, tunnel_timeout, poll_interval, probe_timeout) <= 0:
            raise ValueError("Valid ports and positive timeouts are required")
        self.install_cloudflared = bool(install_cloudflared)
        self.model_autostart = bool(start_model)
        self.startup_timeout, self.tunnel_timeout = startup_timeout, tunnel_timeout
        self.poll_interval, self.probe_timeout = poll_interval, probe_timeout
        self._clock = clock or time.monotonic
        self._environment = dict(os.environ if env is None else env)
        self.api_key = self._environment.get("H3_API_KEY") or secrets.token_urlsafe(32)
        self._control_key = self._environment.get("H3_CONTROL_KEY") or secrets.token_urlsafe(32)
        for token in (self.api_key, self._control_key):
            if len(token) < 32 or not token.isascii() or any(c.isspace() for c in token):
                raise ValueError("Tokens require at least 32 non-whitespace ASCII characters")
        if self.api_key == self._control_key:
            raise ValueError("Public and private control tokens must be distinct")
        self.instance_id = uuid.uuid4().hex
        self.root = Path(self._environment.get("H3_OUTPUT_DIR", self.directory / "h3_outputs")).resolve()
        self.run_directory = self.directory / "runtime" / self.instance_id
        self.state_path = self.run_directory / "status.json"
        self.events_path = self.run_directory / "events.jsonl"
        self.server_log = self.run_directory / "model.log"
        self.api_log = self.run_directory / "api.log"
        self.tunnel_log = self.run_directory / "tunnel.log"
        self.manager_log = self.run_directory / "manager.log"
        self.manager = self._output = self._started_at = self._manager_owner = None
        self._state_lock = threading.Lock()
        self._last = dict(instance_id=self.instance_id, phase="new", ready=False,
                          api_ready=False, model_ready=False, public_ready=False,
                          public_requested=self.public, public_url=None,
                          public_api_ready=False, tunnel_connected=False,
                          server_alive=False, model_alive=False, tunnel_alive=False,
                          manager_alive=False, cleanup_complete=True, shutdown_complete=True, error=None)
        self.manager_url = None

    @staticmethod
    def _cleanup_confirmed(state):
        return (state.get("phase") in {"stopped", "failed"} and state.get("cleanup_complete") is True
                and state.get("shutdown_complete") is True
                and all(state.get(name) is False for name in ("server_alive", "model_alive", "tunnel_alive"))
                and not state.get("process_observation_errors"))

    def _remember(self, state):
        # RPCs can finish out of order while the notebook polls during stop().
        # A verified terminal receipt is monotonic for this manager instance.
        with self._state_lock:
            if not self._cleanup_confirmed(self._last) or self._cleanup_confirmed(state):
                self._last = deepcopy(state)
            return deepcopy(self._last)

    def _read_final_receipt(self):
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if isinstance(state, dict) and state.get("instance_id") == self.instance_id and self._cleanup_confirmed(state):
                self._remember(state)
        except (OSError, ValueError):
            pass

    def _manager_tree_alive(self):
        if self._manager_owner:
            return self._manager_owner.observe()["alive"]
        return self.manager is not None and self.manager.poll() is None

    def _owner_observation(self):
        if self._manager_owner:
            return self._manager_owner.observe()
        alive = self.manager is not None and self.manager.poll() is None
        return {"alive": alive, "cleanup_complete": not alive, "errors": [],
                "pids": [self.manager.pid] if alive else [], "identities": [],
                "scope": "manager_launcher_only", "observed_at": time.time()}

    def _unavailable_snapshot(self, owner_state, *, stopped=False):
        """Keep the last RPC as history; never mix its readiness with current exit evidence."""
        if owner_state.get("alive") is False:
            self._read_final_receipt()
        with self._state_lock:
            history = deepcopy(self._last)
        result = deepcopy(history)
        errors = list(owner_state.get("errors", []))
        tree_complete = (owner_state.get("cleanup_complete") is True and not errors
                         and owner_state.get("scope") != "manager_launcher_only")
        manager_pid = history.get("manager_pid", getattr(self.manager, "pid", None))
        manager_alive = (None if errors else manager_pid in owner_state.get("pids", []))
        if owner_state.get("scope") == "manager_launcher_only":
            manager_alive = owner_state["alive"]
        confirmed = self._cleanup_confirmed(history) and owner_state.get("cleanup_complete") is True and not errors
        children_exited = tree_complete or (self._cleanup_confirmed(history) and manager_alive is False
                                            and owner_state.get("scope") == "manager_launcher_only")
        child_alive = False if children_exited else None
        current_time = time.time()
        owner_observed_at = owner_state.get("observed_at")
        manager_observed_at = history.get("observed_at")
        observation_age = lambda timestamp: (round(max(0, current_time - timestamp), 3)
                                              if isinstance(timestamp, (int, float)) else None)
        modules = {name: {"alive": child_alive, "ready": False,
                          "observation_unknown": child_alive is None}
                   for name in ("api", "model", "tunnel")}
        modules["manager"] = {"alive": manager_alive, "ready": False,
                              "supervisor_alive": False if manager_alive is False else None,
                              "observation_unknown": manager_alive is None}
        current_owned = {}
        for name, previous in history.get("owned_processes", {}).items():
            current_owned[name] = {"alive": child_alive, "cleanup_complete": children_exited,
                                   "pids": [] if children_exited else None,
                                   "identities": deepcopy(previous.get("identities", [])),
                                   "errors": [] if children_exited else ["manager_observation_unavailable"],
                                   "observed_at": current_time}
        result.update(last_manager_observation=history, manager_process_tree=deepcopy(owner_state),
                      manager_observed_at=manager_observed_at, observed_at=current_time,
                      owner_observed_at=owner_observed_at,
                      observation_age_seconds=observation_age(owner_observed_at),
                      manager_observation_age_seconds=observation_age(manager_observed_at),
                      manager_launcher_pid=getattr(self.manager, "pid", None), manager_alive=manager_alive,
                      server_alive=child_alive, model_alive=child_alive, tunnel_alive=child_alive,
                      ready=False, api_ready=False, model_ready=False, public_ready=False,
                      public_api_ready=False, tunnel_connected=False, supervisor_alive=modules["manager"]["supervisor_alive"],
                      modules=modules, owned_processes=current_owned,
                      cleanup_complete=confirmed, observation_stale=True, manager_observation_stale=True,
                      phase="stopped" if confirmed else "failed" if stopped or manager_alive is False else "degraded")
        process_errors = deepcopy(history.get("process_observation_errors", {}))
        if errors:
            process_errors["session_owner"] = errors
        result["process_observation_errors"] = process_errors
        if isinstance(history.get("downloads"), dict):
            result["downloads"] = {**deepcopy(history["downloads"]), "model_ready": False,
                                   "observation_stale": True,
                                   "lifecycle": "stopped" if children_exited else "unavailable"}
        if not confirmed:
            result["error"] = ("Owned process cleanup remains unconfirmed" if errors else
                               "Manager exited before cleanup was confirmed" if manager_alive is False else
                               "Manager control observation unavailable")
        return result

    def start(self, *, background=True):
        if self._started_at is not None:
            raise RuntimeError("Session already started; query or stop it")
        assert_port_available(self.port)
        manager_port = _free_port({self.port})
        model_port = _free_port({self.port, manager_port})
        self.manager_url = f"http://127.0.0.1:{manager_port}"
        self._environment.update(H3_API_KEY=self.api_key, H3_CONTROL_KEY=self._control_key,
                                 H3_INSTANCE_ID=self.instance_id, H3_OUTPUT_DIR=str(self.root),
                                 H3_STARTUP_TIMEOUT=str(self.startup_timeout), H3_TUNNEL_TIMEOUT=str(self.tunnel_timeout))
        self.run_directory.mkdir(parents=True, exist_ok=False)
        self._started_at = self._clock()
        self._last.update(phase="preparing", cleanup_complete=False, shutdown_complete=False)
        command = [sys.executable, "-u", str(self.directory / "global_manager.py"),
                   "--port", str(manager_port), "--api-port", str(self.port),
                   "--model-port", str(model_port), "--directory", str(self.directory),
                   "--instance-id", self.instance_id, "--root", str(self.root), "--parent-pid", str(os.getpid()),
                   "--startup-timeout", str(self.startup_timeout), "--probe-timeout", str(self.probe_timeout),
                   "--poll-interval", str(self.poll_interval), "--tunnel-timeout", str(self.tunnel_timeout)]
        if self.staged:
            command.append("--staged")
        if self.public:
            command.append("--public")
        if self.install_cloudflared:
            command.append("--install-cloudflared")
        if not self.model_autostart:
            command.append("--no-model-autostart")
        try:
            self._output = self.manager_log.open("w", encoding="utf-8")
            self.manager = subprocess.Popen(command, cwd=self.directory, env=self._environment,
                                             stdout=self._output, stderr=subprocess.STDOUT,
                                             start_new_session=os.name != "nt")
            self._manager_owner = ProcessOwner(self.manager)
            deadline = self._clock() + 20
            while self._clock() < deadline:
                if not self._manager_tree_alive():
                    raise RuntimeError("Global manager exited during launch; inspect manager.log")
                try:
                    state = rpc_call(self.manager_url, self._control_key, "status", timeout=1)
                    if state.get("instance_id") != self.instance_id:
                        raise RuntimeError("Manager identity mismatch")
                    self._remember(state)
                    self._manager_owner.observe()
                    return self
                except ControlError:
                    time.sleep(.1)
            raise TimeoutError("Global manager control endpoint did not start")
        except BaseException:
            self.stop()
            raise

    def snapshot(self):
        if self._started_at is None:
            return dict(self._last, elapsed_seconds=0)
        owner_state = self._owner_observation()
        if owner_state["alive"]:
            try:
                state = rpc_call(self.manager_url, self._control_key, "status", timeout=self.probe_timeout)
                if not isinstance(state, dict) or state.get("instance_id") != self.instance_id:
                    raise ControlError("Manager identity mismatch", 502)
                state = self._remember(state)
                owner_state = self._owner_observation()
                manager_missing = (owner_state.get("scope") != "manager_launcher_only"
                                   and state.get("manager_pid") is not None
                                   and state["manager_pid"] not in owner_state.get("pids", []))
                if owner_state.get("errors") or not owner_state["alive"] or manager_missing:
                    return self._unavailable_snapshot(owner_state)
                return dict(state, manager_alive=True, manager_launcher_pid=self.manager.pid,
                            manager_process_tree=owner_state)
            except ControlError:
                return self._unavailable_snapshot(self._owner_observation())
        return self._unavailable_snapshot(owner_state)

    def control(self, action, **parameters):
        if not self.manager or not self._manager_tree_alive():
            raise ControlError("Manager is not running", 503)
        return rpc_call(self.manager_url, self._control_key, "control",
                        {"action": action, "expected_instance_id": self.instance_id, **parameters}, timeout=10)

    def command(self, command_id):
        return rpc_call(self.manager_url, self._control_key, "command", {"command_id": command_id})

    def events(self, after=0, limit=100):
        return rpc_call(self.manager_url, self._control_key, "events", {"after": after, "limit": limit})

    def budget_audit(self, model_instance_id, budget_instance_id, after_sequence=0, through_sequence=None, limit=100):
        payload = {"instance_id": self.instance_id, "model_instance_id": model_instance_id,
                   "budget_instance_id": budget_instance_id, "after_sequence": after_sequence, "limit": limit}
        if through_sequence is not None:
            payload["through_sequence"] = through_sequence
        return rpc_call(self.manager_url, self._control_key, "budget_audit", payload, timeout=self.probe_timeout)

    def wait_ready(self, on_update=None):
        while True:
            state = self.snapshot()
            if on_update:
                on_update(state)
            if state.get("ready"):
                return state
            if state.get("phase") in {"failed", "stopped", "stopping"} or state.get("error") and state.get("model_alive") is False:
                raise RuntimeError(state.get("error") or "Preparation failed; control interface remains available")
            if self._clock() - self._started_at > self.startup_timeout:
                raise TimeoutError("Generation readiness deadline exceeded; query or stop the manager")
            time.sleep(self.poll_interval)

    def stop(self):
        if self.manager is None:
            self._last.update(phase="stopped", cleanup_complete=True, manager_alive=False)
            return dict(self._last)
        if self._manager_tree_alive():
            deadline = time.monotonic() + 30
            requested = False
            command_id = "shutdown-" + uuid.uuid4().hex
            while time.monotonic() < deadline and self._manager_tree_alive():
                try:
                    if not requested:
                        self.control("shutdown", command_id=command_id)
                        requested = True
                    self.snapshot()
                except (ControlError, OSError):
                    # A lost acknowledgement is not permission to interrupt audit
                    # finalization. Retrying this ID cannot enqueue a second stop.
                    pass
                if self.manager.poll() is not None:
                    break
                time.sleep(.2)
            # The manager exits itself only after its final command/audit checkpoint.
            # A clean child snapshot alone must not trigger an early manager signal.
        if self._manager_owner:
            owner_state = self._manager_owner.observe()
            if owner_state["alive"]:
                owner_state = self._manager_owner.terminate_all(timeout=15)
        else:
            if self.manager.poll() is None:
                self.manager.terminate()
                self.manager.wait(timeout=15)
            owner_state = self._owner_observation()
        if self._output and not self._output.closed:
            self._output.close()
        try:
            recorded = json.loads(self.state_path.read_text(encoding="utf-8"))
            if recorded.get("instance_id") == self.instance_id:
                self._remember(recorded)
        except (OSError, ValueError):
            pass
        return self._unavailable_snapshot(owner_state, stopped=True)
