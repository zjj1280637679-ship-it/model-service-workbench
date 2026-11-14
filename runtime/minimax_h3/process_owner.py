"""Track and stop an owned Popen tree by PID and creation identity.

Windows virtual-environment launchers can add a real interpreter descendant.
An exited wrapper alone is not the definition of a fully reclaimed process tree.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import subprocess
import threading
import time

import psutil


class ProcessOwner:
    def __init__(self, popen):
        self.process = popen
        self._identities = {}
        self._lock = threading.RLock()
        self._root_unknown = False
        self._unknown_scopes = set()
        try:
            process = psutil.Process(popen.pid)
            self._identities[process.pid] = process.create_time()
        except psutil.Error:
            self._root_unknown = True
            self._unknown_scopes.add(popen.pid)

    def _same(self, pid, created):
        try:
            process = psutil.Process(pid)
            if abs(process.create_time() - created) >= .001:
                return None
            if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
                return None
            return process
        except psutil.NoSuchProcess:
            return None

    def _capture(self):
        errors = []
        if self._root_unknown:
            try:
                if self.process.poll() is None:
                    root = psutil.Process(self.process.pid)
                    self._identities[root.pid] = root.create_time()
                    self._root_unknown = False
                # An exited wrapper cannot prove an unobserved subtree exited.
            except (OSError, psutil.Error) as error:
                errors.append("root.poll: " + type(error).__name__)
        for pid, created in list(self._identities.items()):
            try:
                process = self._same(pid, created)
                if process is None:
                    continue
                children = process.children(recursive=True)
                complete = True
                for child in children:
                    try:
                        self._identities.setdefault(child.pid, child.create_time())
                    except psutil.NoSuchProcess:
                        pass
                    except psutil.Error as error:
                        errors.append("descendant.identity: " + type(error).__name__)
                        complete = False
                if complete:
                    self._unknown_scopes.discard(pid)
                else:
                    self._unknown_scopes.add(pid)
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                # A known process can exit between identity validation and
                # children(). This is an exit observation, not a new unknown
                # subtree. Previously recorded access/identity uncertainty is
                # deliberately retained until a successful complete recapture.
                pass
            except psutil.Error as error:
                errors.append(str(pid) + ".children: " + type(error).__name__)
                self._unknown_scopes.add(pid)
        errors.extend(str(pid) + ".subtree_exit_unconfirmed" for pid in self._unknown_scopes)
        return errors

    def observe(self):
        with self._lock:
            errors = self._capture()
            alive = []
            for pid, created in self._identities.items():
                try:
                    if self._same(pid, created) is not None:
                        alive.append(pid)
                except psutil.Error as error:
                    alive.append(pid)  # Unknown exit retains ownership.
                    errors.append(str(pid) + ".observe: " + type(error).__name__)
            try:
                root_running = self.process.poll() is None
            except OSError as error:
                root_running = True
                errors.append("root.poll: " + type(error).__name__)
            if root_running and self.process.pid not in alive:
                alive.append(self.process.pid)
                if self.process.pid not in self._identities:
                    errors.append("root_identity_unconfirmed")
            return {"alive": bool(alive) or bool(errors), "cleanup_complete": not alive and not errors,
                    "pids": sorted(set(alive)), "identities": [{"pid": pid, "created_at": created}
                     for pid, created in self._identities.items()], "errors": sorted(set(errors)),
                    "observed_at": time.time()}

    def _signal(self, kill=False):
        with self._lock:
            self._capture()
            root_signalled = False
            for pid, created in list(self._identities.items()):
                try:
                    process = self._same(pid, created)
                    if process:
                        process.kill() if kill else process.terminate()
                        root_signalled |= pid == self.process.pid
                except psutil.Error:
                    pass  # Final observation, not signal success, decides cleanup.
            # The original Popen handle is tied to its original process. This is
            # also the only safe fallback when its psutil identity was unavailable.
            try:
                if not root_signalled and self.process.poll() is None:
                    self.process.kill() if kill else self.process.terminate()
            except (OSError, subprocess.SubprocessError):
                pass

    def request_stop(self):
        self._signal()
        return self.observe()

    def wait_stopped(self, timeout=15):
        deadline = time.monotonic() + max(0, timeout)
        while True:
            state = self.observe()
            if state["cleanup_complete"]:
                return state
            if time.monotonic() >= deadline:
                break
            time.sleep(min(.05, max(0, deadline - time.monotonic())))
        self._signal(kill=True)
        deadline = time.monotonic() + 3
        while True:
            state = self.observe()
            if state["cleanup_complete"] or time.monotonic() >= deadline:
                return state
            time.sleep(.025)

    def terminate_all(self, timeout=15):
        self.request_stop()
        return self.wait_stopped(timeout)


def stop_many(owners, timeout=15):
    """Independent stop requests and waits overlap; caller joins actual evidence."""
    owners = list(owners)
    for owner in owners:
        owner.request_stop()
    if not owners:
        return []
    try:
        with ThreadPoolExecutor(max_workers=len(owners), thread_name_prefix="owned-process-cleanup") as pool:
            futures = [pool.submit(owner.wait_stopped, timeout) for owner in owners]
            return [future.result() for future in futures]
    except (RuntimeError, OSError):
        # Thread creation failure must not prevent already-requested cleanup.
        deadline = time.monotonic() + timeout
        return [owner.wait_stopped(max(0, deadline - time.monotonic())) for owner in owners]
