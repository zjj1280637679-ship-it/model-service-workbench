"""Actual harmless process trees and conservative identity-failure cases."""
import json
import subprocess
import sys
import time

import psutil
import pytest

import process_owner as po


def test_real_parent_child_tree_is_reclaimed(tmp_path):
    child_file = tmp_path / "child.json"
    script = (
        "import subprocess,sys,time,json,pathlib; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
        # Existence is the reader's ready signal, so publish only a closed,
        # complete record. Direct write_text exposes an empty file first.
        f"published=pathlib.Path({str(child_file)!r}); "
        "pending=published.with_suffix('.tmp'); "
        "pending.write_text(json.dumps({'pid':p.pid})); pending.replace(published); "
        "time.sleep(60)"
    )
    parent = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    owner = po.ProcessOwner(parent)
    try:
        deadline = time.monotonic() + 5
        while not child_file.exists() and time.monotonic() < deadline:
            time.sleep(.02)
        assert child_file.exists()
        child_pid = json.loads(child_file.read_text())["pid"]
        child_identity = psutil.Process(child_pid).create_time()
        observed = owner.observe()
        assert parent.pid in observed["pids"] and child_pid in observed["pids"]
        assert len(observed["identities"]) >= 2
        stopped = owner.terminate_all(timeout=2)
        assert stopped["cleanup_complete"] and not stopped["pids"]
        try:
            child = psutil.Process(child_pid)
            assert child.create_time() != child_identity or child.status() == psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            pass
    finally:
        owner.terminate_all(timeout=.1)


class FakeHandle:
    pid = 12345
    alive = True
    def poll(self): return None if self.alive else 0
    def terminate(self): self.alive = False
    def kill(self): self.alive = False


def test_unknown_initial_tree_cannot_be_released_by_wrapper_exit(monkeypatch):
    handle = FakeHandle()
    def denied(_): raise psutil.AccessDenied(handle.pid)
    monkeypatch.setattr(po.psutil, "Process", denied)
    owner = po.ProcessOwner(handle)
    handle.alive = False
    observed = owner.observe()
    assert observed["alive"] and not observed["cleanup_complete"]
    assert any("subtree_exit_unconfirmed" in x for x in observed["errors"])


def test_failed_child_enumeration_remains_unknown_after_root_exit(monkeypatch):
    handle = FakeHandle()
    class Restricted:
        pid = handle.pid
        def create_time(self): return 100.0
        def is_running(self): return handle.alive
        def status(self): return psutil.STATUS_RUNNING
        def children(self, recursive=True): raise psutil.AccessDenied(self.pid)
    monkeypatch.setattr(po.psutil, "Process", lambda _: Restricted())
    owner = po.ProcessOwner(handle)
    assert not owner.observe()["cleanup_complete"]
    handle.alive = False
    result = owner.observe()
    assert not result["cleanup_complete"]
    assert any("subtree_exit_unconfirmed" in x for x in result["errors"])


def test_pid_reuse_is_not_signalled(monkeypatch):
    handle = FakeHandle()
    created = [100.0]
    signals = []
    class Identity:
        pid = handle.pid
        def create_time(self): return created[0]
        def is_running(self): return True
        def status(self): return psutil.STATUS_RUNNING
        def children(self, recursive=True): return []
        def terminate(self): signals.append("terminate")
        def kill(self): signals.append("kill")
    monkeypatch.setattr(po.psutil, "Process", lambda _: Identity())
    owner = po.ProcessOwner(handle)
    created[0] = 200.0
    handle.alive = False
    assert owner.terminate_all(timeout=0)["cleanup_complete"]
    assert signals == []


@pytest.mark.parametrize("kill", [False, True])
@pytest.mark.parametrize("identity_signal_fails", [False, True])
def test_root_gets_one_signal_with_handle_fallback_only_on_failure(monkeypatch, kill, identity_signal_fails):
    handle = FakeHandle()
    handle_signals, identity_signals = [], []
    monkeypatch.setattr(handle, "terminate", lambda: handle_signals.append("terminate"))
    monkeypatch.setattr(handle, "kill", lambda: handle_signals.append("kill"))

    class Identity:
        pid = handle.pid
        def create_time(self): return 100.0
        def is_running(self): return True
        def status(self): return psutil.STATUS_RUNNING
        def children(self, recursive=True): return []
        def signal(self, kind):
            if identity_signal_fails:
                raise psutil.AccessDenied(self.pid)
            identity_signals.append(kind)
        def terminate(self): self.signal("terminate")
        def kill(self): self.signal("kill")

    monkeypatch.setattr(po.psutil, "Process", lambda _: Identity())
    po.ProcessOwner(handle)._signal(kill=kill)
    expected = ["kill" if kill else "terminate"]
    assert identity_signals == ([] if identity_signal_fails else expected)
    assert handle_signals == (expected if identity_signal_fails else [])


def test_successful_recapture_can_resolve_initial_access_failure(monkeypatch):
    handle = FakeHandle()
    denied = [True]
    class Identity:
        pid = handle.pid
        def create_time(self): return 100.0
        def is_running(self): return handle.alive
        def status(self): return psutil.STATUS_RUNNING
        def children(self, recursive=True): return []
    def factory(_):
        if denied[0]: raise psutil.AccessDenied(handle.pid)
        return Identity()
    monkeypatch.setattr(po.psutil, "Process", factory)
    owner = po.ProcessOwner(handle)
    assert owner.observe()["errors"]
    denied[0] = False
    assert not owner.observe()["errors"]
    handle.alive = False
    assert owner.observe()["cleanup_complete"]


@pytest.mark.parametrize("exit_pid", [12345, 12346], ids=["known-root", "known-descendant"])
@pytest.mark.parametrize("exit_error", [psutil.NoSuchProcess, psutil.ZombieProcess], ids=["exited", "zombie"])
def test_known_process_exit_during_child_enumeration_is_not_unknown(monkeypatch, exit_pid, exit_error):
    handle = FakeHandle()
    alive = {12345: True, 12346: True}
    trigger = [False]

    class Identity:
        def __init__(self, pid): self.pid = pid
        def create_time(self): return float(self.pid)
        def is_running(self): return alive[self.pid]
        def status(self): return psutil.STATUS_RUNNING
        def children(self, recursive=True):
            if trigger[0] and self.pid == exit_pid:
                alive[self.pid] = False
                if self.pid == handle.pid:
                    handle.alive = False
                raise exit_error(self.pid)
            return [Identity(12346)] if self.pid == 12345 and alive[12346] else []

    monkeypatch.setattr(po.psutil, "Process", Identity)
    owner = po.ProcessOwner(handle)
    assert owner.observe()["pids"] == [12345, 12346]
    trigger[0] = True
    observed = owner.observe()
    assert observed["errors"] == []
    assert not observed["cleanup_complete"]  # The other known process still lives.
    assert observed["pids"] == [pid for pid in alive if alive[pid]]
    alive.update({12345: False, 12346: False})
    handle.alive = False
    final = owner.observe()
    assert final["cleanup_complete"] and final["pids"] == [] and final["errors"] == []


def test_exit_race_does_not_erase_previous_access_denied_scope(monkeypatch):
    handle = FakeHandle()
    phase = ["denied"]

    class Identity:
        pid = handle.pid
        def create_time(self): return 100.0
        def is_running(self): return handle.alive
        def status(self): return psutil.STATUS_RUNNING
        def children(self, recursive=True):
            if phase[0] == "denied":
                raise psutil.AccessDenied(self.pid)
            handle.alive = False
            raise psutil.NoSuchProcess(self.pid)

    monkeypatch.setattr(po.psutil, "Process", lambda _: Identity())
    owner = po.ProcessOwner(handle)
    assert any("subtree_exit_unconfirmed" in error for error in owner.observe()["errors"])
    phase[0] = "exited"
    observed = owner.observe()
    assert observed["pids"] == []
    assert not observed["cleanup_complete"]
    assert any("subtree_exit_unconfirmed" in error for error in observed["errors"])
