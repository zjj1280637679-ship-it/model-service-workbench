"""Prepare an explicit temporary tunnel independently of model readiness."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import signal
import shutil
import subprocess
import sys
import time
import threading


def watch_owner(parent_pid):
    """Exit through the signal handler so the connector is also reclaimed."""
    if not parent_pid:
        return
    import psutil
    try:
        owner = psutil.Process(int(parent_pid))
        created = owner.create_time()
    except psutil.Error:
        raise RuntimeError("Tunnel owner is already unavailable")

    def observe():
        while True:
            time.sleep(1)
            try:
                alive = (owner.is_running() and owner.create_time() == created
                         and owner.status() != psutil.STATUS_ZOMBIE)
            except psutil.Error:
                alive = False
            if not alive:
                if os.name == "nt":
                    import _thread
                    _thread.interrupt_main()
                else:
                    os.kill(os.getpid(), signal.SIGTERM)
                return
    threading.Thread(target=observe, name="tunnel-owner-watch", daemon=True).start()
import urllib.request


def ready(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as response:
            return json.load(response).get("ready") is True
    except Exception:
        return False


def cloudflared_binary(install: bool = False) -> str:
    found = shutil.which("cloudflared")
    if found:
        return found
    cached = Path(__file__).resolve().parent / ".cache" / "cloudflared"
    if cached.is_file():
        return str(cached)
    if not install:
        raise RuntimeError("cloudflared is not installed. Use --install to fetch an official release with SHA-256 verification.")
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "amd64"}:
        raise RuntimeError("Automatic cloudflared install supports Linux x86_64 only; otherwise install it from Cloudflare first.")
    request = urllib.request.Request(
        "https://api.github.com/repos/cloudflare/cloudflared/releases/latest",
        headers={"User-Agent": "minimax-h3-ref-api", "Accept": "application/vnd.github+json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        release = json.load(response)
    asset = next(a for a in release["assets"] if a["name"] == "cloudflared-linux-amd64")
    digest = asset.get("digest", "")
    if not digest.startswith("sha256:"):
        raise RuntimeError("Official release metadata lacks a SHA-256 digest; install cloudflared manually from Cloudflare.")
    url = asset["browser_download_url"]
    if not url.startswith("https://github.com/cloudflare/cloudflared/releases/download/"):
        raise RuntimeError("Unexpected cloudflared release origin")
    cached.parent.mkdir(parents=True, exist_ok=True)
    temporary = cached.with_suffix(".download")
    hasher = hashlib.sha256()
    try:
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                hasher.update(chunk)
                output.write(chunk)
        if hasher.hexdigest() != digest.split(":", 1)[1]:
            raise RuntimeError("cloudflared SHA-256 mismatch; binary was not installed")
        temporary.chmod(0o700)
        temporary.replace(cached)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Installed official cloudflared {release['tag_name']} (SHA-256 verified).", flush=True)
    return str(cached)


def run_tunnel(port: int = 8000, install: bool = False, *, status_file: Path | None = None,
               instance_id: str | None = None) -> None:
    """Register the transport; a candidate URL never implies application readiness."""
    if status_file and (not instance_id or not re.fullmatch(r"[a-f0-9]{32}", instance_id)):
        raise ValueError("Managed tunnel status requires its owner's 32-character instance ID")
    state = {"instance_id": instance_id, "phase": "preparing", "url": None,
             "connected": False, "pid": None, "error": None}

    def publish(**changes):
        state.update(changes, observed_at=time.time())
        if status_file:
            path = Path(status_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
            temporary.replace(path)

    process = None
    registered = set()
    publish()
    try:
        # A loading origin may not be listening yet. Transport installation and
        # registration do not depend on its /health response.
        binary = cloudflared_binary(install)
        process = subprocess.Popen(
            [binary, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{port}"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        publish(pid=process.pid)
        for line in process.stdout:
            match = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
            if match and not state["url"]:
                publish(url=match.group(0))
                print(f"PUBLIC_API_URL={match.group(0)}", flush=True)
                print("Candidate URL allocated; application readiness still requires the live join.", flush=True)
            connection = re.search(r"connIndex=(\d+)", line)
            connection_id = connection.group(1) if connection else "default"
            if "Registered tunnel connection" in line:
                registered.add(connection_id)
                publish(phase="registered", connected=True)
            elif any(marker in line for marker in (
                "Unregistered tunnel connection", "Connection terminated",
                "failed to serve tunnel connection", "Serve tunnel error",
            )):
                if connection:
                    registered.discard(connection_id)
                else:
                    registered.clear()  # No connection identity: revoke conservatively.
                publish(phase="registered" if registered else "reconnecting", connected=bool(registered))
            print(line, end="", flush=True)
        if process.wait() != 0:
            raise RuntimeError("cloudflared stopped; see the tunnel log above")
    except KeyboardInterrupt:
        publish(phase="stopping", connected=False)
        raise
    except BaseException as error:
        publish(phase="failed", connected=False, error=type(error).__name__)
        raise
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        publish(phase="failed" if state["phase"] == "failed" else "stopped", connected=False)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    watch_owner(os.environ.get("H3_PARENT_PID"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--install", action="store_true", help="Download official cloudflared if missing, checking GitHub release SHA-256")
    parser.add_argument("--status-file", type=Path, help="Atomic transport state checkpoint owned by the startup supervisor")
    parser.add_argument("--instance-id", help="Owning startup session ID; never an API credential")
    args = parser.parse_args()
    try:
        run_tunnel(args.port, args.install, status_file=args.status_file, instance_id=args.instance_id)
    except KeyboardInterrupt:
        print("Tunnel stopped. Local GPU server remains running until you stop it.")
