"""Exercise a REAL running API and verify its downloaded audiovisual MP4.

This is a low-step integration smoke test, not a generation-quality benchmark.
It has no fake backend, synthetic-video fallback, or GPU/model loader.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import uuid
from urllib.parse import urlsplit

PARAMETERS = {
    "seconds": 5, "width": 512, "height": 288, "steps": 8, "seed": 42,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_report(outputdir: Path, report: dict) -> None:
    """Checkpoint the accepted job ID before polling; resuming never resubmits."""
    target = outputdir / "test_report.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(target)


@contextmanager
def content_history_lock(path):
    """OS-held writer lock; process exit releases it without stale lock ownership."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + ".lock").open("a+b") as lock:
        if os.fstat(lock.fileno()).st_size == 0:
            lock.write(b"0")
            lock.flush()
        deadline = time.monotonic() + 5
        while True:
            lock.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                require(time.monotonic() < deadline, "Content history is busy; no generation was submitted")
                time.sleep(.05)
        try:
            yield
        finally:
            lock.seek(0)
            if os.name == "nt":
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def register_content(path: Path, reference_hash: str, prompt: str, *, outputdir: Path, api_base_url: str) -> dict:
    """Reserve fresh content durably before the one paid submission attempt."""
    canonical = json.dumps({"reference_sha256": reference_hash, "prompt": prompt.strip()},
                           sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    with content_history_lock(path):
        history = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"schema_version": 1, "entries": {}}
        require(history.get("schema_version") == 1 and isinstance(history.get("entries"), dict),
                "Invalid content history; refusing to submit without reliable duplicate detection")
        require(fingerprint not in history["entries"],
                "Reference and prompt combination was already registered; provide new content or use --resume-job for query/download only")
        record = {"fingerprint": fingerprint, "reference_sha256": reference_hash, "prompt": prompt.strip(),
                  "registered_at": utc_now(), "request_id": uuid.uuid4().hex,
                  "outputdir": str(outputdir), "api_base_url": api_base_url}
        history["entries"][fingerprint] = record
        temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
        try:
            with temporary.open("x", encoding="utf-8") as output:
                json.dump(history, output, ensure_ascii=False, indent=2)
                output.flush()
                os.fsync(output.fileno())
            temporary.replace(path)
            if os.name != "nt":
                descriptor = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
        finally:
            temporary.unlink(missing_ok=True)
    return {**record, "history_path": str(path)}


def verify_video(path: Path) -> dict:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True, text=True, timeout=120, check=True,
    )
    media = json.loads(probe.stdout)
    video = next((s for s in media["streams"] if s["codec_type"] == "video"), None)
    audio = next((s for s in media["streams"] if s["codec_type"] == "audio"), None)
    require(video is not None and audio is not None, "MP4 must contain both video and audio streams")
    frames = int(video.get("nb_read_frames") or video.get("nb_frames") or 0)
    fps = float(Fraction(video.get("avg_frame_rate", "0")))
    duration = float(video.get("duration") or media["format"]["duration"])
    require(video.get("codec_name") == "h264", "Expected an H.264 video stream")
    require(frames >= 120 and 5 <= duration <= 6, "Expected at least 120 frames and a 5–6 second video")
    require(abs(fps - 24) < 0.01, "Expected 24 fps")
    require((video["width"], video["height"]) == (512, 288), "Unexpected video dimensions")
    require(int(audio.get("channels", 0)) == 2, "Expected stereo generated audio")
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-i", str(path),
         "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-"],
        capture_output=True, text=True, timeout=180, check=True,
    )
    return {
        "video_codec": video["codec_name"], "width": video["width"], "height": video["height"],
        "decoded_frames": frames, "fps": fps, "duration_seconds": duration,
        "audio_codec": audio.get("codec_name"), "audio_channels": audio["channels"],
        "audio_sample_rate": audio.get("sample_rate"), "full_decode_passed": True,
        "bytes": path.stat().st_size, "sha256": sha256(path),
    }


def run(args, report: dict) -> None:
    import requests

    reference = None
    if not args.resume_job:
        require(bool(getattr(args, "reference", None)), "New generation requires an explicit --reference image")
        require(isinstance(getattr(args, "prompt", None), str) and bool(args.prompt.strip()),
                "New generation requires an explicit nonempty --prompt")
        reference = Path(args.reference).resolve()
        require(reference.is_file(), "Reference image does not exist")
        require(reference.stat().st_size <= 8 * 1024 * 1024, "Reference image exceeds smoke-test size limit")
        from PIL import Image
        with Image.open(reference) as image:
            image.verify()
    parsed = urlsplit(args.url)
    require(parsed.scheme in {"https", "http"} and bool(parsed.hostname), "Provide an HTTP(S) API base URL")
    require(not parsed.username and not parsed.password and not parsed.query and not parsed.fragment,
            "Do not embed credentials, query parameters or fragments in the API URL")
    key = os.environ.get("H3_API_KEY", "")
    require(len(key) >= 32, "Set H3_API_KEY to the running server's token")
    for command in ("ffprobe", "ffmpeg"):
        require(shutil.which(command) is not None, f"{command} is required before submitting a paid GPU job")
    base = args.url.rstrip("/")
    report["api_base_url"] = base
    parameters = {**PARAMETERS, "steps": args.steps,
                  "attention_backend": getattr(args, "attention_backend", "auto"),
                  "acceleration_lora": getattr(args, "acceleration_lora", "none")}
    require(parameters["acceleration_lora"] == "none" or parameters["steps"] == 5,
            "Turbo Ref2VA requires explicit --steps 5 (four denoiser evaluations)")
    if not args.resume_job:
        parameters["prompt"] = args.prompt.strip()
        report["parameters"] = parameters
    report["test_scope"] = "real API integration; low-step smoke test, not visual quality acceptance"
    headers = {"Authorization": f"Bearer {key}"}
    session = requests.Session()
    started = time.monotonic()
    deadline = started + args.startup_timeout
    print("Waiting for API and manager availability; model loading may continue while jobs queue...", flush=True)
    while True:
        try:
            response = session.get(base + "/health", timeout=10)
            health = response.json()
            if not isinstance(health, dict):
                raise ValueError("Invalid health response")
            # New APIs expose admission separately from model execution. A legacy
            # aggregate ready flag is only a fallback when capability keys are
            # absent, never an override for an explicitly unavailable API/manager.
            explicit_capabilities = "api_ready" in health or "manager_ready" in health
            admission_ready = response.status_code == 200 and (
                health.get("api_ready") is True and health.get("manager_ready") is True
                if explicit_capabilities else health.get("ready") is True)
            if admission_ready or (args.resume_job and response.status_code in {200, 503}):
                report["health"] = {"http_status": response.status_code,
                    "ready": health.get("ready") is True, "admission_ready": admission_ready,
                    **{name: health.get(name) if type(health.get(name)) is bool else None
                       for name in ("api_ready", "manager_ready", "model_ready")},
                    "observed_at": utc_now()}
                break
        except (requests.RequestException, ValueError):
            pass
        require(time.monotonic() < deadline, "API and manager did not become available before startup timeout")
        time.sleep(args.poll_seconds)
    print(f"/health: HTTP {report['health']['http_status']}, admission_ready={report['health']['admission_ready']}, "
          f"ready={report['health']['ready']}, model_ready={report['health']['model_ready']}", flush=True)

    if args.resume_job:
        job_id = args.resume_job
        report["resumed"] = True
        report["test_scope"] = "resume existing job: status/download only; no new content registration or generation request"
        print(f"Resuming status/download for {job_id}; no generation submission", flush=True)
    else:
        response = session.post(base + "/v1/jobs", timeout=30)
        report["unauthorized_submit_http_status"] = response.status_code
        require(response.status_code in {401, 403}, "Unauthenticated submission was not rejected")
        print(f"Unauthenticated POST rejected: HTTP {response.status_code}", flush=True)
        report["reference_sha256"] = sha256(reference)
        report["reference_source"] = str(reference)
        history_path = Path(getattr(args, "content_history", None) or args.outputdir.parent / "content-history.json").resolve()
        registration = register_content(history_path, report["reference_sha256"], parameters["prompt"],
                                        outputdir=args.outputdir, api_base_url=base)
        report["content_registration"] = registration
        save_report(args.outputdir, report)
        with reference.open("rb") as image:
            require(sha256(reference) == report["reference_sha256"], "Reference changed after content registration; generation not submitted")
            response = session.post(
                base + "/v1/jobs", headers={**headers, "Idempotency-Key": registration["request_id"]}, timeout=120,
                data={**parameters, "kinds": json.dumps(["image"])},
                files=[("files", (reference.name, image, mimetypes.guess_type(reference.name)[0] or "image/jpeg"))],
            )
        report["submit_http_status"] = response.status_code
        require(response.status_code == 202, f"Generation submit failed: HTTP {response.status_code}")
        task = response.json()
        job_id = task["id"]
        report["actual_seconds"] = task.get("actual_seconds")
        print(f"Submitted real job {job_id}: HTTP 202", flush=True)
    require(re.fullmatch(r"[a-f0-9]{32}", job_id) is not None, "Unexpected job ID format")
    report["job_id"] = job_id
    report.setdefault("status_history", [])
    save_report(args.outputdir, report)

    deadline = time.monotonic() + args.generation_timeout
    while True:
        response = session.get(base + f"/v1/jobs/{job_id}", headers=headers, timeout=30)
        require(response.status_code == 200, f"Status query failed: HTTP {response.status_code}")
        task = response.json()
        status = task.get("status")
        if args.resume_job and task.get("generation_parameters"):
            report["parameters"] = task["generation_parameters"]
        report["status_history"].append({"http_status": response.status_code, "status": status,
                                        "queue_position": task.get("queue_position"),
                                        "waiting_for": task.get("waiting_for"),
                                        "active_phase": task.get("active_phase"), "observed_at": utc_now()})
        save_report(args.outputdir, report)
        print(f"{utc_now()} GET job: HTTP 200, status={status}, "
              f"queue_position={task.get('queue_position')}, waiting_for={task.get('waiting_for')}, "
              f"active_phase={task.get('active_phase')}", flush=True)
        if status in {"succeeded", "failed", "cancelled"}:
            report["terminal_status"] = status
            report["metrics"] = task.get("metrics")
            report["lifecycle"] = task.get("lifecycle")
            report["terminal_error"] = task.get("error")
            report["cleanup_complete"] = task.get("cleanup_complete")
            break
        require(status in {"queued", "running", "cancel_requested"}, "Unknown job status")
        require(time.monotonic() < deadline, "Generation timeout; task may still be running on server")
        time.sleep(args.poll_seconds)
    require(status == "succeeded", "Real model generation failed; inspect the server log")

    output = args.outputdir / f"{job_id}.mp4"
    with session.get(base + f"/v1/jobs/{job_id}/file", headers=headers, timeout=120, stream=True) as response:
        report["download_http_status"] = response.status_code
        require(response.status_code == 200, f"Video download failed: HTTP {response.status_code}")
        size = 0
        with output.open("wb") as destination:
            for chunk in response.iter_content(1024 * 1024):
                size += len(chunk)
                require(size <= 128 * 1024 * 1024, "Smoke-test output unexpectedly exceeds 128 MiB")
                destination.write(chunk)
    report["video"] = verify_video(output)
    report["video_path"] = str(output.resolve())
    report["elapsed_seconds"] = round(time.monotonic() - started, 2)
    report["passed"] = True
    print("API_SMOKE_TEST_PASSED", flush=True)
    print(json.dumps(report["video"], indent=2), flush=True)
    print(f"VIDEO_PATH={output.resolve()}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Base URL of an already running real server")
    parser.add_argument("--reference", help="Required for new generation: explicitly chosen local reference image")
    parser.add_argument("--prompt", help="Required for new generation: fresh explicit prompt; no automatic default")
    parser.add_argument("--content-history", type=Path,
                        help="Shared durable content registry; default OUTPUTDIR.parent/content-history.json")
    parser.add_argument("--steps", type=int, choices=range(2, 51), default=8,
                        help="Scheduler grid points; Turbo Ref2VA requires 5 for four evaluations")
    parser.add_argument("--attention-backend", choices=("auto", "native", "_native_flash"), default="auto")
    parser.add_argument("--acceleration-lora", choices=("none", "lightx2v-ref2va-4step"), default="none")
    parser.add_argument("--resume-job", help="Existing job ID: query and download only, never submit another generation")
    parser.add_argument("--outputdir", type=Path, default=Path("smoke_result"))
    parser.add_argument("--startup-timeout", type=int, default=3600)
    parser.add_argument("--generation-timeout", type=int, default=2700)
    parser.add_argument("--poll-seconds", type=int, choices=range(1, 61), default=15, metavar="1..60")
    args = parser.parse_args()
    args.outputdir = args.outputdir.resolve()
    args.outputdir.mkdir(parents=True, exist_ok=True)
    report = {"started_at": utc_now(), "passed": False}
    if args.resume_job:
        require(re.fullmatch(r"[a-f0-9]{32}", args.resume_job) is not None, "Unexpected resume job ID")
        saved = args.outputdir / "test_report.json"
        if saved.exists():
            previous = json.loads(saved.read_text(encoding="utf-8"))
            require(previous.get("job_id") == args.resume_job, "Output directory belongs to a different job")
            require(previous.get("api_base_url") == args.url.rstrip("/"), "Output directory belongs to a different API")
            report.update(previous)
            report.update(passed=False, resumed_at=utc_now())
            report.pop("error", None)
            report.pop("error_type", None)
    try:
        run(args, report)
    except BaseException as error:
        # No response bodies, headers, credentials, or environment are copied into the report.
        report["error_type"] = type(error).__name__
        message = str(error)
        if os.environ.get("H3_API_KEY"):
            message = message.replace(os.environ["H3_API_KEY"], "[REDACTED]")
        report["error"] = message
        raise
    finally:
        report["finished_at"] = utc_now()
        path = args.outputdir / "test_report.json"
        save_report(args.outputdir, report)
        print(f"REPORT_PATH={path}", flush=True)


if __name__ == "__main__":
    main()
