"""Authenticated API adapter. Model execution and durable jobs belong to the manager."""
from __future__ import annotations

import asyncio
import errno
import hashlib
import hmac
import ipaddress
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import threading
from contextlib import asynccontextmanager
from typing import Annotated, Literal
import uuid
from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from control_protocol import ControlError, rpc_call
from adapter_config import load_adapter_settings
from generation_options import generation_capabilities, validate_generation_options
from runtime_options import RuntimeOptions

LOG = logging.getLogger("minimax_api")
MIB = 1024 * 1024
MAX_FILE_BYTES = 32 * MIB
MAX_TOTAL_BYTES = 64 * MIB
MAX_REQUEST_BYTES = MAX_TOTAL_BYTES + MIB
MAX_CONTROL_BYTES = 64 * 1024
EXTENSIONS = {
    "image": {".jpg", ".jpeg", ".png", ".webp"},
    "video": {".mp4", ".mov", ".webm"},
    "audio": {".wav", ".mp3", ".m4a", ".flac", ".ogg"},
}
JOB_ID = re.compile(r"^[a-f0-9]{32}$")
security = HTTPBearer()

# Public receipts are an explicit projection, not arbitrary owner-provided JSON.
# Keep this contract independent of the model module: the API must not load it.
PUBLIC_STAGE_CONDITIONS = ("instance_matches", "attempt_owned", "attempt_live", "model_loaded", "worker_alive",
    "capability_available", "admission_confirmed", "predecessor_completed", "stage_not_started",
    "cancel_flag_clear", "cancel_event_clear", "reservation_owned", "budget_grant_held")
PUBLIC_GATE_CONDITIONS = {
    "manager.dispatch": ("manager_open", "job_queued", "attempt_unassigned", "model_identity_current",
        "model_observation_fresh", "model_process_alive", "model_ready", "worker_alive", "worker_accepting", "supervisor_alive"),
    "model.execute": ("instance_matches", "request_validated", "model_loaded", "worker_alive", "capability_available",
        "not_draining", "not_stopping", "attempt_available", "job_unowned", "artifact_absent",
        "generation_options_valid", "resources_reserved"),
    **{"model.stage." + stage: PUBLIC_STAGE_CONDITIONS for stage in ("prepare", "gpu", "encode_validate")},
}


def public_gate_receipt(value, expected_gate):
    """Expose known decision facts; no free-form context, parameters or paths."""
    if not isinstance(value, dict) or value.get("gate_id") != expected_gate or expected_gate not in PUBLIC_GATE_CONDITIONS:
        return None
    required = PUBLIC_GATE_CONDITIONS[expected_gate]
    raw_conditions = value.get("conditions")
    raw_conditions = raw_conditions if isinstance(raw_conditions, dict) else {}
    states = {"true", "false", "unknown", "conflict"}
    conditions = {name: raw_conditions.get(name) if type(raw_conditions.get(name)) is str
                  and raw_conditions[name] in states else "unknown" for name in required}
    result = {"gate_id": expected_gate, "conditions": conditions,
              "allowed": value.get("allowed") if type(value.get("allowed")) is bool else None}
    if result["allowed"] is True and any(state != "true" for state in conditions.values()):
        # Do not publish a positive authorization whose required evidence is
        # missing, malformed or contradictory; keep uncertainty visible.
        result["allowed"] = None
    for name in ("schema_version", "gate_version"):
        if isinstance(value.get(name), str) and re.fullmatch(r"[0-9]{1,3}\.[0-9]{1,3}", value[name]):
            result[name] = value[name]
    if isinstance(value.get("decision_id"), str) and JOB_ID.fullmatch(value["decision_id"]):
        result["decision_id"] = value["decision_id"]
    observed = value.get("observed_at")
    if type(observed) in {int, float} and math.isfinite(observed) and observed >= 0:
        result["observed_at"] = observed
    monotonic = value.get("monotonic_ns")
    if type(monotonic) is int and 0 <= monotonic < 2 ** 63:
        result["monotonic_ns"] = monotonic
    blocked = value.get("blocked")
    if isinstance(blocked, list):
        result["blocked"] = [name for name in required if name in blocked]
    raw = value.get("context")
    context = {}
    if isinstance(raw, dict):
        for name in ("manager_instance_id", "model_instance_id", "instance_id", "job_id", "attempt_id",
                     "budget_instance_id", "reservation_id", "lease_id", "budget_decision_id"):
            if isinstance(raw.get(name), str) and JOB_ID.fullmatch(raw[name]):
                context[name] = raw[name]
        expected_source = "global_manager.py" if expected_gate == "manager.dispatch" else "model_container.py"
        if raw.get("source_file") == expected_source:
            context["source_file"] = expected_source
        if isinstance(raw.get("source_sha256"), str) and re.fullmatch(r"[a-f0-9]{64}", raw["source_sha256"]):
            context["source_sha256"] = raw["source_sha256"]
        if type(raw.get("pipeline")) is bool:
            context["pipeline"] = raw["pipeline"]
        if raw.get("budget_scope") in ("stage_budget", "legacy_single_active"):
            context["budget_scope"] = raw["budget_scope"]
        if raw.get("stage") in ("prepare", "gpu", "encode_validate"):
            context["stage"] = raw["stage"]
    result["context"] = context
    return result


def public_condition_fields(record):
    fields = {}
    if "condition_gate" in record:
        fields["condition_gate"] = public_gate_receipt(record["condition_gate"], "manager.dispatch")
    if "execution_condition_gates" in record:
        raw = record["execution_condition_gates"]
        fields["execution_condition_gates"] = ({
            name: public_gate_receipt(raw[name], name) for name in PUBLIC_GATE_CONDITIONS
            if name.startswith("model.") and name in raw
        } if isinstance(raw, dict) else None)
    if "stage_completions" in record:
        raw = record["stage_completions"]
        fields["stage_completions"] = ({
            name: raw.get(name) if type(raw.get(name)) is bool else None
            for name in ("prepare", "gpu", "encode_validate")
        } if isinstance(raw, dict) else None)
    if "cancellation_requested" in record:
        fields["cancellation_requested"] = (record["cancellation_requested"]
            if type(record["cancellation_requested"]) is bool else None)
    return fields
DEFAULT_CORS_ORIGINS = (
    "https://zjj1280637679-ship-it.github.io",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
)


def _cors_allowed_origins() -> list[str]:
    """Explicit browser origins, independent of Bearer authorization.

    H3_CORS_ORIGINS replaces the defaults; an empty value disables cross-origin
    reads. Public origins require HTTPS, while HTTP is restricted to loopback.
    """
    configured = os.environ.get("H3_CORS_ORIGINS")
    values = DEFAULT_CORS_ORIGINS if configured is None else configured.split(",")
    origins = []
    for value in values:
        value = value.strip()
        if not value:
            continue
        try:
            if "*" in value or any(char.isspace() or ord(char) < 32 for char in value):
                raise ValueError()
            parsed = urlsplit(value)
            hostname, port = parsed.hostname, parsed.port
            if (parsed.scheme not in {"https", "http"} or not hostname
                    or parsed.username is not None or parsed.password is not None
                    or parsed.path not in {"", "/"} or parsed.query or parsed.fragment
                    or (port is not None and not 1 <= port <= 65535)):
                raise ValueError()
            try:
                address = ipaddress.ip_address(hostname)
            except ValueError:
                address = None
                hostname = hostname.encode("idna").decode("ascii").lower()
                if len(hostname) > 253 or not re.fullmatch(
                    r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)*",
                    hostname,
                ):
                    raise ValueError()
            if parsed.scheme == "http" and not (hostname == "localhost" or (address and address.is_loopback)):
                raise ValueError()
            host = f"[{address.compressed}]" if address and address.version == 6 else hostname
            suffix = f":{port}" if port is not None and port != (443 if parsed.scheme == "https" else 80) else ""
            origin = f"{parsed.scheme}://{host}{suffix}"
        except (ValueError, UnicodeError):
            raise RuntimeError("H3_CORS_ORIGINS must contain explicit HTTPS origins or loopback HTTP origins; wildcards, credentials and URL paths are not allowed.") from None
        if origin not in origins:
            origins.append(origin)
    return origins


class RequestGuard:
    """Authenticate before multipart parsing and bound actual incoming bytes."""

    def __init__(self, app, api_key: str):
        self.app, self.api_key = app, api_key

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not scope["path"].startswith("/v1/"):
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers", []))
        authorization = headers.get(b"authorization", b"").decode("latin1")
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not hmac.compare_digest(token.encode("latin1"), self.api_key.encode("ascii")):
            return await JSONResponse({"detail": "Bearer API key required"}, 401,
                                      headers={"WWW-Authenticate": "Bearer"})(scope, receive, send)
        limit = MAX_REQUEST_BYTES if scope["path"] == "/v1/jobs" and scope["method"] == "POST" else MAX_CONTROL_BYTES
        try:
            content_length = int(headers.get(b"content-length", b"0"))
            if content_length < 0:
                raise ValueError
        except ValueError:
            return await JSONResponse({"detail": "Invalid Content-Length"}, 400)(scope, receive, send)
        if content_length > limit:
            return await JSONResponse({"detail": "Request exceeds body limit"}, 413)(scope, receive, send)
        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            received += len(message.get("body", b""))
            if received > limit:
                raise HTTPException(413, "Request exceeds body limit")
            return message

        await self.app(scope, limited_receive, send)


class ControlRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["drain", "resume", "start_model", "stop_model", "start_api", "stop_api", "shutdown"]
    force: bool = Field(False, strict=True)
    command_id: str | None = Field(None, min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$")


class DriveArchiveReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid")
    file_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    bytes: int = Field(gt=0, strict=True)


def _watch_parent(parent_pid, stop, exit_process=os._exit, process_factory=None, expected_creation=None):
    # Windows venv redirectors are legitimate intermediate OS parents. Ownership
    # is the supplied manager process generation, not direct getppid equality.
    import psutil
    factory = process_factory or psutil.Process
    try:
        if expected_creation is None:
            expected_creation = factory(parent_pid).create_time()
    except psutil.Error:
        exit_process(70)
        return
    while not stop.wait(1):
        try:
            process = factory(parent_pid)
            alive = (process.is_running() and process.status() != psutil.STATUS_ZOMBIE
                     and process.create_time() == expected_creation)
        except psutil.Error:
            alive = False
        if not alive:
            exit_process(70)
            return


def create_app(controller=None, api_key: str | None = None, output_dir: Path | None = None):
    """API creation never initializes CUDA, a model, or a second job-state owner."""
    token = api_key if api_key is not None else os.environ.get("H3_API_KEY", "")
    if len(token) < 32 or not token.isascii() or any(c.isspace() for c in token):
        raise RuntimeError("Set H3_API_KEY to a random ASCII token of at least 32 characters.")
    instance_id = os.environ.get("H3_INSTANCE_ID") or uuid.uuid4().hex
    root = Path(output_dir or os.environ.get("H3_OUTPUT_DIR", "./h3_outputs")).resolve()
    private_key = os.environ.get("H3_CONTROL_KEY", "")
    if controller is None:
        manager_url = os.environ.get("H3_MANAGER_URL", "")
        if not manager_url or len(private_key) < 32:
            raise RuntimeError("The API requires its manager connection and private control token.")
        controller = lambda operation, payload: rpc_call(manager_url, private_key, operation, payload)
    # Only concurrent transport writes are bounded by count; durable jobs/outbox
    # records have no count limit. Disk bytes govern their storage admission.
    upload_lock = threading.Lock()
    upload_state = {"active": 0}
    disk_reserve_bytes = max(0, int(os.environ.get("H3_DISK_RESERVE_BYTES", 64 * MIB)))
    stopped = threading.Event()

    def unknown_count():
        if not root.exists():
            return 0
        return sum(1 for directory in root.iterdir()
                   if JOB_ID.fullmatch(directory.name) and not directory.is_symlink()
                   and (directory / "submission-uncertain.json").is_file())

    def acquire_upload():
        with upload_lock:
            if upload_state["active"] >= 3:
                return False
            if shutil.disk_usage(root).free < disk_reserve_bytes + (upload_state["active"] + 1) * MAX_TOTAL_BYTES:
                raise HTTPException(507, "Insufficient disk space for reference upload")
            upload_state["active"] += 1
            return True

    def release_upload():
        with upload_lock:
            upload_state["active"] -= 1

    def transport_status():
        with upload_lock:
            return {"active_uploads": upload_state["active"], "unknown_submissions": unknown_count(),
                    "concurrent_upload_limit": 3, "durable_submission_count_limit": None,
                    "disk_reserve_bytes": disk_reserve_bytes, "max_reference_bytes_per_request": MAX_TOTAL_BYTES}

    def redact(value):
        if isinstance(value, dict):
            forbidden = {"api_key", "control_key", "private_key", "internal_key", "token", "authorization", "manager_url"}
            return {key: redact(item) for key, item in value.items() if key.lower() not in forbidden}
        if isinstance(value, list):
            return [redact(item) for item in value]
        if isinstance(value, str):
            for secret in (token, private_key):
                if secret:
                    value = value.replace(secret, "[redacted]")
        return value

    def call(operation, payload=None):
        try:
            request = dict(payload or {})
            if operation in {"submit", "cancel", "control", "archive"}:
                request["expected_instance_id"] = instance_id
            result = controller(operation, request)
            if not isinstance(result, dict):
                raise TypeError("Invalid manager response")
            return result
        except ControlError as error:
            status = getattr(error, "status", 503)
            status = status if isinstance(status, int) and 400 <= status <= 599 else 503
            # Transport/internal failures are not evidence of rejection; callers
            # reconcile uncertain submissions before removing their inputs.
            if status >= 500 and status != 507:
                raise HTTPException(503, "Manager unavailable; query the command or job before retrying") from None
            detail = str(redact(str(error)))[:500]
            raise HTTPException(status, detail or "Manager rejected the request") from None
        except Exception as error:
            LOG.warning("Manager operation failed: %s (%s)", operation, type(error).__name__)
            raise HTTPException(503, "Manager unavailable; query the command or job before retrying") from None

    def valid_id(job_id):
        if not JOB_ID.fullmatch(job_id):
            raise HTTPException(404, "Task does not exist")

    def public_job(record):
        identity = record.get("id", record.get("job_id"))
        if not isinstance(identity, str) or not JOB_ID.fullmatch(identity):
            raise HTTPException(503, "Manager returned an invalid task identity")
        record_instance = record.get("instance_id")
        if "instance_id" in record:
            if record_instance != instance_id:
                raise HTTPException(503, "Manager returned a task from a different instance")
        else:
            # Compatibility with an already running older manager: bind only
            # after an authenticated current-manager identity readback. Do not
            # replace an explicit mismatch or infer identity from a job ID.
            state = call("status")
            if state.get("instance_id") != instance_id:
                raise HTTPException(503, "Manager identity is not confirmed for this task")
        allowed = {"status", "created_at", "started_at", "finished_at", "error", "actual_seconds",
                   "generation_parameters", "generation_options", "lifecycle", "metrics", "active_phase", "attempt_id", "instance_id",
                   "cancel_requested", "cleanup_complete", "replayed", "queue_position", "dispatch_state", "dispatch_reason", "waiting_for"}
        result = {"id": identity, **{key: record[key] for key in allowed if key in record}}
        result.update(public_condition_fields(record))
        result["instance_id"] = instance_id
        if result.get("status") not in {"queued", "running", "cancel_requested", "succeeded", "failed", "cancelled"}:
            raise HTTPException(503, "Manager returned an invalid task state")
        if isinstance(record.get("artifact"), dict):
            result["artifact"] = {key: value for key, value in record["artifact"].items() if key != "path"}
        result["status_url"] = f"/v1/jobs/{identity}"
        if result["status"] in {"queued", "running", "cancel_requested"}:
            result["cancel_url"] = f"/v1/jobs/{identity}/cancel"
            result["poll_after_seconds"] = 2
        if result["status"] == "succeeded":
            result["file_url"] = f"/v1/jobs/{identity}/file"
            archive = record.get("archive")
            if isinstance(archive, dict) and archive.get("provider") == "google_drive":
                try:
                    receipt = DriveArchiveReceipt.model_validate({k: archive.get(k) for k in DriveArchiveReceipt.model_fields})
                except ValidationError:
                    raise HTTPException(503, "Manager returned an invalid archive receipt") from None
                download_url = f"https://drive.google.com/uc?export=download&id={receipt.file_id}"
                result["archive"] = {
                    "provider": "google_drive", **receipt.model_dump(), "download_url": download_url,
                    "view_url": f"https://drive.google.com/file/d/{receipt.file_id}/view",
                }
                result["download_url"] = download_url
        return redact(result)

    @asynccontextmanager
    async def lifespan(app):
        root.mkdir(parents=True, exist_ok=True)
        watcher = None
        parent_pid = os.environ.get("H3_PARENT_PID")
        if parent_pid:
            parent_pid = int(parent_pid)
            if parent_pid <= 1:
                raise RuntimeError("Invalid manager parent process identity")
            created = os.environ.get("H3_PARENT_CREATED_AT")
            expected_creation = float(created) if created is not None else None
            watcher = threading.Thread(target=_watch_parent, args=(parent_pid, stopped),
                                       kwargs={"expected_creation": expected_creation},
                                       name="h3-api-owner-watch", daemon=True)
            watcher.start()
        try:
            yield
        finally:
            stopped.set()
            if watcher:
                watcher.join(timeout=2)

    adapter_settings = load_adapter_settings()
    runtime_options = RuntimeOptions.from_env()
    app = FastAPI(title="MiniMax-H3 Ref2VA API", version="0.4.0", lifespan=lifespan,
                  description="Independent authenticated API adapter; durable scheduling is owned by the global manager.")
    app.add_middleware(RequestGuard, api_key=token)
    # Starlette wraps the last-added middleware outermost. Handle valid browser
    # preflights before RequestGuard, and attach CORS headers even to its 401s.
    # Every actual /v1 request still traverses the unchanged Bearer guard.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_allowed_origins(),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type", "Idempotency-Key", "Range"],
        expose_headers=["Content-Disposition", "Content-Length", "Content-Range", "Accept-Ranges", "ETag", "Retry-After"],
        max_age=600,
    )

    def require_auth(credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)]):
        if not hmac.compare_digest(credentials.credentials.encode("utf-8"), token.encode("ascii")):
            raise HTTPException(401, "Invalid API key", headers={"WWW-Authenticate": "Bearer"})

    @app.get("/health/live")
    def live():
        return {"api_ready": True, "alive": True, "instance_id": instance_id}

    @app.get("/health")
    def health():
        try:
            state = call("status")
        except HTTPException:
            return JSONResponse({"api_ready": True, "manager_ready": False, "model_ready": False,
                                 "ready": False, "instance_id": instance_id}, 503)
        same_instance = state.get("instance_id") == instance_id
        return JSONResponse({"api_ready": True, "manager_ready": same_instance,
                             "model_ready": same_instance and state.get("model_ready") is True,
                             "ready": same_instance and state.get("ready") is True,
                             "instance_id": instance_id}, 200 if same_instance else 503)

    @app.get("/v1/system", dependencies=[Depends(require_auth)])
    def system():
        local = transport_status()
        try:
            state = redact(call("status"))
        except HTTPException:
            return JSONResponse({"instance_id": instance_id, "api_ready": True, "manager_ready": False,
                                 "ready": False, "model_ready": False, "api_local": local}, 503)
        return {**state, "api_local": local}

    @app.get("/v1/capabilities", dependencies=[Depends(require_auth)])
    def capabilities():
        # Configuration admits queued work while model_ready is still false.
        # Never infer successful loading from a download percentage.
        state = call("status")
        same_instance = state.get("instance_id") == instance_id
        return {**generation_capabilities(model_ready=same_instance and state.get("model_ready") is True,
                    settings=adapter_settings, runtime_options=runtime_options), "instance_id": instance_id}

    @app.get("/v1/events", dependencies=[Depends(require_auth)])
    def events(after: Annotated[int, Query(ge=0)] = 0, limit: Annotated[int, Query(ge=1, le=1000)] = 100):
        return redact(call("events", {"after": after, "limit": limit}))

    @app.get("/v1/commands/{command_id}", dependencies=[Depends(require_auth)])
    def command(command_id: str):
        if not re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", command_id):
            raise HTTPException(404, "Command does not exist")
        return redact(call("command", {"command_id": command_id}))

    @app.post("/v1/control", status_code=202, dependencies=[Depends(require_auth)])
    def control(request: ControlRequest):
        if request.action in {"stop_api", "shutdown"} and request.command_id is None:
            raise HTTPException(422, "A client command_id is required when stopping this API; query it through the manager")
        return redact(call("control", request.model_dump(exclude_none=True)))

    @app.post("/v1/jobs", status_code=202, dependencies=[Depends(require_auth)])
    async def submit(
        files: Annotated[list[UploadFile], File(description="Reference files in semantic order")],
        kinds: Annotated[str, Form(description='JSON array matching files, e.g. ["image","audio"]')],
        prompt: Annotated[str, Form(min_length=1, max_length=4000)],
        seconds: Annotated[float, Form(ge=5, le=15)] = 5,
        width: Annotated[int, Form(ge=256, le=1344)] = 960,
        height: Annotated[int, Form(ge=256, le=1344)] = 544,
        steps: Annotated[int, Form(ge=2, le=50)] = 50,
        attention_backend: Annotated[str, Form()] = "auto",
        acceleration_lora: Annotated[str, Form()] = "none",
        seed: Annotated[int, Form(ge=0, le=2147483647)] = 42,
        idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key", max_length=128)] = None,
    ):
        directory = None
        reserved = retain_inputs = False
        try:
            if not prompt.strip() or not math.isfinite(seconds):
                raise HTTPException(422, "Prompt and duration must be valid")
            try:
                validate_generation_options({"attention_backend": attention_backend,
                    "acceleration_lora": acceleration_lora, "steps": steps}, adapter_settings)
            except ValueError as error:
                raise HTTPException(422, str(error)) from None
            if idempotency_key is not None and not re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", idempotency_key):
                raise HTTPException(422, "Invalid Idempotency-Key")
            if width % 32 or height % 32 or width * height > 1344 * 768:
                raise HTTPException(422, "Dimensions must be multiples of 32, at most 1,032,192 pixels")
            frames = 5 + 17 * math.ceil((round(seconds * 24) - 5) / 17)
            if frames > 360:
                raise HTTPException(422, "Requested duration rounds beyond 15 seconds; use at most 14.375 seconds")
            try:
                modalities = json.loads(kinds)
            except (ValueError, TypeError):
                raise HTTPException(422, "kinds must be a JSON array") from None
            if not isinstance(modalities, list) or not 1 <= len(files) <= 12 or len(modalities) != len(files):
                raise HTTPException(422, "Supply 1-12 files and one kind per file")
            if any(not isinstance(kind, str) or kind not in EXTENSIONS for kind in modalities):
                raise HTTPException(422, "Kinds must be image, video or audio")
            if modalities.count("image") > 9 or modalities.count("video") > 3 or modalities.count("audio") > 3:
                raise HTTPException(422, "At most 9 images, 3 videos and 3 audio references")
            if all(kind == "audio" for kind in modalities):
                raise HTTPException(422, "At least one image or video reference is required")
            for upload, kind in zip(files, modalities):
                if Path(upload.filename or "").suffix.lower() not in EXTENSIONS[kind]:
                    raise HTTPException(422, f"Unsupported {kind} file extension")
            reserved = await asyncio.to_thread(acquire_upload)
            if not reserved:
                raise HTTPException(429, "Concurrent upload limit reached; retry after uploads finish", headers={"Retry-After": "5"})
            job_id = uuid.uuid4().hex
            directory = root / job_id
            input_dir = directory / "inputs"
            input_dir.mkdir(parents=True, exist_ok=False)
            references, content_identity, total = [], [], 0
            for index, (upload, kind) in enumerate(zip(files, modalities)):
                suffix = Path(upload.filename or "").suffix.lower()
                path = input_dir / f"{index:02d}{suffix}"
                size, digest = 0, hashlib.sha256()
                with path.open("xb") as destination:
                    while chunk := await upload.read(MIB):
                        size += len(chunk)
                        total += len(chunk)
                        if size > MAX_FILE_BYTES or total > MAX_TOTAL_BYTES:
                            raise HTTPException(413, "Each reference <=32 MiB; total references <=64 MiB")
                        digest.update(chunk)
                        await asyncio.to_thread(destination.write, chunk)
                if not size:
                    raise HTTPException(422, "Reference file is empty")
                references.append({"kind": kind, "path": str(path)})
                content_identity.append({"kind": kind, "sha256": digest.hexdigest(), "bytes": size})
            parameters = dict(prompt=prompt, references=references, seconds=seconds,
                              width=width, height=height, steps=steps, seed=seed,
                              attention_backend=attention_backend, acceleration_lora=acceleration_lora)
            semantic_request = {**parameters, "references": content_identity}
            fingerprint = hashlib.sha256(json.dumps(semantic_request, sort_keys=True,
                                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()
            payload = {"job_id": job_id, "parameters": parameters, "fingerprint": fingerprint,
                       "idempotency_key": idempotency_key or job_id}
            def uncertain_submission():
                marker = {"job_id": job_id, "idempotency_key": payload["idempotency_key"],
                          "fingerprint": fingerprint, "state": "submission_outcome_unknown"}
                (directory / "submission-uncertain.json").write_text(json.dumps(marker), encoding="utf-8")
                return HTTPException(503, {"message": "Submission outcome unknown; query this job before retrying",
                                          "job_id": job_id, "status_url": f"/v1/jobs/{job_id}",
                                          "idempotency_key": payload["idempotency_key"]})

            retain_inputs = True  # A dispatched command may commit before its response arrives.
            try:
                record = await asyncio.to_thread(call, "submit", payload)
            except HTTPException as error:
                if error.status_code < 500 or error.status_code == 507:
                    retain_inputs = False
                    raise
                # A dropped response does not prove the durable command failed.
                # Query the identity just sent. Preserve inputs if still unknown.
                retain_inputs = True
                try:
                    record = await asyncio.to_thread(call, "job", {"job_id": job_id})
                except HTTPException:
                    raise uncertain_submission() from None
            try:
                result = public_job(record)
            except HTTPException:
                raise uncertain_submission() from None
            retain_inputs = result["id"] == job_id
            result.setdefault("actual_seconds", frames / 24)
            return result
        except OSError as error:
            if error.errno in {errno.ENOSPC, errno.EDQUOT} and not retain_inputs:
                raise HTTPException(507, "Insufficient disk space for reference upload") from None
            raise
        finally:
            for upload in files:
                await upload.close()
            if directory is not None and not retain_inputs:
                await asyncio.to_thread(shutil.rmtree, directory, ignore_errors=True)
            if reserved:
                release_upload()

    def get_record(job_id):
        valid_id(job_id)
        record = call("job", {"job_id": job_id})
        if record.get("id", record.get("job_id")) != job_id:
            raise HTTPException(503, "Manager returned a different task identity")
        # A durable manager record resolves this upload's uncertain delivery;
        # only the transport marker is removed, never manager-owned inputs.
        directory = root / job_id
        if not directory.is_symlink():
            try:
                (directory / "submission-uncertain.json").unlink(missing_ok=True)
            except OSError:
                LOG.warning("Could not remove resolved upload marker")
        return record

    @app.get("/v1/jobs/{job_id}", dependencies=[Depends(require_auth)])
    def status(job_id: str):
        return public_job(get_record(job_id))

    @app.post("/v1/jobs/{job_id}/cancel", status_code=202, dependencies=[Depends(require_auth)])
    def cancel(job_id: str):
        valid_id(job_id)
        record = call("cancel", {"job_id": job_id})
        if record.get("id", record.get("job_id")) != job_id:
            raise HTTPException(503, "Manager returned a different task identity")
        return public_job(record)

    @app.post("/v1/jobs/{job_id}/drive-archive", dependencies=[Depends(require_auth)])
    def register_drive_archive(job_id: str, receipt: DriveArchiveReceipt):
        """Record the authenticated owner's receipt after external upload and readback."""
        valid_id(job_id)
        record = call("archive", {"job_id": job_id, **receipt.model_dump()})
        if record.get("id", record.get("job_id")) != job_id:
            raise HTTPException(503, "Manager returned a different task identity")
        return public_job(record)

    @app.get("/v1/jobs/{job_id}/file", dependencies=[Depends(require_auth)])
    def download(job_id: str):
        record = get_record(job_id)
        if record.get("status") != "succeeded":
            raise HTTPException(409, "Video is not ready")
        artifact = record.get("artifact")
        if not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str):
            raise HTTPException(409, "No verified artifact is published")
        expected = root / job_id / "result.mp4"
        candidate = Path(artifact["path"])
        # The manager chooses output identity; even a malformed/stale response
        # must not let this public adapter serve arbitrary local files.
        if candidate != expected or candidate.is_symlink() or expected.parent.is_symlink():
            raise HTTPException(409, "Artifact ownership does not match the task")
        try:
            if candidate.resolve() != expected or not candidate.is_file():
                raise HTTPException(404, "Result was removed")
            size = candidate.stat().st_size
            if not size or artifact.get("bytes") != size:
                raise HTTPException(409, "Artifact size no longer matches its committed manifest")
            digest = hashlib.sha256()
            with candidate.open("rb") as source:
                for chunk in iter(lambda: source.read(MIB), b""):
                    digest.update(chunk)
            committed_hash = artifact.get("sha256")
            if (not isinstance(committed_hash, str) or not re.fullmatch(r"[a-f0-9]{64}", committed_hash)
                    or not hmac.compare_digest(digest.hexdigest(), committed_hash)):
                raise HTTPException(409, "Artifact content no longer matches its committed manifest")
        except OSError:
            raise HTTPException(404, "Result was removed or cannot be read") from None
        return FileResponse(candidate, media_type="video/mp4", filename=f"{job_id}.mp4",
                            headers={"ETag": '"' + digest.hexdigest() + '"',
                                     "Cache-Control": "private, no-store"})

    return app
