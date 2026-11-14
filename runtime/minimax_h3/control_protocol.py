"""Small authenticated loopback RPC shared by the three managed processes."""
from __future__ import annotations

import hmac
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

MAX_BODY = 2 * 1024 * 1024
LOG = logging.getLogger("h3.control")


class ControlError(RuntimeError):
    def __init__(self, message: str, status: int = 409):
        super().__init__(message)
        self.status = int(status)


def rpc_call(url, key, operation, payload=None, timeout=10):
    origin = urlsplit(url)
    if origin.scheme != "http" or origin.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ControlError("Private control transport must use loopback HTTP", 400)
    if origin.username or origin.password or origin.query or origin.fragment:
        raise ControlError("Invalid control origin", 400)
    body = json.dumps({"operation": operation, "payload": payload or {}}, allow_nan=False).encode()
    if len(body) > MAX_BODY:
        raise ControlError("Control payload too large", 413)
    request = Request(url.rstrip("/") + "/rpc", data=body, method="POST", headers={
        "Authorization": "Bearer " + key, "Content-Type": "application/json", "Connection": "close"})
    try:
        try:
            response = urlopen(request, timeout=timeout)
        except HTTPError as error:
            response = error
        with response:
            raw = response.read(MAX_BODY + 1)
            status = response.status
        if len(raw) > MAX_BODY:
            raise ControlError("Control response too large", 502)
        data = json.loads(raw)
        if not isinstance(data, dict) or type(data.get("ok")) is not bool:
            raise ValueError("Invalid response envelope")
        if data["ok"] is not True or not 200 <= status < 300:
            raise ControlError(str(data.get("error", "Control operation failed")), data.get("status", status))
        return data.get("result")
    except ControlError:
        raise
    except (URLError, OSError, ValueError, TimeoutError) as error:
        raise ControlError("Control transport unavailable or invalid response", 503) from error


def make_server(host, port, key, dispatch):
    if host not in {"127.0.0.1", "localhost"}:
        raise ValueError("Private controller must bind IPv4 loopback")
    if not isinstance(key, str) or len(key) < 32 or not key.isascii() or any(c.isspace() for c in key):
        raise ValueError("A private control token of at least 32 ASCII characters is required")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def setup(self):
            super().setup()
            self.connection.settimeout(15)

        def log_message(self, *_):
            pass  # Never log authentication headers, payloads or query secrets.

        def reply(self, status, value):
            encoded = json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            try:
                self.wfile.write(encoded)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            if self.path != "/health":
                return self.reply(404, {"ok": False, "error": "Not found", "status": 404})
            self.reply(200, {"alive": True})

        def do_POST(self):
            if self.path != "/rpc":
                return self.reply(404, {"ok": False, "error": "Not found", "status": 404})
            auth = self.headers.get("Authorization", "")
            if not auth.isascii() or not hmac.compare_digest(auth, "Bearer " + key):
                return self.reply(401, {"ok": False, "error": "Private authorization required", "status": 401})
            try:
                if self.headers.get("Transfer-Encoding"):
                    raise ControlError("Chunked control requests are unsupported", 400)
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= MAX_BODY:
                    raise ControlError("Invalid control payload size", 413)
                raw = self.rfile.read(size)
                if len(raw) != size:
                    raise ControlError("Incomplete control request", 400)
                data = json.loads(raw, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
                if not isinstance(data, dict) or not isinstance(data.get("operation"), str):
                    raise ControlError("Invalid control operation", 400)
                payload = data.get("payload", {})
                if not isinstance(payload, dict):
                    raise ControlError("Control payload must be an object", 400)
                result = dispatch(data["operation"], payload)
                self.reply(200, {"ok": True, "result": result})
            except ControlError as error:
                self.reply(error.status, {"ok": False, "error": str(error), "status": error.status})
            except (ValueError, TypeError) as error:
                self.reply(400, {"ok": False, "error": "Invalid control JSON or arguments", "status": 400})
            except Exception:
                LOG.exception("Control dispatch failed")
                self.reply(500, {"ok": False, "error": "Internal control operation failed; inspect owner log", "status": 500})

    server = ThreadingHTTPServer((host, int(port)), Handler)
    server.daemon_threads = True
    return server
