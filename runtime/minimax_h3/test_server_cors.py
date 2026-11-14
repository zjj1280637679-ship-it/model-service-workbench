"""Real ASGI browser transport checks; no model imports, weights or GPU."""
import hashlib

import pytest
from fastapi.testclient import TestClient

import server
from test_server import HEADERS, INSTANCE, KEY, TestController, submit

PAGES = "https://zjj1280637679-ship-it.github.io"


@pytest.fixture
def cors_api(tmp_path, monkeypatch):
    monkeypatch.delenv("H3_CORS_ORIGINS", raising=False)
    monkeypatch.delenv("H3_PARENT_PID", raising=False)
    monkeypatch.setenv("H3_INSTANCE_ID", INSTANCE)
    owner = TestController(tmp_path)
    with TestClient(server.create_app(controller=owner, api_key=KEY, output_dir=tmp_path)) as client:
        yield client, owner, tmp_path


def preflight(client, origin=PAGES, *, method="POST", headers="authorization,idempotency-key,content-type"):
    return client.options("/v1/jobs", headers={"Origin": origin, "Access-Control-Request-Method": method,
                                             "Access-Control-Request-Headers": headers})


@pytest.mark.parametrize("origin", server.DEFAULT_CORS_ORIGINS)
def test_allowed_preflight_is_outside_bearer_guard_and_never_calls_manager(cors_api, origin):
    client, owner, _ = cors_api
    response = preflight(client, origin, headers="authorization,idempotency-key,content-type,range")
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin
    assert set(response.headers["access-control-allow-methods"].split(", ")) == {"GET", "POST"}
    assert "Origin" in response.headers["vary"]
    assert "access-control-allow-credentials" not in response.headers
    assert owner.calls == []


@pytest.mark.parametrize("origin", ["https://attacker.example", "null", "http://zjj1280637679-ship-it.github.io",
                                    PAGES + ".attacker.example", "http://localhost:8765"])
def test_unlisted_origin_preflight_is_rejected(cors_api, origin):
    client, owner, _ = cors_api
    response = preflight(client, origin)
    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers
    assert owner.calls == []


@pytest.mark.parametrize("method,headers", [("DELETE", "authorization"), ("POST", "authorization,x-unapproved-header")])
def test_disallowed_methods_and_headers_are_rejected(cors_api, method, headers):
    client, owner, _ = cors_api
    assert preflight(client, method=method, headers=headers).status_code == 400
    assert owner.calls == []


@pytest.mark.parametrize("method,path", [("GET", "/v1/system"), ("POST", "/v1/jobs"),
                                         ("POST", "/v1/jobs/" + "b" * 32 + "/cancel"),
                                         ("GET", "/v1/jobs/" + "b" * 32 + "/file"),
                                         ("OPTIONS", "/v1/jobs")])
def test_actual_requests_still_require_bearer_and_401_is_readable(cors_api, method, path):
    client, owner, _ = cors_api
    response = client.request(method, path, headers={"Origin": PAGES})
    assert response.status_code == 401
    assert response.headers["access-control-allow-origin"] == PAGES
    assert response.headers["www-authenticate"] == "Bearer"
    assert owner.calls == []


def test_real_upload_query_cancel_and_range_download_with_browser_headers(cors_api):
    client, owner, root = cors_api
    headers = {**HEADERS, "Origin": PAGES, "Idempotency-Key": "browser-upload-once"}
    accepted = submit(client, headers=headers)
    assert accepted.status_code == 202
    assert accepted.headers["access-control-allow-origin"] == PAGES
    job_id = accepted.json()["id"]
    state = client.get(f"/v1/jobs/{job_id}", headers=headers)
    assert state.status_code == 200 and state.json()["status"] == "queued"
    cancelled = client.post(f"/v1/jobs/{job_id}/cancel", headers=headers)
    assert cancelled.status_code == 202 and cancelled.json()["status"] == "cancelled"

    # Download tests only need a committed byte manifest, not GPU-generated media.
    payload = b"browser-range-test-mp4-bytes"
    path = root / job_id / "result.mp4"
    path.write_bytes(payload)
    owner.records[job_id].update(status="succeeded", artifact={"path": str(path), "bytes": len(payload),
                                                              "sha256": hashlib.sha256(payload).hexdigest()})
    ranged = client.get(f"/v1/jobs/{job_id}/file", headers={**headers, "Range": "bytes=0-6"})
    assert ranged.status_code == 206 and ranged.content == payload[:7]
    assert ranged.headers["access-control-allow-origin"] == PAGES
    assert ranged.headers["content-range"] == f"bytes 0-6/{len(payload)}"
    exposed = {name.strip().lower() for name in ranged.headers["access-control-expose-headers"].split(",")}
    assert {"content-disposition", "content-length", "content-range", "accept-ranges", "etag", "retry-after"} <= exposed
    assert ranged.headers["cache-control"] == "private, no-store"
    assert client.get(f"/v1/jobs/{job_id}/file", headers={"Origin": PAGES}).status_code == 401


def test_same_origin_without_origin_header_is_unchanged(cors_api):
    client, _, _ = cors_api
    assert client.get("/health").status_code == 200
    response = client.get("/v1/system", headers=HEADERS)
    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers


def test_unlisted_simple_origin_cannot_read_authenticated_response(cors_api):
    client, _, _ = cors_api
    response = client.get("/v1/system", headers={**HEADERS, "Origin": "https://attacker.example"})
    assert response.status_code == 200  # CORS is not an alternate auth mechanism.
    assert "access-control-allow-origin" not in response.headers


def test_origin_override_normalizes_and_replaces_defaults(monkeypatch, tmp_path):
    monkeypatch.setenv("H3_CORS_ORIGINS", "https://UI.example:443/,http://127.0.0.1:8765,http://[::1]:8765,https://ui.example")
    assert server._cors_allowed_origins() == ["https://ui.example", "http://127.0.0.1:8765", "http://[::1]:8765"]
    with TestClient(server.create_app(controller=lambda *_: {}, api_key=KEY, output_dir=tmp_path)) as client:
        assert preflight(client, "http://127.0.0.1:8765").status_code == 200
        assert preflight(client).status_code == 400


def test_explicit_empty_disables_cross_origin_but_not_api(monkeypatch, tmp_path):
    monkeypatch.setenv("H3_CORS_ORIGINS", "")
    assert server._cors_allowed_origins() == []
    with TestClient(server.create_app(controller=lambda *_: {}, api_key=KEY, output_dir=tmp_path)) as client:
        assert preflight(client).status_code == 400
        assert client.get("/health/live").status_code == 200


@pytest.mark.parametrize("value", ["*", "null", "https://*.example.com", "http://public.example", "https://user:secret@ui.example",
                                  "https://ui.example/path", "https://ui.example?token=secret", "https://ui.example#fragment",
                                  "https://ui.example:0", "https://ui.example:70000", "https://bad host.example", "data:text/plain,test"])
def test_unsafe_or_non_origin_configuration_fails_closed(monkeypatch, value):
    monkeypatch.setenv("H3_CORS_ORIGINS", value)
    with pytest.raises(RuntimeError, match="H3_CORS_ORIGINS") as error:
        server._cors_allowed_origins()
    assert "secret" not in str(error.value)
