from argparse import Namespace
import json

import pytest
import requests

import smoke_test as smoke


class Response:
    def __init__(self, status, data):
        self.status_code, self.data = status, data
    def json(self):
        return self.data


@pytest.mark.parametrize("health_status", [200, 503])
def test_resume_checkpoints_existing_job_and_does_not_resubmit(tmp_path, monkeypatch, health_status):
    job_id = "a" * 32
    posts = []
    class Session:
        def get(self, url, **kwargs):
            if url.endswith("/health"):
                return Response(health_status, {"ready": False})
            return Response(200, {"id": job_id, "status": "failed", "metrics": {"status": "failed"},
                                  "generation_parameters": {"steps": 5}})
        def post(self, url, **kwargs):
            posts.append(kwargs)
            return Response(401, {})
    monkeypatch.setattr(requests, "Session", Session)
    monkeypatch.setattr(smoke.shutil, "which", lambda name: name)
    monkeypatch.setenv("H3_API_KEY", "test" * 12)
    args = Namespace(url="http://127.0.0.1:8000", steps=8, resume_job=job_id, outputdir=tmp_path,
                     startup_timeout=1, generation_timeout=1, poll_seconds=1, reference=None)
    report = {"passed": False}
    with pytest.raises(RuntimeError, match="Real model generation failed"):
        smoke.run(args, report)
    assert posts == []  # Resume is strictly GET-only, including after model stop.
    saved = json.loads((tmp_path / "test_report.json").read_text())
    assert saved["job_id"] == job_id
    assert saved["parameters"] == {"steps": 5}  # no guessed steps on resume
    assert not list(tmp_path.glob("*.jpg"))
    assert report["metrics"]["status"] == "failed"
    assert not (tmp_path.parent / "content-history.json").exists()


def test_report_checkpoint_replaces_atomically_without_temp_files(tmp_path):
    smoke.save_report(tmp_path, {"job_id": "b" * 32, "passed": False})
    smoke.save_report(tmp_path, {"job_id": "b" * 32, "passed": True})
    assert json.loads((tmp_path / "test_report.json").read_text())["passed"]
    assert list(tmp_path.iterdir()) == [tmp_path / "test_report.json"]


@pytest.mark.parametrize("reference,prompt,message", [(None, "Fresh scene", "--reference"),
    ("chosen.png", None, "--prompt"), ("chosen.png", "  ", "--prompt")])
def test_new_generation_requires_explicit_inputs_before_network(tmp_path, monkeypatch, reference, prompt, message):
    monkeypatch.setattr(requests, "Session", lambda: pytest.fail("Missing input must fail before networking"))
    args = Namespace(resume_job=None, reference=reference, prompt=prompt)
    with pytest.raises(RuntimeError, match=message):
        smoke.run(args, {})


def test_content_is_durably_registered_before_submission_and_repetition_is_rejected(tmp_path, monkeypatch):
    from PIL import Image
    reference = tmp_path / "new-reference.png"
    Image.new("RGB", (8, 8), "orange").save(reference)
    outputdir = tmp_path / "smoke"
    outputdir.mkdir()
    history = tmp_path / "content-history.json"
    job_id = "c" * 32
    paid_posts = []
    class Session:
        def get(self, url, **kwargs):
            if url.endswith("/health"):
                return Response(200, {"ready": True})
            return Response(200, {"id": job_id, "status": "failed"})
        def post(self, url, **kwargs):
            if "headers" not in kwargs:
                return Response(401, {})
            recorded = json.loads(history.read_text())
            checkpoint = json.loads((outputdir / "test_report.json").read_text())
            entry = next(iter(recorded["entries"].values()))
            assert entry["request_id"] == kwargs["headers"]["Idempotency-Key"]
            assert checkpoint["content_registration"]["fingerprint"] == entry["fingerprint"]
            assert kwargs["data"]["prompt"] == "A fresh orange sculpture turns slowly."
            paid_posts.append(kwargs)
            return Response(202, {"id": job_id})
    monkeypatch.setattr(requests, "Session", Session)
    monkeypatch.setattr(requests, "get", lambda *a, **kw: pytest.fail("No automatic reference download"))
    monkeypatch.setattr(smoke.shutil, "which", lambda name: name)
    monkeypatch.setenv("H3_API_KEY", "test" * 12)
    args = Namespace(url="http://127.0.0.1:8000", steps=8, resume_job=None, outputdir=outputdir,
                     startup_timeout=1, generation_timeout=1, poll_seconds=1, reference=str(reference),
                     prompt="A fresh orange sculpture turns slowly.", content_history=None)
    with pytest.raises(RuntimeError, match="Real model generation failed"):
        smoke.run(args, {"passed": False})
    with pytest.raises(RuntimeError, match="already registered"):
        smoke.run(args, {"passed": False})
    assert len(paid_posts) == 1
    assert len(json.loads(history.read_text())["entries"]) == 1
    assert not list(tmp_path.glob("*.tmp"))


def test_content_registry_accepts_new_combination_and_normalizes_edge_whitespace(tmp_path):
    path = tmp_path / "history.json"
    kwargs = {"outputdir": tmp_path, "api_base_url": "http://127.0.0.1:8000"}
    first = smoke.register_content(path, "a" * 64, "Fresh prompt", **kwargs)
    with pytest.raises(RuntimeError, match="already registered"):
        smoke.register_content(path, "a" * 64, "  Fresh prompt  ", **kwargs)
    second = smoke.register_content(path, "a" * 64, "Another new prompt", **kwargs)
    third = smoke.register_content(path, "b" * 64, "Fresh prompt", **kwargs)
    assert len({entry["request_id"] for entry in (first, second, third)}) == 3
    assert len(json.loads(path.read_text())["entries"]) == 3


def test_content_registry_serializes_concurrent_duplicate_reservations(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    path = tmp_path / "history.json"
    def register(_):
        try:
            smoke.register_content(path, "a" * 64, "Same fresh prompt", outputdir=tmp_path,
                                   api_base_url="http://127.0.0.1:8000")
            return True
        except RuntimeError as error:
            assert "already registered" in str(error)
            return False
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert sum(pool.map(register, range(4))) == 1
    assert len(json.loads(path.read_text())["entries"]) == 1


def test_loading_model_does_not_delay_submission_and_feedback_is_preserved(tmp_path, monkeypatch, capsys):
    from PIL import Image
    reference = tmp_path / "fresh-cyan-kite.png"
    Image.new("RGB", (8, 8), "cyan").save(reference)
    outputdir = tmp_path / "smoke"
    outputdir.mkdir()
    job_id = "d" * 32
    calls = []
    states = iter([
        {"status": "queued", "queue_position": 3, "waiting_for": "model_ready"},
        {"status": "running", "active_phase": {"phase": "gpu_running"}},
        {"status": "failed", "error": "fixture_encoding_failed", "cleanup_complete": True},
    ])

    class Session:
        def get(self, url, **kwargs):
            calls.append(("GET", url))
            if url.endswith("/health"):
                return Response(200, {"api_ready": True, "manager_ready": True,
                                      "model_ready": False, "ready": False})
            return Response(200, {"id": job_id, **next(states)})

        def post(self, url, **kwargs):
            calls.append(("POST", url, "headers" in kwargs))
            if "headers" not in kwargs:
                return Response(401, {})
            return Response(202, {"id": job_id, "status": "queued", "waiting_for": "model_ready"})

    monkeypatch.setattr(requests, "Session", Session)
    monkeypatch.setattr(smoke.shutil, "which", lambda name: name)
    monkeypatch.setattr(smoke.time, "sleep", lambda _: None)
    monkeypatch.setenv("H3_API_KEY", "test" * 12)
    args = Namespace(url="http://127.0.0.1:8000", steps=8, resume_job=None, outputdir=outputdir,
                     startup_timeout=0, generation_timeout=1, poll_seconds=1, reference=str(reference),
                     prompt="A cyan kite drifts above a new stone lighthouse.", content_history=None)
    report = {"passed": False}
    with pytest.raises(RuntimeError, match="Real model generation failed"):
        smoke.run(args, report)
    assert calls[:3] == [("GET", args.url + "/health"), ("POST", args.url + "/v1/jobs", False),
                         ("POST", args.url + "/v1/jobs", True)]
    assert sum(call[1].endswith("/health") for call in calls) == 1
    assert report["health"]["admission_ready"] is True
    assert report["health"]["ready"] is False and report["health"]["model_ready"] is False
    assert [item["status"] for item in report["status_history"]] == ["queued", "running", "failed"]
    assert report["status_history"][0]["queue_position"] == 3
    assert report["status_history"][0]["waiting_for"] == "model_ready"
    assert report["status_history"][1]["active_phase"] == {"phase": "gpu_running"}
    assert report["terminal_error"] == "fixture_encoding_failed" and report["cleanup_complete"]
    assert "waiting_for=model_ready" in capsys.readouterr().out
    saved = json.loads((outputdir / "test_report.json").read_text())
    assert saved["health"]["ready"] is False


@pytest.mark.parametrize("health_status,health", [(200, {"ready": True, "api_ready": False, "manager_ready": True}),
    (200, {"ready": True, "api_ready": True, "manager_ready": False}), (200, {"ready": True, "api_ready": True}),
    (503, {"ready": True, "api_ready": True, "manager_ready": True})])
def test_aggregate_ready_never_overrides_explicit_unavailable_capability(tmp_path, monkeypatch, health_status, health):
    from PIL import Image
    reference = tmp_path / "new-lilac-clock.png"
    Image.new("RGB", (8, 8), "purple").save(reference)
    class Session:
        def get(self, url, **kwargs):
            return Response(health_status, health)
        def post(self, *args, **kwargs):
            pytest.fail("Unavailable admission must not submit a job")
    monkeypatch.setattr(requests, "Session", Session)
    monkeypatch.setattr(smoke.shutil, "which", lambda name: name)
    monkeypatch.setenv("H3_API_KEY", "test" * 12)
    args = Namespace(url="http://127.0.0.1:8000", steps=8, resume_job=None, outputdir=tmp_path,
                     startup_timeout=0, generation_timeout=1, poll_seconds=1, reference=str(reference),
                     prompt="A new lilac clock floats gently in a glass room.")
    with pytest.raises(RuntimeError, match="API and manager did not become available"):
        smoke.run(args, {})
