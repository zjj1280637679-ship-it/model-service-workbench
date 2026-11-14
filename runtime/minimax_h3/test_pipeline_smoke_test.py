"""Acceptance-client network recovery must never create replacement GPU jobs."""
import argparse
import json

import pytest
import requests

import pipeline_smoke_test as client


class Response:
    def __init__(self, status, body=None):
        self.status_code, self.body = status, body

    def json(self):
        return self.body


def test_read_retries_transport_and_gateway_failures_without_post(monkeypatch):
    answers = iter([requests.Timeout(), Response(503), Response(200, {'id': 'same-job'})])
    calls, records = [], []
    def get(url, **kwargs):
        calls.append(url)
        answer = next(answers)
        if isinstance(answer, Exception):
            raise answer
        return answer
    monkeypatch.setattr(requests, 'get', get)
    monkeypatch.setattr(requests, 'post', lambda *a, **k: pytest.fail('Read retry submitted a job'))
    monkeypatch.setattr(client.time, 'sleep', lambda seconds: None)
    assert client.get_json('https://test.invalid', 'secret', '/v1/jobs/same-job', records=records) == {'id': 'same-job'}
    assert calls == ['https://test.invalid/v1/jobs/same-job'] * 3
    assert records[0]['error_type'] == 'Timeout'
    assert records[1]['http_status'] == 503
    assert 'secret' not in json.dumps(records)


def test_read_rejects_auth_failure_without_retry(monkeypatch):
    calls = []
    monkeypatch.setattr(requests, 'get', lambda *a, **k: calls.append(a) or Response(401))
    with pytest.raises(RuntimeError, match='HTTP 401'):
        client.get_json('https://test.invalid', 'secret', '/v1/system')
    assert len(calls) == 1


def test_read_retry_has_finite_limit(monkeypatch):
    records = []
    monkeypatch.setattr(requests, 'get', lambda *a, **k: Response(502))
    monkeypatch.setattr(client.time, 'sleep', lambda seconds: None)
    with pytest.raises(RuntimeError, match='exhausted bounded read retries'):
        client.get_json('https://test.invalid', 'secret', '/v1/jobs/same-job', records=records)
    assert len(records) == 3


def test_admission_failure_preserves_received_ids_and_redacts_secret(tmp_path, monkeypatch):
    key = 'sensitive-key-' * 3
    monkeypatch.setenv('H3_API_KEY', key)
    args = argparse.Namespace(outputdir=tmp_path)
    def fail(args):
        client.checkpoint(tmp_path, {'jobs': [{'id': 'accepted-before-failure'}]})
        args._active_report = True
        raise RuntimeError('Failure containing ' + key)
    monkeypatch.setattr(client, '_run', fail)
    with pytest.raises(RuntimeError):
        client.run(args)
    report = json.loads((tmp_path/'pipeline_report.json').read_text())
    assert report['jobs'] == [{'id': 'accepted-before-failure'}]
    assert not report['passed'] and '[REDACTED]' in report['error']
    assert key not in json.dumps(report)


def test_invalid_new_run_does_not_overwrite_previous_pass(tmp_path, monkeypatch):
    client.checkpoint(tmp_path, {'passed': True, 'jobs': [{'id': 'old-success'}]})
    original = (tmp_path/'pipeline_report.json').read_bytes()
    def fail(args):
        raise RuntimeError('Existing test report: use --resume')
    monkeypatch.setattr(client, '_run', fail)
    with pytest.raises(RuntimeError):
        client.run(argparse.Namespace(outputdir=tmp_path))
    assert (tmp_path/'pipeline_report.json').read_bytes() == original
