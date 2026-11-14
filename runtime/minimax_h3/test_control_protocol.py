import json
import threading
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import pytest
from control_protocol import ControlError, make_server, rpc_call

KEY = "private-test-token-" + "x" * 40


@pytest.fixture
def control():
    def dispatch(operation, payload):
        if operation == "reject":
            raise ControlError("Lease occupied", 409)
        if operation == "crash":
            raise RuntimeError("SECRET-MUST-NOT-LEAK")
        return {"operation": operation, "payload": payload}
    server = make_server("127.0.0.1", 0, KEY, dispatch)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()
    thread.join(2)


def test_authenticated_round_trip(control):
    assert rpc_call(control, KEY, "status", {"after": 4}) == {"operation": "status", "payload": {"after": 4}}


def test_public_token_cannot_control_private_service(control):
    with pytest.raises(ControlError) as caught:
        rpc_call(control, "wrong-public-key", "shutdown")
    assert caught.value.status == 401


def test_conflict_semantics_survive_transport(control):
    with pytest.raises(ControlError) as caught:
        rpc_call(control, KEY, "reject")
    assert caught.value.status == 409 and str(caught.value) == "Lease occupied"


def test_non_ascii_authorization_is_a_clean_rejection(control):
    request = Request(control + "/rpc", data=b'{}', headers={"Authorization": "Bearer \u00e9"})
    with pytest.raises(HTTPError) as caught:
        urlopen(request)
    assert caught.value.code == 401


def test_unexpected_failure_is_sanitized(control):
    with pytest.raises(ControlError) as caught:
        rpc_call(control, KEY, "crash")
    assert caught.value.status == 500 and "SECRET" not in str(caught.value)


@pytest.mark.parametrize("payload", [b"[]", b'{"operation":"x","payload":[]}', b'{"operation":"x","payload":{"x":NaN}}'])
def test_invalid_envelopes_are_not_dispatched(control, payload):
    request = Request(control + "/rpc", data=payload, headers={"Authorization": "Bearer " + KEY})
    with pytest.raises(HTTPError) as caught:
        urlopen(request)
    assert caught.value.code == 400


def test_remote_private_origin_is_rejected_before_network():
    with pytest.raises(ControlError):
        rpc_call("https://example.org", KEY, "status")


def test_network_failure_has_no_automatic_mutation_retry(monkeypatch):
    import control_protocol
    calls = []
    def unavailable(*a, **kw):
        calls.append(1)
        raise OSError("offline")
    monkeypatch.setattr(control_protocol, "urlopen", unavailable)
    with pytest.raises(ControlError):
        rpc_call("http://127.0.0.1:1", KEY, "execute")
    assert len(calls) == 1
