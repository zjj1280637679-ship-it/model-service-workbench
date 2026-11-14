"""Real loopback command acceptance must not wait for child termination."""
import threading
import time

from control_protocol import make_server, rpc_call
from test_global_manager import rig


def test_rpc_accepts_and_queries_commands_while_prior_api_stop_is_still_running(rig, monkeypatch):
    manager = rig.manager
    entered, release = threading.Event(), threading.Event()
    terminate = manager._terminate
    def delayed(name, *args, **kwargs):
        if name == 'api':
            entered.set()
            assert release.wait(4), 'test did not release the controlled stop'
        return terminate(name, *args, **kwargs)
    monkeypatch.setattr(manager, '_terminate', delayed)
    server = make_server('127.0.0.1', 0, manager.key, manager.dispatch)
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    serving.start()
    origin = 'http://127.0.0.1:' + str(server.server_address[1])
    def request(operation, payload):
        return rpc_call(origin, manager.key, operation, payload, timeout=1)
    try:
        started = time.monotonic()
        stop = request('control', {'action': 'stop_api', 'command_id': 'delayed-stop'})
        assert stop['status'] == 'accepted' and time.monotonic() - started < 1
        assert entered.wait(1)
        assert request('command', {'command_id': stop['command_id']})['status'] == 'accepted'
        start = request('control', {'action': 'start_api', 'command_id': 'queued-restart'})
        assert start['status'] == 'accepted'
        assert request('command', {'command_id': start['command_id']})['status'] == 'accepted'
        assert request('status', {})['manager_alive'] is True
        release.set()
        for command in (stop, start):
            deadline = time.monotonic() + 2
            while True:
                record = request('command', {'command_id': command['command_id']})
                if record['status'] != 'accepted':
                    assert record['status'] == 'succeeded'
                    break
                assert time.monotonic() < deadline
                time.sleep(.01)
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        serving.join(timeout=2)
