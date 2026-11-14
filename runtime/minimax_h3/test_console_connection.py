"""Connection display is explicit; normal observation and saved HTML never contain a key."""
import base64
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from notebook_console import ConsoleController, render_console


@pytest.fixture
def console(tmp_path):
    key = 'board-visible-public-key-' + 'x' * 32
    host = SimpleNamespace(api_key=key, _control_key='private-control-' + 'y' * 32,
                           instance_id='owned-instance', port=8123)
    state = {'instance_id': host.instance_id, 'api_ready': True, 'public_requested': True,
             'public_api_ready': False, 'public_url': 'https://candidate.example.invalid',
             'model_ready': False, 'manager_alive': True, 'phase': 'preparing'}
    host.snapshot = Mock(side_effect=lambda: dict(state))
    controller = ConsoleController(tmp_path, {}, {'startup_host': host})
    yield controller, host, state
    controller.close()


def test_assigned_url_is_distinct_from_reachable_api(console):
    controller, host, state = console
    connection = controller.snapshot()['connection']
    assert connection['url'] is None
    assert connection['candidate_url'] == state['public_url']
    assert connection['local_url'] == 'http://127.0.0.1:8123'
    assert connection['api_ready'] and not connection['public_api_ready']
    assert connection['key_available'] and connection['credential_id']
    state['public_api_ready'] = True
    assert controller.snapshot()['connection']['url'] == state['public_url']
    state['observation_stale'] = True
    stale = controller.snapshot()['connection']
    assert stale['url'] is None and stale['local_url'] is None
    assert not stale['model_ready'] and not stale['api_ready']


def test_key_only_returned_for_explicit_matching_credential_request(console):
    controller, host, state = console
    snapshot = controller.snapshot()
    public_id = snapshot['connection']['credential_id']
    assert public_id == hashlib.sha256(host.api_key.encode()).hexdigest()[:16]
    assert host.api_key not in json.dumps(snapshot)
    assert host.api_key not in render_console('__INITIAL_STATE__', 'callback', snapshot)
    rejected = controller.dispatch('reveal_connection', {'credential_id': 'stale'})
    assert rejected['error'] and 'connection_secret' not in rejected
    response = controller.dispatch('reveal_connection', {'credential_id': public_id})
    secret = response.pop('connection_secret')
    assert secret == {'api_key': host.api_key, 'credential_id': public_id}
    assert host.api_key not in json.dumps(response)
    assert host._control_key not in json.dumps(response)
    audit = controller.audit_path.read_text(encoding='utf-8')
    assert 'credential.revealed' in audit and host.api_key not in audit and host._control_key not in audit
    host.api_key = 'replacement-public-key-' + 'z' * 32
    assert controller.dispatch('reveal_connection', {'credential_id': public_id})['error']


def test_connection_export_and_board_use_same_readiness_and_credential(console):
    controller, host, state = console
    state['public_requested'] = False
    response = controller.dispatch('export_connection')
    exported = json.loads(base64.b64decode(response['download']['data_base64']))
    assert exported['api_key'] == host.api_key
    for key, value in response['connection'].items():
        assert exported[key] == value
    assert exported['url'] == 'http://127.0.0.1:8123'


def test_reveal_before_api_start_and_reject_after_close(tmp_path):
    key = 'new-boot-key-' + 'a' * 32
    controller = ConsoleController(tmp_path, {}, {'BOOTSTRAP_API_KEY': key, 'BOOTSTRAP_PUBLIC': True})
    try:
        initial = controller.snapshot()
        assert initial['connection']['url'] is None and initial['connection']['key_available']
        reply = controller.dispatch('reveal_connection', {'credential_id': initial['connection']['credential_id']})
        assert reply['connection_secret']['api_key'] == key
        assert not reply['connection']['api_ready']
    finally:
        controller.close()
    reply = controller.dispatch('reveal_connection', {'credential_id': initial['connection']['credential_id']})
    assert reply['ui']['closed'] and 'connection_secret' not in reply


def test_no_new_key_is_combined_with_old_owner_url(console):
    controller, host, state = console
    state['public_api_ready'] = True
    old_state = dict(state)
    controller.scope.pop('startup_host')
    controller.scope['BOOTSTRAP_API_KEY'] = 'different-owned-key-' + 'b' * 32
    controller.scope['BOOTSTRAP_PUBLIC'] = True
    connection, key = controller._connection_context(old_state)
    assert connection['url'] is None and connection['candidate_url'] is None
    assert key == controller.scope['BOOTSTRAP_API_KEY']


def test_download_projection_keeps_unknown_total_and_loading_distinct_from_ready(console):
    controller, host, state = console
    state['downloads'] = {'schema_version': 1, 'model_instance_id': 'model-generation',
        'model_ready': False, 'observation_stale': True, 'items': {
            'model': {'phase': 'loaded', 'bytes_completed': 42, 'bytes_total': None, 'percent': None,
                      'detail': 'diagnostic ' + host._control_key},
            'acceleration': {'phase': 'disabled'}, 'style': {'phase': 'waiting'}}}
    reply = controller.snapshot()
    assert reply['downloads']['items']['model']['percent'] is None
    assert reply['downloads']['items']['model']['bytes_total'] is None
    assert reply['downloads']['items']['model']['phase'] == 'loaded'
    assert not reply['downloads']['model_ready'] and reply['downloads']['observation_stale']
    assert not reply['state']['model_ready']
    assert host._control_key not in json.dumps(reply)


def test_before_host_downloads_distinguish_selected_and_disabled_lora(tmp_path):
    controller = ConsoleController(tmp_path, {})
    try:
        reply = controller.snapshot()['downloads']
        assert reply['items']['model']['phase'] == 'waiting'
        assert reply['items']['acceleration']['phase'] == 'disabled'
        assert reply['items']['style']['phase'] == 'disabled'
        controller.requested_options = {'acceleration': 'lightx2v-ref2va-4step', 'style': 'none'}
        reply = controller.snapshot()['downloads']
        assert reply['items']['acceleration']['phase'] == 'waiting'
        assert reply['items']['acceleration']['percent'] is None
        assert reply['items']['style']['phase'] == 'disabled'
    finally:
        controller.close()
