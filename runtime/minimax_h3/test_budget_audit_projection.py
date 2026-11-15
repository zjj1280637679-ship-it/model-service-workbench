"""Real durable budget pages through manager ownership and authenticated HTTP."""
import copy
import threading

from fastapi.testclient import TestClient
import pytest

from condition_projection import public_budget_page
from control_protocol import ControlError
from pipeline_scheduler import PipelineBudget
import server
import startup
from test_global_manager import rig
from test_server import HEADERS, KEY


@pytest.fixture
def audit(rig, monkeypatch):
    budget = PipelineBudget(rig.manager.root, limits={'ram_bytes': 1000, 'disk_bytes': 1000, 'cpu_threads': 2})
    identity = 'c' * 32
    budget.reserve(identity, {'ram_bytes': 1, 'disk_bytes': 1, 'cpu_threads': 1})
    with budget.stage(identity, 'cpu'):
        pass
    model_id = rig.manager.model_instance_id
    original = rig.manager._rpc
    def rpc(url, key, operation, payload, timeout):
        if operation != 'budget_audit':
            return original(url, key, operation, payload, timeout)
        assert payload['instance_id'] == model_id
        if payload['budget_instance_id'] != budget.instance_id:
            raise ControlError('Budget identity changed', 409)
        # Another observer can acquire the state lock while the slow RPC is open.
        acquired = threading.Event()
        def observe():
            with rig.manager._lock:
                acquired.set()
        observer = threading.Thread(target=observe); observer.start()
        assert acquired.wait(1), 'Manager retained the main state lock across RPC'
        observer.join()
        try:
            page = budget.audit_page(after_sequence=payload.get('after_sequence', 0),
                through_sequence=payload.get('through_sequence'), limit=payload.get('limit', 100))
        except ValueError as error:
            raise ControlError(str(error), 422) from None
        return {'model_instance_id': model_id, **page}
    rig.manager._rpc = rpc
    monkeypatch.setenv('H3_INSTANCE_ID', rig.manager.instance_id)
    monkeypatch.delenv('H3_PARENT_PID', raising=False)
    with TestClient(server.create_app(controller=rig.manager.dispatch, api_key=KEY, output_dir=rig.manager.root)) as client:
        yield budget, client, {'model_instance_id': model_id, 'budget_instance_id': budget.instance_id}


def test_authenticated_bounded_pages_keep_full_hash_chain(audit):
    budget, client, query = audit
    assert client.get('/v1/audit/budget', params=query).status_code == 401
    first = client.get('/v1/audit/budget', params={**query, 'limit': 1}, headers=HEADERS)
    assert first.status_code == 200
    page = first.json()
    assert page['more'] is True and page['through_sequence'] == 2 and page['next_sequence'] == 1
    expected = {'model_instance_id': query['model_instance_id'], **budget.audit_page(limit=1)}
    assert page == expected
    last = client.get('/v1/audit/budget', params={**query, 'after_sequence': 1,
        'through_sequence': page['through_sequence'], 'limit': 1}, headers=HEADERS).json()
    assert last['events'][0]['previous_sha256'] == page['events'][0]['sha256']
    assert last['more'] is False and last['page_complete'] is True
    assert last['tail_sha256'] == last['events'][-1]['sha256']


def test_private_default_page_size_is_explicit_across_owner_defaults(audit, rig):
    budget, _, query = audit
    for _ in range(50):
        with budget.stage('c'*32, 'cpu'):
            pass
    assert budget.audit_page(limit=128)['through_sequence'] == 102
    requests = []
    original = rig.manager._rpc
    def model_default(url, key, operation, payload, timeout):
        if operation != 'budget_audit':
            return original(url, key, operation, payload, timeout)
        requests.append(payload.copy())
        return {'model_instance_id': query['model_instance_id'],
                **budget.audit_page(after_sequence=payload.get('after_sequence', 0),
                                    through_sequence=payload.get('through_sequence'),
                                    limit=payload.get('limit', 128))}
    rig.manager._rpc = model_default
    first = rig.manager.dispatch('budget_audit', {'instance_id': rig.manager.instance_id, **query})
    assert requests[0]['limit'] == 100 and requests[0]['after_sequence'] == 0
    assert len(first['events']) == 100 and first['more'] is True and first['next_sequence'] == 100
    final = rig.manager.dispatch('budget_audit', {'instance_id': rig.manager.instance_id, **query,
        'after_sequence': first['next_sequence'], 'through_sequence': first['through_sequence']})
    assert len(final['events']) == 2 and final['more'] is False
    assert final['events'][0]['previous_sha256'] == first['events'][-1]['sha256']


@pytest.mark.parametrize('extra', [{'limit': 257}, {'limit': 0}, {'after_sequence': -1},
    {'after_sequence': 3}, {'through_sequence': 0, 'after_sequence': 1}])
def test_audit_cursor_rejection_does_not_return_false_complete(audit, extra):
    _, client, query = audit
    assert client.get('/v1/audit/budget', params={**query, **extra}, headers=HEADERS).status_code == 422


@pytest.mark.parametrize('field', ['model_instance_id', 'budget_instance_id'])
def test_audit_old_owner_is_not_current(audit, field):
    _, client, query = audit
    assert client.get('/v1/audit/budget', params={**query, field: 'f'*32}, headers=HEADERS).status_code == 409


def test_replacement_during_audit_rpc_rejects_late_page(audit, rig):
    _, _, query = audit
    original = rig.manager._rpc
    def replace(*args, **kwargs):
        result = original(*args, **kwargs)
        rig.manager.model_instance_id = 'f'*32
        return result
    rig.manager._rpc = replace
    with pytest.raises(ControlError) as error:
        rig.manager.dispatch('budget_audit', {'instance_id': rig.manager.instance_id, **query})
    assert error.value.status == 409


@pytest.mark.parametrize('query_change,page_change', [
    ({'after_sequence': 1, 'through_sequence': 2, 'limit': 1}, {}),
    ({'after_sequence': 0, 'through_sequence': 1, 'limit': 2}, {}),
    ({'after_sequence': 0, 'through_sequence': 2, 'limit': 1}, {}),
    ({}, {'events': [], 'next_sequence': 0, 'more': True})])
def test_valid_but_wrong_audit_range_cannot_answer_requested_page(audit, rig, query_change, page_change):
    budget, client, query = audit
    cached = {'model_instance_id': query['model_instance_id'], **budget.audit_page(limit=2), **page_change}
    original = rig.manager._rpc
    rig.manager._rpc = lambda url, key, operation, payload, timeout: cached if operation == 'budget_audit' else original(url, key, operation, payload, timeout)
    response = client.get('/v1/audit/budget', params={**query, **query_change}, headers=HEADERS)
    assert response.status_code == 503


def test_empty_nonterminal_audit_page_is_not_complete(audit):
    budget, _, query = audit
    page = {'model_instance_id': query['model_instance_id'], **budget.audit_page(),
            'events': [], 'next_sequence': 0, 'more': True}
    with pytest.raises(ValueError, match='completeness'):
        public_budget_page(page)


def test_http_itself_binds_page_to_requested_range(audit, rig):
    budget, _, query = audit
    page = {'model_instance_id': query['model_instance_id'], **budget.audit_page()}
    with TestClient(server.create_app(controller=lambda operation, payload: page, api_key=KEY, output_dir=rig.manager.root)) as client:
        response = client.get('/v1/audit/budget', params={**query, 'after_sequence': 1,
            'through_sequence': 2, 'limit': 1}, headers=HEADERS)
        assert response.status_code == 503


def test_audit_projection_drops_extra_secrets_without_rehashing(audit):
    budget, _, query = audit
    page = {'model_instance_id': query['model_instance_id'], **budget.audit_page()}
    contaminated = copy.deepcopy(page)
    contaminated['path'] = '/private/secret'
    contaminated['events'][0]['grant_receipt'] = {'api_key': KEY}
    assert public_budget_page(contaminated) == page
    contaminated['events'][0]['cpu_threads'] = True
    with pytest.raises(ValueError):
        public_budget_page(contaminated)


def test_startup_audit_preserves_selected_instances_and_cutoff(tmp_path, monkeypatch):
    host = startup.StartupSession(tmp_path, env={})
    calls = []
    monkeypatch.setattr(startup, 'rpc_call', lambda *args, **kwargs: calls.append((args, kwargs)) or {'test': True})
    assert host.budget_audit('b'*32, 'c'*32, after_sequence=2, through_sequence=8, limit=3) == {'test': True}
    args, kwargs = calls[0]
    assert args[2] == 'budget_audit'
    assert args[3] == {'instance_id': host.instance_id, 'model_instance_id': 'b'*32,
        'budget_instance_id': 'c'*32, 'after_sequence': 2, 'through_sequence': 8, 'limit': 3}
