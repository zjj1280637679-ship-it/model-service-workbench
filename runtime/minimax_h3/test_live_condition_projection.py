"""Identity, freshness and current-vs-historical evidence across manager/API/UI."""
import copy
import json
from pathlib import Path
import time
from types import SimpleNamespace
from fastapi.testclient import TestClient

import pytest

from condition_guard import evaluate_gate
from condition_projection import PUBLIC_GATE_CONDITIONS, public_condition_summary, public_queue_policy, public_execution_observation, public_execution_summary
from test_global_manager import rig, job, Rig
from test_notebook_console import controllers
import server
from test_server import KEY, HEADERS

FIXTURES = json.loads((Path(__file__).parents[2] / 'tests/fixtures/live-condition-projection.json').read_text(encoding='utf-8'))


@pytest.mark.parametrize('fixture', FIXTURES, ids=lambda row: row['name'])
def test_shared_python_condition_projection(fixture, controllers):
    state = copy.deepcopy(fixture['state'])
    state['condition_summary']['private_path'] = '/private/weights/secret'
    safe = public_condition_summary(state['condition_summary'], state['instance_id'])
    assert safe['status'] == fixture['status']
    if fixture.get('unknown_version'):
        assert safe['last_decision']['allowed'] is None
    assert '/private/' not in json.dumps(safe)
    host = SimpleNamespace(snapshot=lambda: state)
    controller = controllers(scope={'startup_host': host})
    public = controller.snapshot()
    assert public['state']['condition_summary'] == safe
    assert '/private/' not in json.dumps(public)
    assert '调度观察' in public['logs']['text']


@pytest.mark.parametrize('phase', ['prepare_wait', 'preparing', 'gpu_wait', 'gpu_running', 'encode_wait', 'encode_validate'])
def test_current_attempt_stage_projection_is_bound_to_owned_identity(rig, phase):
    accepted = rig.submit()
    rig.ready()
    rig.manager.refresh()
    attempt = rig.manager._model['attempts'][0]
    attempt.update(phase=phase, stage_completions={'prepare': True, 'gpu': None, 'encode_validate': None},
                   cancel_requested=False)
    gate_id = 'model.stage.prepare'
    receipt = evaluate_gate(gate_id, PUBLIC_GATE_CONDITIONS[gate_id],
        dict.fromkeys(PUBLIC_GATE_CONDITIONS[gate_id], True),
        {'instance_id': rig.manager.model_instance_id, 'job_id': accepted['job_id'], 'attempt_id': attempt['attempt_id']})
    attempt['condition_gates'] = {gate_id: receipt}
    rig.manager._model['observed_at'] = time.time()
    public = job(rig, accepted['job_id'])
    assert public['active_phase']['phase'] == phase
    assert public['execution_observation']['status'] == 'current'
    assert public['stage_completions']['prepare'] is True
    assert public['execution_condition_gates'][gate_id]['decision_id'] == receipt['decision_id']
    assert public['execution_condition_gates'][gate_id]['allowed'] is True


@pytest.mark.parametrize('mutation,expected', [
    ('old_model', 'identity_mismatch'), ('old_attempt', 'unknown'), ('old_job', 'unknown'),
    ('old_attempt_model', 'unknown'), ('stale', 'stale'), ('missing_time', 'unknown'), ('nan_time', 'unknown')])
def test_late_or_unbound_model_observation_never_becomes_current(rig, mutation, expected):
    accepted = rig.submit(); rig.ready(); rig.manager.refresh()
    attempt = rig.manager._model['attempts'][0]
    if mutation == 'old_model': rig.manager._model['instance_id'] = 'f' * 32
    elif mutation == 'old_attempt': attempt['attempt_id'] = 'f' * 32
    elif mutation == 'old_job': attempt['job_id'] = 'f' * 32
    elif mutation == 'old_attempt_model': attempt['instance_id'] = 'f' * 32
    elif mutation == 'stale': rig.manager._last_observation -= 100
    elif mutation == 'missing_time': rig.manager._model.pop('observed_at')
    else: rig.manager._model['observed_at'] = float('nan')
    public = job(rig, accepted['job_id'])
    assert public['execution_observation']['status'] == expected
    assert public['active_phase'] is None
    assert 'execution_condition_gates' not in public


@pytest.mark.parametrize('fault', ['job_id', 'attempt_id', 'instance_id', 'future', 'missing_decision'])
def test_live_attempt_cannot_borrow_receipt_from_other_identity_or_time(rig, fault):
    accepted = rig.submit(); rig.ready(); rig.manager.refresh()
    attempt = rig.manager._model['attempts'][0]
    gate_id = 'model.stage.gpu'
    receipt = evaluate_gate(gate_id, PUBLIC_GATE_CONDITIONS[gate_id],
        dict.fromkeys(PUBLIC_GATE_CONDITIONS[gate_id], True),
        {'instance_id': rig.manager.model_instance_id, 'job_id': accepted['job_id'], 'attempt_id': attempt['attempt_id']})
    rig.manager._model['observed_at'] = time.time()
    if fault == 'future': receipt['observed_at'] += 3600
    elif fault == 'missing_decision': receipt.pop('decision_id')
    else: receipt['context'][fault] = 'f' * 32
    attempt['condition_gates'] = {gate_id: receipt}
    public = job(rig, accepted['job_id'])
    assert public['execution_condition_gates'][gate_id] is None


def test_current_unknown_reason_does_not_reuse_previous_positive_gate(rig):
    accepted = rig.submit(); rig.ready(); rig.manager.refresh()
    second = rig.submit()
    rig.manager._model['accepting'] = 'true'
    public = job(rig, second['job_id'])
    assert public['waiting_for'] == 'condition_verification'
    assert public['condition_summary']['conditions']['worker_accepting'] == 'unknown'
    rig.complete(); rig.manager.refresh()
    terminal = job(rig, accepted['job_id'])
    assert terminal['execution_observation']['status'] == 'historical'
    assert rig.results == []
    assert job(rig, accepted['job_id']) == terminal


def test_safe_queue_policy_keeps_types_and_does_not_invent_zero():
    value = {'manager_instance_id': 'a'*32, 'mode': 'bounded_backfill', 'bypasses': True,
             'deferrals': 3, 'admitted_at': float('inf'), 'last_deferred_at': 2.0,
             'clock': 'manager_monotonic', 'deferral_to_admission_seconds': None, 'secret': '/private'}
    public = public_queue_policy(value, 'a'*32)
    assert public['deferrals'] == 3 and public['last_deferred_at'] == 2.0
    assert public['deferral_to_admission_seconds'] is None
    assert not {'bypasses', 'admitted_at', 'secret'} & public.keys()
    assert public_queue_policy(value, 'b'*32) is None


def test_cancel_intent_survives_an_old_model_cancel_flag(rig):
    accepted = rig.submit(); rig.ready(); rig.manager.refresh()
    rig.manager._model['attempts'][0]['cancel_requested'] = False
    result = rig.manager.cancel(accepted['job_id'])
    assert result['cancellation_requested'] is True
    assert result['status'] == 'cancel_requested'
    assert not result['cleanup_complete']


def test_panel_default_log_contains_one_bound_active_attempt(rig, controllers):
    accepted = rig.submit(); rig.ready(); rig.manager.refresh()
    rig.manager._model['attempts'][0].update(phase='gpu_wait', stage_completions={'prepare': True},
        condition_gates={}, api_key='private-state')
    controller = controllers(scope={'startup_host': SimpleNamespace(snapshot=rig.manager.snapshot)})
    snapshot = controller.snapshot()
    execution = snapshot['state']['execution_summary']
    assert execution['scope'] == 'first_live_attempt' and execution['job_id'] == accepted['job_id']
    assert execution['active_phase']['phase'] == 'gpu_wait'
    assert execution['stage_completions']['prepare'] is True
    assert '首个执行中任务' in snapshot['logs']['text']
    assert 'private-state' not in json.dumps(snapshot)


def test_terminal_condition_evidence_recovers_without_becoming_live(tmp_path):
    first = Rig(tmp_path).start()
    accepted = first.submit(); first.ready()
    gate_id = 'model.stage.gpu'
    receipt = evaluate_gate(gate_id, PUBLIC_GATE_CONDITIONS[gate_id],
        {**dict.fromkeys(PUBLIC_GATE_CONDITIONS[gate_id], True), 'predecessor_completed': None},
        {'instance_id': first.manager.model_instance_id, 'job_id': accepted['job_id'], 'attempt_id': first.active['attempt_id']})
    first.complete(status='failed', condition_gates={gate_id: receipt}, stage_completions={'prepare': True, 'gpu': None})
    first.manager.refresh()
    assert first.results == []
    first.manager.close()
    recovered = Rig(tmp_path).start()
    try:
        public = job(recovered, accepted['job_id'])
        assert public['status'] == 'failed' and public['execution_observation']['status'] == 'historical'
        assert public['execution_condition_gates'][gate_id]['decision_id'] == receipt['decision_id']
        assert public['stage_completions']['gpu'] is None
        assert public['active_phase'] is None and not recovered.executions
    finally:
        recovered.manager.close()


@pytest.mark.parametrize('fault', ['job_id', 'attempt_id', 'instance_id', 'missing_identity', 'future'])
def test_terminal_http_receipt_cannot_borrow_another_attempt(rig, monkeypatch, fault):
    accepted = rig.submit(); rig.ready()
    gate_id = 'model.stage.gpu'
    receipt = evaluate_gate(gate_id, PUBLIC_GATE_CONDITIONS[gate_id], dict.fromkeys(PUBLIC_GATE_CONDITIONS[gate_id], True),
        {'instance_id': rig.manager.model_instance_id, 'job_id': accepted['job_id'], 'attempt_id': rig.active['attempt_id']})
    if fault == 'missing_identity': receipt['context'].pop('job_id')
    elif fault == 'future': receipt['observed_at'] += 3600
    else: receipt['context'][fault] = 'f'*32
    rig.complete(condition_gates={gate_id: receipt})
    rig.manager.refresh()
    assert rig.results == []
    monkeypatch.setenv('H3_INSTANCE_ID', rig.manager.instance_id)
    monkeypatch.delenv('H3_PARENT_PID', raising=False)
    with TestClient(server.create_app(controller=rig.manager.dispatch, api_key=KEY, output_dir=rig.manager.root)) as client:
        response = client.get('/v1/jobs/'+accepted['job_id'], headers=HEADERS)
        assert response.status_code == 200
        assert response.json()['execution_condition_gates'][gate_id] is None


@pytest.mark.parametrize('field,value', [('schema_version', '99.0'), ('gate_version', '99.0'),
    ('schema_version', None), ('gate_version', None)])
def test_unknown_protocol_version_cannot_publish_permission(field, value):
    gate = 'model.stage.gpu'
    receipt = evaluate_gate(gate, PUBLIC_GATE_CONDITIONS[gate], dict.fromkeys(PUBLIC_GATE_CONDITIONS[gate], True), {})
    receipt[field] = value
    public = server.public_gate_receipt(receipt, gate)
    assert public['allowed'] is None
    assert set(public['conditions'].values()) == {'true'}


@pytest.mark.parametrize('bad', [{}, [], True, 1])
def test_malformed_summary_enums_are_unknown_without_crashing(bad):
    assert public_execution_observation({'status': bad})['status'] == 'unknown'
    assert public_condition_summary({'status': bad})['status'] == 'unknown'
    assert 'mode' not in public_queue_policy({'manager_instance_id': 'a'*32, 'mode': bad}, 'a'*32)
    assert public_execution_summary({'instance_id': 'a'*32, 'active_phase': {'phase': bad}}, 'a'*32)['active_phase'] is None
