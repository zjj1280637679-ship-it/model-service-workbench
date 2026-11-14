"""Foreground delivery is observation, not a second lifecycle scheduler."""
import json
import threading
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import notebook_console as console
from test_notebook_console import controllers, bootstrap_fixture, fake_colab_display


def controller_with_host(controllers, states):
    owner = SimpleNamespace(instance_id='current-instance', public=True, snapshot=Mock(side_effect=states),
                            api_key='', _control_key='', stop=Mock())
    controller = controllers(scope={'startup_host': owner})
    controller.operations['delivery'] = {'request_id': 'delivery', 'action': 'bootstrap', 'status': 'running'}
    controller._bind_bootstrap_host('delivery', owner)
    return controller, owner


def ready_state(**changes):
    return {'instance_id': 'current-instance', 'phase': 'preparing', 'observation_stale': False,
            'api_ready': True, 'public_api_ready': True, 'model_ready': False, 'ready': False,
            'observed_at': 1234, **changes}


def test_api_delivery_requires_matching_fresh_public_observation_but_not_model(controllers):
    controller, owner = controller_with_host(controllers, [
        ready_state(instance_id='old-instance'), ready_state(observation_stale=True),
        ready_state(public_api_ready=False), ready_state(),
    ])
    controller.begin_startup_delivery('delivery', 2)
    result = controller.wait_startup_delivery('delivery')
    assert owner.snapshot.call_count == 4
    assert result['status'] == 'api_ready' and result['active'] is False
    assert result['api_ready'] is True and result['model_ready'] is False
    assert controller.scope['startup_host'] is owner
    assert controller.operations['delivery']['status'] == 'running'
    owner.stop.assert_not_called()
    events = [json.loads(line) for line in controller.audit_path.read_text(encoding='utf-8').splitlines()]
    assert [item['type'] for item in events] == ['startup.host_bound', 'startup.delivery_started', 'startup.delivery_finished']
    assert events[-1]['status'] == 'api_ready'


def test_local_delivery_does_not_invent_public_dependency(controllers):
    controller, owner = controller_with_host(controllers, [ready_state(public_api_ready=False)])
    owner.public = False
    controller.begin_startup_delivery('delivery', 1)
    assert controller.wait_startup_delivery('delivery')['status'] == 'api_ready'


@pytest.mark.parametrize('state', [ready_state(api_ready=False), ready_state(observation_stale=True),
                                  ready_state(instance_id='old-instance')])
def test_expiry_retains_owner_and_running_job_without_claiming_failure(controllers, state):
    controller, owner = controller_with_host(controllers, lambda: dict(state))
    controller.begin_startup_delivery('delivery', .02)
    result = controller.wait_startup_delivery('delivery')
    assert result['status'] == 'observation_expired' and result['active'] is False
    assert result['cleanup_complete'] is None
    assert controller.scope['startup_host'] is owner
    assert controller.operations['delivery']['status'] == 'running'
    owner.stop.assert_not_called()


@pytest.mark.parametrize('outcome', ['cancelled', 'failed', 'observation_failed'])
def test_failure_and_cancel_return_controls_without_discarding_owner(controllers, outcome):
    controller, owner = controller_with_host(controllers, [ready_state(api_ready=False)])
    if outcome == 'cancelled':
        controller.cancel_startup.set()
    elif outcome == 'failed':
        controller.operations['delivery']['status'] = 'failed'
    else:
        owner.snapshot.side_effect = OSError('observation unavailable')
    controller.begin_startup_delivery('delivery', 2)
    result = controller.wait_startup_delivery('delivery')
    assert result['status'] == outcome and result['active'] is False
    assert controller.scope['startup_host'] is owner
    owner.stop.assert_not_called()


def test_keyboard_interrupt_joins_real_bootstrap_then_stops_owned_host(tmp_path, controllers, monkeypatch):
    controller, owner, state, steps, observed = bootstrap_fixture(tmp_path, controllers, monkeypatch)
    state.update(api_ready=False, public_api_ready=False, observation_stale=False)
    reply = controller.dispatch('bootstrap', {'request_id': 'interrupt-delivery', 'public': True})
    assert reply['accepted'] and observed.wait(2)
    ordinary = owner.snapshot.side_effect
    interrupted = False
    def snapshot():
        nonlocal interrupted
        if threading.current_thread() is threading.main_thread() and not interrupted:
            interrupted = True
            raise KeyboardInterrupt()
        return ordinary()
    owner.snapshot.side_effect = snapshot
    controller.begin_startup_delivery('interrupt-delivery', 2)
    with pytest.raises(KeyboardInterrupt):
        controller.wait_startup_delivery('interrupt-delivery')
    controller.executor.finish()
    assert controller.startup_delivery['active'] is False
    assert controller.startup_delivery['status'] == 'interrupted'
    assert controller.startup_delivery['cleanup_complete'] is True
    assert controller.operations['interrupt-delivery']['status'] == 'cancelled'
    assert not controller.active
    owner.stop.assert_called_once()
    assert controller.scope['startup_host'] is owner
    assert state['phase'] == 'stopped' and state['cleanup_complete'] is True


def test_interrupt_does_not_claim_cleanup_when_stop_fails(controllers, monkeypatch):
    controller, owner = controller_with_host(controllers, KeyboardInterrupt())
    monkeypatch.setattr(controller, '_operate', Mock(side_effect=RuntimeError('owned process exit unknown')))
    controller.begin_startup_delivery('delivery', 2)
    with pytest.raises(KeyboardInterrupt):
        controller.wait_startup_delivery('delivery')
    result = controller.startup_delivery
    assert result['status'] == 'interrupted_cleanup_unconfirmed' and result['active'] is False
    assert result['cleanup_complete'] is False and controller.scope['startup_host'] is owner


@pytest.mark.parametrize('timeout', [True, 0, -1, float('inf'), float('nan'), 601, '180'])
def test_foreground_observation_must_be_finite_and_bounded(controllers, timeout):
    controller = controllers()
    with pytest.raises(ValueError):
        controller.begin_startup_delivery('delivery', timeout)


def test_mount_publishes_delivery_state_and_returns_before_model_ready(tmp_path, monkeypatch):
    callbacks, shown = fake_colab_display(monkeypatch)
    (tmp_path / 'console.html').write_text('__CALLBACK_NAME__ __INITIAL_STATE__', encoding='utf-8')
    state = ready_state()
    owner = SimpleNamespace(instance_id='current-instance', public=True, api_key='', _control_key='',
                            snapshot=lambda: dict(state))
    def bootstrap(controller, action, payload):
        assert shown  # Full panel is displayed before any worker starts.
        controller.scope['startup_host'] = owner
        controller._bind_bootstrap_host(payload['request_id'], owner)
        controller.cancel_startup.wait(5)  # The independent model branch remains active.
    monkeypatch.setattr(console.ConsoleController, '_operate', bootstrap)
    scope = {}
    controller = console.mount_console(tmp_path, {}, scope, auto_start=True, foreground_bootstrap_timeout=2)
    try:
        assert len(shown) >= 2
        assert '"active": true' in shown[0] and '"status": "observing"' in shown[0]
        assert '"status": "api_ready"' in shown[-1] and '"active": false' in shown[-1]
        assert controller.startup_delivery['model_ready'] is False
        assert controller.active
        callback = callbacks[next(iter(callbacks))]
        assert callback('snapshot')['ui']['startup_delivery']['active'] is False
    finally:
        controller.cancel_startup.set()
        for finished in controller.finished.values():
            assert finished.wait(2)
        controller.close()


@pytest.mark.parametrize('old_phase', ['stopped', 'failed'])
def test_warm_bootstrap_delivery_ignores_previous_terminal_host_until_request_binding(
        tmp_path, controllers, monkeypatch, old_phase):
    controller, owner, state, steps, observed = bootstrap_fixture(tmp_path, controllers, monkeypatch)
    state.update(observation_stale=False)
    worker_entered, worker_release, observer_published, observer_done = [threading.Event() for _ in range(4)]
    old_reads = []
    old_state = dict(instance_id='previous-instance', phase=old_phase, cleanup_complete=True,
        observation_stale=False, manager_alive=False, server_alive=False, model_alive=False, tunnel_alive=False,
        api_ready=False, public_api_ready=False, model_ready=False)
    def old_snapshot():
        old_reads.append(threading.current_thread().name)
        if threading.current_thread().name.startswith('h3-console'):
            worker_entered.set()
            assert worker_release.wait(3), 'test did not release previous owner validation'
        return dict(old_state)
    previous = SimpleNamespace(instance_id='previous-instance', public=True, staged=False,
                               api_key='', _control_key='', snapshot=old_snapshot)
    controller.scope['startup_host'] = previous
    controller.begin_startup_delivery('warm-delivery', 2)
    receipt = controller.dispatch('bootstrap', {'request_id': 'warm-delivery', 'public': True})
    assert receipt['accepted'] and worker_entered.wait(1)
    result = {}
    def observe():
        result.update(controller.wait_startup_delivery('warm-delivery', publish=observer_published.set))
        observer_done.set()
    observer = threading.Thread(target=observe, name='foreground-delivery-test')
    observer.start()
    try:
        assert observer_published.wait(1)
        assert not observer_done.is_set()
        assert controller.startup_delivery['active'] is True
        assert controller.startup_delivery['instance_id'] is None
        assert controller.scope['startup_host'] is previous  # Retain old ownership until checked.
        assert 'foreground-delivery-test' not in old_reads
        worker_release.set()
        assert observer_done.wait(1)
        assert result['status'] == 'api_ready'
        assert result['instance_id'] == owner.instance_id
        assert controller.operations['warm-delivery']['instance_id'] == owner.instance_id
        assert result['model_ready'] is False  # Delivery still does not wait for the model.
        assert controller.operations['warm-delivery']['status'] == 'running'
    finally:
        worker_release.set()
        state.update(phase='ready', model_ready=True, ready=True)
        observer.join(timeout=3)
        controller.executor.finish()


@pytest.mark.parametrize('window', ['display', 'dispatch'])
def test_interrupt_before_wait_observer_starts_also_reclaims_ownership(tmp_path, monkeypatch, window):
    fake_colab_display(monkeypatch)
    (tmp_path / 'console.html').write_text('__CALLBACK_NAME__ __INITIAL_STATE__', encoding='utf-8')
    if window == 'display':
        monkeypatch.setattr(sys.modules['IPython.display'], 'display', Mock(side_effect=KeyboardInterrupt()))
    else:
        monkeypatch.setattr(console.ConsoleController, 'dispatch', Mock(side_effect=KeyboardInterrupt()))
    cleanup = Mock()
    monkeypatch.setattr(console.ConsoleController, '_operate', cleanup)
    scope = {}
    with pytest.raises(KeyboardInterrupt):
        console.mount_console(tmp_path, {}, scope, auto_start=True, foreground_bootstrap_timeout=2)
    owner = scope['h3_console']
    try:
        cleanup.assert_called_once_with('stop', {})
        assert owner.startup_delivery['active'] is False
        assert owner.startup_delivery['status'] == 'interrupted'
        assert owner.startup_delivery['cleanup_complete'] is True
    finally:
        owner.close()
