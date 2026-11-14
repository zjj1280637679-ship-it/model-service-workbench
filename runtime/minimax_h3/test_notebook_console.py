"""Exercise the owned console bridge without Colab, network, model imports or GPU."""
from concurrent.futures import ThreadPoolExecutor as RealExecutor
import base64
import json
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

import notebook_console as console
import notebook_ui as ui


class TrackedExecutor(RealExecutor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.futures = []

    def submit(self, *args, **kwargs):
        future = super().submit(*args, **kwargs)
        self.futures.append(future)
        return future

    def finish(self):
        for future in list(self.futures):
            future.result(timeout=5)


@pytest.fixture
def controllers(tmp_path, monkeypatch):
    monkeypatch.setattr(console, 'ThreadPoolExecutor', TrackedExecutor)
    ui._known_secrets = ()
    ui._diagnostics.clear()
    created = []

    def create(programs=None, scope=None):
        controller = console.ConsoleController(tmp_path / str(len(created)), programs or {}, scope)
        created.append(controller)
        return controller

    yield create
    for controller in created:
        controller.executor.finish()
        if not controller.closed:
            controller.close()
    ui._known_secrets = ()
    ui._diagnostics.clear()


def host(tmp_path):
    value = SimpleNamespace(api_key='public-secret-UNIQUE-123', _control_key='private-secret-UNIQUE-456',
                            instance_id='instance-1')
    value.snapshot = Mock(return_value={
        'instance_id': value.instance_id, 'phase': 'preparing', 'api_ready': True,
        'model_ready': False, 'queue': {'queued': 2, 'running': 0, 'admission_open': True},
        'error': 'diagnostic ' + value.api_key + ' ' + value._control_key,
        'api_key': value.api_key, 'private_control_key': value._control_key,
    })
    value.control = Mock()
    value.wait_ready = Mock()
    value.start = Mock()
    value.stop = Mock()
    value.events = Mock(return_value=[{'note': 'Bearer ' + value._control_key}])
    for name in ('server_log', 'api_log', 'manager_log', 'tunnel_log'):
        path = tmp_path / (name + '.log')
        path.write_text('opaque ' + value.api_key + '\nopaque ' + value._control_key, encoding='utf-8')
        setattr(value, name, path)
    return value


def test_stop_lane_runs_while_generation_is_blocked(controllers, monkeypatch):
    controller = controllers()
    generating, release, stopping = threading.Event(), threading.Event(), threading.Event()

    def operation(action, payload):
        if action == 'generate':
            generating.set()
            assert release.wait(5), 'test did not release generation'
        elif action == 'stop':
            stopping.set()

    monkeypatch.setattr(controller, '_operate', operation)
    try:
        assert controller.dispatch('generate', {'request_id': 'generation'})['accepted']
        assert generating.wait(2)
        assert controller.dispatch('stop', {'request_id': 'stop'})['accepted']
        assert stopping.wait(2), 'stop was queued behind long generation'
        assert not release.is_set()
    finally:
        release.set()
    controller.executor.finish()
    assert not controller.active
    assert controller.operations['generation']['status'] == 'succeeded'
    assert controller.operations['stop']['status'] == 'succeeded'


def test_duplicate_request_reuses_one_execution_and_rejects_changed_content(controllers, monkeypatch):
    controller = controllers()
    started, release = threading.Event(), threading.Event()
    calls = []

    def operation(action, payload):
        calls.append((action, payload))
        started.set()
        assert release.wait(5)

    monkeypatch.setattr(controller, '_operate', operation)
    payload = {'request_id': 'same-id', 'prompt': 'new content'}
    try:
        first = controller.dispatch('generate', payload)
        assert started.wait(2)
        repeated = controller.dispatch('generate', dict(payload))
        conflict = controller.dispatch('generate', dict(payload, prompt='changed content'))
        assert first['accepted'] and repeated['accepted']
        assert first['operation']['request_id'] == repeated['operation']['request_id']
        assert '内容发生变化' in conflict['error']
        assert len(calls) == 1
    finally:
        release.set()
    controller.executor.finish()
    controller.dispatch('generate', payload)
    assert len(calls) == 1


@pytest.mark.parametrize('source', ['diagnostics', 'model', 'events'])
def test_snapshot_only_observes_and_redacts_both_credentials(tmp_path, controllers, source):
    owner = host(tmp_path)
    controller = controllers(scope={'startup_host': owner})
    controller._print('message', owner.api_key, owner._control_key)
    reply = controller.dispatch('snapshot', {'source': source})
    serialized = json.dumps(reply, ensure_ascii=False)
    assert owner.snapshot.call_count == 1
    assert owner.api_key not in serialized and owner._control_key not in serialized
    assert reply['state']['api_ready'] is True and reply['state']['model_ready'] is False
    for action in (owner.control, owner.wait_ready, owner.start, owner.stop):
        action.assert_not_called()


def test_snapshot_nested_fields_cannot_leak_known_credentials(tmp_path, controllers):
    owner = host(tmp_path)
    owner.snapshot.return_value['queue']['future_diagnostic'] = owner._control_key
    controller = controllers(scope={'startup_host': owner,
        'RESOURCE_REPORT': {'gpu': 'test ' + owner.api_key, 'available_ram_gib': 1}})
    reply = json.dumps(controller.dispatch('snapshot'), ensure_ascii=False)
    assert owner.api_key not in reply and owner._control_key not in reply


def test_worker_error_is_recorded_without_escaping_to_notebook(tmp_path, controllers, monkeypatch, capsys):
    owner = host(tmp_path)
    controller = controllers(scope={'startup_host': owner})

    def fail(action, payload):
        raise RuntimeError('meaningful failure ' + owner.api_key + ' ' + owner._control_key)

    monkeypatch.setattr(controller, '_operate', fail)
    assert controller.dispatch('install', {'request_id': 'failed-install'})['accepted']
    controller.executor.finish()
    reply = controller.dispatch('snapshot')
    assert controller.operations['failed-install']['status'] == 'failed'
    assert reply['notice']['kind'] == 'error'
    assert 'meaningful failure' in reply['notice']['message']
    assert not controller.active
    log = controller.log_path.read_text(encoding='utf-8')
    assert 'Traceback' in log and 'meaningful failure' in log
    assert owner.api_key not in log and owner._control_key not in log
    assert capsys.readouterr().out == ''


def test_callback_validation_failure_is_an_inline_notice(controllers):
    controller = controllers()
    for action, payload in [('arbitrary_python', {'request_id': 'x'}),
                            ('start', {'request_id': 'x', 'public': 'true'}),
                            ('start', {'request_id': 'x', 'python': 'print(1)'})]:
        reply = controller.dispatch(action, payload)
        assert reply['error'] and reply['notice']['kind'] == 'error'
    assert not controller.operations and not controller.executor.futures


def test_action_code_changes_only_named_defaults_and_removes_nested_downloads():
    download = Mock()
    source = '''START = False
KEEP = "untouched"
if START:
    files.download("secret.json")
    result = KEEP
'''
    scope = {'files': SimpleNamespace(download=download)}
    exec(console.action_code(source, {'START': True}), scope)
    assert scope['START'] is True and scope['result'] == 'untouched'
    download.assert_not_called()
    with pytest.raises(ValueError, match='操作参数'):
        console.action_code(source, {'UNKNOWN': True})


def test_action_code_does_not_modify_nested_local_variables():
    source = '''VALUE = 1
def inner():
    VALUE = 2
    return VALUE
result = inner()
'''
    scope = {}
    exec(console.action_code(source, {'VALUE': 9}), scope)
    assert scope['VALUE'] == 9 and scope['result'] == 2


def test_monitor_stop_terminates_only_generation_process(controllers):
    controller = controllers()
    process = Mock()
    process.poll.return_value = None
    process.wait.return_value = 0
    identity = threading.get_ident()
    controller.active['generate'] = {'request_id': 'monitor', 'thread_id': identity}
    controller.process_started(process)
    try:
        controller._stop_client_monitor()
        process.terminate.assert_called_once_with()
        process.wait.assert_called_once_with(timeout=10)
        process.kill.assert_not_called()
        assert 'client.monitor_stopped' in controller.audit_path.read_text(encoding='utf-8')
    finally:
        controller.process_finished(process)
        controller.active.clear()
    assert not controller.processes


def test_monitor_termination_timeout_escalates_and_records_cleanup(controllers):
    controller = controllers()
    process = Mock()
    process.poll.return_value = None
    process.wait.side_effect = [TimeoutError('still running'), 0]
    controller.active['generate'] = {'request_id': 'monitor', 'thread_id': threading.get_ident()}
    controller.process_started(process)
    try:
        controller._stop_client_monitor()
        process.kill.assert_called_once_with()
        assert process.wait.call_args.kwargs == {'timeout': 5}
    finally:
        controller.process_finished(process)
        controller.active.clear()
    assert not controller.processes


def test_close_rejects_active_owner_then_disables_future_calls(controllers, monkeypatch):
    controller = controllers()
    started, release = threading.Event(), threading.Event()

    def operation(*args):
        started.set()
        assert release.wait(5)

    monkeypatch.setattr(controller, '_operate', operation)
    try:
        controller.dispatch('generate', {'request_id': 'active'})
        assert started.wait(2)
        with pytest.raises(RuntimeError, match='受管操作'):
            controller.close()
    finally:
        release.set()
    controller.executor.finish()
    controller.close()
    assert controller.closed and controller.epoch is None
    assert ui._dashboard is not controller and ui._process_observer is not controller
    assert controller.dispatch('start', {'request_id': 'later'})['ui']['closed'] is True


def test_mount_has_no_operation_and_old_epoch_callback_becomes_inert(tmp_path, monkeypatch):
    callbacks, displayed = {}, []
    google = ModuleType('google')
    colab = ModuleType('google.colab')
    colab.output = SimpleNamespace(register_callback=lambda name, callback: callbacks.__setitem__(name, callback))
    google.colab = colab
    ipython = ModuleType('IPython')
    display = ModuleType('IPython.display')
    display.HTML = display.JSON = lambda value: value
    display.display = lambda value: displayed.append(value)
    ipython.display = display
    for name, value in [('google', google), ('google.colab', colab), ('IPython', ipython), ('IPython.display', display)]:
        monkeypatch.setitem(sys.modules, name, value)
    (tmp_path / 'console.html').write_text('__CALLBACK_NAME__ __INITIAL_STATE__', encoding='utf-8')
    scope = {}
    operation = Mock(side_effect=AssertionError('mount must not operate'))
    monkeypatch.setattr(console.ConsoleController, '_operate', operation)
    controller = console.mount_console(tmp_path, {}, scope)
    try:
        old = callbacks[next(iter(callbacks))]
        same = console.mount_console(tmp_path, {}, scope)
        assert same is controller and len(displayed) == 2
        assert old('start', {'request_id': 'stale-start'})['ui']['closed'] is True
        latest = callbacks[next(reversed(callbacks))]
        assert latest('snapshot', {})['ui']['connected'] is True
        assert not controller.operations
        operation.assert_not_called()
    finally:
        controller.close()


def test_render_escapes_script_terminators_in_initial_json():
    rendered = console.render_console('__CALLBACK_NAME__ __INITIAL_STATE__', 'safe',
                                      {'notice': {'message': '</script><script>unsafe()</script>'}})
    assert '</script>' not in rendered and '<script>' not in rendered
    assert '\\u003c' in rendered


def bootstrap_fixture(tmp_path, controllers, monkeypatch, before_step=None):
    """Run the real bootstrap orchestration with controlled stage implementations."""
    controller = controllers()
    owner = host(tmp_path)
    owner.poll_interval = .01
    owner.startup_timeout = 5
    owner.public, owner.staged = True, False
    state = {'instance_id': owner.instance_id, 'phase': 'preparing', 'api_ready': True,
             'model_ready': False, 'ready': False, 'public_requested': True,
             'public_api_ready': True, 'public_url': 'https://fixture.invalid',
             'manager_alive': True, 'model_alive': True, 'server_alive': True,
             'queue': {'queued': 1, 'running': 0, 'admission_open': True}}
    observed = threading.Event()

    def snapshot():
        observed.set()
        return dict(state)

    owner.snapshot.side_effect = snapshot
    steps = []

    def execute(name, parameters=None):
        steps.append(name)
        if before_step:
            before_step(name, controller)
        if name == 'preflight':
            assert len(controller.scope['BOOTSTRAP_API_KEY']) >= 32
            controller.scope['PREFLIGHT_OK'] = True
        elif name == 'install':
            controller.scope['INSTALL_OK'] = True
        elif name == 'start':
            owner.api_key = controller.scope['BOOTSTRAP_API_KEY']
            controller.scope['startup_host'] = owner
            controller.scope['API_KEY'] = owner.api_key
        elif name == 'stop':
            owner.stop()
            state.update(phase='stopped', ready=False, api_ready=False, model_ready=False,
                         manager_alive=False, model_alive=False, server_alive=False,
                         tunnel_alive=False, cleanup_complete=True)

    monkeypatch.setattr(controller, '_execute', execute)
    return controller, owner, state, steps, observed


def test_bootstrap_automates_stages_but_waits_for_real_model_readiness(tmp_path, controllers, monkeypatch):
    controller, owner, state, steps, observed = bootstrap_fixture(tmp_path, controllers, monkeypatch)
    try:
        reply = controller.dispatch('bootstrap', {'request_id': 'one-click', 'public': True})
        assert reply['accepted']
        assert observed.wait(2)
        assert [step for step in steps if step in {'preflight', 'install', 'start'}] == ['preflight', 'install', 'start']
        assert controller.operations['one-click']['status'] == 'running'
        snapshot = controller.dispatch('snapshot')
        assert snapshot['ui']['bootstrap_running'] is True
        assert snapshot['state']['api_ready'] is True
        assert snapshot['state']['model_ready'] is False
        assert snapshot['ui']['connection_available'] is True
        state.update(model_ready=True, ready=True, phase='ready')
    finally:
        state.update(model_ready=True, ready=True, phase='ready')
    controller.executor.finish()
    assert controller.operations['one-click']['status'] == 'succeeded'
    assert controller.dispatch('snapshot')['ui']['startup_stage'] == 'ready'
    owner.wait_ready.assert_not_called()


@pytest.mark.parametrize('missing', ['api_ready', 'public_api_ready', 'fresh_observation'])
def test_bootstrap_rejects_partial_or_stale_ready_claims(tmp_path, controllers, monkeypatch, missing):
    controller, owner, state, steps, observed = bootstrap_fixture(tmp_path, controllers, monkeypatch)
    state.update(model_ready=True, ready=True, phase='ready')
    state['observation_stale' if missing == 'fresh_observation' else missing] = missing == 'fresh_observation'
    polled_twice = threading.Event()
    reads = []

    def snapshot():
        reads.append(True)
        if len(reads) >= 2:
            polled_twice.set()
        return dict(state)

    owner.snapshot.side_effect = snapshot
    try:
        assert controller.dispatch('bootstrap', {'request_id': 'partial-ready', 'public': True})['accepted']
        assert polled_twice.wait(2), 'partial readiness was accepted instead of continuing observation'
        assert controller.operations['partial-ready']['status'] == 'running'
    finally:
        state.update(api_ready=True, public_api_ready=True, observation_stale=False)
    controller.executor.finish()
    assert controller.operations['partial-ready']['status'] == 'succeeded'


def test_bootstrap_public_default_reaches_start_action_consistently(tmp_path, controllers, monkeypatch):
    controller, owner, state, steps, observed = bootstrap_fixture(tmp_path, controllers, monkeypatch)
    state.update(model_ready=True, ready=True, phase='ready')
    execute = controller._execute
    start_parameters = []

    def recording(name, parameters=None):
        if name == 'start':
            start_parameters.append(parameters)
        return execute(name, parameters)

    monkeypatch.setattr(controller, '_execute', recording)
    assert controller.dispatch('bootstrap', {'request_id': 'default-public'})['accepted']
    controller.executor.finish()
    assert start_parameters == [{'START_GPU_API': True, 'OPEN_PUBLIC_API': True}]
    assert controller.operations['default-public']['status'] == 'succeeded'


def test_repeat_bootstrap_observes_owned_instance_without_reinstall_or_key_rotation(tmp_path, controllers, monkeypatch):
    controller, owner, state, steps, observed = bootstrap_fixture(tmp_path, controllers, monkeypatch)
    state.update(model_ready=True, ready=True, phase='ready')
    payload = {'request_id': 'initial-start', 'public': True}
    assert controller.dispatch('bootstrap', payload)['accepted']
    controller.executor.finish()
    old_steps, old_key = list(steps), owner.api_key
    assert controller.dispatch('bootstrap', dict(payload, request_id='repeated-start'))['accepted']
    controller.executor.finish()
    assert controller.operations['repeated-start']['status'] == 'succeeded'
    assert steps == old_steps and owner.api_key == old_key
    assert controller.scope['BOOTSTRAP_API_KEY'] == old_key


@pytest.mark.parametrize('failure_stage', ['preflight', 'install'])
def test_bootstrap_stops_at_failed_prerequisite(tmp_path, controllers, monkeypatch, failure_stage):
    def fail(name, controller):
        if name == failure_stage:
            raise RuntimeError('fixture prerequisite failed: ' + name)

    controller, owner, state, steps, observed = bootstrap_fixture(tmp_path, controllers, monkeypatch, fail)
    assert controller.dispatch('bootstrap', {'request_id': 'failed-boot'})['accepted']
    controller.executor.finish()
    assert controller.operations['failed-boot']['status'] == 'failed'
    assert 'start' not in steps
    if failure_stage == 'preflight':
        assert 'install' not in steps
    assert not controller.active
    snapshot = controller.dispatch('snapshot')
    assert snapshot['ui']['startup_stage'] == 'failed'
    assert snapshot['notice']['kind'] == 'error'
    assert 'fixture prerequisite failed' in snapshot['notice']['message']
    owner.start.assert_not_called()


def test_bootstrap_model_failure_keeps_api_observable_and_preserves_error(tmp_path, controllers, monkeypatch):
    controller, owner, state, steps, observed = bootstrap_fixture(tmp_path, controllers, monkeypatch)
    state.update(phase='degraded', model_alive=False, error='fixture model initialization failed')
    assert controller.dispatch('bootstrap', {'request_id': 'model-failed', 'public': True})['accepted']
    controller.executor.finish()
    assert controller.operations['model-failed']['status'] == 'failed'
    snapshot = controller.dispatch('snapshot')
    assert snapshot['state']['api_ready'] is True
    assert snapshot['state']['model_ready'] is False
    assert snapshot['notice']['kind'] == 'error'
    assert 'fixture model initialization failed' in snapshot['notice']['message']
    assert controller.scope['startup_host'] is owner
    owner.stop.assert_not_called()
    assert controller.dispatch('control', {'request_id': 'inspect-after-failure', 'action': 'drain'})['accepted']
    controller.executor.finish()
    owner.control.assert_called_once_with('drain', force=False)


def test_bootstrap_credentials_exist_before_setup_without_claiming_api_ready(tmp_path, controllers, monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def pause(name, controller):
        if name == 'preflight':
            controller._print('fixture secret diagnostic', controller.scope['BOOTSTRAP_API_KEY'])
            entered.set()
            assert release.wait(4)
            raise RuntimeError('fixture ends before installation')

    controller, owner, state, steps, observed = bootstrap_fixture(tmp_path, controllers, monkeypatch, pause)
    try:
        assert controller.dispatch('bootstrap', {'request_id': 'early-key', 'public': True})['accepted']
        assert entered.wait(2)
        key = controller.scope['BOOTSTRAP_API_KEY']
        snapshot = controller.dispatch('snapshot')
        assert snapshot['ui']['connection_available'] is True
        assert key not in json.dumps(snapshot)
        downloaded = controller.dispatch('export_connection')
        connection = json.loads(base64.b64decode(downloaded['download']['data_base64']))
        assert connection['api_key'] == key
        assert connection['url'] is None
        assert connection['api_ready'] is False
        assert connection['model_ready'] is False
        assert key not in controller.log_path.read_text(encoding='utf-8')
        assert key not in controller.audit_path.read_text(encoding='utf-8')
        assert 'install' not in steps and 'start' not in steps
    finally:
        release.set()
    controller.executor.finish()


def test_bootstrap_model_wait_does_not_block_api_generation_or_control(tmp_path, controllers, monkeypatch):
    generated = threading.Event()

    def signal(name, controller):
        if name == 'generate':
            generated.set()

    controller, owner, state, steps, observed = bootstrap_fixture(tmp_path, controllers, monkeypatch, signal)
    try:
        assert controller.dispatch('bootstrap', {'request_id': 'wait-for-model', 'public': True})['accepted']
        assert observed.wait(2)
        assert controller.dispatch('generate', {'request_id': 'queued-generate', 'endpoint': 'public'})['accepted']
        assert generated.wait(2)
        assert controller.dispatch('control', {'request_id': 'pause-admission', 'action': 'drain'})['accepted']
        # Both actions have independent workers; bootstrap still awaits readiness.
        for future in list(controller.executor.futures)[1:]:
            future.result(timeout=2)
        owner.control.assert_called_once_with('drain', force=False)
        assert controller.operations['wait-for-model']['status'] == 'running'
    finally:
        state.update(model_ready=True, ready=True, phase='ready')
    controller.executor.finish()


@pytest.mark.parametrize('register_process_late', [False, True])
def test_bootstrap_stop_cancels_install_before_any_late_service_start(tmp_path, controllers, monkeypatch,
                                                                   register_process_late):
    entered, released = threading.Event(), threading.Event()
    process = Mock()
    process.poll.return_value = None
    process.terminate.side_effect = released.set
    process.wait.return_value = 0

    def install(name, controller):
        if name == 'install':
            if register_process_late:
                entered.set()
                assert controller.cancel_startup.wait(2), 'stop did not reach cancellation barrier'
            controller.process_started(process)
            entered.set()
            try:
                assert released.wait(4), 'stop did not cancel the owned installer'
            finally:
                controller.process_finished(process)

    controller, owner, state, steps, observed = bootstrap_fixture(tmp_path, controllers, monkeypatch, install)
    try:
        assert controller.dispatch('bootstrap', {'request_id': 'cancel-startup'})['accepted']
        assert entered.wait(2)
        assert controller.dispatch('stop', {'request_id': 'stop-startup'})['accepted']
        controller.executor.finish()
        assert 'start' not in steps
        assert controller.operations['cancel-startup']['status'] == 'cancelled'
        assert controller.operations['stop-startup']['status'] == 'succeeded'
        process.terminate.assert_called_once_with()
        assert not controller.active
        finished = [json.loads(line) for line in controller.audit_path.read_text(encoding='utf-8').splitlines()
                    if json.loads(line)['type'] == 'operation.finished']
        assert [event['action'] for event in finished] == ['bootstrap', 'stop']
    finally:
        released.set()


def test_auto_start_mount_bootstraps_only_once_and_never_from_saved_html(tmp_path, monkeypatch):
    callbacks, shown = {}, []
    google, colab = ModuleType('google'), ModuleType('google.colab')
    colab.output = SimpleNamespace(register_callback=lambda name, callback: callbacks.__setitem__(name, callback))
    google.colab = colab
    ipython, display = ModuleType('IPython'), ModuleType('IPython.display')
    display.HTML = display.JSON = lambda value: value
    display.display = shown.append
    ipython.display = display
    for name, value in [('google', google), ('google.colab', colab), ('IPython', ipython), ('IPython.display', display)]:
        monkeypatch.setitem(sys.modules, name, value)
    (tmp_path / 'console.html').write_text('__CALLBACK_NAME__ __INITIAL_STATE__', encoding='utf-8')
    submit = Mock(return_value={'request_id': 'auto', 'status': 'accepted'})
    monkeypatch.setattr(console.ConsoleController, 'submit', submit)
    # Rendering saved output has no transport or active startup side effect.
    offline = console.render_console('__CALLBACK_NAME__ __INITIAL_STATE__')
    assert '"offline": true' in offline
    submit.assert_not_called()
    scope = {}
    controller = console.mount_console(tmp_path, {}, scope, auto_start=True, bootstrap_payload={'public': True})
    try:
        same = console.mount_console(tmp_path, {}, scope, auto_start=True, bootstrap_payload={'public': True})
        assert same is controller and len(shown) == 2
        assert submit.call_count == 1
        action, payload = submit.call_args.args
        assert action == 'bootstrap' and payload['public'] is True
        assert payload['request_id']
    finally:
        controller.close()


def fake_colab_display(monkeypatch):
    callbacks, shown = {}, []
    google, colab = ModuleType('google'), ModuleType('google.colab')
    colab.output = SimpleNamespace(register_callback=lambda name, callback: callbacks.__setitem__(name, callback))
    google.colab = colab
    ipython, display = ModuleType('IPython'), ModuleType('IPython.display')
    display.HTML = display.JSON = lambda value: value

    def capture(value, **kwargs):
        shown.append(value)
        return SimpleNamespace(update=shown.append)

    display.display = capture
    ipython.display = display
    for name, value in [('google', google), ('google.colab', colab), ('IPython', ipython), ('IPython.display', display)]:
        monkeypatch.setitem(sys.modules, name, value)
    return callbacks, shown


def test_mount_returns_with_full_controls_while_api_unready_and_stop_remains_callable(tmp_path, monkeypatch):
    callbacks, shown = fake_colab_display(monkeypatch)
    monkeypatch.setattr(console, 'ThreadPoolExecutor', TrackedExecutor)
    template = Path(console.__file__).with_name('console.html').read_text(encoding='utf-8')
    (tmp_path / 'console.html').write_text(template, encoding='utf-8')
    owner = host(tmp_path)
    owner.public, owner.staged = True, False
    owner.poll_interval, owner.startup_timeout = .01, 30
    state = {'instance_id': owner.instance_id, 'phase': 'preparing', 'api_ready': False,
             'public_requested': True, 'public_api_ready': False, 'model_ready': False,
             'ready': False, 'manager_alive': True, 'server_alive': True, 'model_alive': True}
    awaiting_api_logged = threading.Event()
    original_startup_stage = console.ConsoleController._startup_stage

    def observe():
        return dict(state)

    owner.snapshot.side_effect = observe

    def record_startup_stage(controller, stage):
        result = original_startup_stage(controller, stage)
        if stage == 'awaiting_api':
            # A snapshot starts the observation; the stage transition publishes
            # its log afterwards. Wait for that effect before inspecting logs.
            awaiting_api_logged.set()
        return result

    monkeypatch.setattr(console.ConsoleController, '_startup_stage', record_startup_stage)

    def execute(controller, name, parameters=None):
        # The controls must already exist before any potentially slow startup step.
        assert shown and 'id="h3-console"' in shown[0]
        if name == 'preflight':
            controller.scope['PREFLIGHT_OK'] = True
        elif name == 'install':
            controller.scope['INSTALL_OK'] = True
        elif name == 'start':
            owner.api_key = controller.scope['BOOTSTRAP_API_KEY']
            controller.scope['startup_host'] = owner
        elif name == 'stop':
            owner.stop()
            state.update(phase='stopped', manager_alive=False, server_alive=False, model_alive=False,
                         tunnel_alive=False, cleanup_complete=True)

    monkeypatch.setattr(console.ConsoleController, '_execute', execute)
    scope = {}
    with RealExecutor(max_workers=1) as cell:
        mounted = cell.submit(console.mount_console, tmp_path, {}, scope,
                              auto_start=True)
        try:
            controller = mounted.result(timeout=2)
            assert awaiting_api_logged.wait(2)
            assert len(shown) == 1
            for component in ('id="board"', 'id="logs-heading"', 'data-action="stop"'):
                assert component in shown[0]
            request_id = next(iter(controller.operations))
            assert not controller.finished[request_id].is_set()
            callback = callbacks[next(iter(callbacks))]
            snapshot = callback('snapshot')
            assert snapshot['ui']['bootstrap_running'] is True
            assert snapshot['state']['api_ready'] is False
            assert snapshot['state']['model_ready'] is False
            assert '等待 API' in snapshot['logs']['text']
            assert callback('stop', {'request_id': 'stop-before-api'})['accepted']
            controller.executor.finish()
            assert controller.operations[request_id]['status'] == 'cancelled'
            assert controller.operations['stop-before-api']['status'] == 'succeeded'
            owner.stop.assert_called_once()
        finally:
            controller = scope.get('h3_console')
            if controller:
                controller.cancel_startup.set()
                controller.executor.finish()
                controller.close()
            mounted.result(timeout=4)


def test_mount_does_not_wait_for_existing_host_observation(tmp_path, controllers, monkeypatch):
    callbacks, shown = fake_colab_display(monkeypatch)
    owner = host(tmp_path)
    controller = controllers(scope={'startup_host': owner})
    controller.scope['h3_console'] = controller
    (controller.directory / 'console.html').write_text('__CALLBACK_NAME__ __INITIAL_STATE__', encoding='utf-8')
    console.mount_console(controller.directory, {}, controller.scope)
    owner.snapshot.assert_not_called()
    assert '"observation_stale": true' in shown[0]
    assert '"model_ready": true' not in shown[0] and '"api_ready": true' not in shown[0]
    # Actual observations still occur when the returned panel invokes its callback.
    callback = callbacks[next(iter(callbacks))]
    assert callback('snapshot')['state']['api_ready'] is True
    owner.snapshot.assert_called_once()
