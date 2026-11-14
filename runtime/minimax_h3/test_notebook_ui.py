"""UI boundary checks: read-only observation, isolated logs, and credential-safe text."""
import json
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

import notebook_ui as ui


@pytest.fixture(autouse=True)
def reset_ui(monkeypatch):
    ui._dashboard = None
    ui._known_secrets = ()
    ui._diagnostics.clear()
    handle = Mock()
    display = Mock(return_value=handle)
    module = ModuleType('IPython.display')
    module.HTML = lambda text: text
    module.display = display
    monkeypatch.setitem(sys.modules, 'IPython', ModuleType('IPython'))
    monkeypatch.setitem(sys.modules, 'IPython.display', module)
    yield display, handle
    ui._dashboard = None
    ui._known_secrets = ()
    ui._diagnostics.clear()


def live_state(**overrides):
    return dict(phase='preparing', manager_alive=True, server_alive=True, api_ready=True,
                model_alive=True, model_ready=False, public_requested=True, public_api_ready=True,
                tunnel_connected=True, ready=False, observed_at=1794617000, observation_stale=False,
                queue={'queued': 2, 'running': 0, 'admission_open': True}, **overrides)


def test_dashboard_refresh_reads_once_and_never_controls(reset_ui):
    display, handle = reset_ui
    host = SimpleNamespace(snapshot=Mock(return_value=live_state()), api_key='unique-key-123',
                           wait_ready=Mock(), control=Mock(), start=Mock(), stop=Mock())
    board = ui.mount_dashboard(lambda: {'host': host})
    assert host.snapshot.call_count == 1
    assert display.call_count == 1
    assert display.call_args.kwargs['display_id'] == 'h3-dashboard'
    board.refresh()
    assert host.snapshot.call_count == 2
    assert display.call_count == 1 and handle.update.call_count == 1
    for action in (host.wait_ready, host.control, host.start, host.stop):
        action.assert_not_called()


def test_status_updates_only_mounted_dashboard_not_control_cell(reset_ui):
    display, handle = reset_ui
    ui.mount_dashboard(lambda: {'state': live_state()})
    card = ui.StatusCard('安装', badge='执行中', details='raw-log-not-for-dashboard')
    card.update('安装完成', badge='完成', details='separate-diagnostic')
    assert display.call_count == 1 and handle.update.call_count == 2
    output = handle.update.call_args.args[0]
    assert '安装完成' in output
    assert 'raw-log-not-for-dashboard' not in output and 'separate-diagnostic' not in output
    assert 'raw-log-not-for-dashboard' in ui.last_diagnostics()


def test_live_and_ready_are_separate_and_public_does_not_imply_generation():
    output = ui.render_dashboard({'state': live_state()})
    assert 'API 可用 · 生成尚未就绪' in output
    assert 'API 探测通过' in output
    model = output.split('大模型容器', 1)[1].split('</div></div>', 1)[0]
    assert '准备中' in model and '能力未就绪' in model
    assert '可以生成' not in output


def test_stale_observations_do_not_display_old_ready_as_current():
    state = live_state()
    state.update(ready=True, model_ready=True, observation_stale=True)
    output = ui.render_dashboard({'state': state})
    assert '状态需要刷新' in output and '待刷新' in output
    assert '可以生成' not in output and 'API 探测通过' not in output


def test_pure_render_never_reads_host_and_escapes_markup():
    host = Mock()
    output = ui.render_dashboard({'host': host, 'resource_report': {'gpu': '<script>alert(1)</script>'}},
        {'title': '<img src=x onerror=x>', 'note': '&unsafe', 'badge': '"bad"', 'fields': []})
    assert not host.mock_calls
    assert '<script>' not in output and '<img src=x' not in output
    assert '&lt;script&gt;' in output and '&amp;unsafe' in output


@pytest.mark.parametrize('secret_text', [
    'Authorization: Bearer super-secret-value',
    'API_KEY=super-secret-value',
    '{"api_key": "super-secret-value"}',
    "{'access_token': 'super-secret-value'}",
    'H3_API_KEY="super-secret-value"',
])
def test_common_credential_forms_are_hidden(secret_text):
    assert 'super-secret-value' not in ui.redact_text(secret_text)


def test_known_secret_redaction_precedes_truncation_and_escapes_logs():
    output = ui.diagnostics_html('<logs>', '<script>secret-key</script>' + 'x' * 15000,
                                 secrets=('secret-key',))
    assert 'secret-key' not in output and '<script>' not in output
    assert '&lt;logs&gt;' in output and 'tabindex="0"' in output
    assert output.count('x') < 12500


def test_diagnostics_are_bounded_and_never_inside_cards():
    for i in range(40):
        ui.StatusCard('event', details=str(i) + ':' + 'x' * 16000)
    assert len(ui._diagnostics) == 20
    assert len(ui.last_diagnostics()) <= ui._DIAGNOSTIC_LIMIT
    output = ui.card_html('summary', details='raw-log-marker')
    assert 'raw-log-marker' not in output and '<pre>' not in output


def test_snapshot_failure_records_detail_without_claiming_ready(reset_ui):
    display, _ = reset_ui
    host = SimpleNamespace(api_key='credential-opaque', snapshot=Mock(side_effect=
        RuntimeError('transport error credential-opaque')))
    ui.mount_dashboard(lambda: {'host': host})
    output = display.call_args.args[0]
    assert '状态读取失败' in output and 'transport error' not in output
    assert 'credential-opaque' not in output and 'credential-opaque' not in ui.last_diagnostics()
    assert 'transport error' in ui.last_diagnostics()


def test_dependency_error_does_not_dump_raw_stderr_in_control_output(tmp_path, monkeypatch):
    monkeypatch.setattr(ui.subprocess, 'Popen', lambda *a, **k:
        SimpleNamespace(returncode=1, communicate=lambda **kw: ('', 'raw-private-diagnostic-marker')))
    with pytest.raises(RuntimeError) as error:
        ui.json_probe(['unused'], tmp_path / 'install.log', card=ui.StatusCard('依赖检查'))
    assert 'raw-private-diagnostic-marker' not in str(error.value)
    assert 'raw-private-diagnostic-marker' in (tmp_path / 'install.log').read_text(encoding='utf-8')
    assert 'raw-private-diagnostic-marker' in ui.last_diagnostics()


def test_json_probe_is_owned_and_detaches_on_timeout(tmp_path, monkeypatch):
    process = Mock()
    process.communicate.side_effect = [ui.subprocess.TimeoutExpired('probe', 180), ('', '')]
    process.poll.return_value = None
    observer = Mock()
    monkeypatch.setattr(ui.subprocess, 'Popen', lambda *a, **k: process)
    monkeypatch.setattr(ui, '_process_observer', observer)
    with pytest.raises(ui.subprocess.TimeoutExpired):
        ui.json_probe(['unused'], tmp_path / 'install.log')
    observer.process_started.assert_called_once_with(process)
    observer.process_finished.assert_called_once_with(process)
    process.kill.assert_called_once_with()
