"""Exercise the generated notebook's UI boundaries without Colab, network or GPU."""
import ast
import builtins
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime/minimax_h3'))
import notebook_ui as ui

PROGRAM_TITLES = {
    'preflight': '资源与驻留方案',
    'install': '安装与检查',
    'start': '启动总管理、API与模型容器',
    'generate': '生成验收与恢复查询',
    'stop': '关闭选项',
    'advanced': '性能与 LoRA',
    'wait': '等待就绪或重新查询',
    'control': '查询、事件与模型控制',
    'export': '导出 API 连接信息',
    'logs': '查看日志',
}


def notebook():
    return json.loads((ROOT / 'notebooks/minimax-h3-ref2va.ipynb').read_text(encoding='utf-8'))


def console_source():
    cell, = (cell for cell in notebook()['cells'] if cell['cell_type'] == 'code')
    return ''.join(cell['source'])


def console_bundle(source):
    return {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in ast.parse(source).body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in {'PACKAGE_FILES', 'CONSOLE_PROGRAMS'}
    }


def program_tree(name):
    return ast.parse(console_bundle(console_source())['CONSOLE_PROGRAMS'][name])


def execute_program(name, scope, **parameters):
    tree = program_tree(name)
    replaced = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            target_name = node.targets[0].id
            if target_name in parameters:
                node.value = ast.Constant(parameters[target_name])
                replaced.add(target_name)
    assert replaced == set(parameters)
    exec(compile(ast.fix_missing_locations(tree), f'generated-notebook-program-{name}', 'exec'), scope)


def host_fixture(tmp_path):
    host = SimpleNamespace(api_key='test-secret-should-never-display', instance_id='test-instance',
        api_log=tmp_path / 'api.log', server_log=tmp_path / 'model.log',
        manager_log=tmp_path / 'manager.log', tunnel_log=tmp_path / 'tunnel.log')
    host.snapshot = Mock(return_value={'api_ready': True, 'public_api_ready': True,
        'public_url': 'https://example.invalid', 'api_key': host.api_key})
    host.events = Mock(return_value={'events': [{'message': 'Bearer ' + host.api_key}]})
    host.command = Mock(return_value={'command_id': 'existing-command', 'status': 'succeeded'})
    host.control = Mock(return_value={'command_id': 'new-command', 'status': 'accepted'})
    host.start = Mock()
    host.stop = Mock()
    host.wait_ready = Mock()
    for path in (host.api_log, host.server_log, host.manager_log, host.tunnel_log):
        path.write_text('line before\nAPI_KEY=' + host.api_key + '\n<script>unsafe markup</script>', encoding='utf-8')
    return host


def base_scope(tmp_path, host):
    return dict(PROJECT_DIR=tmp_path, startup_host=host, StatusCard=Mock(),
                json=json, Path=Path, log_tail=ui.log_tail)


@pytest.fixture
def fake_display(monkeypatch):
    shown = []
    package = ModuleType('IPython')
    display_module = ModuleType('IPython.display')
    display_module.HTML = lambda text: text
    display_module.display = lambda value, **kwargs: shown.append(value)
    package.display = display_module
    monkeypatch.setitem(sys.modules, 'IPython', package)
    monkeypatch.setitem(sys.modules, 'IPython.display', display_module)
    return shown


def test_notebook_renders_three_regions_in_the_single_start_cell():
    class ConsoleFields(HTMLParser):
        def __init__(self):
            super().__init__()
            self.sections = []
            self.open_sections = []
            self.fields = {}
            self.log_viewport_depth = 0
            self.log_content = []

        def handle_starttag(self, tag, attrs):
            attributes = dict(attrs)
            element_id = attributes.get('id')
            if element_id == 'log-viewport':
                assert tag == 'div' and self.open_sections == ['logs-heading']
                assert not self.log_viewport_depth
                self.log_viewport_depth = 1
            elif tag == 'div' and self.log_viewport_depth:
                self.log_viewport_depth += 1
            if element_id in {'log-intro', 'log-text'}:
                assert self.log_viewport_depth and self.open_sections == ['logs-heading']
                self.log_content.append(element_id)
            if tag == 'section':
                panel = attributes.get('aria-labelledby')
                assert 'panel' in attributes.get('class', '').split()
                assert not self.open_sections
                self.sections.append(panel)
                self.open_sections.append(panel)
            elif tag in {'button', 'input', 'select', 'textarea', 'summary', 'progress'}:
                assert self.open_sections, f'Field outside a panel: {tag} {attributes}'
                field = element_id or attributes.get('data-action')
                if field:
                    self.fields[field] = self.open_sections[-1]

        def handle_endtag(self, tag):
            if tag == 'section':
                self.open_sections.pop()
            elif tag == 'div' and self.log_viewport_depth:
                self.log_viewport_depth -= 1

    data = notebook()
    assert [cell['cell_type'] for cell in data['cells']] == ['code']
    assert not data['metadata']['colab'].get('collapsed_sections')
    startup = data['cells'][0]
    assert ''.join(startup['source']).startswith('#@title 一键启动\n')
    assert startup['execution_count'] is None
    assert len(startup['outputs']) == 1
    placeholder = ''.join(startup['outputs'][0]['data']['text/html'])
    fields = ConsoleFields()
    fields.feed(placeholder)
    fields.close()
    assert not fields.open_sections and not fields.log_viewport_depth
    assert fields.sections == ['headline', 'control-heading', 'logs-heading']
    assert fields.fields['api-url'] == fields.fields['model-download-progress'] == 'headline'
    assert fields.fields['bootstrap'] == fields.fields['reference_file'] == 'control-heading'
    assert fields.fields['acceleration'] == fields.fields['component-action'] == 'control-heading'
    assert fields.fields['log-source'] == fields.fields['follow-logs'] == 'logs-heading'
    assert fields.log_content == ['log-intro', 'log-text']
    assert '"offline": true' in placeholder
    assert 'window.alert(' not in placeholder and 'window.confirm(' not in placeholder


def test_console_embedded_programs_do_not_leak_colab_form_directives():
    source = console_source()
    # Colab scans form markers even inside quoted Python strings. A valid AST
    # alone does not prevent embedded action comments from becoming visible UI.
    directives = [line for line in source.splitlines() if '#@' in line]
    assert directives == ['#@title 一键启动']
    assert source.count('#@') == 1

    embedded = console_bundle(source)
    assert set(embedded) == {'PACKAGE_FILES', 'CONSOLE_PROGRAMS'}
    files = embedded['PACKAGE_FILES']
    assert len(files) == 35
    assert 'condition_guard.py' in files
    assert {'condition_projection.py', 'queue_policy.py'} <= files.keys()
    assert 'generation_options.py' in files
    for name, content in files.items():
        assert content == (ROOT / 'runtime/minimax_h3' / name).read_text(encoding='utf-8')

    programs = embedded['CONSOLE_PROGRAMS']
    assert set(programs) == set(PROGRAM_TITLES)
    for name, title in PROGRAM_TITLES.items():
        assert programs[name].startswith('#@title ' + title + '\n')
        ast.parse(programs[name])
    # The install action embeds a second copy. Escaping must survive both
    # literal layers while reproducing the same source files at execution.
    install_files = next(ast.literal_eval(node.value)
                         for node in ast.parse(programs['install']).body
                         if isinstance(node, ast.Assign)
                         and isinstance(node.targets[0], ast.Name)
                         and node.targets[0].id == 'PACKAGE_FILES')
    assert install_files == files


def test_command_id_query_does_not_send_new_command(tmp_path):
    host = host_fixture(tmp_path)
    scope = base_scope(tmp_path, host)
    execute_program('control', scope, COMMAND_ID=' existing-command ')
    host.command.assert_called_once_with('existing-command')
    host.control.assert_not_called()
    host.start.assert_not_called()
    host.stop.assert_not_called()


def test_default_component_control_is_noop(tmp_path):
    host = host_fixture(tmp_path)
    execute_program('control', base_scope(tmp_path, host))
    for name in ('command', 'control', 'start', 'stop', 'wait_ready'):
        getattr(host, name).assert_not_called()


def test_drain_label_describes_admission_not_dispatch(tmp_path):
    host = host_fixture(tmp_path)
    execute_program('control', base_scope(tmp_path, host), CONTROL_ACTION='暂停接收')
    host.control.assert_called_once_with('drain', force=False)
    host.command.assert_not_called()


@pytest.mark.parametrize('program', ['start', 'generate', 'stop', 'export'])
def test_action_programs_default_to_no_gpu_no_shutdown_no_export(tmp_path, program):
    host = host_fixture(tmp_path)
    execute_program(program, base_scope(tmp_path, host))
    for name in ('snapshot', 'command', 'control', 'start', 'stop', 'wait_ready'):
        getattr(host, name).assert_not_called()
    assert not (tmp_path / 'api-connection.private.json').exists()


@pytest.mark.parametrize('source', ['最近诊断', 'API后台', '调度事件', '完整状态'])
def test_log_program_only_reads_and_redacts(tmp_path, fake_display, source):
    host = host_fixture(tmp_path)
    execute_program('logs', base_scope(tmp_path, host), LOG_SOURCE=source)
    for name in ('command', 'control', 'start', 'stop', 'wait_ready'):
        getattr(host, name).assert_not_called()
    assert len(fake_display) == 1
    html = fake_display[0]
    assert 'aria-label="日志"' in html
    assert host.api_key not in html and '<script>' not in html
    if source != '最近诊断':
        assert '[已隐藏]' in html


def test_connection_export_contains_key_only_in_private_file(tmp_path, monkeypatch, capsys):
    host = host_fixture(tmp_path)
    files = SimpleNamespace(download=Mock())
    colab = ModuleType('google.colab')
    colab.files = files
    monkeypatch.setitem(sys.modules, 'google.colab', colab)
    scope = base_scope(tmp_path, host)
    execute_program('export', scope, EXPORT_CONNECTION=True)
    path = tmp_path / 'api-connection.private.json'
    exported = json.loads(path.read_text(encoding='utf-8'))
    assert exported == {'url': 'https://example.invalid', 'api_key': host.api_key,
                        'instance_id': host.instance_id}
    files.download.assert_called_once_with(str(path))
    assert host.api_key not in capsys.readouterr().out
    assert host.api_key not in repr(scope['StatusCard'].call_args_list)
    host.control.assert_not_called()


@pytest.mark.parametrize('phase,error', [('failed', 'cleanup_incomplete'), ('degraded', 'model failed')])
def test_failure_has_priority_over_draining_or_ready(phase, error):
    html = ui.render_dashboard({'state': {'phase': phase, 'error': error, 'draining': True, 'ready': True}})
    assert '有模块需要处理' in html
    assert '已暂停接收新任务' not in html and '可以生成' not in html


def test_stopping_does_not_promise_queue_continues():
    html = ui.render_dashboard({'state': {'phase': 'stopping', 'draining': True, 'ready': False}})
    assert '正在停止' in html
    assert '已有任务继续处理' not in html and '可以生成' not in html


def test_stale_observation_hides_old_ready_capabilities():
    html = ui.render_dashboard({'state': {'phase': 'ready', 'ready': True, 'observation_stale': True,
        'api_ready': True, 'model_ready': True, 'public_api_ready': True, 'public_requested': True,
        'queue': {'running': 8, 'queued': 10, 'admission_open': True}}})
    assert '状态需要刷新' in html
    assert '可以生成' not in html and 'API 探测通过' not in html
    assert '排队 10' not in html and '执行 8' not in html


def execute_real_generation_admission_guard(state, endpoint):
    """Execute the actual generated program up to its first output-directory mutation."""
    branch = next(node for node in program_tree('generate').body
                  if isinstance(node, ast.If) and isinstance(node.test, ast.Name)
                  and node.test.id == 'RUN_SMOKE_TEST')
    cutoff = next(index for index, node in enumerate(branch.body)
                  if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                  and node.targets[0].id == 'result_dir')
    # Imports do not participate in these guards; omit them to avoid loading Colab/GPU packages.
    guards = [node for node in branch.body[:cutoff] if not isinstance(node, (ast.Import, ast.ImportFrom))]
    scope = {'startup_host': SimpleNamespace(snapshot=lambda: dict(state)),
             'RESUME_JOB_ID': '', 'TEST_API_URL': endpoint, 'API_KEY': 'fixture-key'}
    exec(compile(ast.fix_missing_locations(ast.Module(body=guards, type_ignores=[])),
                 'actual-notebook-generation-admission', 'exec'), scope)
    return scope


@pytest.mark.parametrize('public', [False, True])
def test_real_generation_program_can_queue_before_model_ready(public):
    state = {'api_ready': True, 'public_api_ready': public,
             'ready': False, 'model_ready': False, 'public_ready': False,
             'public_url': 'https://fixture.invalid', 'observation_stale': False}
    scope = execute_real_generation_admission_guard(state, 'PUBLIC_API_URL' if public else 'http://127.0.0.1:8000')
    assert scope['test_base_url'] == ('https://fixture.invalid' if public else 'http://127.0.0.1:8000')


@pytest.mark.parametrize('failure', ['stale', 'api_unavailable', 'public_unavailable'])
def test_real_generation_program_rejects_unobserved_or_unavailable_api(failure):
    # All generation-ready flags deliberately remain True: only current API evidence should admit submission.
    state = {'api_ready': failure != 'api_unavailable', 'public_api_ready': failure != 'public_unavailable',
             'ready': True, 'model_ready': True, 'public_ready': True,
             'public_url': 'https://fixture.invalid', 'observation_stale': failure == 'stale'}
    with pytest.raises(RuntimeError):
        execute_real_generation_admission_guard(state, 'PUBLIC_API_URL')


@pytest.fixture
def run_real_main_cell(tmp_path, monkeypatch):
    """Run the actual main cell with only its target directory and mount isolated."""
    project = tmp_path / 'colab-project'
    project.mkdir()
    tree = ast.parse(console_source())
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'PROJECT_DIR':
            node.value = ast.Call(func=ast.Name(id='Path', ctx=ast.Load()), args=[ast.Constant(str(project))], keywords=[])
    compiled = compile(ast.fix_missing_locations(tree), 'actual-notebook-main-cell', 'exec')
    calls, created = [], []

    def mount(directory, programs, scope, *, auto_start=False, bootstrap_payload=None, foreground_bootstrap_timeout=None):
        calls.append((directory, programs, auto_start, bootstrap_payload, foreground_bootstrap_timeout))
        owner = scope.get('h3_console')
        if owner is None or owner.closed:
            owner = SimpleNamespace(active={}, closed=False, close=Mock(), scope=scope)
            scope['h3_console'] = owner
            created.append(owner)
        return owner

    module = ModuleType('notebook_console')
    module.mount_console = mount
    real_import = builtins.__import__

    def controlled_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == 'notebook_console':
            return module
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, '__import__', controlled_import)
    monkeypatch.setattr(sys, 'path', list(sys.path))

    def run(scope):
        exec(compiled, scope)
        return scope

    return SimpleNamespace(run=run, directory=project, calls=calls, created=created)


def test_real_main_first_run_mounts_and_autostarts_with_content_hash(run_real_main_cell):
    fixture = run_real_main_cell
    scope = fixture.run({})
    assert len(fixture.calls) == 1 and len(fixture.created) == 1
    directory, programs, auto_start, payload, foreground_timeout = fixture.calls[0]
    assert directory == fixture.directory and auto_start is True
    assert payload == scope['H3_BOOTSTRAP_OPTIONS']
    assert payload['public'] is True
    assert payload['acceleration'] in {'none', 'lightx2v-ref2va-4step'}
    # The foreground handoff is bounded and only observes API delivery; the
    # independent model branch continues after the panel regains RPC controls.
    assert foreground_timeout == 180
    assert set(programs) == set(PROGRAM_TITLES)
    expected = hashlib.sha256(json.dumps({'files': scope['PACKAGE_FILES'], 'programs': programs,
        'bootstrap_options': scope['H3_BOOTSTRAP_OPTIONS']},
        ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()
    assert scope['H3_LOADED_BUNDLE_HASH'] == scope['H3_CONSOLE_BUNDLE_HASH'] == expected
    for name, content in scope['PACKAGE_FILES'].items():
        assert (directory / name).read_text(encoding='utf-8') == content


def test_real_main_same_version_preserves_active_controller_and_files(run_real_main_cell):
    fixture = run_real_main_cell
    scope = fixture.run({})
    owner = scope['h3_console']
    owner.active = {'setup': {'action': 'bootstrap', 'status': 'running'}}
    existing = fixture.directory / 'console.html'
    existing.write_text('fixture existing file', encoding='utf-8')
    fixture.run(scope)
    assert scope['h3_console'] is owner and len(fixture.created) == 1
    owner.close.assert_not_called()
    assert existing.read_text(encoding='utf-8') == 'fixture existing file'


def test_real_main_upgrades_idle_old_controller_and_evicts_only_project_modules(run_real_main_cell, monkeypatch, tmp_path):
    fixture = run_real_main_cell
    old = SimpleNamespace(active={}, closed=False, close=Mock(), scope={})
    host = SimpleNamespace(snapshot=Mock(return_value={
        'phase': 'stopped', 'cleanup_complete': True, 'observation_stale': True,
        'manager_alive': False, 'server_alive': False, 'model_alive': False, 'tunnel_alive': False,
        'process_observation_errors': {}}))
    project_module, external_module = ModuleType('fixture_old_h3'), ModuleType('fixture_external')
    project_module.__file__ = str(fixture.directory / 'notebook_console.py')
    external_module.__file__ = str(tmp_path / 'unrelated.py')
    monkeypatch.setitem(sys.modules, 'fixture_old_h3', project_module)
    monkeypatch.setitem(sys.modules, 'fixture_external', external_module)
    scope = {'h3_console': old, 'startup_host': host, 'H3_LOADED_BUNDLE_HASH': 'old-version'}
    fixture.run(scope)
    old.close.assert_called_once_with()
    host.snapshot.assert_called_once_with()
    assert scope['h3_console'] is not old and len(fixture.created) == 1
    assert 'fixture_old_h3' not in sys.modules
    assert sys.modules['fixture_external'] is external_module
    assert scope['startup_host'] is host  # Ownership is retained for the new controller's clean-start check.
    assert scope['H3_LOADED_BUNDLE_HASH'] == scope['H3_CONSOLE_BUNDLE_HASH']


@pytest.mark.parametrize('blocked', ['operation', 'running_host', 'unknown_exit', 'unclean', 'observation_error'])
def test_real_main_refuses_upgrade_without_destroying_old_owner(run_real_main_cell, monkeypatch, blocked):
    fixture = run_real_main_cell
    state = {'phase': 'stopped', 'cleanup_complete': True,
             'manager_alive': False, 'server_alive': False, 'model_alive': False, 'tunnel_alive': False}
    active = {'setup': {'action': 'bootstrap'}} if blocked == 'operation' else {}
    if blocked == 'running_host':
        state.update(phase='ready', model_alive=True)
    elif blocked == 'unknown_exit':
        state['model_alive'] = None
    elif blocked == 'unclean':
        state['cleanup_complete'] = False
    elif blocked == 'observation_error':
        state['process_observation_errors'] = {'model': 'access denied'}
    host = SimpleNamespace(snapshot=Mock(return_value=state))
    old = SimpleNamespace(active=active, closed=False, close=Mock(), scope={})
    old_module = ModuleType('fixture_old_h3')
    old_module.__file__ = str(fixture.directory / 'notebook_console.py')
    monkeypatch.setitem(sys.modules, 'fixture_old_h3', old_module)
    existing = fixture.directory / 'console.html'
    existing.write_text('fixture prior source', encoding='utf-8')
    scope = {'h3_console': old, 'startup_host': host, 'H3_LOADED_BUNDLE_HASH': 'old-version'}
    with pytest.raises(RuntimeError, match='旧'):
        fixture.run(scope)
    assert scope['h3_console'] is old and scope['startup_host'] is host
    assert scope['H3_LOADED_BUNDLE_HASH'] == 'old-version'
    old.close.assert_not_called()
    assert existing.read_text(encoding='utf-8') == 'fixture prior source'
    assert sys.modules['fixture_old_h3'] is old_module
    assert not fixture.calls and not fixture.created
