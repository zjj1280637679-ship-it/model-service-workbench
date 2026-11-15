"""Owned notebook UI bridge. It observes the manager; it is not a GPU scheduler."""
from __future__ import annotations

import ast
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
import secrets
from pathlib import Path
import threading
import sys
import time
import traceback
import uuid

import notebook_ui as ui
from condition_projection import public_condition_summary, public_execution_summary


ACTION_TITLES = {'bootstrap': '一键启动', 'preflight': '资源检查', 'install': '安装依赖', 'start': '启动服务',
                 'generate': '生成验收', 'stop': '停止服务', 'release': '释放实例', 'control': '组件控制'}
STARTUP_LABELS = {
    'preflight': '检查资源',
    'install': '安装依赖',
    'start': '启动独立服务',
    'awaiting_api': '等待 API 及所选公网可达',
    'awaiting_model': 'API 已可接收排队任务；等待模型完整加载',
    'ready': '模型与 API 已就绪；排队任务由总管理自动调度',
    'failed': '启动未完成，查看实际错误；已启动的后台仍可查询和停止',
    'cancelled': '启动已取消',
}
LOG_NAMES = {'model': 'server_log', 'api': 'api_log', 'manager': 'manager_log', 'tunnel': 'tunnel_log'}
BOOLS = {'staged', 'cached', 'public', 'force'}
STRINGS = {'reference_file', 'prompt', 'resume_job_id', 'endpoint', 'acceleration', 'style',
           'attention', 'offload', 'action', 'command_id', 'request_id', 'source', 'credential_id'}
CHOICES = {'endpoint': {'public', 'local'}, 'acceleration': {'none', 'lightx2v-ref2va-4step'},
           'style': {'none', 'fal-realism-people'}, 'attention': {'auto', 'native', '_native_flash'},
           'offload': {'stream', 'sync'}}
SAFE_STATE = {'instance_id', 'model_instance_id', 'phase', 'ready', 'api_ready', 'model_ready', 'public_requested',
              'public_api_ready', 'public_ready', 'tunnel_connected', 'server_alive', 'model_alive',
              'manager_alive', 'tunnel_alive', 'cleanup_complete', 'observed_at', 'observation_stale',
              'elapsed_seconds', 'draining', 'queue'}


class OperationCancelled(RuntimeError):
    pass


def validated(payload):
    if not isinstance(payload, dict) or len(json.dumps(payload)) > 65536:
        raise ValueError('控制参数必须是有限的对象。')
    result = {}
    for key, value in payload.items():
        if key in BOOLS:
            if type(value) is not bool:
                raise ValueError(f'{key} 必须为布尔值。')
        elif key in STRINGS:
            if not isinstance(value, str) or len(value) > (6000 if key == 'prompt' else 2048):
                raise ValueError(f'{key} 不是有效的短文本。')
        elif key == 'style_scale':
            if type(value) not in (int, float) or not .1 <= value <= 1:
                raise ValueError('风格强度应在 0.1 与 1 之间。')
        elif key in {'cursor'}:
            continue
        else:
            raise ValueError(f'未知控制字段：{key}')
        if key in CHOICES and value not in CHOICES[key]:
            raise ValueError(f'{key} 选项无效。')
        result[key] = value
    return result


def action_code(source, parameters):
    """Replace only known top-level notebook defaults, never execute browser-supplied code."""
    tree = ast.parse(source)
    found = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in parameters:
                node.value = ast.Constant(parameters[name])
                found.add(name)
    if found != set(parameters):
        raise ValueError('操作参数与冻结的操作代码不匹配。')

    class NoAutomaticDownloads(ast.NodeTransformer):
        def visit_Expr(self, node):
            call = node.value
            if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Name) and call.func.value.id == 'files'
                    and call.func.attr == 'download'):
                return ast.copy_location(ast.Pass(), node)
            return self.generic_visit(node)
    tree = NoAutomaticDownloads().visit(tree)
    return compile(ast.fix_missing_locations(tree), 'owned-notebook-operation', 'exec')


class ConsoleController:
    def __init__(self, directory, programs, scope=None, release_callback=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.programs = dict(programs)
        self.scope = scope if scope is not None else {}
        self.scope['PROJECT_DIR'] = self.directory
        self.scope['dashboard'] = self
        self.scope['print'] = self._print
        self.scope.update(StatusCard=ui.StatusCard, elapsed_text=ui.elapsed_text, log_tail=ui.log_tail,
                          run_logged=ui.run_logged, json_probe=ui.json_probe, os=os, sys=sys, json=json, Path=Path)
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix='h3-console')
        self.operations = {}
        self.active = {}
        self.finished = {}
        self.cancel_startup = threading.Event()
        self.startup_stage = None
        self.startup_delivery = None
        self._bootstrap_hosts = {}
        self.requested_options = {}
        self.processes = {}
        self.process_owners = {}
        self.closed = False
        self.epoch = None
        self.activity = None
        self._last_activity = None
        self.notice = None
        self.release_callback = release_callback
        self.audit_path = self.directory / 'console-events.jsonl'
        self.log_path = self.directory / 'console.log'
        ui._dashboard = self
        ui._process_observer = self

    def _secrets(self):
        host = self.scope.get('startup_host')
        return [getattr(host, 'api_key', ''), getattr(host, '_control_key', ''), self.scope.get('API_KEY', '') or '',
                self.scope.get('BOOTSTRAP_API_KEY', '') or '']

    def _clean(self, value):
        return ui.redact_text(str(value), secrets=self._secrets())

    def _sanitize(self, value):
        if isinstance(value, str):
            return self._clean(value)
        if isinstance(value, dict):
            return {key: self._sanitize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._sanitize(item) for item in value]
        return value

    def _event(self, kind, **values):
        record = {'type': kind, 'observed_at': time.time(), **values}
        with self.lock:
            with self.audit_path.open('a', encoding='utf-8') as stream:
                stream.write(self._clean(json.dumps(record, ensure_ascii=False)) + '\n')

    def _print(self, *values, sep=' ', end='\n', **kwargs):
        text = self._clean(sep.join(str(v) for v in values))
        stamp = time.strftime('%m-%d %H:%M:%S', time.gmtime(time.time() + 8 * 3600))
        with self.lock:
            with self.log_path.open('a', encoding='utf-8') as stream:
                stream.write('[' + stamp + ' 北京时间] ' + text + end)

    def refresh(self):
        """StatusCard sink: update data and log, never create a notebook output."""
        item = self.activity or {}
        marker = json.dumps(item, sort_keys=True, ensure_ascii=False)
        if marker != self._last_activity:
            self._last_activity = marker
            self._print(item.get('title', ''), item.get('badge', ''), item.get('note', ''))

    def process_started(self, process):
        owner = None
        if type(getattr(process, 'pid', None)) is int:
            from process_owner import ProcessOwner
            owner = ProcessOwner(process)
        with self.lock:
            self.processes[threading.get_ident()] = process
            if owner is not None:
                self.process_owners[id(process)] = (owner, threading.get_ident())
            cancelled = any(record.get('thread_id') == threading.get_ident()
                            and (record.get('cancel_monitor') or record.get('cancel_requested'))
                            for record in self.active.values())
        if cancelled:
            self._terminate_client_process(process)

    def process_progress(self, process):
        with self.lock:
            entry = self.process_owners.get(id(process))
        if entry:
            entry[0].observe()  # Capture descendants before wrappers exit.

    def process_finished(self, process):
        with self.lock:
            entry = self.process_owners.get(id(process))
        if entry:
            state = entry[0].observe()
            if not state['cleanup_complete']:
                state = entry[0].terminate_all(timeout=10)
            self._event('client.process_cleanup', **state)
            if not state['cleanup_complete']:
                raise RuntimeError('安装或客户端的子进程树尚未确认退出；请停止后检查日志。')
        with self.lock:
            self.process_owners.pop(id(process), None)
            if self.processes.get(threading.get_ident()) is process:
                self.processes.pop(threading.get_ident(), None)

    def _terminate_client_process(self, process):
        with self.lock:
            entry = self.process_owners.get(id(process))
        if entry:
            state = entry[0].terminate_all(timeout=10)
            self._event('client.process_cleanup', **state)
            if not state['cleanup_complete']:
                raise RuntimeError('子进程树仍未退出；未声明停止或释放完成。')
        elif process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except Exception:
                process.kill()
                process.wait(timeout=5)

    def _stop_client_monitor(self):
        with self.lock:
            active = self.active.get('generate')
            if active:
                active['cancel_monitor'] = True
            process = self.processes.get(active.get('thread_id')) if active else None
        if process is not None and process.poll() is None:
            self._terminate_client_process(process)
            self._event('client.monitor_stopped', request_id=active['request_id'])

    def _execute(self, name, parameters=None):
        exec(action_code(self.programs[name], parameters or {}), self.scope)

    def _configuration(self, payload):
        self.requested_options = {name: payload.get(name, 'none') for name in ('acceleration', 'style')}
        self._execute('advanced', {'ACCELERATION_LORA': payload.get('acceleration', 'none'),
            'STYLE_LORA': payload.get('style', 'none'), 'STYLE_LORA_SCALE': payload.get('style_scale', 1.0),
            'ATTENTION_BACKEND': payload.get('attention', 'auto'), 'DENOISER_OFFLOAD': payload.get('offload', 'stream')})

    def _check_startup_cancelled(self):
        if self.cancel_startup.is_set():
            raise OperationCancelled('启动已取消；不再进入后续阶段。')

    def _startup_stage(self, stage):
        if self.startup_stage != stage:
            self.startup_stage = stage
            self._event('startup.stage', stage=stage)
            self._print(STARTUP_LABELS.get(stage, stage))

    def _await_bootstrap(self, host, public):
        deadline = time.monotonic() + float(getattr(host, 'startup_timeout', 3600))
        interval = max(.01, min(2, float(getattr(host, 'poll_interval', 1))))
        while True:
            self._check_startup_cancelled()
            state = host.snapshot()
            fresh = not state.get('observation_stale', False)
            if state.get('phase') in {'failed', 'stopped', 'stopping'} or (fresh and state.get('error')):
                raise RuntimeError(state.get('error') or '服务准备未完成；API 可用时仍可查询与取消任务。')
            api = fresh and state.get('api_ready') is True and (not public or state.get('public_api_ready') is True)
            if api and state.get('model_ready') is True and state.get('ready') is True:
                self._startup_stage('ready')
                return
            self._startup_stage('awaiting_model' if api else 'awaiting_api')
            if time.monotonic() >= deadline:
                raise TimeoutError('等待服务就绪超时；请查看日志。已有任务和后台保留，可查询、取消或停止。')
            self.cancel_startup.wait(interval)

    def _bootstrap(self, payload):
        payload = dict(payload, public=payload.get('public', True))
        self._check_startup_cancelled()
        host = self.scope.get('startup_host')
        state = host.snapshot() if host else {}
        public = payload.get('public', True)
        if host and any(state.get(name) for name in ('manager_alive', 'server_alive', 'model_alive', 'tunnel_alive')):
            # A repeated start observes the owned instance; it cannot reinstall or rotate its token.
            if bool(getattr(host, 'public', False)) != public or bool(getattr(host, 'staged', False)) != payload.get('staged', False):
                raise RuntimeError('现有实例配置不同；请先停止，再以新配置启动。')
            self._bind_bootstrap_host(payload['request_id'], host)
            return self._await_bootstrap(host, public)
        clean_terminal = (state.get('phase') in {'failed', 'stopped'} and state.get('cleanup_complete') is True
                          and all(state.get(name) is False for name in ('manager_alive', 'server_alive', 'model_alive', 'tunnel_alive'))
                          and not state.get('process_observation_errors'))
        if host and not clean_terminal and (state.get('observation_stale') or state.get('phase') != 'new'):
            raise RuntimeError('旧生命周期尚未确认清理完成；先停止并检查状态。')
        self.scope.pop('startup_host', None)
        self.scope.update(PREFLIGHT_OK=False, INSTALL_OK=False, RESOURCE_REPORT={})
        self.requested_options = {name: payload.get(name, 'none') for name in ('acceleration', 'style')}
        self.scope['BOOTSTRAP_API_KEY'] = secrets.token_urlsafe(32)
        self.scope['API_KEY'] = self.scope['BOOTSTRAP_API_KEY']
        self.scope['BOOTSTRAP_PUBLIC'] = public
        self._event('credential.generated', purpose='public_api')
        for stage in ('preflight', 'install', 'start'):
            self._check_startup_cancelled()
            self._startup_stage(stage)
            self._operate(stage, payload)
            self._check_startup_cancelled()
            if stage == 'preflight' and self.scope.get('PREFLIGHT_OK') is not True:
                raise RuntimeError('资源检查没有提供通过凭据，停止启动。')
            if stage == 'install' and self.scope.get('INSTALL_OK') is not True:
                raise RuntimeError('依赖检查没有提供通过凭据，停止启动。')
        host = self.scope.get('startup_host')
        if not host:
            raise RuntimeError('总管理未创建，停止启动。')
        self._bind_bootstrap_host(payload['request_id'], host)
        return self._await_bootstrap(host, public)

    def _bind_bootstrap_host(self, request_id, host):
        """Bind observations to the owner accepted by this bootstrap request."""
        instance_id = getattr(host, 'instance_id', None)
        with self.lock:
            operation = self.operations.get(request_id)
            if not instance_id or not operation or operation.get('action') != 'bootstrap':
                raise RuntimeError('启动请求与实例绑定无效。')
            previous = self._bootstrap_hosts.get(request_id)
            if previous and (previous[0] is not host or previous[1] != instance_id):
                raise RuntimeError('启动请求不能更换已绑定实例。')
            self._bootstrap_hosts[request_id] = (host, instance_id)
            operation['instance_id'] = instance_id
        self._event('startup.host_bound', request_id=request_id, instance_id=instance_id)

    def _cancel_bootstrap_and_wait(self):
        with self.lock:
            record = self.active.get('setup')
            if not record or record['action'] != 'bootstrap':
                return
            record['cancel_requested'] = True
            self.cancel_startup.set()
            process = self.processes.get(record.get('thread_id'))
            finished = self.finished[record['request_id']]
        self._event('startup.cancel_requested', request_id=record['request_id'])
        if process is not None:
            self._terminate_client_process(process)
        # Join before stopping the owner: a late start must not resurrect a released service.
        if not finished.wait(60):
            raise TimeoutError('启动步骤尚未退出；未声明停止或释放完成，请检查日志后重试。')

    def begin_startup_delivery(self, request_id, timeout):
        """Observe delivery on the calling cell; never schedule dependent work."""
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or not 0 < timeout <= 600:
            raise ValueError('启动交付观察时间须为 0 到 600 秒之间的有限数值。')
        with self.lock:
            self.startup_delivery = {'request_id': request_id, 'status': 'observing', 'active': True,
                'started_at': time.time(), 'deadline_at': time.time() + timeout, 'timeout_seconds': timeout,
                'instance_id': None, 'last_observed_at': None, 'api_ready': False, 'model_ready': False,
                'cleanup_complete': None}
        self._event('startup.delivery_started', request_id=request_id, timeout_seconds=timeout,
                    dependency='current_instance_api_and_selected_public_api; model_ready_not_required')
        self._print('正在交付独立 API；最多观察', timeout,
                    '秒，不等待模型。观察期间请用 Colab 本单元的停止按钮取消，程序会回收已启动进程。')

    def wait_startup_delivery(self, request_id, *, publish=None):
        """A bounded foreground handoff, independent from model preparation.

        Expiry only ends observation and retains the worker/owner. Interrupting
        the cell explicitly cancels startup, joins it, then reclaims its owned
        processes through the same audited stop path as the control panel.
        """
        delivery = self.startup_delivery
        if not delivery or delivery['request_id'] != request_id or not delivery['active']:
            raise RuntimeError('启动交付观察身份不匹配。')
        deadline = time.monotonic() + delivery['timeout_seconds']
        status, error, last_publish = 'observation_failed', None, -float('inf')
        def update():
            nonlocal last_publish
            if publish:
                try:
                    publish()
                except Exception as caught:
                    self._print('启动快照显示未更新：', type(caught).__name__)
            last_publish = time.monotonic()
        try:
            while True:
                with self.lock:
                    record = dict(self.operations.get(request_id) or {})
                    binding = self._bootstrap_hosts.get(request_id)
                if self.cancel_startup.is_set() or record.get('status') == 'cancelled':
                    status = 'cancelled'
                    break
                host, expected = binding if binding else (None, None)
                # An upgraded controller can still hold a clean, stopped host
                # while its new worker validates and replaces it. That previous
                # owner's terminal state cannot finish this request's handoff.
                if (host is not None and self.scope.get('startup_host') is host
                        and getattr(host, 'instance_id', None) == expected):
                    state = host.snapshot()
                    same = bool(expected and state.get('instance_id') == expected)
                    fresh = same and state.get('observation_stale') is False
                    selected_public = bool(getattr(host, 'public', False))
                    with self.lock:
                        delivery.update(instance_id=expected, last_observed_at=state.get('observed_at'),
                                        api_ready=bool(fresh and state.get('api_ready') is True),
                                        model_ready=bool(fresh and state.get('model_ready') is True))
                    if fresh and state.get('api_ready') is True and (not selected_public or state.get('public_api_ready') is True):
                        status = 'api_ready'
                        break
                    if same and state.get('phase') in {'failed', 'stopped', 'stopping'}:
                        status, error = 'failed', state.get('error') or '启动生命周期已结束'
                        break
                if record.get('status') == 'failed':
                    status, error = 'failed', record.get('error') or '启动未完成'
                    break
                if time.monotonic() >= deadline:
                    status = 'observation_expired'
                    break
                if time.monotonic() - last_publish >= 2:
                    update()
                self.cancel_startup.wait(min(.2, max(0, deadline - time.monotonic())))
        except KeyboardInterrupt:
            status, error = self._interrupt_delivery(delivery, publish=update)
            raise
        except Exception as caught:
            error = str(caught)
            self._print('启动交付观察未完成；后台所有权保留：', caught)
        finally:
            with self.lock:
                delivery.update(status=status, active=False, finished_at=time.time())
                if error:
                    delivery['error'] = self._clean(error)[:500]
            self._event('startup.delivery_finished', **dict(delivery))
            messages = {'api_ready': 'API 交付已确认；模型可继续下载，任务按实际就绪条件处理。',
                        'observation_expired': '前台观察到期；后台仍受管，面板可继续查询、取消启动或停止。',
                        'cancelled': '启动交付观察已取消；进程清理以停止回执为准。',
                        'failed': '启动未完成；控制器及已有后台保留，查看日志后处理。',
                        'interrupted': '启动已中断，所拥有进程清理已确认。',
                        'interrupted_cleanup_unconfirmed': '启动已中断，但清理未确认；继续保留所有权。'}
            self._print(messages.get(status, '前台观察已结束；继续通过面板查询实际状态。'))
            update()
        return dict(delivery)

    def _interrupt_delivery(self, delivery, *, publish=None):
        self.cancel_startup.set()
        with self.lock:
            delivery.update(status='cancelling', deadline_at=time.time() + 90)
        self._event('startup.delivery_interrupted', request_id=delivery['request_id'])
        self._print('收到停止本单元请求，正在取消启动并回收已拥有的进程。')
        try:
            if publish:
                publish()
        except BaseException:
            pass  # Another interrupt/display failure must not skip cleanup.
        try:
            self._operate('stop', {})
            with self.lock:
                delivery['cleanup_complete'] = True
            return 'interrupted', None
        except BaseException as caught:
            with self.lock:
                delivery['cleanup_complete'] = False
            self._print('启动中断后的清理尚未确认；保留控制器，请继续查询或停止：', caught)
            return 'interrupted_cleanup_unconfirmed', str(caught)

    def _operate(self, action, payload):
        if action == 'bootstrap':
            return self._bootstrap(payload)
        if action == 'preflight':
            return self._execute('preflight', {'STAGED_BF16': payload.get('staged', False),
                                             'MODEL_ALREADY_CACHED': payload.get('cached', False)})
        if action == 'install':
            self._configuration(payload)
            return self._execute('install')
        if action == 'start':
            self._configuration(payload)
            return self._execute('start', {'START_GPU_API': True, 'OPEN_PUBLIC_API': payload.get('public', False)})
        if action == 'generate':
            host = self.scope.get('startup_host')
            state = host.snapshot() if host else {}
            self.scope['PUBLIC_API_URL'] = state.get('public_url') if state.get('public_api_ready') else None
            return self._execute('generate', {'RUN_SMOKE_TEST': True,
                '验收接口': '本机API' if payload.get('endpoint') == 'local' else '临时公网',
                'RESUME_JOB_ID': payload.get('resume_job_id', '').strip(),
                'TEST_REFERENCE_FILE': payload.get('reference_file', ''), 'TEST_PROMPT': payload.get('prompt', '')})
        if action in {'stop', 'release'}:
            self._cancel_bootstrap_and_wait()
            self._stop_client_monitor()
            with self.lock:
                remaining = list(self.process_owners.values())
            for owner, _ in remaining:
                state = owner.terminate_all(timeout=10)
                self._event('client.process_cleanup', **state)
                if not state['cleanup_complete']:
                    raise RuntimeError('仍有客户端子进程，不能声明停止或释放完成。')
                with self.lock:
                    self.process_owners.pop(id(owner.process), None)
            self._execute('stop', {'STOP_SERVER': True, 'RELEASE_COLAB_RUNTIME': False})
            if action == 'release':
                self._event('runtime.release_requested')
                if self.release_callback:
                    self.release_callback()
                else:
                    from google.colab import runtime
                    runtime.unassign()
            return
        if action == 'control':
            host = self.scope.get('startup_host')
            if not host:
                raise RuntimeError('尚未启动总管理。')
            command = payload.get('action')
            if command == 'query':
                if not payload.get('command_id', '').strip():
                    raise ValueError('先填写命令 ID。')
                result = host.command(payload['command_id'].strip())
            elif command in {'drain', 'resume', 'stop_model', 'start_model', 'stop_api', 'start_api'}:
                result = host.control(command, force=payload.get('force', False))
            else:
                raise ValueError('组件控制动作无效。')
            self._print(json.dumps(result, ensure_ascii=False))
            return result
        raise ValueError('未知操作。')

    def _worker(self, record, lane, action, payload):
        with self.lock:
            record.update(status='running', started_at=time.time(), thread_id=threading.get_ident())
        self._event('operation.started', action=action, request_id=record['request_id'])
        try:
            result = self._operate(action, payload)
            if action == 'bootstrap':
                self._check_startup_cancelled()
            with self.lock:
                record.update(status='succeeded', finished_at=time.time())
                if isinstance(result, dict):
                    record['command'] = {k: result[k] for k in ('command_id', 'status') if k in result}
                message = ('已发起平台释放；是否完成以 Colab 断开为准。' if action == 'release'
                           else '操作已完成。' if action != 'control' else '命令回执已记录；接收不等于执行完成。')
                self.notice = None
                self._print(ACTION_TITLES[action], message)
        except BaseException as error:
            cancelled = action == 'bootstrap' and self.cancel_startup.is_set()
            if action == 'bootstrap':
                self._startup_stage('cancelled' if cancelled else 'failed')
            self._print(str(error) if cancelled else traceback.format_exc())
            with self.lock:
                record.update(status='cancelled' if cancelled else 'failed', finished_at=time.time(), error=self._clean(error)[:500])
                if not cancelled:
                    self.notice = {'id': record['request_id'], 'kind': 'error', 'title': ACTION_TITLES[action] + '未完成',
                                   'message': self._clean(error)[:500]}
        finally:
            self._event('operation.finished', action=action, request_id=record['request_id'], status=record['status'])
            with self.lock:
                self.active.pop(lane, None)
                self.processes.pop(threading.get_ident(), None)
                self.finished[record['request_id']].set()

    def submit(self, action, payload):
        request_id = payload.get('request_id')
        if not request_id or len(request_id) > 128:
            raise ValueError('操作必须携带唯一 request_id。')
        fingerprint = hashlib.sha256(json.dumps([action, payload], sort_keys=True).encode()).hexdigest()
        lane = 'generate' if action == 'generate' else 'control' if action in {'stop', 'release', 'control'} else 'setup'
        with self.lock:
            if self.closed:
                raise RuntimeError('控制台已关闭。')
            previous = self.operations.get(request_id)
            if previous:
                if previous['fingerprint'] != fingerprint:
                    raise ValueError('同一请求 ID 的内容发生变化。')
                return dict(previous)
            setup = self.active.get('setup')
            setup_allows_control = setup and setup['action'] == 'bootstrap' and (
                action in {'stop', 'release'} or self.startup_stage in {'awaiting_api', 'awaiting_model', 'ready'})
            if lane in self.active or (setup and not setup_allows_control) or (lane == 'setup' and self.active):
                raise RuntimeError('相关操作尚在执行；可继续观察日志，不重复提交。')
            if action == 'bootstrap':
                self.cancel_startup.clear()
                self.notice = None
            record = {'request_id': request_id, 'action': action, 'status': 'accepted',
                      'accepted_at': time.time(), 'fingerprint': fingerprint}
            self.operations[request_id] = record
            self.active[lane] = record
            self.finished[request_id] = threading.Event()
            self._event('operation.accepted', action=action, request_id=request_id, fingerprint=fingerprint)
            self.executor.submit(self._worker, record, lane, action, payload)
            return dict(record)

    def _connection_context(self, state=None):
        host = self.scope.get('startup_host')
        with self.lock:
            preparing = (self.startup_stage in {'preflight', 'install', 'start'}
                         and any(r['action'] == 'bootstrap' for r in self.active.values()))
        if preparing:
            host, state = None, {}
        if state is None:
            state = host.snapshot() if host else {}
        if state.get('instance_id') and state['instance_id'] != getattr(host, 'instance_id', None):
            state = {}  # A read raced with replacement of the owned lifecycle.
        key = getattr(host, 'api_key', None) or self.scope.get('BOOTSTRAP_API_KEY')
        public = bool(state.get('public_requested', self.scope.get('BOOTSTRAP_PUBLIC', False)))
        fresh = not state.get('observation_stale', False)
        api_ready = state.get('api_ready') is True and fresh
        public_ready = api_ready and state.get('public_api_ready') is True
        local = 'http://127.0.0.1:' + str(getattr(host, 'port', 8000)) if api_ready else None
        candidate = state.get('public_url') if public else None
        url = candidate if public_ready else local if not public else None
        connection = {'url': url, 'candidate_url': candidate, 'local_url': local,
                      'api_ready': api_ready, 'public_requested': public,
                      'public_api_ready': public_ready,
                      'model_ready': state.get('model_ready') is True and fresh,
                      'observation_stale': not fresh, 'key_available': bool(key),
                      'credential_id': hashlib.sha256(key.encode()).hexdigest()[:16] if key else None,
                      'instance_id': getattr(host, 'instance_id', None)}
        return connection, key

    def _downloads(self, state):
        progress = state.get('downloads')
        if isinstance(progress, dict) and progress.get('schema_version') == 1:
            return progress
        items = {}
        for name, variable in (('model', None), ('acceleration', 'ACCELERATION_LORA'), ('style', 'STYLE_LORA')):
            selection = self.requested_options.get(name, self.scope.get(variable, 'none')) if variable else None
            phase = 'waiting' if name == 'model' or selection != 'none' else 'disabled'
            items[name] = {'phase': phase, 'bytes_completed': 0, 'bytes_total': None, 'percent': None,
                           'files_completed': 0, 'files_total': None, 'active_files': 0,
                           'scope': 'observed_transfers' if name == 'model' else 'asset',
                           'detail': '等待模型容器提供真实进度' if phase == 'waiting' else '未启用'}
        return {'schema_version': 1, 'model_instance_id': None, 'observed_at': None,
                'observation_stale': False, 'model_ready': False, 'lifecycle': 'not_started', 'items': items}

    def snapshot(self, source='diagnostics', *, observe_host=True):
        host = self.scope.get('startup_host')
        for key in self._secrets():
            if key and key not in ui._known_secrets:
                ui._known_secrets = (*ui._known_secrets[-4:], key)
        try:
            # Mounting the panel must not depend on an API/model RPC response.
            # The browser requests a fresh observation after the cell returns.
            state = (host.snapshot() if observe_host else {'observation_stale': True}) if host else {}
        except Exception as error:
            state = {'observation_stale': True, 'phase': 'unavailable'}
            self._print('状态读取失败：', error)
        safe = {k: state[k] for k in SAFE_STATE if k in state}
        summary = public_condition_summary(state.get('condition_summary'), state.get('instance_id'))
        if summary is not None:
            safe['condition_summary'] = summary
        execution = public_execution_summary(state.get('execution_summary'), state.get('instance_id'))
        if execution is not None:
            safe['execution_summary'] = execution
        if state.get('error'):
            safe['error'] = self._clean(state['error'])[:300]
        if source == 'events':
            text = json.dumps(host.events(limit=35), ensure_ascii=False, indent=2) if host else '尚无运行实例。'
        elif source in LOG_NAMES:
            text = ui.log_tail(getattr(host, LOG_NAMES[source]), 16000) if host else '尚无运行实例。'
        elif source == 'install':
            text = ui.log_tail(self.directory / 'install.log', 16000)
        elif source == 'preflight':
            text = ui.log_tail(self.directory / 'preflight.log', 16000)
        elif source == 'state':
            text = json.dumps(self._sanitize(safe), ensure_ascii=False, indent=2, sort_keys=True) if host else '尚无运行实例。'
        elif source == 'smoke':
            smoke_log = self.scope.get('smoke_log')
            text = (ui.log_tail(smoke_log, 16000) if smoke_log else '') or '暂无生成验收日志。'
        else:
            text = ui.log_tail(self.log_path, 10000) + '\n' + ui.last_diagnostics()[-4000:]
            if summary:
                text += '\n调度观察（只读；历史决定不代表当前许可）：\n' + json.dumps(summary, ensure_ascii=False, sort_keys=True)
            if execution:
                text += '\n首个执行中任务（有界摘要；阶段回执是决策历史）：\n' + json.dumps(execution, ensure_ascii=False, sort_keys=True)
        with self.lock:
            active = [{k: v for k, v in record.items() if k not in {'fingerprint', 'thread_id'}} for record in self.active.values()]
            notice = dict(self.notice) if self.notice else None
        resource = self.scope.get('RESOURCE_REPORT') or {}
        bundle = self.scope.get('bundle')
        connection, _ = self._connection_context(state)
        return self._sanitize({'state': safe, 'resources': {k: resource[k] for k in ('gpu', 'available_ram_gib', 'residency') if k in resource},
                'connection': connection,
                'downloads': self._downloads(state),
                'ui': {'connected': not self.closed, 'closed': self.closed, 'busy': bool(active),
                       'operation': ', '.join(ACTION_TITLES.get(r['action'], r['action']) for r in active),
                       'operations': active, 'preflight_ok': self.scope.get('PREFLIGHT_OK', False),
                       'install_ok': self.scope.get('INSTALL_OK', False),
                       'startup_stage': self.startup_stage,
                       'startup_delivery': dict(self.startup_delivery) if self.startup_delivery else None,
                       'bootstrap_running': any(r['action'] == 'bootstrap' for r in active),
                       'connection_available': connection['key_available']},
                'logs': {'text': self._clean(text)[-16000:], 'source': source, 'observed_at': time.time()},
                'notice': notice, 'result': {'available': bool(bundle and Path(bundle).is_file())}})

    def _download(self, action):
        if action == 'export_connection':
            connection, key = self._connection_context()
            if not key:
                raise RuntimeError('密钥尚未生成；先执行一键启动。')
            data = json.dumps({**connection, 'api_key': key,
                               'note': 'url 为空时尚不可连接；API 可达后重新导出。模型未就绪时任务只排队。'}, ensure_ascii=False).encode()
            name, mime = 'api-connection.private.json', 'application/json'
        else:
            path = self.scope.get('bundle')
            if not path or not Path(path).is_file():
                raise RuntimeError('本次尚无已验证结果包。')
            path = Path(path)
            if path.stat().st_size > 32 * 1024**2:
                raise ValueError('结果包超过内联下载上限；请使用 Colab 文件栏下载。')
            data, name, mime = path.read_bytes(), path.name, 'application/zip'
        return {'name': name, 'mime': mime, 'data_base64': base64.b64encode(data).decode('ascii')}

    def dispatch(self, action, payload=None):
        try:
            payload = validated(payload or {})
            if self.closed:
                return {'ui': {'connected': False, 'closed': True}, 'error': '控制台已关闭。'}
            if action in {'snapshot', 'refresh_board'}:
                return self.snapshot(payload.get('source', 'diagnostics'))
            if action == 'reveal_connection':
                result = self.snapshot(payload.get('source', 'diagnostics'))
                connection, key = self._connection_context()
                if not key:
                    raise RuntimeError('密钥尚未生成；先执行一键启动。')
                if payload.get('credential_id') != connection['credential_id']:
                    raise RuntimeError('连接凭据已改变；请刷新看板后重新显示或复制。')
                result['connection'] = self._sanitize(connection)
                # This dedicated user-triggered response is never embedded into saved HTML or logs.
                result['connection_secret'] = {'api_key': key, 'credential_id': connection['credential_id']}
                self._event('credential.revealed', credential_id=connection['credential_id'])
                return result
            if action in {'export_connection', 'download_result'}:
                result = self.snapshot(payload.get('source', 'diagnostics'))
                result['download'] = self._download(action)
                self._event('download.prepared', action=action)
                return result
            if action not in ACTION_TITLES:
                raise ValueError('未知控制动作。')
            receipt = self.submit(action, payload)
            return {'accepted': True, 'operation': {k: v for k, v in receipt.items() if k not in {'fingerprint', 'thread_id'}}}
        except Exception as error:
            return {'error': self._clean(error)[:500], 'notice': {'id': uuid.uuid4().hex,
                'kind': 'error', 'title': '操作未执行', 'message': self._clean(error)[:500]}}

    def close(self):
        with self.lock:
            if self.active or self.process_owners:
                raise RuntimeError('仍有受管操作，不能丢弃控制器。')
            self.closed = True
            self.epoch = None
        self.executor.shutdown(wait=True, cancel_futures=True)
        if ui._dashboard is self:
            ui._dashboard = None
        if getattr(ui, '_process_observer', None) is self:
            ui._process_observer = None
        self._event('console.closed')


def render_console(template, callback_name='', initial=None):
    state = initial or {'ui': {'connected': False, 'offline': True}, 'state': {}, 'resources': {},
                        'logs': {'text': '尚未连接控制台。运行本单元后，日志在这里持续更新。', 'source': 'diagnostics'}}
    safe_json = lambda data: json.dumps(data, ensure_ascii=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    return template.replace('__CALLBACK_NAME__', safe_json(callback_name)).replace('__INITIAL_STATE__', safe_json(state))


def mount_console(directory, programs, scope=None, *, auto_start=False, bootstrap_payload=None,
                  foreground_bootstrap_timeout=None):
    """Display the unchanged panel, optionally holding the cell until API delivery.

    The owner still runs independent startup branches concurrently. The finite
    foreground observer never waits for the model or treats expiry as failure.
    """
    from google.colab import output
    from IPython.display import HTML, JSON, display
    scope = scope if scope is not None else {}
    controller = scope.get('h3_console')
    created = False
    if not isinstance(controller, ConsoleController) or controller.closed:
        controller = ConsoleController(directory, programs, scope)
        scope['h3_console'] = controller
        created = True
    controller.epoch = uuid.uuid4().hex
    epoch = controller.epoch
    name = 'h3.console.' + epoch
    def callback(action, payload=None):
        if controller.epoch != epoch:
            return JSON({'ui': {'connected': False, 'closed': True}, 'error': '旧面板已停用。'})
        return JSON(controller.dispatch(action, payload))
    output.register_callback(name, callback)
    template = (Path(directory) / 'console.html').read_text(encoding='utf-8')
    request_id = 'mount-' + epoch
    foreground = created and auto_start and foreground_bootstrap_timeout is not None
    active_delivery = lambda: foreground and bool(controller.startup_delivery and controller.startup_delivery.get('active'))
    try:
        if foreground:
            controller.begin_startup_delivery(request_id, foreground_bootstrap_timeout)
        initial = HTML(render_console(template, name, controller.snapshot(observe_host=False)))
        handle = display(initial, display_id=True) if foreground else display(initial)
        if created and auto_start:
            payload = {'public': True, **(bootstrap_payload or {}), 'request_id': request_id}
            receipt = controller.dispatch('bootstrap', payload)
            if foreground:
                def publish():
                    if callable(getattr(handle, 'update', None)):
                        handle.update(HTML(render_console(template, name, controller.snapshot())))
                if receipt.get('accepted') is not True:
                    with controller.lock:
                        controller.startup_delivery.update(active=False, status='failed',
                            error=receipt.get('error', '启动未被接收'), finished_at=time.time())
                    publish()
                else:
                    controller.wait_startup_delivery(request_id, publish=publish)
    except KeyboardInterrupt:
        # Cover the display/dispatch window before wait_startup_delivery owns
        # interrupts. Its own handler sets active=False, preventing double stop.
        if active_delivery():
            status, error = controller._interrupt_delivery(controller.startup_delivery)
            controller.startup_delivery.update(status=status, error=error, active=False, finished_at=time.time())
            controller._event('startup.delivery_finished', **dict(controller.startup_delivery))
        raise
    finally:
        if active_delivery():
            controller.startup_delivery.update(status='observation_failed', active=False, finished_at=time.time())
            controller._event('startup.delivery_finished', **dict(controller.startup_delivery))
    return controller
