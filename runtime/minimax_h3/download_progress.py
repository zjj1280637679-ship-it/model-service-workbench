"""Observed Hugging Face transfer progress, separate from model readiness.

No extra downloads, repository scans, or scheduling are introduced. An owned
load temporarily supplies a progress-bar observer to HF's existing HTTP/Xet
transfer functions. Unsupported signatures degrade to unknown progress. The
download algorithm, retries, validation, caching and worker counts stay in HF.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import functools
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import threading
import time
import uuid

MAX_STATE_BYTES = 256 * 1024
PHASES = {'waiting', 'downloading', 'verifying', 'loading', 'loaded', 'disabled', 'failed'}
_ACTIVE = None
_RECORDER = None
_HOOK_LOCK = threading.RLock()


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def read_progress(path, expected_instance_id=None):
    unavailable = {'schema_version': 1, 'model_instance_id': expected_instance_id,
                   'observed_at': None, 'observation_stale': True, 'items': {}}
    try:
        path = Path(path)
        if path.stat().st_size > MAX_STATE_BYTES:
            raise ValueError('oversized')
        with path.open('r', encoding='utf-8') as stream:
            raw = stream.read(MAX_STATE_BYTES + 1)
        if len(raw.encode('utf-8')) > MAX_STATE_BYTES:
            raise ValueError('oversized')
        data = json.loads(raw)
        if data.get('schema_version') != 1 or not isinstance(data.get('items'), dict):
            raise ValueError('invalid_schema')
        if expected_instance_id is not None and data.get('model_instance_id') != expected_instance_id:
            raise ValueError('generation_mismatch')
        if not _number(data.get('observed_at')):
            raise ValueError('invalid_timestamp')
        items = {}
        for key in ('model', 'acceleration', 'style'):
            item = data['items'].get(key)
            if item is None:
                continue
            if not isinstance(item, dict) or item.get('phase') not in PHASES:
                raise ValueError('invalid_item')
            clean = {'phase': item['phase']}
            for field in ('bytes_completed', 'bytes_total', 'percent', 'files_completed', 'files_total', 'active_files'):
                value = item.get(field)
                if value is not None and not _number(value):
                    raise ValueError('invalid_counter')
                if field == 'percent' and value is not None and value > 100:
                    raise ValueError('invalid_percent')
                clean[field] = value
            for field in ('scope', 'detail'):
                clean[field] = str(item.get(field, ''))[:240]
            clean['cache_hit'] = item.get('cache_hit') if type(item.get('cache_hit')) is bool else None
            items[key] = clean
        return {'schema_version': 1, 'model_instance_id': data.get('model_instance_id'),
                'observed_at': data['observed_at'], 'observation_stale': False, 'items': items}
    except (OSError, ValueError, TypeError, AttributeError) as error:
        return dict(unavailable, error=str(error) if isinstance(error, ValueError) and str(error) in {
            'oversized', 'invalid_schema', 'generation_mismatch', 'invalid_timestamp', 'invalid_item',
            'invalid_counter', 'invalid_percent'} else 'progress_unavailable')


class DownloadProgress:
    def __init__(self, path=None, *, instance_id=None, settings=None, interval=.5, clock=time.monotonic):
        self.path = Path(path) if path else None
        self.instance_id = instance_id
        self.lock = threading.RLock()
        self.interval, self.clock, self.last_write = interval, clock, -math.inf
        self.category = 'model'
        self.transfers = {}
        self.transfer_identities = {}
        self.measurement_available = None
        self.items = {}
        self._phase_details = {}
        for name in ('model', 'acceleration', 'style'):
            enabled = name == 'model' or (settings is not None and getattr(settings, name, 'none') != 'none')
            self.items[name] = dict(phase='waiting' if enabled else 'disabled', bytes_completed=0,
                bytes_total=None, percent=None, files_completed=0, files_total=None,
                active_files=0, scope='observed_transfers' if name == 'model' else 'asset',
                detail='尚未开始' if enabled else '未启用', cache_hit=None)
        self._write(force=True)

    def _document(self):
        return {'schema_version': 1, 'model_instance_id': self.instance_id,
                'observed_at': time.time(), 'observation_stale': False, 'items': deepcopy(self.items)}

    def snapshot(self):
        with self.lock:
            return self._document()

    def _write(self, *, force=False, event=None):
        if not self.path:
            return
        now = self.clock()
        if not force and now - self.last_write < self.interval:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_name(self.path.name + '.' + uuid.uuid4().hex + '.tmp')
            try:
                temporary.write_text(json.dumps(self._document(), ensure_ascii=False), encoding='utf-8')
                os.replace(temporary, self.path)
            finally:
                temporary.unlink(missing_ok=True)
            if event:
                with self.path.with_suffix('.events.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps({'observed_at': time.time(), 'model_instance_id': self.instance_id,
                                             **event}, ensure_ascii=False) + '\n')
            self.last_write = now
        except OSError:
            # UI telemetry must never abort or retry an otherwise valid download.
            pass

    def phase(self, category, phase, detail='', *, total=None, cache_hit=None):
        if phase not in PHASES or category not in self.items:
            raise ValueError('Invalid progress phase/category')
        with self.lock:
            item = self.items[category]
            if self.measurement_available is False and phase in {'loading', 'loaded'}:
                detail += '；当前下载库未提供兼容进度回调'
            item.update(phase=phase, detail=detail[:240])
            self._phase_details[category] = (phase, detail[:240])
            if total is not None and category != 'model':
                item.update(bytes_total=int(total), files_total=1)
            if cache_hit is not None:
                item['cache_hit'] = bool(cache_hit)
            if phase in {'verifying', 'loaded'} and total is not None and category != 'model':
                item.update(bytes_completed=int(total), percent=100, files_completed=1)
            self._write(force=True, event={'type': 'phase', 'category': category, 'phase': phase})

    def begin_transfer(self, category, total=None, initial=0, identity=None):
        with self.lock:
            identity = (category, identity) if identity is not None else None
            transfer = self.transfer_identities.get(identity) if identity is not None else None
            if transfer is None:
                transfer = uuid.uuid4().hex
                self.transfers[transfer] = {'category': category, 'n': max(0, initial),
                                           'total': total, 'finished': False, 'active': 0}
                if identity is not None:
                    self.transfer_identities[identity] = transfer
            entry = self.transfers[transfer]
            entry.update(n=max(entry['n'], initial), finished=False, active=entry['active'] + 1)
            self._update_item(category)
            self._write(force=True, event={'type': 'transfer.started', 'category': category,
                                          'transfer_id': transfer, 'total': total, 'initial': initial})
        return transfer

    def update_transfer(self, transfer, completed):
        with self.lock:
            entry = self.transfers[transfer]
            # Callback retries cannot decrease or double-count a transfer's absolute progress.
            entry['n'] = max(entry['n'], completed)
            self._update_item(entry['category'])
            self._write()

    def finish_transfer(self, transfer, *, success):
        with self.lock:
            entry = self.transfers[transfer]
            entry['active'] = max(0, entry['active'] - 1)
            entry.update(finished=entry['active'] == 0, success=success)
            self._update_item(entry['category'])
            self._write(force=True, event={'type': 'transfer.finished', 'category': entry['category'],
                                          'transfer_id': transfer, 'success': success})

    def _update_item(self, category):
        entries = [e for e in self.transfers.values() if e['category'] == category]
        item = self.items[category]
        item['active_files'] = sum(not e['finished'] for e in entries)
        item['files_completed'] = sum(e.get('success') is True and e['finished'] for e in entries)
        value = sum(e['n'] for e in entries)
        item['bytes_completed'] = max(item['bytes_completed'], value)
        if category != 'model' and item['bytes_total']:
            item['bytes_completed'] = min(item['bytes_total'], item['bytes_completed'])
            item['percent'] = round(100 * item['bytes_completed'] / item['bytes_total'], 1)
        if item['active_files']:
            item.update(phase='downloading', detail='已观察文件量（含续传与重建）；整模型总量未知' if category == 'model'
                        else '正在下载/重建已选 LoRA 文件')
        elif item['phase'] == 'downloading':
            phase, detail = self._phase_details.get(category, ('loading', '继续加载；下载完成不代表模型就绪'))
            item.update(phase=phase, detail=detail)


class _ProgressBar:
    """Protocol proxy used by HF HTTP and Xet reconstruction callbacks."""
    def __init__(self, recorder, transfer, *, initial=0, total=None, delegate=None):
        self.recorder, self.transfer, self.delegate = recorder, transfer, delegate
        self.n, self.total = initial, total
        self.disable = False
        self.lock = threading.RLock()

    def update(self, amount=1):
        with self.lock:
            self.n = max(0, self.n + (amount or 0))
            try:
                self.recorder.update_transfer(self.transfer, self.n)
            except Exception:
                pass  # Observation cannot fail an otherwise valid payload transfer.
            if self.delegate is not None:
                return self.delegate.update(amount)

    def __getattr__(self, name):
        if self.delegate is not None:
            return getattr(self.delegate, name)
        if name == 'format_dict':
            return {'n': self.n, 'total': self.total, 'rate': None}
        if name in {'refresh', 'set_description', 'set_description_str', 'set_postfix_str', 'close'}:
            return lambda *a, **kw: None
        raise AttributeError(name)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _wrap_transfer(original, recorder):
    try:
        signature = inspect.signature(original)
    except (ValueError, TypeError):
        return original
    if '_tqdm_bar' not in signature.parameters:
        return original

    @functools.wraps(original)
    def observed(*args, **kwargs):
        bound = signature.bind_partial(*args, **kwargs)
        existing = bound.arguments.get('_tqdm_bar')
        if isinstance(existing, _ProgressBar):
            return original(*args, **kwargs)  # HF's nested HTTP retry owns the same byte counter.
        total, initial = bound.arguments.get('expected_size'), bound.arguments.get('resume_size', 0)
        total = int(total) if _number(total) else None
        initial = int(initial) if _number(initial) else 0
        # Hash an owned temporary-file identity, never persist a signed URL or local path.
        path = bound.arguments.get('incomplete_path') or bound.arguments.get('downloaded_file')
        if path is None:
            path = getattr(bound.arguments.get('temp_file'), 'name', None)
        identity = hashlib.sha256(os.fsencode(path)).hexdigest() if isinstance(path, (str, bytes, os.PathLike)) else None
        transfer = recorder.begin_transfer(recorder.category, total, initial, identity=identity)
        bound.arguments['_tqdm_bar'] = _ProgressBar(recorder, transfer, initial=initial, total=total, delegate=existing)
        try:
            result = original(*bound.args, **bound.kwargs)
        except BaseException:
            recorder.finish_transfer(transfer, success=False)
            raise
        recorder.finish_transfer(transfer, success=True)
        return result
    return observed


@contextmanager
def observe_hub_downloads(recorder):
    """A model process owns these hooks; no thread pool or concurrency limit changes."""
    global _ACTIVE
    with _HOOK_LOCK:
        if _ACTIVE is recorder:
            yield recorder
            return
        previous, patched = _ACTIVE, []
        _ACTIVE = recorder
        try:
            try:
                module = importlib.import_module('huggingface_hub.file_download')
                for name in ('http_get', 'xet_get'):
                    original = getattr(module, name, None)
                    if callable(original):
                        wrapper = _wrap_transfer(original, recorder)
                        if wrapper is not original:
                            setattr(module, name, wrapper)
                            patched.append((module, name, original, wrapper))
            except (ImportError, AttributeError, TypeError, ValueError):
                pass
            recorder.measurement_available = bool(patched)
            yield recorder
        finally:
            for module, name, original, wrapper in reversed(patched):
                if getattr(module, name) is wrapper:
                    setattr(module, name, original)
            _ACTIVE = previous


def current_progress():
    return _ACTIVE or _RECORDER


@contextmanager
def adapter_progress(category, spec):
    recorder = current_progress()
    if recorder is None:
        yield None
        return
    with observe_hub_downloads(recorder):
        previous = recorder.category
        recorder.category = category
        try:
            recorder.phase(category, 'loading', '准备所选 LoRA', total=spec.size)
            yield recorder
        except BaseException:
            recorder.phase(category, 'failed', 'LoRA 下载、校验或加载失败；详见模型日志')
            raise
        finally:
            recorder.category = previous


def tracked_model_load(function):
    @functools.wraps(function)
    def tracked(self, *args, **kwargs):
        global _RECORDER
        path = os.environ.get('H3_DOWNLOAD_PROGRESS_PATH')
        if not path:
            return function(self, *args, **kwargs)
        recorder = DownloadProgress(path, instance_id=os.environ.get('H3_MODEL_INSTANCE_ID'),
                                    settings=getattr(self, 'adapter_settings', None))
        _RECORDER = recorder
        with observe_hub_downloads(recorder):
            recorder.phase('model', 'loading', '检查缓存并加载所需组件；整模型总量未知')
            try:
                result = function(self, *args, **kwargs)
            except BaseException:
                recorder.phase('model', 'failed', '下载或加载失败；详见模型日志')
                raise
            no_transfers = not any(e['category'] == 'model' for e in recorder.transfers.values())
            recorder.phase('model', 'loaded', '组件加载完成；等待容器就绪检查' +
                           ('。未观察到传输：缓存命中或当前库未提供回调' if no_transfers else ''))
            return result
    return tracked


def tracked_adapter_application(function):
    @functools.wraps(function)
    def tracked(pipe, settings, *args, **kwargs):
        try:
            return function(pipe, settings, *args, **kwargs)
        except BaseException:
            recorder = current_progress()
            if recorder:
                for category in ('acceleration', 'style'):
                    if getattr(settings, category, 'none') != 'none':
                        recorder.phase(category, 'failed', 'LoRA 下载、验证或组合加载失败；详见模型日志')
            raise
    return tracked
