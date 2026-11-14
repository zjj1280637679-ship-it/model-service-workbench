"""CPU callback contracts only; no Hub requests, weights, or GPU imports."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest

import download_progress as progress


@pytest.fixture(autouse=True)
def isolate_monitor(monkeypatch):
    monkeypatch.setattr(progress, '_ACTIVE', None)
    monkeypatch.setattr(progress, '_RECORDER', None)


def recorder(tmp_path, **kw):
    return progress.DownloadProgress(tmp_path / 'download.json', instance_id='generation-a', **kw)


def fake_hub(monkeypatch, operation):
    package = ModuleType('huggingface_hub')
    module = ModuleType('huggingface_hub.file_download')
    module.http_get = module.xet_get = operation
    package.file_download = module
    monkeypatch.setitem(sys.modules, 'huggingface_hub', package)
    monkeypatch.setitem(sys.modules, 'huggingface_hub.file_download', module)
    return package, module


def test_reader_missing_oversized_malformed_and_wrong_generation(tmp_path):
    path = tmp_path / 'state.json'
    assert progress.read_progress(path, 'a')['observation_stale']
    path.write_text('x' * (progress.MAX_STATE_BYTES + 1))
    assert progress.read_progress(path, 'a')['error'] == 'oversized'
    path.write_text('{bad')
    assert progress.read_progress(path, 'a')['items'] == {}
    monitor = recorder(tmp_path)
    assert progress.read_progress(monitor.path, 'other')['error'] == 'generation_mismatch'
    assert progress.read_progress(monitor.path, 'generation-a')['items']['model']['phase'] == 'waiting'


@pytest.mark.parametrize('field,value', [('percent', 101), ('bytes_completed', -1), ('files_total', True)])
def test_reader_rejects_untrusted_counters(tmp_path, field, value):
    monitor = recorder(tmp_path)
    data = monitor.snapshot()
    data['items']['model'][field] = value
    monitor.path.write_text(json.dumps(data))
    result = progress.read_progress(monitor.path, 'generation-a')
    assert result['observation_stale'] and result['items'] == {}


def test_parallel_file_callbacks_aggregate_without_whole_model_percentage(tmp_path):
    monitor = recorder(tmp_path)
    barrier = threading.Barrier(4)
    def transfer(_):
        identity = monitor.begin_transfer('model', total=100, initial=10)
        barrier.wait(timeout=3)
        for n in range(20, 101, 10):
            monitor.update_transfer(identity, n)
        monitor.finish_transfer(identity, success=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(transfer, range(4)))
    result = progress.read_progress(monitor.path, 'generation-a')['items']['model']
    assert result['bytes_completed'] == 400 and result['files_completed'] == 4
    assert result['active_files'] == 0 and result['bytes_total'] is None and result['percent'] is None
    assert result['phase'] != 'loaded'


def test_updates_are_throttled_but_stage_boundaries_flush(tmp_path):
    clock = [0.0]
    monitor = recorder(tmp_path, clock=lambda: clock[0])
    identity = monitor.begin_transfer('model', total=100)
    before = monitor.path.read_text(encoding="utf-8")
    monitor.update_transfer(identity, 10)
    assert monitor.path.read_text(encoding="utf-8") == before
    clock[0] = .6
    monitor.update_transfer(identity, 20)
    assert progress.read_progress(monitor.path)['items']['model']['bytes_completed'] == 20
    monitor.phase('model', 'loaded', 'Actual loader returned')
    assert progress.read_progress(monitor.path)['items']['model']['phase'] == 'loaded'
    assert 'model_ready' not in progress.read_progress(monitor.path)


def test_hook_restores_original_and_nested_http_retry_reuses_counter(tmp_path, monkeypatch):
    monitor = recorder(tmp_path)
    calls = []
    module = None
    def original(url, temp_file, *, expected_size=None, resume_size=0, _tqdm_bar=None):
        calls.append((url, resume_size))
        _tqdm_bar.update(40 if resume_size == 0 else 60)
        if resume_size == 0:
            return module.http_get(url, temp_file, expected_size=100, resume_size=40, _tqdm_bar=_tqdm_bar)
        return 'original-result'
    _, module = fake_hub(monkeypatch, original)
    with progress.observe_hub_downloads(monitor):
        assert module.http_get('private-signed-url', object(), expected_size=100) == 'original-result'
        assert module.http_get is not original
    assert module.http_get is original and module.xet_get is original
    assert len(monitor.transfers) == 1
    assert monitor.items['model']['bytes_completed'] == 100
    assert monitor.items['model']['files_completed'] == 1
    assert 'private-signed-url' not in monitor.path.read_text(encoding="utf-8")
    assert 'private-signed-url' not in monitor.path.with_suffix('.events.jsonl').read_text(encoding="utf-8")


def test_nested_lora_scope_does_not_reassign_existing_model_transfers(tmp_path, monkeypatch):
    monitor = recorder(tmp_path, settings=SimpleNamespace(acceleration='turbo', style='none'))
    pending = []
    def original(*, expected_size=None, _tqdm_bar=None):
        pending.append(_tqdm_bar)
        _tqdm_bar.update(expected_size)
    _, module = fake_hub(monkeypatch, original)
    with progress.observe_hub_downloads(monitor):
        module.xet_get(expected_size=50)
        with progress.adapter_progress('acceleration', SimpleNamespace(size=200)):
            module.xet_get(expected_size=200)
        module.xet_get(expected_size=25)
    assert monitor.items['model']['bytes_completed'] == 75
    assert monitor.items['acceleration']['bytes_completed'] == 200
    assert monitor.items['acceleration']['percent'] == 100
    assert monitor.items['model']['percent'] is None
    assert monitor.items['style']['phase'] == 'disabled'


def test_unsupported_hook_signature_keeps_original_download_and_unknown_progress(tmp_path, monkeypatch):
    monitor = recorder(tmp_path)
    calls = []
    def original(url):
        calls.append(url)
        return 'unchanged'
    _, module = fake_hub(monkeypatch, original)
    with progress.observe_hub_downloads(monitor):
        assert module.http_get is original and module.http_get('source') == 'unchanged'
    assert calls == ['source'] and not monitor.transfers
    assert monitor.items['model']['percent'] is None


def test_original_failure_propagates_and_hook_is_restored(tmp_path, monkeypatch):
    monitor = recorder(tmp_path)
    def original(*, _tqdm_bar=None):
        _tqdm_bar.update(8)
        raise RuntimeError('network failure')
    _, module = fake_hub(monkeypatch, original)
    with pytest.raises(RuntimeError, match='network failure'):
        with progress.observe_hub_downloads(monitor):
            module.http_get()
    assert module.http_get is original and monitor.items['model']['files_completed'] == 0


def test_cached_lora_reports_cache_and_verification_not_network_transfer(tmp_path, monkeypatch):
    import adapter_config as adapters
    content = b'checked adapter fixture'
    path = tmp_path / 'adapter.safetensors'
    path.write_bytes(content)
    spec = adapters.AdapterSpec('fixture', 'fixture/repo', path.name, 'a' * 40,
                                hashlib.sha256(content).hexdigest(), len(content), 1)
    monitor = recorder(tmp_path, settings=SimpleNamespace(acceleration='none', style='fixture'))
    package, _ = fake_hub(monkeypatch, lambda **kw: None)
    package.hf_hub_download = lambda *a, **kw: str(path)
    package.try_to_load_from_cache = lambda *a, **kw: str(path)
    with progress.observe_hub_downloads(monitor), progress.adapter_progress('style', spec):
        assert adapters._download_verified(spec) == path
    item = monitor.items['style']
    assert item['cache_hit'] is True and item['phase'] == 'verifying'
    assert item['percent'] == 100 and item['bytes_completed'] == len(content)
    assert not monitor.transfers


def test_model_load_completion_is_loaded_not_ready_and_failure_is_visible(tmp_path, monkeypatch):
    monkeypatch.setenv('H3_DOWNLOAD_PROGRESS_PATH', str(tmp_path / 'progress.json'))
    monkeypatch.setenv('H3_MODEL_INSTANCE_ID', 'my-model-generation')
    fake_hub(monkeypatch, lambda **kw: None)
    class Backend:
        adapter_settings = SimpleNamespace(acceleration='none', style='none')
        @progress.tracked_model_load
        def load(self, fail=False):
            if fail:
                raise ValueError('failure')
            return 'loaded-components'
    assert Backend().load() == 'loaded-components'
    state = progress.read_progress(tmp_path / 'progress.json', 'my-model-generation')
    assert state['items']['model']['phase'] == 'loaded' and 'ready' not in state
    assert state['items']['model']['bytes_completed'] == 0
    assert state['items']['model']['cache_hit'] is None
    with pytest.raises(ValueError):
        Backend().load(fail=True)
    assert progress.read_progress(tmp_path / 'progress.json')['items']['model']['phase'] == 'failed'


def test_write_failure_does_not_abort_transfer(tmp_path, monkeypatch):
    monitor = recorder(tmp_path)
    def fail(*args, **kwargs):
        raise PermissionError('fixture')
    monkeypatch.setattr(progress.os, 'replace', fail)
    identity = monitor.begin_transfer('model', total=100)
    monitor.update_transfer(identity, 100)
    monitor.finish_transfer(identity, success=True)
    assert monitor.items['model']['bytes_completed'] == 100


def test_top_level_resume_reuses_file_identity_without_double_count(tmp_path, monkeypatch):
    monitor = recorder(tmp_path)
    def original(url, temp_file, *, expected_size=None, resume_size=0, _tqdm_bar=None):
        if resume_size == 0:
            _tqdm_bar.update(40)
            raise RuntimeError('transfer interrupted')
        _tqdm_bar.update(60)
    _, module = fake_hub(monkeypatch, original)
    temporary = SimpleNamespace(name=str(tmp_path / 'private-cache-file'))
    with progress.observe_hub_downloads(monitor):
        with pytest.raises(RuntimeError):
            module.http_get('signed-url-secret', temporary, expected_size=100)
        module.http_get('changed-signed-url-secret', temporary, expected_size=100, resume_size=40)
    assert len(monitor.transfers) == 1
    assert monitor.items['model']['bytes_completed'] == 100
    assert monitor.items['model']['files_completed'] == 1
    assert monitor.items['model']['active_files'] == 0
    evidence = monitor.path.read_text(encoding='utf-8') + monitor.path.with_suffix('.events.jsonl').read_text(encoding='utf-8')
    assert 'private-cache-file' not in evidence and 'signed-url-secret' not in evidence


def test_xet_reconstruction_retry_uses_same_identity(tmp_path, monkeypatch):
    monitor = recorder(tmp_path)
    def original(*, incomplete_path, expected_size=None, _tqdm_bar=None):
        _tqdm_bar.update(expected_size)
    _, module = fake_hub(monkeypatch, original)
    with progress.observe_hub_downloads(monitor):
        for _ in range(2):
            module.xet_get(incomplete_path=tmp_path / 'same-partial', expected_size=50)
    assert monitor.items['model']['bytes_completed'] == 50
    assert monitor.items['model']['files_completed'] == 1


def test_same_transfer_parallel_callbacks_are_exact(tmp_path):
    monitor = recorder(tmp_path)
    identity = monitor.begin_transfer('model', total=4000)
    bar = progress._ProgressBar(monitor, identity, total=4000)
    def emit(_):
        for _ in range(1000):
            bar.update(1)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(emit, range(4)))
    monitor.finish_transfer(identity, success=True)
    assert bar.n == 4000 and monitor.items['model']['bytes_completed'] == 4000


def test_observer_callback_failure_does_not_interrupt_original_download(tmp_path, monkeypatch):
    monitor = recorder(tmp_path)
    monkeypatch.setattr(monitor, 'update_transfer', lambda *a: (_ for _ in ()).throw(RuntimeError('observer-only')))
    def original(*, _tqdm_bar=None):
        _tqdm_bar.update(50)
        return 'download-still-succeeds'
    _, module = fake_hub(monkeypatch, original)
    with progress.observe_hub_downloads(monitor):
        assert module.http_get() == 'download-still-succeeds'
