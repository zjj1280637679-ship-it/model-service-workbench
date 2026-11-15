"""Finite publication counterexamples; all media/source bytes below are synthetic."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


exporter = load('condition_public_export_test', ROOT / 'research/export_condition_evidence.py')
fixtures = load('condition_public_fixture', ROOT / 'tests/test_condition_gate_analysis.py')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def declare_experiment(source):
    protocol = {'protocol_id': 'synthetic-protocol', 'status': 'planned_not_a_result'}
    write(source / 'protocol.json', protocol)
    trace = exporter.read_json(source / 'trace-complete.private.json')
    state = exporter.read_json(source / 'client/state.json')
    intervention = source / 'cloud/collection/api-query-update.json'
    declaration = {'schema_version': 1, 'experiment_id': state['run_id'],
        'protocol': {'id': 'synthetic-protocol', 'file': 'protocol.json', 'sha256': exporter.digest(source / 'protocol.json')},
        'baseline_commit': '1' * 40, 'source_revision': {'commit': '2' * 40, 'state': 'modified'},
        'runtime_changes': 'Synthetic fixture with explicit source declaration; no executed GPU claim.',
        'runtime_manifest_sha256': exporter.digest(source / 'cloud/collection/runtime-source-manifest.json'),
        'manager_instance_id': trace['manager_instance_id'], 'model_instance_id': trace['model_instance_id'],
        'interventions': ([{'kind': 'managed_api_replacement', 'evidence_sha256': exporter.digest(intervention)}] if intervention.exists() else [])}
    write(source / 'experiment.json', declaration)
    return declaration


def synthetic_run(tmp_path):
    runtime = tmp_path / 'runtime'
    runtime.mkdir()
    trace = fixtures.fixture(runtime)
    source = tmp_path / 'private-run'
    source.mkdir()
    manifest = {'scope': 'Synthetic CPU-only exporter fixture, no generated video or GPU claim',
                'parameters': {'seconds': 5, 'width': 512, 'height': 288, 'steps': 5,
                               'attention_backend': 'native', 'acceleration_lora': 'lightx2v-ref2va-4step'}, 'jobs': []}
    state = {'schema_version': 1, 'run_id': 'fixture-run', 'instance_id': 'manager-1',
             'state': 'completed', 'jobs': []}
    (source / 'client').mkdir()
    for name, job_id, disposition in [('first', 'a', 'retain'), ('second', 'b', 'retain'), ('third', 'cancel-job', 'cancel')]:
        reference = source / (name + '.png')
        reference.write_bytes(('synthetic reference ' + name).encode())
        manifest['jobs'].append({'name': name, 'reference_file': reference.name,
                                'seed': 100 + len(manifest['jobs']), 'disposition': disposition,
                                'prompt': 'Entirely synthetic illustration test ' + name})
        record = next(job for job in trace['jobs'] if job['id'] == job_id)
        row = {'name': name, 'job_id': job_id, 'disposition': disposition, 'state': record['status'],
               'attempt_id': record['attempt_id'], 'reference_sha256': exporter.digest(reference), 'cleanup_complete': True}
        if disposition == 'retain':
            video = source / 'client' / (job_id + '.mp4')
            video.write_bytes(('synthetic fixture bytes, not playable media ' + name).encode())
            sha = exporter.digest(video)
            row['download'] = {'verified': True, 'sha256': sha, 'expected_sha256': sha,
                               'bytes': video.stat().st_size, 'state': 'verified', 'media': {'full_decode_passed': True}}
            record['artifact'] = {'bytes': video.stat().st_size, 'sha256': sha,
                                  'path': '/content/private-output/' + job_id + '.mp4',
                                  'media': {'full_decode_passed': True}}
        state['jobs'].append(row)
    write(source / 'content-manifest.json', manifest)
    state['manifest_sha256'] = exporter.digest(source / 'content-manifest.json')
    write(source / 'client/state.json', state)
    write(source / 'trace-complete.private.json', trace)
    write(source / 'cloud/collection/runtime-source-manifest.json', trace['runtime_source_identity'])
    write(source / 'cloud/gpu-after-cleanup.json', {'instance_id': 'manager-1', 'returncode': 0, 'compute_processes_output': ''})
    write(source / 'cloud/collection/gpu_samples.jsonl', {'manager_instance_id': 'manager-1', 'observation': {'devices': [], 'source': 'synthetic'}})
    # JSONL must have one complete JSON object per physical line.
    (source / 'cloud/collection/gpu_samples.jsonl').write_text(json.dumps({'manager_instance_id': 'manager-1',
        'observation': {'devices': [], 'source': 'synthetic'}}) + '\n', encoding='utf-8')
    declare_experiment(source)
    return source, runtime, trace


def test_public_projection_preserves_both_analyzers_and_atomic_lease_endpoints(tmp_path):
    runtime = tmp_path / 'runtime'
    runtime.mkdir()
    trace = fixtures.fixture(runtime)
    public, analyses = exporter.project_trace(trace, runtime)
    assert analyses['condition_analysis']['counts'] == {'pass': 7, 'fail': 0, 'unknown': 2}
    assert public['model_events'][-1]['budget_snapshot'] == trace['model_events'][-1]['budget_snapshot']
    assert public['runtime_source_identity'] == trace['runtime_source_identity']
    for raw, projected in zip(trace['model_events'], public['model_events']):
        if 'lease_witness' in raw:
            assert raw['lease_witness'] == projected['lease_witness']
        if 'receipt' in raw:
            assert raw['receipt'] == projected['receipt']


def test_nested_private_fields_are_not_copied_under_an_allowed_object(tmp_path):
    runtime = tmp_path / 'runtime'
    runtime.mkdir()
    trace = fixtures.fixture(runtime)
    private = {'api_key': 'private-key-marker', 'path': 'C:/Users/private/file',
               'url': 'https://private.trycloudflare.com', 'authorization': 'Bearer private-token'}
    trace.update(private)
    trace['jobs'][0]['artifact'] = {**private, 'bytes': 10, 'sha256': 'a' * 64,
                                  'media': {**private, 'full_decode_passed': True}}
    for event in trace['model_events']:
        event.update(private)
        if 'receipt' in event:
            event['receipt']['context'].update(private)
            event['receipt'].update(private)
        if 'lease_witness' in event:
            event['lease_witness'].update(private)
            event['lease_witness']['grant_receipt']['context'].update(private)
    trace['system_snapshots'][0]['modules']['model']['budget']['cpu_capacity'] = private
    public, _ = exporter.project_trace(trace, runtime)
    encoded = json.dumps(public)
    for secret in private.values():
        assert secret not in encoded
    assert 'api_key' not in encoded and 'authorization' not in encoded


@pytest.mark.parametrize('state', ['unknown', 'conflict'])
def test_real_recorded_blocked_state_survives_projection(tmp_path, state):
    runtime = tmp_path / 'runtime'
    runtime.mkdir()
    trace = fixtures.fixture(runtime)
    fixtures.add_blocked(trace, state)
    public, analyses = exporter.project_trace(trace, runtime)
    assert analyses['condition_analysis']['checks'][state + '_blocked_without_effect']['status'] == 'pass'
    assert any(event.get('data', {}).get('receipt', {}).get('conditions', {}).get('model_ready') == state
               for event in public['manager_events'])


def test_projection_cannot_drop_a_required_lease_field_and_keep_success(tmp_path, monkeypatch):
    runtime = tmp_path / 'runtime'
    runtime.mkdir()
    trace = fixtures.fixture(runtime)
    monkeypatch.delitem(exporter.LEASE, 'released_monotonic_ns')
    with pytest.raises(ValueError, match='changes an analyzer conclusion'):
        exporter.project_trace(trace, runtime)


@pytest.mark.parametrize('secret', ['https://private.example/token', 'Bearer hidden-key',
                                   'C:/Users/private/token', '/content/private/key.json', '/opt/custom/token'])
def test_sensitive_text_in_a_retained_field_is_rejected_not_silently_published(tmp_path, secret):
    runtime = tmp_path / 'runtime'
    runtime.mkdir()
    trace = fixtures.fixture(runtime)
    trace['platform_release']['source'] = secret
    with pytest.raises(ValueError, match='public finite scalar'):
        exporter.project_trace(trace, runtime)


def test_credential_cannot_escape_by_being_reused_as_an_allowed_identifier(tmp_path):
    runtime = tmp_path / 'runtime'
    runtime.mkdir()
    trace = fixtures.fixture(runtime)
    trace['api_key'] = 'private-key-pretending-to-be-an-id'
    trace['platform_release']['source'] = trace['api_key']
    with pytest.raises(ValueError, match='private credential'):
        exporter.project_trace(trace, runtime)


def test_full_export_has_only_explicit_artifacts_and_no_historical_calibration(tmp_path):
    source, runtime, trace = synthetic_run(tmp_path)
    (source / 'calibration.json').write_text('private calibration must not be copied')
    (source / 'finite-model.json').write_text('old finite test must not be copied')
    target = tmp_path / 'public'
    result = exporter.export(source, target, source_root=runtime, experiment=source / 'experiment.json')
    assert result['publication_only'] is True
    assert result['condition_analysis'] == {'pass': 7, 'fail': 0, 'unknown': 2}
    assert result['public_files'] == 14
    assert not (target / 'calibration.json').exists() and not (target / 'finite-model.json').exists()
    assert sorted(path.name for path in (target / 'assets').iterdir()) == [
        'first.mp4', 'first.png', 'second.mp4', 'second.png', 'third.png']
    provenance = exporter.read_json(target / 'provenance.json')
    assert provenance['analysis_equivalence'] == {'historical_trace': True, 'condition_gates': True}
    assert provenance['baseline_commit'] == '1' * 40
    assert provenance['protocol_id'] == 'synthetic-protocol'
    assert provenance['source_revision'] == {'commit': '2' * 40, 'state': 'modified'}
    exported = exporter.read_json(target / 'trace.json')
    assert exporter.analysis_equivalence(trace, exported, runtime)
    assert exporter.read_json(target / 'gpu-after-cleanup.json')['compute_processes_output_empty'] is True


@pytest.mark.parametrize('damage', ['client_instance', 'client_manifest', 'reference', 'video',
                                    'video_artifact', 'attempt_identity', 'source_bytes', 'source_manifest',
                                    'gpu_instance', 'gpu_telemetry_instance'])
def test_cross_bound_identity_or_byte_change_stops_export_before_publication(tmp_path, damage):
    source, runtime, trace = synthetic_run(tmp_path)
    state = exporter.read_json(source / 'client/state.json')
    if damage == 'client_instance':
        state['instance_id'] = 'different-manager'
    elif damage == 'client_manifest':
        state['manifest_sha256'] = '0' * 64
    elif damage == 'reference':
        (source / 'first.png').write_bytes(b'changed reference')
    elif damage == 'video':
        (source / 'client/a.mp4').write_bytes(b'changed video')
    elif damage == 'video_artifact':
        trace['jobs'][0]['artifact']['sha256'] = '0' * 64
    elif damage == 'attempt_identity':
        state['jobs'][0]['attempt_id'] = 'some-other-attempt'
    elif damage == 'source_bytes':
        (runtime / 'condition_guard.py').write_text('changed source bytes')
    elif damage == 'source_manifest':
        value = exporter.read_json(source / 'cloud/collection/runtime-source-manifest.json')
        value['files']['condition_guard.py']['sha256'] = '0' * 64
        write(source / 'cloud/collection/runtime-source-manifest.json', value)
    elif damage == 'gpu_instance':
        write(source / 'cloud/gpu-after-cleanup.json', {'instance_id': 'another-manager',
                                                      'returncode': 0, 'compute_processes_output': ''})
    elif damage == 'gpu_telemetry_instance':
        (source / 'cloud/collection/gpu_samples.jsonl').write_text(
            json.dumps({'manager_instance_id': 'another-manager', 'observation': {'devices': []}}) + '\n')
    write(source / 'client/state.json', state)
    write(source / 'trace-complete.private.json', trace)
    target = tmp_path / 'public'
    with pytest.raises(ValueError):
        exporter.export(source, target, source_root=runtime, experiment=source / 'experiment.json')
    assert not target.exists()


def test_incomplete_or_missing_input_never_creates_an_acceptance_directory(tmp_path):
    source = tmp_path / 'incomplete'
    source.mkdir()
    with pytest.raises(ValueError, match='missing'):
        exporter.export(source, tmp_path / 'public')
    assert not (tmp_path / 'public').exists()


def test_existing_public_evidence_is_not_overwritten(tmp_path):
    source, runtime, _ = synthetic_run(tmp_path)
    target = tmp_path / 'public'
    target.mkdir()
    marker = target / 'existing.json'
    marker.write_text('retain original evidence')
    with pytest.raises(ValueError, match='never overwritten'):
        exporter.export(source, target, source_root=runtime, experiment=source / 'experiment.json')
    assert marker.read_text() == 'retain original evidence'


def test_source_paths_cannot_escape_or_replace_private_tree(tmp_path):
    source, runtime, _ = synthetic_run(tmp_path)
    with pytest.raises(ValueError, match='separate directories'):
        exporter.export(source, source / 'public', source_root=runtime, experiment=source / 'experiment.json')
    with pytest.raises(ValueError, match='relative path'):
        exporter.owned_file(source, '../outside.txt')


def test_partial_trace_unknowns_are_preserved_instead_of_manufacturing_a_pass(tmp_path):
    runtime = tmp_path / 'runtime'
    runtime.mkdir()
    trace = fixtures.fixture(runtime)
    trace['collection']['model_events_complete'] = False
    trace['model_events'][-1]['budget_snapshot']['lease_history']['history_complete'] = False
    _, analyses = exporter.project_trace(trace, runtime)
    assert analyses['analysis']['checks']['trace_integrity']['status'] == 'unknown'
    assert analyses['condition_analysis']['checks']['atomic_budget_ledger']['status'] == 'unknown'


def add_api_update(source, runtime, trace):
    server = runtime / 'server.py'
    server.write_bytes(b'synthetic new public API projection')
    trace['runtime_source_identity']['files']['server.py'] = {'sha256': exporter.digest(server), 'bytes': server.stat().st_size}
    gate_sources = {name: row['sha256'] for name, row in trace['runtime_source_identity']['files'].items() if name != 'server.py'}
    receipt = {'schema_version': 1, 'phase': 'completed', 'started_at': 100, 'finished_at': 110,
        'manager_instance_id': 'manager-1', 'model_instance_id': 'model-1', 'source_file': 'server.py',
        'before_sha256': hashlib.sha256(b'synthetic original API').hexdigest(), 'after_sha256': exporter.digest(server),
        'model_pid_before': 1000, 'model_pid_after': 1000, 'api_pid_before': 2000, 'api_pid_after': 2001,
        'public_api_ready': True, 'manager_identity_preserved': True, 'model_identity_preserved': True,
        'job_identities_preserved': True, 'gate_sources_preserved': True,
        'gate_source_sha256_before': gate_sources.copy(), 'gate_source_sha256_after': gate_sources.copy(),
        'job_ids_before': ['a', 'b', 'cancel-job'], 'job_ids_after': ['a', 'b', 'cancel-job'],
        'commands': [{'command_id': 'stop-api', 'action': 'stop_api', 'status': 'succeeded', 'created_at': 101, 'finished_at': 102, 'error': None},
                     {'command_id': 'start-api', 'action': 'start_api', 'status': 'succeeded', 'created_at': 103, 'finished_at': 104, 'error': None}]}
    write(source / 'trace-complete.private.json', trace)
    write(source / 'cloud/collection/runtime-source-manifest.json', trace['runtime_source_identity'])
    write(source / 'cloud/collection/api-query-update.json', receipt)
    declare_experiment(source)
    return receipt


def test_api_only_replacement_keeps_both_source_versions_and_exact_command_receipts(tmp_path):
    source, runtime, trace = synthetic_run(tmp_path)
    receipt = add_api_update(source, runtime, trace)
    target = tmp_path / 'public'
    exporter.export(source, target, source_root=runtime, experiment=source / 'experiment.json')
    exported = exporter.read_json(target / 'api-query-update.json')
    assert exported == receipt
    provenance = exporter.read_json(target / 'provenance.json')['api_query_update']
    assert provenance['initial_server_sha256'] == receipt['before_sha256']
    assert provenance['replacement_server_sha256'] == receipt['after_sha256']
    assert all(provenance['gate_source_endpoint_matches'].values())


@pytest.mark.parametrize('damage', ['model_instance', 'gate_file_changed', 'replacement_hash', 'model_pid', 'job_ids'])
def test_api_update_cannot_hide_changed_model_or_gate_source(tmp_path, damage):
    source, runtime, trace = synthetic_run(tmp_path)
    receipt = add_api_update(source, runtime, trace)
    if damage == 'model_instance':
        receipt['model_instance_id'] = 'another-model'
    elif damage == 'gate_file_changed':
        receipt['gate_source_sha256_before']['global_manager.py'] = '0' * 64
    elif damage == 'replacement_hash':
        receipt['after_sha256'] = '0' * 64
    elif damage == 'model_pid':
        receipt['model_pid_after'] += 1
    elif damage == 'job_ids':
        receipt['job_ids_after'] = ['unrelated-job']
    write(source / 'cloud/collection/api-query-update.json', receipt)
    with pytest.raises(ValueError):
        exporter.export(source, tmp_path / 'public', source_root=runtime, experiment=source / 'experiment.json')
    assert not (tmp_path / 'public').exists()


def test_public_api_query_keeps_historical_running_state_and_excludes_private_urls(tmp_path):
    source, runtime, trace = synthetic_run(tmp_path)
    receipt = next(event['data']['receipt'] for event in trace['manager_events']
                   if event['type'] == 'condition.checked' and event['job_id'] == 'a')
    observation = {'id': 'a', 'instance_id': 'manager-1', 'attempt_id': 'a-attempt', 'status': 'running',
                   'condition_gate': receipt, 'status_url': 'https://private.test/v1/jobs/a', 'api_key': 'never-publish'}
    write(source / 'api-query-observation.private.json', observation)
    target = tmp_path / 'public'
    exporter.export(source, target, source_root=runtime, experiment=source / 'experiment.json')
    result = exporter.read_json(target / 'api-query-observation.json')
    assert result['status'] == 'running'
    assert result['condition_gate'] == receipt
    assert 'status_url' not in result and 'api_key' not in result
    assert exporter.read_json(target / 'client-receipt.json')['jobs'][0]['state'] == 'succeeded'


@pytest.mark.parametrize('field,value', [('id', 'unrelated-job'), ('instance_id', 'old-manager'), ('attempt_id', 'wrong-attempt')])
def test_public_api_query_cannot_be_from_another_job_or_attempt(tmp_path, field, value):
    source, runtime, trace = synthetic_run(tmp_path)
    query = {'id': 'a', 'instance_id': 'manager-1', 'attempt_id': 'a-attempt', 'status': 'running'}
    query[field] = value
    write(source / 'api-query-observation.private.json', query)
    with pytest.raises(ValueError):
        exporter.export(source, tmp_path / 'public', source_root=runtime, experiment=source / 'experiment.json')


def add_precision(source, runtime, trace):
    backend = runtime / 'model_backend.py'
    backend.write_bytes(b'synthetic parameter audit source, not executed GPU code')
    trace['runtime_source_identity']['files']['model_backend.py'] = {'sha256': exporter.digest(backend), 'bytes': backend.stat().st_size}
    report = {'schema_version': 1, 'manager_instance_id': 'manager-1', 'model_instance_id': 'model-1',
        'observed_at': 110, 'status': 'passed', 'scope': 'loaded_parameters_only',
        'source_file': 'model_backend.py', 'source_sha256': exporter.digest(backend),
        'source_log_snapshot_sha256': hashlib.sha256(b'synthetic audit log, no GPU').hexdigest(),
        'gpu_query': 'NVIDIA A100-SXM4-80GB, 81920, 550.54.15', 'python_version': '3.12.12',
        'interpretation': 'Synthetic loaded-parameter metadata; not activation or arithmetic verification.', 'components': {}}
    for name in ('transformer_ref', 'text_encoder', 'vae', 'audio_vae'):
        dtype = 'torch.float32' if name == 'audio_vae' else 'torch.bfloat16'
        report['components'][name] = {'component': name, 'status': 'passed', 'parameter_counts': {dtype: 1000},
            'tensor_counts': {dtype: 10}, 'keep_in_fp32_modules': ['encoder', 'decoder', 'quant_conv', 'post_quant_conv'] if name == 'audio_vae' else []}
    write(source / 'trace-complete.private.json', trace)
    write(source / 'cloud/collection/runtime-source-manifest.json', trace['runtime_source_identity'])
    write(source / 'cloud/collection/precision-audit.json', report)
    declare_experiment(source)
    return report


def test_precision_export_preserves_loaded_parameter_scope_and_official_fp32_exceptions(tmp_path):
    source, runtime, trace = synthetic_run(tmp_path)
    report = add_precision(source, runtime, trace)
    target = tmp_path / 'public'
    exporter.export(source, target, source_root=runtime, experiment=source / 'experiment.json')
    assert exporter.read_json(target / 'precision-audit.json') == report
    provenance = exporter.read_json(target / 'provenance.json')['parameter_precision_audit']
    assert provenance['scope'] == 'loaded_parameters_only'
    assert 'unmeasured' in provenance['interpretation']
    assert provenance['source_log_snapshot_sha256'] == report['source_log_snapshot_sha256']


@pytest.mark.parametrize('damage', ['old_manager', 'old_model', 'source_hash', 'unknown_component',
    'component_mismatch', 'float16_hidden', 'noninteger_count', 'negative_count', 'fp32_path',
    'fp32_private_field', 'gpu_url', 'gpu_extra_column', 'python_private_path', 'arithmetic_scope'])
def test_precision_evidence_cannot_expand_scope_hide_dtype_or_expose_private_fields(tmp_path, damage):
    source, runtime, trace = synthetic_run(tmp_path)
    report = add_precision(source, runtime, trace)
    component = report['components']['transformer_ref']
    if damage == 'old_manager':
        report['manager_instance_id'] = 'another-manager'
    elif damage == 'old_model':
        report['model_instance_id'] = 'another-model'
    elif damage == 'source_hash':
        report['source_sha256'] = '0' * 64
    elif damage == 'unknown_component':
        report['components']['api_key'] = deepcopy(component)
    elif damage == 'component_mismatch':
        component['component'] = 'vae'
    elif damage == 'float16_hidden':
        component['parameter_counts']['torch.float16'] = 1
    elif damage == 'noninteger_count':
        component['parameter_counts']['torch.bfloat16'] = True
    elif damage == 'negative_count':
        component['tensor_counts']['torch.bfloat16'] = -1
    elif damage == 'fp32_path':
        component['keep_in_fp32_modules'] = ['/opt/private/key']
    elif damage == 'fp32_private_field':
        component['keep_in_fp32_modules'] = ['api_key']
    elif damage == 'gpu_url':
        report['gpu_query'] = 'https://private.test, 81920, 550.54.15'
    elif damage == 'gpu_extra_column':
        report['gpu_query'] += ', private-key'
    elif damage == 'python_private_path':
        report['python_version'] = '/opt/private/python'
    elif damage == 'arithmetic_scope':
        report['scope'] = 'all_activations_and_kernels'
    write(source / 'cloud/collection/precision-audit.json', report)
    with pytest.raises(ValueError):
        exporter.export(source, tmp_path / 'public', source_root=runtime, experiment=source / 'experiment.json')
    assert not (tmp_path / 'public').exists()


def test_platform_iso_time_is_exactly_normalized_without_changing_original_observation(tmp_path):
    source, runtime, trace = synthetic_run(tmp_path)
    release = {'released': True, 'manager_instance_id': 'manager-1',
               'observed_at': '2026-11-15T01:20:30.703Z', 'source': 'Independent session UI observation'}
    normalized = exporter.normalize_platform_release(release)
    assert release['observed_at'] == '2026-11-15T01:20:30.703Z'
    assert normalized['observed_at'] == 1794705630.703
    assert normalized['observed_at_iso8601'] == release['observed_at']
    trace['platform_release'] = normalized
    write(source / 'trace-complete.private.json', trace)
    write(source / 'platform-release.json', release)
    target = tmp_path / 'public'
    exporter.export(source, target, source_root=runtime, experiment=source / 'experiment.json')
    public = exporter.read_json(target / 'trace.json')['platform_release']
    assert public == normalized
    assert exporter.read_json(source / 'platform-release.json') == release
    assert exporter.read_json(target / 'analysis.json')['checks']['platform_release']['status'] == 'pass'


def test_platform_naive_time_is_not_assigned_an_invented_timezone():
    with pytest.raises(ValueError, match='explicit timezone'):
        exporter.normalize_platform_release({'observed_at': '2026-11-15T01:20:30.703'})


def test_attached_release_time_must_match_the_actual_source_record(tmp_path):
    source, runtime, trace = synthetic_run(tmp_path)
    write(source / 'platform-release.json', {'released': True, 'manager_instance_id': 'manager-1',
                                          'observed_at': '2026-11-15T01:20:30.703Z'})
    with pytest.raises(ValueError, match='does not match'):
        exporter.export(source, tmp_path / 'public', source_root=runtime, experiment=source / 'experiment.json')


@pytest.mark.parametrize('field', ['protocol', 'baseline_commit', 'source_revision', 'runtime_changes',
    'runtime_manifest_sha256', 'experiment_id', 'manager_instance_id', 'model_instance_id', 'interventions'])
def test_experiment_declaration_may_not_inherit_missing_fields(tmp_path, field):
    source, runtime, _ = synthetic_run(tmp_path)
    declaration = exporter.read_json(source / 'experiment.json')
    del declaration[field]
    write(source / 'experiment.json', declaration)
    with pytest.raises(ValueError, match='exact declared fields'):
        exporter.export(source, tmp_path / 'public', source_root=runtime, experiment=source / 'experiment.json')
    assert not (tmp_path / 'public').exists()


@pytest.mark.parametrize('damage', ['baseline', 'revision', 'manager', 'model', 'experiment', 'manifest',
    'protocol_id', 'protocol_hash', 'protocol_bytes', 'intervention', 'private_description', 'empty_description'])
def test_experiment_declaration_conflicts_are_not_published(tmp_path, damage):
    source, runtime, _ = synthetic_run(tmp_path)
    declaration = exporter.read_json(source / 'experiment.json')
    if damage == 'baseline':
        declaration['baseline_commit'] = 'main'
    elif damage == 'revision':
        declaration['source_revision']['state'] = 'unknown'
    elif damage in ('manager', 'model', 'experiment'):
        declaration[damage + ('_id' if damage == 'experiment' else '_instance_id')] = 'another-run'
    elif damage == 'manifest':
        declaration['runtime_manifest_sha256'] = '0' * 64
    elif damage == 'protocol_id':
        declaration['protocol']['id'] = 'another-protocol'
    elif damage == 'protocol_hash':
        declaration['protocol']['sha256'] = '0' * 64
    elif damage == 'protocol_bytes':
        write(source / 'protocol.json', {'protocol_id': 'synthetic-protocol', 'changed': True})
    elif damage == 'intervention':
        declaration['interventions'] = [{'kind': 'managed_api_replacement', 'evidence_sha256': '0' * 64}]
    elif damage == 'private_description':
        declaration['runtime_changes'] = 'Bearer super-private-token'
    else:
        declaration['runtime_changes'] = ' '
    write(source / 'experiment.json', declaration)
    with pytest.raises(ValueError):
        exporter.export(source, tmp_path / 'public', source_root=runtime, experiment=source / 'experiment.json')
    assert not (tmp_path / 'public').exists()


def test_export_requires_both_explicit_experiment_and_source_root(tmp_path):
    source, runtime, _ = synthetic_run(tmp_path)
    with pytest.raises(ValueError, match='missing'):
        exporter.export(source, tmp_path / 'public', source_root=runtime)
    with pytest.raises(ValueError, match='source-root'):
        exporter.export(source, tmp_path / 'public', experiment=source / 'experiment.json')


def test_later_experiment_has_its_own_baseline_protocol_and_changes(tmp_path):
    source, runtime, _ = synthetic_run(tmp_path)
    declaration = exporter.read_json(source / 'experiment.json')
    declaration.update(baseline_commit='a' * 40, source_revision={'commit': 'b' * 40, 'state': 'committed'},
                       runtime_changes='A separate declared release without an API intervention.')
    write(source / 'protocol.json', {'protocol_id': 'new-protocol'})
    declaration['protocol'].update(id='new-protocol', sha256=exporter.digest(source / 'protocol.json'))
    write(source / 'experiment.json', declaration)
    exporter.export(source, tmp_path / 'public', source_root=runtime, experiment=source / 'experiment.json')
    provenance = exporter.read_json(tmp_path / 'public/provenance.json')
    assert provenance['schema_version'] == 2
    assert provenance['baseline_commit'] == 'a' * 40
    assert provenance['source_revision'] == declaration['source_revision']
    assert provenance['protocol_id'] == 'new-protocol'
    assert provenance['runtime_changes'] == declaration['runtime_changes']
    assert provenance['interventions'] == []


def test_actual_api_intervention_cannot_be_omitted_from_declaration(tmp_path):
    source, runtime, trace = synthetic_run(tmp_path)
    add_api_update(source, runtime, trace)
    declaration = exporter.read_json(source / 'experiment.json')
    declaration['interventions'] = []
    write(source / 'experiment.json', declaration)
    with pytest.raises(ValueError, match='interventions'):
        exporter.export(source, tmp_path / 'public', source_root=runtime, experiment=source / 'experiment.json')


def add_durable_pages(trace, page_size=4):
    """Synthetic hash-chained ledger with bounded snapshot tails."""
    budget = trace['model_events'][-1]['budget_snapshot']
    rows = deepcopy(budget['lease_history']['events'])
    previous = '0' * 64
    for row in rows:
        row['previous_sha256'] = previous
        row['sha256'] = hashlib.sha256(json.dumps(row, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        previous = row['sha256']
    header = {key: deepcopy(budget[key]) for key in ('budget_instance_id', 'capacity_version', 'limits', 'gpu_host_cpu_threads')}
    header.update(schema_version=1, model_instance_id=trace['model_instance_id'])
    trace['budget_audit_pages'] = [{**deepcopy(header), 'after_sequence': start,
        'through_sequence': len(rows), 'next_sequence': min(start + page_size, len(rows)),
        'after_sha256': rows[start - 1]['sha256'] if start else '0' * 64,
        'tail_sha256': previous, 'events': deepcopy(rows[start:start + page_size]),
        'page_complete': True, 'more': start + page_size < len(rows)} for start in range(0, len(rows), page_size)]
    for target in (budget, trace['system_snapshots'][0]['modules']['model']['budget']):
        target['lease_history'].update(events=deepcopy(rows[-page_size:]), capacity=page_size,
            count=len(rows), dropped=len(rows) - page_size, history_complete=False)
        target['lease_journal'] = {'schema_version': 1, 'budget_instance_id': budget['budget_instance_id'],
            'count': len(rows), 'tail_sha256': previous, 'complete': True, 'error': None}
    return trace


def test_durable_journal_pages_preserve_analysis_and_strip_unhashed_private_metadata(tmp_path):
    runtime = tmp_path / 'runtime'
    runtime.mkdir()
    trace = add_durable_pages(fixtures.fixture(runtime))
    trace['budget_audit_pages'][0]['journal_path'] = '/private/budget.sqlite3'
    trace['model_events'][-1]['budget_snapshot']['lease_journal']['private_path'] = '/private/budget.sqlite3'
    public, result = exporter.project_trace(trace, runtime)
    assert result['condition_analysis']['checks']['atomic_budget_ledger']['status'] == 'pass'
    assert len(public['budget_audit_pages']) == 3
    assert '/private/budget.sqlite3' not in json.dumps(public)
    assert public['budget_audit_pages'][1]['events'][0]['sha256'] == trace['budget_audit_pages'][1]['events'][0]['sha256']


def test_partial_durable_pages_stay_unknown_after_publication(tmp_path):
    runtime = tmp_path / 'runtime'
    runtime.mkdir()
    trace = add_durable_pages(fixtures.fixture(runtime))
    trace['budget_audit_pages'].pop(1)
    _, result = exporter.project_trace(trace, runtime)
    assert result['condition_analysis']['checks']['atomic_budget_ledger']['status'] == 'unknown'


def test_exporter_does_not_rehash_private_or_unknown_ledger_fields(tmp_path):
    runtime = tmp_path / 'runtime'
    runtime.mkdir()
    trace = fixtures.fixture(runtime)
    trace['model_events'][-1]['budget_snapshot']['lease_history']['events'][0]['private_path'] = '/private/file'
    add_durable_pages(trace)
    with pytest.raises(ValueError, match='changes an analyzer conclusion'):
        exporter.project_trace(trace, runtime)
