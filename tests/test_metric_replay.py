"""Evidence replay and adversarial claims; synthetic traces are not GPU runs."""
from copy import deepcopy
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


metrics = load('metric_replay_test', ROOT / 'research/replay_metrics.py')
fixtures = load('metric_export_fixture', ROOT / 'tests/test_condition_evidence_export.py')
PROFILE = {'width': 512, 'height': 288, 'frames': 124, 'fps': 24.0, 'audio_channels': 2,
           'video_codec': 'h264', 'audio_codec': 'aac'}


def selection(trace, client, protocol_id='synthetic-protocol'):
    return metrics.contract_for(trace, client, experiment_id=client['run_id'], protocol_id=protocol_id,
        job_ids=[row['job_id'] for row in client['jobs']], artifact_profile=PROFILE)


def fixture(tmp_path):
    source, runtime, trace = fixtures.synthetic_run(tmp_path)
    trace['cleanup_receipt']['owner_observation']['observed_at'] = 200
    trace['platform_release']['observed_at'] = 210
    client = fixtures.exporter.read_json(source / 'client/state.json')
    for job in trace['jobs']:
        if job['status'] == 'succeeded':
            job['model_instance_id'] = 'model-1'
            job['artifact']['media'].update(PROFILE)
    for job in client['jobs']:
        if job['state'] == 'succeeded':
            job['download']['media'].update(PROFILE)
    return trace, client, selection(trace, client), runtime


def by_id(report):
    return {row['metric_id']: row for row in report['metrics']}


def test_replays_raw_events_artifacts_and_closed_batch_not_saved_summaries(tmp_path):
    trace, client, contract, runtime = fixture(tmp_path)
    result = metrics.replay(trace, client, contract, source_root=runtime)
    assert result['counts'] == {'measured': 4, 'unknown': 1, 'conflict': 0}
    rows = by_id(result)
    assert (rows['MC-CORE-ARTIFACT']['numerator'], rows['MC-CORE-ARTIFACT']['denominator']) == (2, 2)
    assert (rows['EO-CORE-CLOSURE']['numerator'], rows['EO-CORE-CLOSURE']['denominator']) == (3, 3)
    assert rows['MC-REC-GPU-OVERLAP']['value'] == 0
    assert rows['MC-REC-GPU-OVERLAP']['lease_coverage']['atomic_endpoints'] == 12
    assert rows['EO-REC-CPU-GPU-OVERCOMMIT']['value'] == 0
    assert rows['EO-REC-CPU-GPU-OVERCOMMIT']['peak_cpu_lease_charge'] == 2
    assert rows['EO-REC-OVERCOMMIT']['value'] is None
    assert metrics.verify_claims(result, deepcopy(result['metrics']))['status'] == 'pass'


@pytest.mark.parametrize('damage', ['identity', 'window', 'trace_hash', 'client_hash', 'denominator_subset',
    'duplicate_jobs', 'duplicate_metrics', 'missing_protocol', 'missing_profile', 'bad_profile', 'bool_window', 'nonfinite_cutoff'])
def test_contract_cannot_change_identity_scope_or_omit_denominator(tmp_path, damage):
    trace, client, contract, runtime = fixture(tmp_path)
    if damage == 'identity':
        contract['manager_instance_id'] = 'some-other-manager'
    elif damage == 'window':
        contract['window']['manager_sequence_end'] -= 1
    elif damage in ('trace_hash', 'client_hash'):
        contract[damage.replace('_hash', '_semantic_sha256')] = '0' * 64
    elif damage == 'denominator_subset':
        contract['job_ids'].pop()
    elif damage == 'duplicate_jobs':
        contract['job_ids'].append(contract['job_ids'][0])
    elif damage == 'duplicate_metrics':
        contract['metric_ids'].append(contract['metric_ids'][0])
    elif damage == 'missing_protocol':
        del contract['protocol_id']
    elif damage == 'missing_profile':
        del contract['artifact_profile']
    elif damage == 'bad_profile':
        contract['artifact_profile']['frames'] = True
    elif damage == 'bool_window':
        contract['window']['model_sequence_start'] = True
    else:
        contract['window']['owner_observed_at'] = float('inf')
    with pytest.raises(ValueError):
        metrics.replay(trace, client, contract, source_root=runtime)


@pytest.mark.parametrize('damage', ['drop_final', 'drop_client', 'drop_artifact', 'drop_decode', 'drop_dimensions',
    'truncated_ledger', 'unknown_source', 'no_owner', 'duplicate_client', 'stale_attempt', 'wrong_hash',
    'wrong_size', 'wrong_profile', 'false_decode', 'boolean_size', 'nan_fps', 'event_after_cutoff'])
def test_rebound_damaged_evidence_is_unknown_conflicting_or_rejected_not_measured(tmp_path, damage):
    trace, client, _, runtime = fixture(tmp_path)
    if damage == 'drop_final':
        trace['jobs'].pop(0)
    elif damage == 'drop_client':
        client['jobs'].pop(0)
    elif damage == 'drop_artifact':
        trace['jobs'][0].pop('artifact')
    elif damage == 'drop_decode':
        client['jobs'][0]['download']['media'].pop('full_decode_passed')
    elif damage == 'drop_dimensions':
        trace['jobs'][0]['artifact']['media'].pop('width')
    elif damage == 'truncated_ledger':
        trace['model_events'][-1]['budget_snapshot']['lease_history']['history_complete'] = False
        trace['model_events'][-1]['budget_snapshot']['lease_history']['dropped'] = 1
        trace['system_snapshots'][0]['modules']['model']['budget']['lease_history']['history_complete'] = False
        trace['system_snapshots'][0]['modules']['model']['budget']['lease_history']['dropped'] = 1
    elif damage == 'unknown_source':
        trace['runtime_source_identity']['files'].pop('condition_guard.py')
    elif damage == 'no_owner':
        trace['cleanup_receipt'].pop('owner_observation')
    elif damage == 'duplicate_client':
        client['jobs'].append(deepcopy(client['jobs'][0]))
    elif damage == 'stale_attempt':
        client['jobs'][0]['attempt_id'] = 'another-attempt'
    elif damage == 'wrong_hash':
        client['jobs'][0]['download']['sha256'] = 'b' * 64
    elif damage == 'wrong_size':
        client['jobs'][0]['download']['bytes'] += 1
    elif damage == 'wrong_profile':
        client['jobs'][0]['download']['media']['width'] = 1024
    elif damage == 'false_decode':
        client['jobs'][0]['download']['media']['full_decode_passed'] = False
    elif damage == 'boolean_size':
        client['jobs'][0]['download']['bytes'] = True
    elif damage == 'nan_fps':
        client['jobs'][0]['download']['media']['fps'] = float('nan')
    else:
        trace['model_events'][-1]['observed_at'] = 300
    raises = damage in {'duplicate_client', 'stale_attempt', 'boolean_size', 'nan_fps', 'event_after_cutoff'}
    if raises:
        with pytest.raises(ValueError):
            contract = metrics.contract_for(trace, client, experiment_id='fixture-run', protocol_id='synthetic-protocol',
                job_ids=['a', 'b', 'cancel-job'], artifact_profile=PROFILE)
            metrics.replay(trace, client, contract, source_root=runtime)
    else:
        contract = metrics.contract_for(trace, client, experiment_id='fixture-run', protocol_id='synthetic-protocol',
            job_ids=['a', 'b', 'cancel-job'], artifact_profile=PROFILE)
        rows = by_id(metrics.replay(trace, client, contract, source_root=runtime))
        key = ('EO-REC-CPU-GPU-OVERCOMMIT' if damage == 'truncated_ledger' else
               'EO-CORE-CLOSURE' if damage == 'no_owner' else 'MC-CORE-ARTIFACT')
        assert rows[key]['status'] != 'measured'
        assert rows[key]['value'] is None


@pytest.mark.parametrize('damage', ['value', 'same_ratio_wrong_counts', 'coverage', 'window', 'instance',
    'resource_scope', 'binding', 'false_full_resource_zero', 'missing_metric', 'duplicate_metric', 'missing_evidence', 'bool_zero'])
def test_existing_claims_must_match_recomputed_evidence_not_only_numeric_range(tmp_path, damage):
    trace, client, contract, runtime = fixture(tmp_path)
    report = metrics.replay(trace, client, contract, source_root=runtime)
    claims = deepcopy(report['metrics'])
    if damage == 'value':
        claims[0]['value'] = .5
    elif damage == 'same_ratio_wrong_counts':
        claims[0].update(numerator=9, denominator=9)
    elif damage == 'coverage':
        claims[0]['coverage']['admissions'] = 99
    elif damage == 'window':
        claims[0]['window']['owner_observed_at'] += 1
    elif damage == 'instance':
        claims[0]['model_instance_id'] = 'another-model'
    elif damage == 'resource_scope':
        claims[3]['resource_scope'].append('ram_bytes')
    elif damage == 'binding':
        claims[0]['evidence_binding']['trace_semantic_sha256'] = '0' * 64
    elif damage == 'false_full_resource_zero':
        claims[4].update(status='measured', value=0)
    elif damage == 'missing_metric':
        claims.pop()
    elif damage == 'duplicate_metric':
        claims.append(deepcopy(claims[0]))
    elif damage == 'missing_evidence':
        claims[0].pop('evidence_binding')
    else:
        claims[3]['value'] = False
    with pytest.raises(ValueError):
        metrics.verify_claims(report, claims)


def test_unknown_metric_and_full_resource_metric_remain_unknown(tmp_path):
    trace, client, contract, runtime = fixture(tmp_path)
    contract['metric_ids'].append('UNSUPPORTED-METRIC')
    report = metrics.replay(trace, client, contract, source_root=runtime)
    assert report['metrics'][-1]['status'] == 'unknown'
    assert report['metrics'][-1]['value'] is None
    assert report['metrics'][-2]['status'] == 'unknown'


def test_source_root_is_explicit_and_changed_source_prevents_a_measured_claim(tmp_path):
    trace, client, contract, runtime = fixture(tmp_path)
    with pytest.raises(ValueError, match='source_root'):
        metrics.replay(trace, client, contract, source_root=None)
    (runtime / 'condition_guard.py').write_text('changed source', encoding='utf-8')
    report = metrics.replay(trace, client, contract, source_root=runtime)
    assert all(row['status'] != 'measured' for row in report['metrics'])
    with pytest.raises(ValueError, match='Conflicting'):
        metrics.verify_claims(report, report['metrics'])


def test_original_v040_evidence_replays_against_immutable_source_without_modifying_it():
    directory = ROOT / 'evidence/runtime-conditions-2026-11-15'
    trace = json.loads((directory / 'trace.json').read_text(encoding='utf-8'))
    client = json.loads((directory / 'client-receipt.json').read_text(encoding='utf-8'))
    binding = load('metric_historical_source_binding', ROOT / 'research/source_bindings.py')
    source_root = binding.bound_source_root(ROOT / 'research/source-bindings/runtime-conditions-v040.json', root=ROOT, trace=trace)
    contract = selection(trace, client, 'runtime-conditions-a100-bf16-2026-11-15')
    report = metrics.replay(trace, client, contract, source_root=source_root)
    assert report['counts'] == {'measured': 4, 'unknown': 1, 'conflict': 0}
    rows = by_id(report)
    assert (rows['MC-CORE-ARTIFACT']['numerator'], rows['MC-CORE-ARTIFACT']['denominator']) == (2, 2)
    assert (rows['EO-CORE-CLOSURE']['numerator'], rows['EO-CORE-CLOSURE']['denominator']) == (3, 3)
    assert rows['MC-REC-GPU-OVERLAP']['value'] == rows['EO-REC-CPU-GPU-OVERCOMMIT']['value'] == 0
    assert rows['EO-REC-OVERCOMMIT']['status'] == 'unknown'
    assert metrics.verify_claims(report, deepcopy(report['metrics']))['status'] == 'pass'


def test_durable_pages_reconstruct_truncated_tail_but_missing_page_never_becomes_zero(tmp_path):
    trace, client, _, runtime = fixture(tmp_path)
    fixtures.add_durable_pages(trace)
    result = metrics.replay(trace, client, selection(trace, client), source_root=runtime)
    assert by_id(result)['EO-REC-CPU-GPU-OVERCOMMIT']['value'] == 0
    assert by_id(result)['EO-REC-CPU-GPU-OVERCOMMIT']['lease_coverage']['atomic_endpoints'] == 12
    trace['budget_audit_pages'].pop(1)
    partial = metrics.replay(trace, client, selection(trace, client), source_root=runtime)
    assert by_id(partial)['EO-REC-CPU-GPU-OVERCOMMIT']['status'] == 'unknown'
    assert by_id(partial)['MC-REC-GPU-OVERLAP']['value'] is None


def test_conflicting_capacity_cannot_become_zero_or_accepted_claim(tmp_path):
    trace, client, _, runtime = fixture(tmp_path)
    trace['model_events'][-1]['budget_snapshot']['lease_history']['events'][0]['limits']['cpu_threads'] = 999
    report = metrics.replay(trace, client, selection(trace, client), source_root=runtime)
    assert by_id(report)['EO-REC-CPU-GPU-OVERCOMMIT']['status'] == 'conflict'
    assert by_id(report)['EO-REC-CPU-GPU-OVERCOMMIT']['value'] is None
    with pytest.raises(ValueError, match='Conflicting'):
        metrics.verify_claims(report, report['metrics'])


@pytest.fixture(scope='module')
def historical_knowledge_case():
    directory = ROOT / 'evidence/runtime-conditions-2026-11-15'
    trace = json.loads((directory / 'trace.json').read_text(encoding='utf-8'))
    client = json.loads((directory / 'client-receipt.json').read_text(encoding='utf-8'))
    contract = json.loads((ROOT / 'research/metric-replays/runtime-conditions-v040.json').read_text(encoding='utf-8'))
    bindings = load('knowledge_metric_source_bindings', ROOT / 'research/source_bindings.py')
    source_root = bindings.bound_source_root(ROOT / 'research/source-bindings/runtime-conditions-v040.json', root=ROOT, trace=trace)
    report = metrics.replay(trace, client, contract, source_root=source_root)
    modules = [json.loads((ROOT / ('knowledge/modules/' + name + '.json')).read_text(encoding='utf-8'))
               for name in ('model-container', 'event-orchestrator')]
    # Add only structured binding metadata. Existing published status, value,
    # ratio counts, outcomes and lease coverage are independent asserted claims.
    rows = by_id(report)
    for module in modules:
        for perspective in module['perspectives'].values():
            for metric in perspective['metrics']:
                if metric['id'] not in rows:
                    continue
                actual = rows[metric['id']]
                for observation in metric.get('observations', []):
                    if observation.get('experiment_id') != report['experiment_id']:
                        continue
                    observation.update(measurement_contract_id='h3-runtime-conditions-v040-closed-batch',
                        evidence_window=deepcopy(actual['window']), resource_scope=deepcopy(actual['resource_scope']),
                        measurement_coverage=deepcopy(actual['coverage']), measurement_evidence_binding=deepcopy(actual['evidence_binding']))
    return report, modules


def selected_observation(modules, metric_id):
    for module in modules:
        for perspective in module['perspectives'].values():
            for metric in perspective['metrics']:
                if metric['id'] == metric_id:
                    return next(row for row in metric['observations'] if row.get('measurement_contract_id') == 'h3-runtime-conditions-v040-closed-batch')
    raise AssertionError('test metric not found')


def test_actual_published_knowledge_numbers_and_coverage_match_independent_recomputation(historical_knowledge_case):
    report, modules = historical_knowledge_case
    result = metrics.verify_knowledge_observations(report, modules, contract_id='h3-runtime-conditions-v040-closed-batch')
    assert result['verified_metric_ids'] == list(metrics.METRIC_IDS)
    assert result['status'] == 'pass'


@pytest.mark.parametrize('damage', ['ratio_counts', 'closed_denominator', 'window', 'instance', 'contract',
    'resource_scope', 'coverage', 'budget_instance', 'capacity_version', 'duration', 'unknown_zero',
    'evidence_binding', 'drop_selection', 'duplicate_selection', 'wrong_units', 'boolean_value'])
def test_knowledge_observations_cannot_pass_with_wrong_math_scope_or_partial_zero(historical_knowledge_case, damage):
    report, original = historical_knowledge_case
    modules = deepcopy(original)
    artifact = selected_observation(modules, 'MC-CORE-ARTIFACT')
    closure = selected_observation(modules, 'EO-CORE-CLOSURE')
    cpu = selected_observation(modules, 'EO-REC-CPU-GPU-OVERCOMMIT')
    full = selected_observation(modules, 'EO-REC-OVERCOMMIT')
    if damage == 'ratio_counts':
        artifact.update(numerator=9, denominator=9)
    elif damage == 'closed_denominator':
        closure['denominator'] = 2
    elif damage == 'window':
        artifact['evidence_window']['owner_observed_at'] -= 1
    elif damage == 'instance':
        artifact['model_instance_id'] = 'different-model'
    elif damage == 'contract':
        artifact['measurement_contract_id'] = 'different-contract'
    elif damage == 'resource_scope':
        cpu['resource_scope'].append('ram_bytes')
    elif damage == 'coverage':
        cpu['coverage']['endpoint_events'] = 6
    elif damage == 'budget_instance':
        cpu['budget_instance_id'] = 'different-budget'
    elif damage == 'capacity_version':
        cpu['capacity_versions'] = ['0' * 64]
    elif damage == 'duration':
        cpu['violation_duration_seconds'] = 10
    elif damage == 'unknown_zero':
        full.update(status='measured', value=0)
    elif damage == 'evidence_binding':
        artifact['measurement_evidence_binding']['trace_semantic_sha256'] = '0' * 64
    elif damage == 'drop_selection':
        artifact.pop('measurement_contract_id')
        artifact['experiment_id'] = 'different-run'
    elif damage == 'duplicate_selection':
        modules[0]['perspectives']['core']['metrics'][0]['observations'].append(deepcopy(artifact))
    elif damage == 'wrong_units':
        modules[0]['perspectives']['core']['metrics'][0]['unit'] = 'count'
    else:
        cpu['value'] = False
    with pytest.raises(ValueError):
        metrics.verify_knowledge_observations(report, modules, contract_id='h3-runtime-conditions-v040-closed-batch')


def test_free_text_windows_are_not_pretended_to_be_machine_verifiable(historical_knowledge_case):
    report, original = historical_knowledge_case
    modules = deepcopy(original)
    selected_observation(modules, 'MC-CORE-ARTIFACT')['window'] = 'Narrative remains outside this mechanical check.'
    result = metrics.verify_knowledge_observations(report, modules, contract_id='h3-runtime-conditions-v040-closed-batch')
    assert 'free-text' in result['scope']


def test_violation_time_regions_are_not_counted_as_pair_combinations_or_endpoints():
    assert metrics._merge_intervals([(1, 5), (2, 6), (3, 7), (9, 10), (10, 12)]) == [[1, 7], [9, 12]]
