"""Recompute selected H3 metrics from identity-bound, closed-batch evidence.

This is a finite adapter for the collected H3 trace/client schemas, not a generic
metric interpreter. The explicit contract binds experiment/protocol/instances,
semantic input hashes, the exact admitted job set, complete event sequence
windows and final owner cutoff. A contract is an analysis selection, not proof
of execution or a preregistered hypothesis. Recorded media receipts are checked;
this command does not decode files again, start a GPU, or assess video quality.

CLI: replay_metrics.py --trace TRACE --client CLIENT --contract CONTRACT
     (--source-binding BINDING | --source-root EXECUTED_SOURCE_ROOT)
     [--claims CLAIMS] [--output REPORT]
Historical runs require their fixed binding/snapshot. No current source default
is used. CLAIMS is a list of metric records from the report format; values,
coverage, numerator/denominator and bindings are independently compared.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
METRIC_IDS = ('MC-CORE-ARTIFACT', 'EO-CORE-CLOSURE', 'MC-REC-GPU-OVERLAP',
              'EO-REC-CPU-GPU-OVERCOMMIT', 'EO-REC-OVERCOMMIT')
TOKEN = re.compile(r'[A-Za-z0-9_.:-]{1,256}\Z')
SHA = re.compile(r'[0-9a-f]{64}\Z')
TERMINAL = {'succeeded', 'failed', 'cancelled'}


def semantic_sha256(value):
    """Hash parsed JSON, not its formatting; the field name exposes this scope."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def load_module(name):
    spec = importlib.util.spec_from_file_location('metric_' + name, Path(__file__).with_name(name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _token(value):
    return isinstance(value, str) and bool(TOKEN.fullmatch(value))


def contract_for(trace, client, *, experiment_id, protocol_id, job_ids, artifact_profile, metric_ids=METRIC_IDS):
    """Create a reviewable selection; replay still validates all evidence.

    Protocol and batch identities are supplied explicitly. Never call this on
    modified evidence to pretend that an existing contract still matches it.
    """
    return {'schema_version': 1, 'experiment_id': experiment_id, 'protocol_id': protocol_id,
            'manager_instance_id': trace['manager_instance_id'], 'model_instance_id': trace['model_instance_id'],
            'trace_semantic_sha256': semantic_sha256(trace), 'client_semantic_sha256': semantic_sha256(client),
            'job_ids': list(job_ids), 'artifact_profile': deepcopy(artifact_profile), 'metric_ids': list(metric_ids), 'window': {
                'manager_sequence_start': 1, 'manager_sequence_end': trace['collection']['manager_event_sequence_final'],
                'model_sequence_start': 1, 'model_sequence_end': trace['collection']['model_event_sequence_final'],
                'owner_observed_at': trace.get('cleanup_receipt', {}).get('owner_observation', {}).get('observed_at')}}


def _validate_contract(trace, client, contract):
    required = {'schema_version', 'experiment_id', 'protocol_id', 'manager_instance_id', 'model_instance_id',
                'trace_semantic_sha256', 'client_semantic_sha256', 'job_ids', 'artifact_profile', 'metric_ids', 'window'}
    if not isinstance(contract, dict) or set(contract) != required or type(contract['schema_version']) is not int or contract['schema_version'] != 1:
        raise ValueError('Explicit metric contract requires its exact schema fields')
    for key in ('experiment_id', 'protocol_id', 'manager_instance_id', 'model_instance_id'):
        if not _token(contract[key]):
            raise ValueError('Metric contract identity is invalid')
    if (contract['experiment_id'] != client.get('run_id')
            or contract['manager_instance_id'] != trace.get('manager_instance_id')
            or contract['manager_instance_id'] != client.get('instance_id')
            or contract['model_instance_id'] != trace.get('model_instance_id')):
        raise ValueError('Metric experiment or instance identity conflicts with evidence')
    for name, evidence in (('trace', trace), ('client', client)):
        if contract[name + '_semantic_sha256'] != semantic_sha256(evidence):
            raise ValueError('Metric input semantic hash does not match its contract')
    for name in ('job_ids', 'metric_ids'):
        values = contract[name]
        if not isinstance(values, list) or not values or not all(_token(v) for v in values) or len(set(values)) != len(values):
            raise ValueError('Metric contract requires a nonempty unique explicit ' + name)
    profile = contract['artifact_profile']
    if (not isinstance(profile, dict) or set(profile) != {'width', 'height', 'frames', 'fps', 'audio_channels', 'video_codec', 'audio_codec'}
            or any(type(profile[k]) is not int or profile[k] <= 0 for k in ('width', 'height', 'frames', 'audio_channels'))
            or not _finite(profile['fps']) or profile['fps'] <= 0
            or not _token(profile['video_codec']) or not _token(profile['audio_codec'])):
        raise ValueError('Explicit H3 audiovisual artifact profile is missing or malformed')
    window = contract['window']
    keys = {'manager_sequence_start', 'manager_sequence_end', 'model_sequence_start', 'model_sequence_end', 'owner_observed_at'}
    if not isinstance(window, dict) or set(window) != keys:
        raise ValueError('Metric contract requires explicit complete-stream window and owner cutoff')
    for stream in ('manager', 'model'):
        start, end = window[stream + '_sequence_start'], window[stream + '_sequence_end']
        if type(start) is not int or start != 1 or type(end) is not int or end < 1:
            raise ValueError('Only complete instance-lifetime sequence windows are supported')
        if end != trace.get('collection', {}).get(stream + '_event_sequence_final'):
            raise ValueError('Metric event window differs from the declared final sequence')
    cutoff = window['owner_observed_at']
    observed = trace.get('cleanup_receipt', {}).get('owner_observation', {}).get('observed_at')
    if cutoff is not None and not _finite(cutoff):
        raise ValueError('Metric cutoff must be finite or explicitly unknown')
    if cutoff != observed:
        raise ValueError('Metric cutoff differs from the final owner observation')
    if cutoff is not None:
        for event in trace.get('manager_events', []) + trace.get('model_events', []):
            stamp = event.get('observed_at')
            if not _finite(stamp) or stamp > cutoff:
                raise ValueError('Event timestamp is invalid or outside the final owner cutoff')


def _indexed(rows, *keys):
    result = {}
    for row in rows:
        key = next((row.get(k) for k in keys if row.get(k) is not None), None)
        if not _token(key) or key in result:
            raise ValueError('Missing or duplicate job identity in metric evidence')
        result[key] = row
    return result


def _status(checks):
    states = [check['status'] for check in checks]
    return 'conflict' if 'fail' in states else 'unknown' if any(s != 'pass' for s in states) else 'measured'


def _ledger_rows(trace):
    """Select observed endpoints, after analyzer validates complete coverage.

    New durable pages supersede bounded history; legacy complete histories remain
    supported. This selection never establishes completeness by itself.
    """
    collected = load_module('analyze_condition_gates').collect_budget_ledgers(trace)
    if collected['errors']:
        raise ValueError('Conflicting durable ledger evidence')
    if collected['missing']:
        return []
    return [row for identity in sorted(collected['ledgers'])
            for row in collected['ledgers'][identity]['events']]


def _merge_intervals(intervals):
    """Count time regions, not pair combinations or sampling endpoints."""
    merged = []
    for start, end in sorted(intervals):
        if start >= end:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def replay(trace, client, contract, *, source_root):
    if source_root is None:
        raise ValueError('Explicit executed source_root required; historical runs cannot use current runtime implicitly')
    _validate_contract(trace, client, contract)
    base = load_module('analyze_trace').analyze(trace)['checks']
    gates = load_module('analyze_condition_gates').analyze(trace, source_root)['checks']
    expected = set(contract['job_ids'])
    admitted_rows = [e for e in trace.get('manager_events', []) if e.get('type') == 'job.queued']
    admitted = _indexed(admitted_rows, 'job_id')
    jobs = _indexed(trace.get('jobs', []), 'job_id', 'id')
    clients = _indexed(client.get('jobs', []), 'job_id')
    if (set(admitted) - expected) or (set(jobs) - expected) or (set(clients) - expected):
        raise ValueError('Explicit closed batch omits observed jobs; denominator cannot be narrowed')
    for job_id in set(jobs) & set(clients):
        job, observed = jobs[job_id], clients[job_id]
        if job.get('instance_id') != contract['manager_instance_id']:
            raise ValueError('Job belongs to a different manager')
        if (job.get('model_instance_id') not in (None, contract['model_instance_id'])
                or job.get('attempt_id') != observed.get('attempt_id') or job.get('status') != observed.get('state')):
            raise ValueError('Client/server terminal identity or outcome conflicts')
    batch_complete = set(admitted) == set(jobs) == set(clients) == expected
    common = {key: deepcopy(contract[key]) for key in ('experiment_id', 'protocol_id', 'manager_instance_id', 'model_instance_id', 'window')}
    binding = {key: contract[key] for key in ('trace_semantic_sha256', 'client_semantic_sha256')}
    binding['contract_semantic_sha256'] = semantic_sha256(contract)
    coverage = {'expected_jobs': len(expected), 'admissions': len(admitted), 'job_receipts': len(jobs),
                'client_receipts': len(clients), 'manager_events': len(trace.get('manager_events', [])),
                'model_events': len(trace.get('model_events', [])), 'batch_complete': batch_complete}

    def record(metric_id, status, reason, *, unit='count', resource_scope, value=None, **extra):
        return {**deepcopy(common), 'metric_id': metric_id, 'status': status, 'value': value,
                'unit': unit, 'resource_scope': resource_scope, 'reason': reason,
                'coverage': deepcopy(coverage), 'evidence_binding': deepcopy(binding), **extra}

    prerequisite = [base[name] for name in ('trace_integrity', 'terminal_receipt_coverage', 'stage_causality')]
    shared_status = _status(prerequisite + [gates['gate_source_bytes']])
    if not batch_complete and shared_status != 'conflict':
        shared_status = 'unknown'
    records = {}
    successes = {key: job for key, job in jobs.items() if job.get('status') == 'succeeded'}
    valid, missing, invalid = [], [], []
    for key, job in successes.items():
        artifact = job.get('artifact')
        download = clients.get(key, {}).get('download')
        if not isinstance(artifact, dict) or not isinstance(download, dict):
            missing.append(key)
            continue
        required = ('bytes', 'sha256', 'media')
        if any(k not in artifact or k not in download for k in required) or 'verified' not in download:
            missing.append(key)
            continue
        if (type(artifact['bytes']) is not int or artifact['bytes'] <= 0
                or type(download['bytes']) is not int or download['bytes'] <= 0
                or not isinstance(artifact['sha256'], str) or not SHA.fullmatch(artifact['sha256'])
                or not isinstance(download['sha256'], str) or not SHA.fullmatch(download['sha256'])):
            raise ValueError('Malformed artifact size or SHA256')
        server_media, client_media = artifact['media'], download['media']
        if not isinstance(server_media, dict) or not isinstance(client_media, dict):
            raise ValueError('Malformed artifact media evidence')
        server_decode, client_decode = server_media.get('full_decode_passed'), client_media.get('full_decode_passed')
        profile = contract['artifact_profile']
        server_keys = ('width', 'height', 'fps', 'audio_channels', 'video_codec', 'audio_codec')
        client_keys = ('width', 'height', 'frames', 'fps', 'audio_channels')
        if any(k not in server_media for k in server_keys) or any(k not in client_media for k in client_keys):
            missing.append(key)
            continue
        for media, keys in ((server_media, server_keys), (client_media, client_keys)):
            if any(type(media[k]) is not int or media[k] <= 0 for k in keys if k in ('width', 'height', 'frames', 'audio_channels')):
                raise ValueError('Malformed integer media property')
            if not _finite(media['fps']) or media['fps'] <= 0:
                raise ValueError('Malformed finite media frame rate')
        if server_decode is None or client_decode is None or 'expected_sha256' not in download:
            missing.append(key)
        elif (artifact['bytes'] != download['bytes'] or artifact['sha256'] != download['sha256']
              or artifact['sha256'] != download['expected_sha256'] or download['verified'] is not True
              or server_decode is not True or client_decode is not True
              or any(server_media[k] != profile[k] for k in server_keys)
              or any(client_media[k] != profile[k] for k in client_keys)):
            invalid.append(key)
        else:
            valid.append(key)
    artifact_status = 'conflict' if invalid or shared_status == 'conflict' else 'unknown' if missing or not successes or shared_status == 'unknown' else 'measured'
    records['MC-CORE-ARTIFACT'] = record('MC-CORE-ARTIFACT', artifact_status,
        'Successful artifact identities, sizes, checksums and recorded server/client full-decode receipts reconciled; no new decode or semantic assessment.',
        unit='ratio', resource_scope=['declared_success_artifacts'],
        value=len(valid) / len(successes) if artifact_status == 'measured' else None,
        numerator=len(valid), denominator=len(successes) if batch_complete else None,
        verified_job_ids=sorted(valid), missing_job_ids=sorted(missing), conflicting_job_ids=sorted(invalid),
        artifact_profile=deepcopy(contract['artifact_profile']))

    closure_status = _status(prerequisite + [base['terminal_job_cleanup'], base['service_process_cleanup'], gates['gate_source_bytes']])
    if (not batch_complete or contract['window']['owner_observed_at'] is None) and closure_status != 'conflict':
        closure_status = 'unknown'
    closed = [key for key, job in jobs.items() if job.get('status') in TERMINAL and job.get('cleanup_complete') is True
              and clients.get(key, {}).get('cleanup_complete') is True]
    if len(closed) != len(expected) and closure_status == 'measured':
        closure_status = 'unknown'
    outcomes = Counter(job.get('status', 'unknown') for job in jobs.values())
    records['EO-CORE-CLOSURE'] = record('EO-CORE-CLOSURE', closure_status,
        'Explicit batch admissions, causal stages, terminal outcomes and job/service owner cleanup reconciled at the common cutoff.',
        unit='ratio', resource_scope=['admitted_jobs'], value=len(closed) / len(expected) if closure_status == 'measured' else None,
        numerator=len(closed), denominator=len(expected), closed_job_ids=sorted(closed), outcomes=dict(sorted(outcomes.items())))

    lease_status = _status([base['trace_integrity'], gates['gate_source_bytes'], gates['actual_stage_lease_ownership'], gates['atomic_budget_ledger']])
    if not batch_complete and lease_status != 'conflict':
        lease_status = 'unknown'
    intervals, endpoints, peak_cpu, violation_intervals = [], [], 0, []
    if lease_status == 'measured':
        endpoints = _ledger_rows(trace)
        if not endpoints:
            lease_status = 'unknown'
        active, cpu, gpu = {}, 0, set()
        for index, row in enumerate(endpoints):
            identity = (row['budget_instance_id'], row['lease_id'])
            charge, limit = row['cpu_threads'], row['limits']['cpu_threads']
            if type(charge) is not int or charge <= 0 or type(limit) is not int or limit <= 0:
                raise ValueError('Malformed atomic lease CPU charge or limit')
            if row['type'] == 'lease.acquired':
                if identity in active:
                    raise ValueError('Duplicate live lease')
                active[identity] = row
                cpu += charge
                if row['kind'] == 'gpu':
                    gpu.add(identity)
            elif row['type'] == 'lease.released':
                start = active.pop(identity, None)
                if start is None:
                    raise ValueError('Atomic lease release has no acquisition')
                cpu -= charge
                if row['kind'] == 'gpu':
                    gpu.remove(identity)
                    intervals.append({'lease_id': row['lease_id'], 'start_ns': start['monotonic_ns'], 'end_ns': row['monotonic_ns']})
            else:
                raise ValueError('Unsupported atomic lease endpoint')
            peak_cpu = max(peak_cpu, cpu)
            if (cpu > limit or cpu < 0 or len(gpu) > 1) and index + 1 < len(endpoints):
                violation_intervals.append((row['monotonic_ns'], endpoints[index + 1]['monotonic_ns']))
        if active or cpu or gpu:
            lease_status = 'unknown'
    gpu_overlaps = _merge_intervals([(max(a['start_ns'], b['start_ns']), min(a['end_ns'], b['end_ns']))
                                  for index, a in enumerate(intervals) for b in intervals[index + 1:]])
    budget_overlaps = _merge_intervals(violation_intervals)
    lease_coverage = {'atomic_endpoints': len(endpoints), 'complete_gpu_intervals': len(intervals),
                      'paired_model_leases': gates['actual_stage_lease_ownership'].get('paired_leases', 0),
                      'atomic_ledger_status': gates['atomic_budget_ledger']['status'],
                      'model_lease_status': gates['actual_stage_lease_ownership']['status']}
    records['MC-REC-GPU-OVERLAP'] = record('MC-REC-GPU-OVERLAP', lease_status,
        'Strict interval intersection of complete exclusive GPU leases in one physical model pool; lease time is not kernel busy time.',
        resource_scope=['exclusive_gpu_lease'], value=len(gpu_overlaps) if lease_status == 'measured' else None,
        lease_coverage=lease_coverage, intervals=intervals,
        violation_duration_seconds=sum(end - start for start, end in gpu_overlaps) / 1e9 if lease_status == 'measured' else None)
    records['EO-REC-CPU-GPU-OVERCOMMIT'] = record('EO-REC-CPU-GPU-OVERCOMMIT', lease_status,
        'Independent atomic endpoint replay merges CPU charge or exclusive GPU owner overcommit time regions; declared accounting only.',
        resource_scope=['declared_cpu_threads', 'exclusive_gpu_lease'],
        value=len(budget_overlaps) if lease_status == 'measured' else None,
        lease_coverage=deepcopy(lease_coverage), peak_cpu_lease_charge=peak_cpu if lease_status == 'measured' else None,
        violation_duration_seconds=sum(end - start for start, end in budget_overlaps) / 1e9 if lease_status == 'measured' else None)
    if lease_status == 'measured':
        grants = [row for row in endpoints if row['type'] == 'lease.acquired']
        gpu_grants = [row for row in grants if row['kind'] == 'gpu']
        records['MC-REC-GPU-OVERLAP']['knowledge_coverage'] = {
            'all_stage_leases': len(grants), 'all_endpoint_events': len(endpoints), 'gpu_leases': len(gpu_grants),
            'gpu_endpoint_events': len(gpu_grants) * 2, 'history_complete': True, 'dropped_events': 0}
        records['EO-REC-CPU-GPU-OVERCOMMIT'].update(
            budget_instance_id=grants[0]['budget_instance_id'], capacity_versions=sorted({row['capacity_version'] for row in endpoints}),
            knowledge_coverage={'leases': len(grants), 'cpu_leases': len(grants) - len(gpu_grants), 'gpu_leases': len(gpu_grants),
                'endpoint_events': len(endpoints), 'history_complete': True, 'dropped_events': 0,
                'peak_cpu_lease_charge': peak_cpu, 'declared_cpu_limit': grants[0]['limits']['cpu_threads'], 'final_active_leases': 0})
    records['EO-REC-OVERCOMMIT'] = record('EO-REC-OVERCOMMIT', 'unknown',
        'The supported trace adapter has no complete RAM and disk reservation/acknowledgement endpoint ledger. CPU/GPU evidence cannot establish a full-resource zero.',
        resource_scope=['declared_cpu_threads', 'exclusive_gpu_lease', 'ram_bytes', 'disk_bytes'],
        missing_resource_coverage=['ram_bytes', 'disk_bytes'])
    selected = [records.get(key) or record(key, 'unknown', 'No registered recomputation adapter for the selected metric.',
                resource_scope=['unsupported']) for key in contract['metric_ids']]
    return {'schema_version': 1, **common, 'evidence_binding': binding, 'metrics': selected,
            'scope': 'Finite evidence recomputation, not a new GPU experiment, media decode, execution attestation, semantic-quality or population-reliability proof.',
            'counts': {status: sum(row['status'] == status for row in selected) for status in ('measured', 'unknown', 'conflict')}}


def verify_claims(report, claims):
    """Reject stale/mis-scoped/incorrect observations, even if ratios coincide."""
    if not isinstance(claims, list):
        raise ValueError('Claims must be a list of complete metric records')
    expected = _indexed(report['metrics'], 'metric_id')
    provided = _indexed(claims, 'metric_id')
    if set(expected) != set(provided):
        raise ValueError('Claim metric coverage differs from the explicit selection')
    for key, actual in expected.items():
        claim = provided[key]
        # All derived fields matter, including coverage, identities and ratio
        # counts. Explanatory wording can differ but cannot substitute evidence.
        if set(claim) != set(actual):
            raise ValueError('Claim fields or evidence coverage differ for ' + key)
        if not isinstance(claim.get('reason'), str) or not claim['reason'].strip():
            raise ValueError('Claim must explain its measured or unknown scope')
        for field in set(actual) - {'reason'}:
            if semantic_sha256(claim[field]) != semantic_sha256(actual[field]):
                raise ValueError('Claim differs from evidence for ' + key + ': ' + field)
        if claim['status'] == 'conflict':
            raise ValueError('Conflicting evidence cannot be accepted as a verified metric claim')
    return {'status': 'pass', 'verified_metric_ids': list(expected),
            'scope': 'Exact evidence-derived values, identities, windows, resource scopes and coverage; unknown remains unknown.'}


def verify_knowledge_observations(report, modules, *, contract_id):
    """Validate selected actual module observations against recomputed evidence.

    Free-text window/scope prose is deliberately not interpreted. New structured
    bindings make the selection reviewable and prevent prose from impersonating
    evidence. Unselected historical observations are outside this check's scope.
    """
    if not _token(contract_id):
        raise ValueError('Explicit measurement contract identity required')
    available = {}
    for module in modules:
        for perspective in module.get('perspectives', {}).values():
            for metric in perspective.get('metrics', []):
                key = metric.get('id')
                if key in available:
                    raise ValueError('Duplicate knowledge metric identity')
                available[key] = metric
    verified = []
    identity_fields = ('experiment_id', 'protocol_id', 'manager_instance_id', 'model_instance_id')
    for actual in report['metrics']:
        key = actual['metric_id']
        if key not in available:
            raise ValueError('Selected knowledge metric missing: ' + key)
        candidates = [observation for observation in available[key].get('observations', [])
                      if observation.get('experiment_id') == report['experiment_id']
                      or observation.get('measurement_contract_id') == contract_id]
        if len(candidates) != 1:
            raise ValueError('Knowledge selection missing or ambiguous: ' + key)
        observation = candidates[0]
        required = {name: actual[name] for name in identity_fields}
        required.update(measurement_contract_id=contract_id, evidence_window=actual['window'],
                        resource_scope=actual['resource_scope'], measurement_coverage=actual['coverage'],
                        measurement_evidence_binding=actual['evidence_binding'], status=actual['status'], value=actual['value'])
        for field in ('numerator', 'denominator', 'budget_instance_id', 'capacity_versions', 'violation_duration_seconds'):
            if field in actual:
                required[field] = actual[field]
        if 'outcomes' in actual:
            required['outcomes'] = {name: actual['outcomes'].get(name, 0)
                                    for name in ('succeeded', 'cancelled', 'failed', 'running', 'unknown')}
        if 'knowledge_coverage' in actual:
            required['coverage'] = actual['knowledge_coverage']
        for field, expected in required.items():
            if field not in observation:
                raise ValueError('Knowledge observation missing structured evidence field: ' + key + ': ' + field)
            given = observation[field]
            # Existing integer zero and computed floating zero have equal units;
            # booleans/NaN are never numeric measurements. Ratios still require
            # exact independently derived numerator and denominator.
            if field in ('value', 'violation_duration_seconds') and expected is not None:
                equal = _finite(given) and given == expected
            else:
                equal = semantic_sha256(given) == semantic_sha256(expected)
            if not equal:
                raise ValueError('Knowledge observation conflicts with recomputed evidence: ' + key + ': ' + field)
        if available[key].get('unit') != actual['unit']:
            raise ValueError('Knowledge metric unit differs from adapter')
        if actual['status'] == 'conflict':
            raise ValueError('Conflicting evidence cannot verify a knowledge observation')
        if actual['status'] == 'unknown' and (not isinstance(observation.get('reason'), str) or not observation['reason'].strip()):
            raise ValueError('Unknown knowledge observation needs an explicit limitation')
        verified.append(key)
    return {'status': 'pass', 'contract_id': contract_id, 'verified_metric_ids': verified,
            'scope': 'Only selected experiment observations with explicit structured windows and resource coverage; older observations and free-text semantic equivalence are not verified.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('trace', 'client', 'contract'):
        parser.add_argument('--' + name, type=Path, required=True)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument('--source-root', type=Path, help='Exact executed runtime source tree; never defaults to current code')
    sources.add_argument('--source-binding', type=Path, help='Immutable historical binding descriptor verified against trace and snapshot')
    parser.add_argument('--claims', type=Path)
    parser.add_argument('--knowledge-module', type=Path, action='append', default=[], help='Actual knowledge module to verify; repeat for each module')
    parser.add_argument('--contract-id', help='Explicit measurement_contract_id required for knowledge verification')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    read = lambda path: json.loads(path.read_text(encoding='utf-8'))
    trace, client, contract = read(args.trace), read(args.client), read(args.contract)
    source_root = (load_module('source_bindings').bound_source_root(args.source_binding, root=ROOT, trace=trace)
                   if args.source_binding else args.source_root)
    result = replay(trace, client, contract, source_root=source_root)
    if args.claims:
        result['claim_verification'] = verify_claims(result, read(args.claims))
    if args.knowledge_module:
        result['knowledge_verification'] = verify_knowledge_observations(result, [read(path) for path in args.knowledge_module], contract_id=args.contract_id)
    rendered = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.write_text(rendered, encoding='utf-8', newline='\n')
    else:
        print(rendered, end='')
    return 1 if result['counts']['conflict'] else (0 if args.knowledge_module or args.claims else 2 if result['counts']['unknown'] else 0)


if __name__ == '__main__':
    raise SystemExit(main())
