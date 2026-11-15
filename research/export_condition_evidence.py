"""Explicit public projection for a declared single-instance H3 experiment.

Usage: python research/export_condition_evidence.py PRIVATE_RUN PUBLIC_DIRECTORY
       --experiment PRIVATE_RUN/experiment.json --source-root EXECUTED_SOURCE_ROOT
The destination must be new. No network, GPU, old experiments or implicit glob
copies are used. Both analyzers must return exactly the same output before and
after projection, including unknown and failure results. Export is publication
preparation, not evidence that an incomplete run succeeded.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MAX_JSON_BYTES = 256 * 1024 * 1024
MAX_ASSET_BYTES = 256 * 1024 * 1024
TOKEN = re.compile(r'[A-Za-z0-9_.:-]{1,256}\Z')
ASSET_NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,80}\Z')
HASH = re.compile(r'[0-9a-f]{64}\Z')
PRIVATE_TEXT = re.compile(r'(?i)(?:[a-z][a-z0-9+.-]*://|\bBearer\s+|[A-Z]:[\\/]|\\\\|'
                          r'(?:^|[\s\"\'=])/[A-Za-z0-9_.~-]+(?:/|\b))')
SENSITIVE_KEY = re.compile(r'(?i)(?:apikey|controlkey|authorization|password|accesstoken|secret|bearer)')


def scalar(value):
    if value is None or type(value) in {bool, int}:
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if type(value) is str and len(value) <= 16384 and not PRIVATE_TEXT.search(value):
        return value
    raise ValueError('Retained field is not a public finite scalar; review its declared schema')


def names(text):
    return {name: scalar for name in text.split()}


def sequence(schema):
    return ('sequence', schema)


def mapping(schema):
    return ('mapping', schema)


def project(value, schema):
    """Retain declared fields only, never recursively copy an arbitrary object."""
    if callable(schema):
        return schema(value)
    if isinstance(schema, tuple):
        mode, child = schema
        if mode == 'sequence':
            if type(value) is not list:
                return scalar(value)
            return [project(item, child) for item in value]
        if type(value) is not dict:
            return scalar(value)
        if any(not isinstance(key, str) or not TOKEN.fullmatch(key) for key in value):
            raise ValueError('Invalid public observation-map field identity')
        return {key: project(item, child) for key, item in value.items()}
    if type(value) is not dict:
        return scalar(value)
    return {key: project(value[key], child) for key, child in schema.items() if key in value}


def credential_values(value):
    """Extra refusal check; field allowlists remain the primary privacy boundary."""
    found = set()
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = re.sub(r'[^a-z0-9]', '', str(key).lower())
            if SENSITIVE_KEY.search(normalized) and isinstance(item, str) and item:
                found.add(item)
            found.update(credential_values(item))
    elif isinstance(value, list):
        for item in value:
            found.update(credential_values(item))
    return found


def refuse_credentials(value, credentials):
    if isinstance(value, dict):
        for key, child in value.items():
            refuse_credentials(key, credentials)
            refuse_credentials(child, credentials)
    elif isinstance(value, list):
        for child in value:
            refuse_credentials(child, credentials)
    elif isinstance(value, str) and any(secret == value or len(secret) >= 8 and secret in value for secret in credentials):
        raise ValueError('A private credential was reused in a retained field; publication refused')


RESOURCE = names('ram_bytes disk_bytes cpu_threads')
OPTIONS = names('attention_backend acceleration_lora steps attention_scope precision denoiser_evaluations style_lora style_scale')
GENERATION = {'requested': OPTIONS, 'effective': OPTIONS}
GATE_CONTEXT = names('manager_instance_id model_instance_id instance_id job_id attempt_id source_file source_sha256 '
                     'pipeline budget_scope budget_reason stage lease_id budget_instance_id budget_decision_id '
                     'resource_kind reservation_id capacity_version')
GATE = {**names('schema_version gate_version gate_id decision_id observed_at monotonic_ns allowed'),
        'context': GATE_CONTEXT, 'conditions': mapping(scalar), 'blocked': sequence(scalar)}
RESERVATION = {**names('owned budget_instance_id attempt_id reservation_id capacity_version'), 'resources': RESOURCE}
LEASE = {**names('owned lease_id attempt_id kind budget_instance_id reservation_id cpu_threads capacity_version '
                 'decision_id granted_at granted_monotonic_ns released_at released_monotonic_ns'), 'grant_receipt': GATE}
LEASE_EVENT = {**names('event_id sequence type observed_at monotonic_ns lease_id budget_instance_id attempt_id kind '
                       'cpu_threads capacity_version decision_id reservation_id cpu_threads_in_use gpu_attempt_id '
                       'sha256 previous_sha256'), 'limits': RESOURCE}
AUDIT_PAGE = {**names('schema_version model_instance_id budget_instance_id capacity_version gpu_host_cpu_threads '
                     'after_sequence through_sequence next_sequence after_sha256 tail_sha256 page_complete more'),
              'limits': RESOURCE, 'events': sequence(LEASE_EVENT)}
BUDGET = {**names('budget_instance_id capacity_version cpu_threads_in_use gpu_attempt_id gpu_host_cpu_threads_in_use '
                  'cpu_stage_threads_in_use gpu_host_cpu_threads requested_cpu_threads requested_gpu_host_cpu_threads '
                  'admitted_attempts unacknowledged_result_bytes model_peak_ram_bytes os_reserve_bytes live_pressure_checks'),
          'limits': RESOURCE, 'reserved': RESOURCE, 'free': RESOURCE, 'last_gate_receipts': mapping(GATE),
          'active_leases': sequence(LEASE),
          'lease_history': {**names('capacity count dropped history_complete'), 'events': sequence(LEASE_EVENT)},
          'lease_journal': names('schema_version budget_instance_id count tail_sha256 complete error'),
          'io_budget': names('scope bandwidth_calibrated bytes_per_second_limit optimality')}
MEDIA = names('frames fps width height audio_channels full_decode_passed duration_seconds video_codec audio_codec')
ARTIFACT = {**names('bytes sha256'), 'media': MEDIA}
COMPLETIONS = names('prepare gpu encode_validate')
RESULT = {**names('instance_id job_id attempt_id status error started_at finished_at cleanup_complete'),
          'artifact': ARTIFACT, 'generation_options': GENERATION, 'condition_gates': mapping(GATE),
          'stage_completions': COMPLETIONS}
EVENT_DATA = {**names('module pid model_instance_id api_ready model_ready public_api_ready ready phase cleanup_complete '
                      'model_alive server_alive tunnel_alive command_id action condition_decision_id'), 'receipt': GATE}
EVENT = {**names('event_id sequence instance_id job_id attempt_id type monotonic_ns observed_at stage succeeded accepting '
                 'phase worker_cleanup_complete condition_decision_id lease_id budget_instance_id gate_id'),
         'resources': RESOURCE, 'generation_options': GENERATION, 'data': EVENT_DATA, 'receipt': GATE,
         'reservation_receipt': GATE, 'reservation_witness': RESERVATION, 'lease_witness': LEASE,
         'budget_snapshot': BUDGET, 'result': RESULT}
JOB = {**names('id job_id status created_at started_at finished_at attempt_id model_instance_id instance_id '
               'cleanup_complete model_lease_released dispatch_state cancellation_requested error'),
       'generation_options': GENERATION, 'artifact': ARTIFACT, 'condition_gate': GATE,
       'execution_condition_gates': mapping(GATE), 'stage_completions': COMPLETIONS,
       'lifecycle': sequence(names('state observed_at'))}
SNAPSHOT = {**names('instance_id model_instance_id phase observed_at observation_age_seconds observation_stale '
                    'api_ready model_ready public_api_ready'), 'condition_gates': mapping(GATE),
            'modules': {'model': {**names('instance_id observation_unknown ready accepting worker_alive phase'),
                                  'budget': BUDGET, 'last_condition_gate': GATE}}}
SOURCES = {'files': mapping(names('sha256 bytes')), 'unavailable': sequence(names('file error_type')), 'scope': scalar}
TRACE = {**names('schema_version manager_instance_id model_instance_id'), 'queued_cancel_job_ids': sequence(scalar),
         'collection': {**names('manager_events_complete model_events_complete manager_event_sequence_final '
                               'model_event_sequence_final basis collector_id'), 'problems': sequence(names('source error_type reason'))},
         'manager_events': sequence(EVENT), 'model_events': sequence(EVENT), 'jobs': sequence(JOB),
         'budget_audit_pages': sequence(AUDIT_PAGE),
         'system_snapshots': sequence(SNAPSHOT), 'runtime_source_identity': SOURCES,
         'cleanup_receipt': {'instance_id': scalar,
             'state': names('instance_id cleanup_complete shutdown_complete manager_alive server_alive model_alive '
                            'tunnel_alive manager_launcher_pid'),
             'owner_observation': {**names('alive cleanup_complete observed_at'), 'pids': sequence(scalar),
                                   'identities': sequence(names('pid created_at')), 'errors': sequence(scalar)}},
         'platform_release': names('released manager_instance_id instance_id observed_at observed_at_iso8601 source evidence_sha256')}
PARAMETERS = names('seconds width height steps acceleration_lora attention_backend')
DOWNLOAD = {**names('verified state sha256 expected_sha256 bytes error_type'), 'media': MEDIA}
CLIENT_JOB = {**names('name disposition state job_id reference_sha256 submission_http_status admission_state '
                      'attempt_id cleanup_complete cancel_http_status'),
              'pre_submit_observation': names('model_ready observed_at'), 'download': DOWNLOAD,
              'cancel_intent': names('source observed_at')}
CLIENT = {**names('schema_version run_id instance_id manifest_sha256 state started_at updated_at finished_at reason'),
          'parameters': PARAMETERS, 'cold_queue_precondition': names('model_ready public_api_ready observed_at'),
          'jobs': sequence(CLIENT_JOB)}
GPU_SAMPLE = {**names('collector_id manager_instance_id monotonic_ns finished_monotonic_ns observed_at probe_duration_seconds'),
              'observation': {**names('source synchronizes_cuda interpretation'), 'devices': sequence(names(
                  'device_timestamp index name memory_used_mib memory_total_mib utilization_gpu_percent power_watts'))}}
API_UPDATE = {**names('schema_version phase started_at finished_at manager_instance_id model_instance_id '
                      'model_pid_before model_pid_after api_pid_before api_pid_after source_file before_sha256 after_sha256 '
                      'reason error_type public_api_ready model_identity_preserved manager_identity_preserved '
                      'job_identities_preserved gate_sources_preserved'),
              'commands': sequence(names('command_id action status created_at finished_at error')),
              'gate_source_sha256_before': mapping(scalar), 'gate_source_sha256_after': mapping(scalar),
              'job_ids_before': sequence(scalar), 'job_ids_after': sequence(scalar)}
API_QUERY = {**names('id instance_id status attempt_id cancellation_requested'),
             'condition_gate': GATE, 'execution_condition_gates': mapping(GATE), 'stage_completions': COMPLETIONS}
PRECISION_COMPONENTS = {'transformer_ref', 'text_encoder', 'vae', 'audio_vae'}
PARAMETER_DTYPES = {'torch.bfloat16', 'torch.float32'}
MODULE_NAME = re.compile(r'[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)*\Z')
PRECISION_COMPONENT = {**names('component status'),
    'parameter_counts': names('torch.bfloat16 torch.float32'),
    'tensor_counts': names('torch.bfloat16 torch.float32'), 'keep_in_fp32_modules': sequence(scalar)}
PRECISION = {**names('schema_version manager_instance_id model_instance_id observed_at status scope '
    'source_log_snapshot_sha256 source_file source_sha256 gpu_query python_version interpretation'),
    'components': mapping(PRECISION_COMPONENT)}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def owned_file(root, relative, limit=MAX_JSON_BYTES):
    root = Path(root).resolve()
    requested = Path(relative)
    if requested.is_absolute() or '..' in requested.parts:
        raise ValueError('Evidence input must be a relative path inside the declared source')
    lexical = root / requested
    resolved = lexical.resolve()
    if resolved != lexical or not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError('Evidence input missing, symlinked, or outside its owner')
    if resolved.stat().st_size > limit:
        raise ValueError('Evidence input exceeds its bounded export size')
    return resolved


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def normalize_platform_release(release):
    """Convert an explicitly zoned ISO observation to epoch seconds without a new observation."""
    result = dict(release)
    original = result.get('observed_at')
    if isinstance(original, str):
        stamp = datetime.fromisoformat(original.replace('Z', '+00:00'))
        if stamp.tzinfo is None or stamp.utcoffset() is None:
            raise ValueError('Platform release time requires an explicit timezone')
        result.update(observed_at=stamp.timestamp(), observed_at_iso8601=original)
    return result


def load_analyzer(name):
    spec = importlib.util.spec_from_file_location('public_' + name, Path(__file__).with_name(name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.analyze


def analysis_equivalence(raw, public, source_root):
    base = load_analyzer('analyze_trace')
    gates = load_analyzer('analyze_condition_gates')
    original = {'analysis': base(raw), 'condition_analysis': gates(raw, source_root)}
    projected = {'analysis': base(public), 'condition_analysis': gates(public, source_root)}
    if original != projected:
        raise ValueError('Public projection changes an analyzer conclusion; publication refused')
    return projected


def project_trace(trace, source_root=None):
    public = project(trace, TRACE)
    refuse_credentials(public, credential_values(trace))
    analyses = analysis_equivalence(trace, public, source_root)
    public['publication_projection'] = ('Explicit nested field allowlist; addresses, private paths, keys, raw logs '
        'and unneeded metrics omitted. Both analyzer outputs exactly match the private trace. '
        'Unknown, failure and unexercised conditions retain their original status.')
    return public, analyses


def _validate_content(source, trace, client, manifest, manifest_sha256):
    if client.get('instance_id') != trace.get('manager_instance_id'):
        raise ValueError('Client and trace manager identities differ')
    if client.get('manifest_sha256') != manifest_sha256:
        raise ValueError('Client manifest identity differs from this content manifest')
    jobs, observations = manifest.get('jobs'), client.get('jobs')
    if (not isinstance(jobs, list) or len(jobs) != 3 or not isinstance(observations, list) or len(observations) != 3
            or sorted(job.get('disposition', '') for job in jobs) != ['cancel', 'retain', 'retain']):
        raise ValueError('This experiment requires two retained jobs and one cancellation target')
    if len({job.get('name') for job in jobs}) != 3 or {job.get('name') for job in jobs} != {job.get('name') for job in observations}:
        raise ValueError('Content/client job names do not match uniquely')
    records = {job.get('job_id', job.get('id')): job for job in trace.get('jobs', [])}
    public = project(manifest, {**names('schema_version scope content_scope'), 'parameters': PARAMETERS})
    public['jobs'], assets = [], []
    for job in jobs:
        name = job.get('name')
        if not isinstance(name, str) or not ASSET_NAME.fullmatch(name):
            raise ValueError('Unsafe experiment asset name')
        row = next(item for item in observations if item.get('name') == name)
        if row.get('disposition') != job.get('disposition'):
            raise ValueError('Content/client dispositions differ')
        reference = owned_file(source, job['reference_file'], MAX_ASSET_BYTES)
        if reference.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.webp'}:
            raise ValueError('Unexpected reference media type')
        reference_sha = digest(reference)
        if reference_sha != row.get('reference_sha256'):
            raise ValueError('Reference bytes changed since client admission')
        reference_name = 'assets/' + name + reference.suffix.lower()
        item = project(job, names('name seed disposition prompt'))
        item.update(reference_file=reference_name, reference_sha256=reference_sha)
        public['jobs'].append(item)
        assets.append((reference_name, reference, reference_sha))
        identity = row.get('job_id')
        if identity is None:
            if row.get('state') in {'succeeded', 'cancelled'}:
                raise ValueError('Terminal client claim lacks a job identity')
            continue
        if not isinstance(identity, str) or not TOKEN.fullmatch(identity):
            raise ValueError('Unsafe client job identity')
        record = records.get(identity)
        if (not isinstance(record, dict) or record.get('instance_id') != client.get('instance_id')
                or record.get('status') != row.get('state') or record.get('attempt_id') != row.get('attempt_id')):
            raise ValueError('Client terminal record and trace job/attempt identities disagree')
        if row.get('state') != 'succeeded' or row.get('disposition') != 'retain':
            continue
        download = row.get('download') or {}
        video = owned_file(source, 'client/' + identity + '.mp4', MAX_ASSET_BYTES)
        observed, size = digest(video), video.stat().st_size
        artifact = record.get('artifact') or {}
        if (download.get('verified') is not True or (download.get('media') or {}).get('full_decode_passed') is not True
                or observed != download.get('sha256') or observed != download.get('expected_sha256')
                or observed != artifact.get('sha256') or size != download.get('bytes') or size != artifact.get('bytes')):
            raise ValueError('Video identity or recorded client decoding validation is unconfirmed')
        video_name = 'assets/' + name + '.mp4'
        item.update(video_file=video_name, video_sha256=observed, video_bytes=size)
        assets.append((video_name, video, observed))
    return public, assets


def project_api_update(raw, trace, files):
    public = project(raw, API_UPDATE)
    refuse_credentials(public, credential_values(raw))
    if (public.get('manager_instance_id') != trace.get('manager_instance_id')
            or public.get('model_instance_id') != trace.get('model_instance_id')
            or public.get('source_file') != 'server.py'):
        raise ValueError('API update does not belong to this manager/model/server')
    for field in ('before_sha256', 'after_sha256'):
        if not isinstance(public.get(field), str) or not HASH.fullmatch(public[field]):
            raise ValueError('API update lacks a valid before/after source hash')
    if public['after_sha256'] != files.get('server.py', {}).get('sha256'):
        raise ValueError('Replacement API bytes differ from final runtime source identity')
    gate_names = {'condition_guard.py', 'global_manager.py', 'model_container.py', 'pipeline_scheduler.py'}
    before, after = public.get('gate_source_sha256_before'), public.get('gate_source_sha256_after')
    if not isinstance(before, dict) or not isinstance(after, dict) or set(before) != gate_names or set(after) != gate_names:
        raise ValueError('API update requires all four gate source identities on both sides')
    unchanged = {name: before[name] == after[name] == files.get(name, {}).get('sha256') for name in sorted(gate_names)}
    if not all(unchanged.values()):
        raise ValueError('Gate source identity changed across the API-only update')
    if public.get('model_identity_preserved') is True and (
            type(public.get('model_pid_before')) is not int or public['model_pid_before'] <= 0
            or public.get('model_pid_after') != public['model_pid_before']):
        raise ValueError('API update preservation flag contradicts the recorded model PIDs')
    if public.get('job_identities_preserved') is True and (
            type(public.get('job_ids_before')) is not list
            or public.get('job_ids_after') != public['job_ids_before']):
        raise ValueError('API update preservation flag contradicts the recorded job identities')
    return public, {'initial_server_sha256': public['before_sha256'], 'replacement_server_sha256': public['after_sha256'],
        'gate_source_endpoint_matches': unchanged, 'recorded_phase': public.get('phase'),
        'scope': 'Two recorded source/instance observations around managed API replacement; unchanged endpoint hashes do not attest every intermediate memory state.'}


def project_api_query(raw, trace, client):
    public = project(raw, API_QUERY)
    refuse_credentials(public, credential_values(raw))
    identity = public.get('id')
    row = next((row for row in client.get('jobs', []) if row.get('job_id') == identity), None)
    if row is None or public.get('instance_id') != trace.get('manager_instance_id'):
        raise ValueError('Public API query is not bound to a job in this client experiment')
    # This may be an earlier queued/running observation, not the final outcome.
    if public.get('attempt_id') is not None and public['attempt_id'] != row.get('attempt_id'):
        raise ValueError('Public API query belongs to a different execution attempt')
    if public.get('status') not in {'queued', 'running', 'cancel_requested', 'cancelled', 'succeeded', 'failed'}:
        raise ValueError('Public API query has an unknown job state')
    gate = public.get('condition_gate')
    if isinstance(gate, dict):
        context = gate.get('context') or {}
        if context.get('job_id') != identity or context.get('manager_instance_id') != trace.get('manager_instance_id'):
            raise ValueError('Public API decision receipt belongs to a different manager/job')
        if gate.get('allowed') is True and (context.get('model_instance_id') != trace.get('model_instance_id')
                or context.get('attempt_id') != public.get('attempt_id')):
            raise ValueError('Allowed public API decision receipt belongs to another model/attempt')
    return public


def project_precision(raw, trace, files):
    public = project(raw, PRECISION)
    refuse_credentials(public, credential_values(raw))
    if (any(type(public.get(field)) is not str or not TOKEN.fullmatch(public[field])
            for field in ('manager_instance_id', 'model_instance_id'))
            or type(public.get('source_sha256')) is not str or not HASH.fullmatch(public['source_sha256'])
            or public.get('manager_instance_id') != trace.get('manager_instance_id')
            or public.get('model_instance_id') != trace.get('model_instance_id')
            or public.get('source_file') != 'model_backend.py'
            or public.get('source_sha256') != files.get('model_backend.py', {}).get('sha256')):
        raise ValueError('Parameter audit is not bound to this model/backend source')
    if public.get('scope') != 'loaded_parameters_only' or public.get('status') != 'passed':
        raise ValueError('Unsupported parameter audit scope/status; do not infer compute precision')
    if not isinstance(public.get('source_log_snapshot_sha256'), str) or not HASH.fullmatch(public['source_log_snapshot_sha256']):
        raise ValueError('Parameter audit lacks its private source log byte identity')
    components = raw.get('components')
    if not isinstance(components, dict) or not components or set(components) - PRECISION_COMPONENTS:
        raise ValueError('Parameter audit contains an unreviewed component identity')
    for name, component in components.items():
        if not isinstance(component, dict) or component.get('component') != name or component.get('status') != 'passed':
            raise ValueError('Parameter audit component identity/status disagrees')
        for field in ('parameter_counts', 'tensor_counts'):
            values = component.get(field)
            if (not isinstance(values, dict) or not values or set(values) - PARAMETER_DTYPES
                    or any(type(value) is not int or value < 0 for value in values.values())):
                raise ValueError('Parameter audit dtype/count cannot be silently projected away')
        if set(component['parameter_counts']) != set(component['tensor_counts']):
            raise ValueError('Parameter and tensor dtype counts cover different fields')
        modules = component.get('keep_in_fp32_modules')
        if (type(modules) is not list or len(modules) > 256 or any(type(value) is not str
                or len(value) > 256 or not MODULE_NAME.fullmatch(value)
                or SENSITIVE_KEY.search(re.sub(r'[^a-z0-9]', '', value.lower())) for value in modules)):
            raise ValueError('FP32 exception list contains an unreviewed module name or private field')
    query = public.get('gpu_query')
    if not isinstance(query, str) or not query or len(query) > 4096:
        raise ValueError('GPU precision-audit query is missing or oversized')
    rows = list(csv.reader(query.splitlines()))
    if not 1 <= len(rows) <= 8:
        raise ValueError('GPU precision-audit query must contain bounded device rows')
    for row in rows:
        if len(row) != 3:
            raise ValueError('GPU precision-audit query requires exactly three columns')
        name, memory, driver = [value.strip() for value in row]
        if (not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9 ()_.+-]{0,100}', name)
                or not re.fullmatch(r'[0-9]+(?:\.[0-9]+)?', memory)
                or not re.fullmatch(r'[0-9]+(?:\.[0-9]+){1,3}', driver)):
            raise ValueError('GPU precision-audit query contains unexpected or private text')
    if not isinstance(public.get('python_version'), str) or not re.fullmatch(r'[0-9]+(?:\.[0-9]+){1,2}', public['python_version']):
        raise ValueError('Parameter audit Python version must be a numeric version')
    return public


def declared_experiment(source, descriptor_path, captured, raw, state):
    """Bind provenance to explicit inputs; never inherit an earlier experiment.

    Commit strings are declared identities, not Git attestations. Executed source
    bytes, protocol bytes, instance identity and supported interventions are
    independently checked against captured inputs before publication.
    """
    try:
        relative = Path(descriptor_path).resolve().relative_to(source)
    except (TypeError, ValueError):
        raise ValueError('Explicit experiment descriptor missing or outside the private run') from None
    path = owned_file(source, relative)
    data = path.read_bytes()
    descriptor = json.loads(data)
    fields = {'schema_version', 'experiment_id', 'protocol', 'baseline_commit', 'source_revision',
              'runtime_changes', 'runtime_manifest_sha256', 'manager_instance_id', 'model_instance_id', 'interventions'}
    if not isinstance(descriptor, dict) or set(descriptor) != fields or type(descriptor['schema_version']) is not int or descriptor['schema_version'] != 1:
        raise ValueError('Experiment descriptor requires its exact declared fields')
    if (descriptor['experiment_id'] != state.get('run_id')
            or descriptor['manager_instance_id'] != raw.get('manager_instance_id')
            or descriptor['model_instance_id'] != raw.get('model_instance_id')):
        raise ValueError('Experiment identity differs from client or runtime')
    for name in ('experiment_id', 'manager_instance_id', 'model_instance_id'):
        if not isinstance(descriptor[name], str) or not TOKEN.fullmatch(descriptor[name]):
            raise ValueError('Experiment identity must be a bounded public identifier')
    if not isinstance(descriptor['baseline_commit'], str) or not re.fullmatch(r'[0-9a-f]{40}', descriptor['baseline_commit']):
        raise ValueError('Explicit baseline commit must be a full commit identity')
    revision = descriptor['source_revision']
    if (not isinstance(revision, dict) or set(revision) != {'commit', 'state'}
            or revision['state'] not in {'committed', 'modified'}
            or not isinstance(revision['commit'], str) or not re.fullmatch(r'[0-9a-f]{40}', revision['commit'])):
        raise ValueError('Explicit executed source revision and modification state required')
    changes = scalar(descriptor['runtime_changes'])
    if not isinstance(changes, str) or not changes.strip() or len(changes) > 2048:
        raise ValueError('Explicit public runtime change description required')
    if descriptor['runtime_manifest_sha256'] != hashlib.sha256(captured['runtime']).hexdigest():
        raise ValueError('Experiment runtime manifest binding does not match captured bytes')
    protocol = descriptor['protocol']
    if not isinstance(protocol, dict) or set(protocol) != {'id', 'file', 'sha256'}:
        raise ValueError('Explicit protocol identity, file and SHA256 required')
    protocol_path = owned_file(source, protocol['file'])
    protocol_bytes = protocol_path.read_bytes()
    protocol_body = json.loads(protocol_bytes)
    if (not isinstance(protocol['id'], str) or not TOKEN.fullmatch(protocol['id'])
            or not isinstance(protocol_body, dict) or protocol_body.get('protocol_id') != protocol['id']
            or protocol['sha256'] != hashlib.sha256(protocol_bytes).hexdigest()):
        raise ValueError('Protocol identity or byte binding conflicts with its declared source')
    interventions = descriptor['interventions']
    if not isinstance(interventions, list) or len(interventions) > 1:
        raise ValueError('Declare the bounded supported intervention list, including an explicit empty list')
    expected = [] if 'api_update' not in captured else [{'kind': 'managed_api_replacement',
        'evidence_sha256': hashlib.sha256(captured['api_update']).hexdigest()}]
    if interventions != expected:
        raise ValueError('Declared interventions do not match supported captured intervention evidence')
    public = {key: descriptor[key] for key in ('experiment_id', 'baseline_commit', 'source_revision', 'runtime_changes',
        'manager_instance_id', 'model_instance_id', 'runtime_manifest_sha256', 'interventions')}
    public.update(protocol_id=protocol['id'], protocol_sha256=protocol['sha256'],
                  experiment_descriptor_sha256=hashlib.sha256(data).hexdigest())
    return public, {'experiment': path, 'protocol': protocol_path}, {'experiment': data, 'protocol': protocol_bytes}


def export(source, destination, *, source_root=None, experiment=None):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError('Private source and public destination must be separate directories')
    if destination.exists():
        raise ValueError('Destination must be new; existing evidence is never overwritten')
    if source_root is None or experiment is None:
        raise ValueError('Explicit experiment descriptor and executed source-root are required; no missing input inherits current or historical defaults')
    source_root = Path(source_root).resolve()
    inputs = {name: owned_file(source, relative) for name, relative in {
        'trace': 'trace-complete.private.json', 'runtime': 'cloud/collection/runtime-source-manifest.json',
        'client': 'client/state.json', 'manifest': 'content-manifest.json',
        'gpu_after_cleanup': 'cloud/gpu-after-cleanup.json', 'gpu_samples': 'cloud/collection/gpu_samples.jsonl',
    }.items()}
    if (source / 'cloud/collection/api-query-update.json').exists():
        inputs['api_update'] = owned_file(source, 'cloud/collection/api-query-update.json')
    if (source / 'api-query-observation.private.json').exists():
        inputs['api_query'] = owned_file(source, 'api-query-observation.private.json')
    if (source / 'cloud/collection/precision-audit.json').exists():
        inputs['precision'] = owned_file(source, 'cloud/collection/precision-audit.json')
    if (source / 'platform-release.json').exists():
        inputs['platform_release'] = owned_file(source, 'platform-release.json')
    captured = {name: path.read_bytes() for name, path in inputs.items()}
    input_hashes = {name: hashlib.sha256(data).hexdigest() for name, data in captured.items()}
    raw = json.loads(captured['trace'])
    if 'platform_release' in captured:
        platform = normalize_platform_release(json.loads(captured['platform_release']))
        if any((raw.get('platform_release') or {}).get(key) != value for key, value in platform.items()):
            raise ValueError('Attached platform release does not match its retained source observation')
    runtime, state, manifest = (json.loads(captured[name]) for name in ('runtime', 'client', 'manifest'))
    declaration, extra_inputs, extra_bytes = declared_experiment(source, experiment, captured, raw, state)
    inputs.update(extra_inputs)
    captured.update(extra_bytes)
    input_hashes.update({name: hashlib.sha256(data).hexdigest() for name, data in extra_bytes.items()})
    if runtime.get('files') != raw.get('runtime_source_identity', {}).get('files'):
        raise ValueError('Runtime source manifest differs from the trace source identity')
    source_manifest = project(runtime, SOURCES)
    # Reuse the collector's finite source catalog, not a caller-controlled path.
    spec = importlib.util.spec_from_file_location('export_source_catalog', Path(__file__).with_name('collect_runtime.py'))
    catalog = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(catalog)
    files = runtime.get('files', {})
    if not files or set(files) - set(catalog.SOURCE_ALLOWLIST):
        raise ValueError('Runtime manifest includes undeclared source files')
    source_matches = {}
    for name, item in files.items():
        expected = item.get('sha256')
        if not isinstance(expected, str) or not HASH.fullmatch(expected):
            raise ValueError('Invalid runtime source hash')
        path = owned_file(source_root, name)
        source_matches[name] = digest(path) == expected
    if not all(source_matches.values()):
        raise ValueError('Runtime code bytes differ from the recorded experiment source')
    public_trace, analyses = project_trace(raw, source_root)
    public_client = project(state, CLIENT)
    public_manifest, assets = _validate_content(source, raw, state, manifest, input_hashes['manifest'])
    gpu_samples = [project(json.loads(line), GPU_SAMPLE) for line in captured['gpu_samples'].decode('utf-8').splitlines() if line.strip()]
    if any(sample.get('manager_instance_id') != raw.get('manager_instance_id') for sample in gpu_samples):
        raise ValueError('GPU telemetry belongs to a different manager lifetime')
    after = json.loads(captured['gpu_after_cleanup'])
    if after.get('instance_id') != raw.get('manager_instance_id'):
        raise ValueError('GPU cleanup observation belongs to a different manager lifetime')
    gpu_after = project(after, names('returncode instance_id observed_at'))
    raw_processes = after.get('compute_processes_output')
    gpu_after.update(compute_processes_output_empty=(raw_processes == '' if isinstance(raw_processes, str) else None),
                     compute_processes_output_sha256=(hashlib.sha256(raw_processes.encode()).hexdigest() if isinstance(raw_processes, str) else None),
                     scope='Exit code and emptiness of the recorded process query; raw process strings are private. Not platform deallocation.')
    provenance = {'schema_version': 2, **declaration,
        'runtime_source_match': source_matches, 'private_input_sha256': input_hashes,
        'analysis_equivalence': {'historical_trace': True, 'condition_gates': True},
        'asset_sha256': {name: expected for name, _, expected in assets},
        'research_source_sha256': {name: digest(Path(__file__).with_name(name)) for name in
            ('export_condition_evidence.py', 'analyze_trace.py', 'analyze_condition_gates.py')},
        'scope': 'Explicit public projection of the declared single-manager/model run. Commit identities and change descriptions are declarations, not execution attestation; protocol and executed source bytes are checked. Supported intervention is one managed API replacement; unsupported histories are rejected. No new GPU or media decode test, calibration or finite-model results reused.'}
    documents = {'trace.json': public_trace, 'analysis.json': analyses['analysis'],
        'condition-analysis.json': analyses['condition_analysis'], 'client-receipt.json': public_client,
        'content-manifest.json': public_manifest, 'runtime-source-manifest.json': source_manifest,
        'gpu-samples.json': gpu_samples, 'gpu-after-cleanup.json': gpu_after, 'provenance.json': provenance}
    if 'api_update' in captured:
        api_update, api_provenance = project_api_update(json.loads(captured['api_update']), raw, files)
        documents['api-query-update.json'] = api_update
        provenance['api_query_update'] = api_provenance
        provenance['api_query_update']['evidence_file'] = 'api-query-update.json'
    if 'api_query' in captured:
        api_query = project_api_query(json.loads(captured['api_query']), raw, state)
        documents['api-query-observation.json'] = api_query
        provenance['api_query_observation'] = {'evidence_file': 'api-query-observation.json',
            'job_id': api_query['id'], 'recorded_status': api_query['status'],
            'scope': 'Historical single-job HTTP response in this lifetime; state may precede final completion. No endpoint or credential published.'}
    if 'precision' in captured:
        precision = project_precision(json.loads(captured['precision']), raw, files)
        documents['precision-audit.json'] = precision
        provenance['parameter_precision_audit'] = {'evidence_file': 'precision-audit.json',
            'status': precision['status'], 'scope': precision['scope'],
            'source_log_snapshot_sha256': precision['source_log_snapshot_sha256'],
            'interpretation': 'Reported loaded native parameter metadata and upstream FP32 exceptions only; activation, buffer and kernel arithmetic precision are unmeasured.'}
    if 'platform_release' in captured:
        provenance['platform_time_representation'] = {
            'original_observed_at': json.loads(captured['platform_release']).get('observed_at'),
            'trace_observed_at_epoch_seconds': raw['platform_release'].get('observed_at'),
            'scope': 'Exact timezone-aware representation conversion of the original platform observation; no new observation or clock estimate.'}
    secrets = set().union(*(credential_values(value) for value in (raw, runtime, state, manifest, after)))
    refuse_credentials(documents, secrets)
    # Complete validation before touching the destination. Publish the prepared
    # directory by rename so an error never leaves apparently complete evidence.
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.condition-evidence-', dir=destination.parent) as temporary:
        staging = Path(temporary) / 'public'
        staging.mkdir()
        for name, value in documents.items():
            (staging / name).write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n', encoding='utf-8', newline='\n')
        for name, original, expected in assets:
            output = staging / name
            output.parent.mkdir(exist_ok=True)
            data = original.read_bytes()
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError('Asset changed during export; publication refused')
            output.write_bytes(data)
        if any(digest(path) != input_hashes[name] for name, path in inputs.items()):
            raise ValueError('Private evidence changed during export; publication refused')
        staging.rename(destination)
    return {'publication_only': True, 'analysis': analyses['analysis']['counts'],
            'condition_analysis': analyses['condition_analysis']['counts'], 'runtime_source_matches': len(source_matches),
            'public_files': len(documents) + len(assets), 'client_state': public_client.get('state')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--source-root', type=Path, required=True, help='Exact executed source tree; use the bound historical snapshot for historical export, never current-runtime fallback')
    parser.add_argument('--experiment', type=Path, required=True, help='Explicit descriptor inside PRIVATE_RUN, with protocol, baseline, executed-source and intervention bindings')
    args = parser.parse_args()
    print(json.dumps(export(args.source, args.destination, source_root=args.source_root, experiment=args.experiment), ensure_ascii=False))


if __name__ == '__main__':
    main()
