"""Real public-API queue/cancel/two-content acceptance. No model or fake video.

Start while the model is loading: accept eight jobs, cancel six queued jobs,
then fetch the two distinct survivors. --resume only queries existing identities.
Stage overlap and GPU-release proofs are checked separately from the cloud audit.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import uuid
from urllib.parse import urlsplit


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint(folder, report):
    temporary = folder / 'pipeline_report.json.tmp'
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(folder / 'pipeline_report.json')


def download_ranges(base, key, job, folder):
    import requests
    artifact = job['artifact']
    size = artifact['bytes']
    require(isinstance(size, int) and 0 < size <= 128 * 1024**2, 'Unexpected output size')
    require(artifact.get('media', {}).get('full_decode_passed') is True, 'Server did not validate the complete video')
    url = base + '/v1/jobs/' + job['id'] + '/file'
    def part(bounds):
        start, end = bounds
        records = []
        for number in range(1, 4):
            record = {'attempt': number, 'start': start, 'end': end}
            records.append(record)
            try:
                with requests.get(url, headers={'Authorization': 'Bearer '+key,
                        'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'},
                        timeout=(15, 25), allow_redirects=False, stream=True) as response:
                    record.update(http_status=response.status_code,
                                  content_range=response.headers.get('Content-Range'))
                    require(response.status_code == 206, 'Range request was not accepted')
                    require(record['content_range'] == f'bytes {start}-{end}/{size}', 'Incorrect Content-Range')
                    data = response.raw.read(end-start+2)
                    record['bytes'] = len(data)
                    require(len(data) == end-start+1, 'Truncated range')
                record['passed'] = True
                return data, records
            except Exception as error:
                record.update(passed=False, error_type=type(error).__name__)
        return None, records
    bounds = [(start, min(size-1, start+65535)) for start in range(0, size, 65536)]
    with ThreadPoolExecutor(max_workers=3) as pool:
        parts = list(pool.map(part, bounds))
    result = {'job_id': job['id'], 'ranges': [records for _, records in parts], 'passed': False}
    (folder / (job['id']+'-download.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
    require(all(data is not None for data,_ in parts), 'File transfer exhausted bounded range retries; resume without generating again')
    content = b''.join(data for data,_ in parts)
    actual = hashlib.sha256(content).hexdigest()
    require(len(content)==size and actual==artifact['sha256'], 'External file differs from committed artifact')
    path = folder / (job['id']+'.mp4')
    path.write_bytes(content)
    result.update(passed=True, bytes=size, sha256=actual, path=str(path),
                  full_decode_basis='Server full audiovisual decode, with exact external byte count and SHA256 match')
    (folder / (job['id']+'-download.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


def get_json(base, key, route, *, authenticated=True, records=None):
    """Bounded read retries never submit or replace an existing generation."""
    import requests
    for attempt in range(1, 4):
        record = {'route': route, 'attempt': attempt, 'observed_at': now()}
        if records is not None:
            records.append(record)
        try:
            response = requests.get(base+route,
                headers={'Authorization': 'Bearer '+key} if authenticated else {},
                timeout=(10,20), allow_redirects=False)
            record['http_status'] = response.status_code
            if response.status_code == 200:
                return response.json()
            require(response.status_code in (429,502,503,504),
                    'GET '+route+' returned HTTP '+str(response.status_code))
        except requests.RequestException as error:
            record['error_type'] = type(error).__name__
        if attempt < 3:
            time.sleep(attempt)
    raise RuntimeError('GET '+route+' exhausted bounded read retries; resume existing task identities')


def _run(args):
    import requests
    key = os.environ.get('H3_API_KEY', '')
    require(len(key)>=32, 'Set H3_API_KEY without placing it in command-line arguments')
    base = args.url.rstrip('/')
    parsed = urlsplit(base)
    require(parsed.scheme in {'http','https'} and parsed.hostname and not any(
        (parsed.username, parsed.password, parsed.query, parsed.fragment)), 'Invalid API base URL')
    args.outputdir.mkdir(parents=True, exist_ok=True)
    target = args.outputdir/'pipeline_report.json'
    headers = {'Authorization':'Bearer '+key}
    reads = []
    def get(route, authenticated=True):
        return get_json(base,key,route,authenticated=authenticated,records=reads)
    if args.resume:
        report = json.loads(target.read_text(encoding='utf-8'))
        require(report['api_base_url']==base and len(report.get('jobs',[]))==8,
                'Resume needs the original API URL and all eight recorded job identities')
        require(all(job.get('id') for job in report['jobs']), 'Resolve uncertain submission identities before resume')
        reads = report.setdefault('read_attempts', [])
        report['resumed_at'] = now()
        args._active_report = True
    else:
        require(not target.exists(), 'Existing test report: use --resume; never resubmit an existing run')
        require(args.reference_a and args.reference_b and args.prompt_a and args.prompt_b,
                'Two fresh reference images and distinct explicit prompts are required')
        refs = [args.reference_a.resolve(),args.reference_b.resolve()]
        require(all(p.is_file() for p in refs), 'Reference files do not exist')
        hashes = [digest(p) for p in refs]
        require(hashes[0]!=hashes[1] and args.prompt_a.strip()!=args.prompt_b.strip(),
                'The two survivors must use different references and prompts')
        report = {'schema_version':1, 'run_id':uuid.uuid4().hex, 'started_at':now(), 'api_base_url':base,
                  'passed':False, 'jobs':[], 'downloads':[], 'observations':[], 'read_attempts':reads,
                  'scope':'Eight durable queued jobs; six queued cancellations; two new-content real generations. Stage overlap and shutdown are independently audited.'}
        checkpoint(args.outputdir,report)
        args._active_report = True
        deadline=time.monotonic()+args.startup_timeout
        while True:
            try:
                system=get('/v1/system')
                if system.get('api_ready') and not system.get('observation_stale'):
                    break
            except Exception:
                pass
            require(time.monotonic()<deadline, 'API did not become queryable')
            time.sleep(3)
        require(system.get('model_ready') is False, 'Run queue admission/cancel test during cold loading, before any GPU job can start')
        report['instance_id']=system['instance_id']
        report['system_before_submission']=system
        from smoke_test import register_content
        registry=args.content_history or args.outputdir.parent/'content-history.json'
        report['content_reservations']=[]
        for index,prompt in enumerate((args.prompt_a,args.prompt_b)):
            report['content_reservations'].append(register_content(registry,hashes[index],prompt,
                outputdir=args.outputdir,api_base_url=base))
            checkpoint(args.outputdir,report)
        unauth=requests.get(base+'/v1/system',timeout=15,allow_redirects=False)
        report['unauthorized_system_http_status']=unauth.status_code
        require(unauth.status_code==401,'Unauthorized query was not rejected')
        for index in range(8):
            survivor=index in (0,2)
            content_index=1 if index==2 else 0
            prompt=(args.prompt_a,args.prompt_b)[content_index]
            if not survivor:
                prompt+=f' Alternate cancelled queue test {index}: a new viewpoint with slow lateral camera motion.'
            entry={'index':index,'survivor':survivor,'reference':str(refs[content_index]),
                   'reference_sha256':hashes[content_index], 'prompt':prompt,
                   'idempotency_key':report['run_id']+'-'+str(index), 'submission_attempts':[]}
            report['jobs'].append(entry)
            checkpoint(args.outputdir,report)
            for attempt in range(1,4):
                try:
                    with refs[content_index].open('rb') as media:
                        response=requests.post(base+'/v1/jobs',headers={**headers,'Idempotency-Key':entry['idempotency_key']},
                            data={'prompt':prompt,'kinds':'["image"]','seconds':5,'width':512,'height':288,'steps':8,'seed':42+index},
                            files=[('files',(refs[content_index].name,media,'image/png'))],
                            timeout=(15,90),allow_redirects=False)
                    entry['submission_attempts'].append({'attempt':attempt,'http_status':response.status_code})
                    if response.status_code==202:
                        value=response.json()
                        entry.update(id=value['id'],submitted_status=value['status'],receipt=value)
                        break
                    require(response.status_code in (429,502,503,504), 'Submission definitively rejected: HTTP '+str(response.status_code))
                except requests.RequestException as error:
                    entry['submission_attempts'].append({'attempt':attempt,'error_type':type(error).__name__})
                finally:
                    checkpoint(args.outputdir,report)
                time.sleep(1)
            require(entry.get('id'), 'Submission uncertain: keep this report and idempotency key; do not start a new run')
            require(entry['submitted_status']=='queued','Model started before queue cancellation acceptance window')
            print('accepted',index,entry['id'],flush=True)
        report['all_eight_accepted_before_cancellation']=True
        checkpoint(args.outputdir,report)
    try:
        require(get('/v1/system')['instance_id']==report['instance_id'],'Manager instance changed; do not replay generation')
        # Repeating cancellation for the recorded identities is idempotent, including on resume.
        for entry in report['jobs']:
            if not entry['survivor']:
                before=get('/v1/jobs/'+entry['id'])
                require(before['status'] in {'queued','cancelled'},'Cancellation target already began executing; preserve failure evidence')
                response=requests.post(base+'/v1/jobs/'+entry['id']+'/cancel',headers=headers,timeout=30,allow_redirects=False)
                require(response.status_code==202,'Cancellation request was not accepted')
                after=get('/v1/jobs/'+entry['id'])
                entry['cancel_receipt']=response.json()
                entry['terminal']=after
                require(after['status']=='cancelled' and after.get('started_at') is None,
                        'Queued cancellation did not complete before dispatch')
                checkpoint(args.outputdir,report)
        deadline=time.monotonic()+args.generation_timeout
        previous=None
        while True:
            survivors=[entry for entry in report['jobs'] if entry['survivor']]
            for entry in survivors:
                entry['terminal']=get('/v1/jobs/'+entry['id'])
            summary=[{'id':entry['id'],'status':entry['terminal']['status'],
                      'stage':entry['terminal'].get('active_phase')} for entry in survivors]
            signature=json.dumps(summary,sort_keys=True)
            if signature!=previous:
                observation={'observed_at':now(),'jobs':summary}
                report['observations'].append(observation)
                print(json.dumps(observation),flush=True)
                previous=signature
            checkpoint(args.outputdir,report)
            require(all(entry['terminal']['status'] not in {'failed','cancelled'} for entry in survivors),
                    'A generation did not succeed; inspect the retained report and model log')
            if all(entry['terminal']['status']=='succeeded' for entry in survivors):
                break
            require(time.monotonic()<deadline,'Generation deadline exceeded; query or cancel existing IDs without resubmitting')
            time.sleep(args.poll_seconds)
        for entry in survivors:
            report['downloads'].append(download_ranges(base,key,entry['terminal'],args.outputdir))
            checkpoint(args.outputdir,report)
        report.update(passed=True,finished_at=now())
        checkpoint(args.outputdir,report)
        print('QUEUE_CANCEL_TWO_GENERATIONS_PASSED',flush=True)
        return report
    except Exception as error:
        report.update(passed=False,error_type=type(error).__name__,error=str(error),last_updated_at=now())
        checkpoint(args.outputdir,report)
        raise


def run(args):
    """Checkpoint failures during admission as well as during generation polling."""
    args._active_report = False
    try:
        return _run(args)
    except Exception as error:
        target = args.outputdir/'pipeline_report.json'
        if args._active_report and target.is_file():
            try:
                report = json.loads(target.read_text(encoding='utf-8'))
                message = str(error)
                key = os.environ.get('H3_API_KEY')
                if key:
                    message = message.replace(key, '[REDACTED]')
                report.update(passed=False,error_type=type(error).__name__,error=message,last_updated_at=now())
                checkpoint(args.outputdir,report)
            except (OSError, ValueError, TypeError):
                pass  # Preserve the original error and any earlier durable identities.
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',required=True)
    parser.add_argument('--outputdir',type=Path,required=True)
    parser.add_argument('--reference-a',type=Path)
    parser.add_argument('--reference-b',type=Path)
    parser.add_argument('--prompt-a')
    parser.add_argument('--prompt-b')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--content-history',type=Path)
    parser.add_argument('--startup-timeout',type=float,default=300)
    parser.add_argument('--generation-timeout',type=float,default=5400)
    parser.add_argument('--poll-seconds',type=float,default=10)
    run(parser.parse_args())


if __name__=='__main__':
    main()
