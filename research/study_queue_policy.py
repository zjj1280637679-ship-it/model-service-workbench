"""Reproducible finite admission-capacity comparisons, not hardware benchmarks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime/minimax_h3"))
from queue_policy import QueuePolicy


def simulate(jobs, capacity, policy):
    if type(capacity) is not int or capacity < 1:
        raise ValueError("Positive declared capacity required")
    rows = [dict(row, order=i) for i, row in enumerate(jobs)]
    if len({row['id'] for row in rows}) != len(rows):
        raise ValueError("Unique job identities required")
    for row in rows:
        if (type(row['arrival']) is not int or row['arrival'] < 0 or type(row['duration']) is not int
                or row['duration'] < 1 or type(row['size']) is not int or not 1 <= row['size'] <= capacity):
            raise ValueError("Finite nonnegative arrivals, positive durations and feasible sizes required")
    now = 0
    pending = sorted(rows, key=lambda x: (x['arrival'], x['order']))
    queue, active, completed, admissions = [], [], [], []
    first_deferred, overtakes = {}, {}
    blocked_job_ticks, blocked_union_ticks = 0, 0
    while pending or queue or active:
        completed.extend(row for row in active if row['finish'] <= now)
        active = [row for row in active if row['finish'] > now]
        while pending and pending[0]['arrival'] <= now:
            queue.append(pending.pop(0))
        queue.sort(key=lambda row: (row['arrival'], row['order']))
        while queue:
            free = capacity - sum(row['size'] for row in active)
            head = queue[0]
            candidate = head if head['size'] <= free else None
            predecessors = []
            if candidate is None:
                # Match production's bounded offer order: each skipped job has
                # an explicit resource deferral before a later job can start.
                for row in queue[:policy.lookahead]:
                    if row['size'] <= free:
                        if policy.may_bypass_all([{'bypasses': overtakes.get(older['id'], 0),
                                'first_deferred_at': first_deferred[older['id']]} for older in predecessors], now=now):
                            candidate = row
                        break
                    first_deferred.setdefault(row['id'], now)
                    predecessors.append(row)
                    if not policy.may_bypass_all([{'bypasses': overtakes.get(older['id'], 0),
                            'first_deferred_at': first_deferred[older['id']]} for older in predecessors], now=now):
                        break
            if candidate is None:
                break
            if candidate is not head:
                for predecessor in predecessors:
                    overtakes[predecessor['id']] = overtakes.get(predecessor['id'], 0) + 1
            queue.remove(candidate)
            started = dict(candidate, start=now, finish=now + candidate['duration'])
            admissions.append(started)
            active.append(started)
        future = [row['finish'] for row in active] + ([pending[0]['arrival']] if pending else [])
        if not future:
            if queue:
                raise RuntimeError("Unexpected deadlock in finite feasible model")
            break
        next_time = min(future)
        free = capacity - sum(row['size'] for row in active)
        runnable_blocked = sum(row['size'] <= free for row in queue[1:]) if queue and queue[0]['size'] > free else 0
        blocked_job_ticks += runnable_blocked * (next_time - now)
        blocked_union_ticks += (next_time - now) if runnable_blocked else 0
        now = next_time
    waits = {row['id']: row['start'] - row['arrival'] for row in admissions}
    return {'policy': policy.as_dict(), 'admissions': admissions, 'wait_ticks_by_job': waits,
            'max_wait_ticks': max(waits.values(), default=0),
            'mean_wait_ticks': sum(waits.values()) / len(waits) if waits else None,
            'makespan_ticks': max((row['finish'] for row in completed), default=0),
            'runnable_successor_blocked_job_ticks': blocked_job_ticks,
            'runnable_successor_blocked_union_ticks': blocked_union_ticks,
            'successful_overtakes_by_job': overtakes,
            'completed': len(completed), 'unfinished': len(queue) + len(active) + len(pending)}


def study():
    cases = {
        'short_successors': [{'id':'resident','arrival':0,'duration':10,'size':8},
                            {'id':'head','arrival':0,'duration':4,'size':8},
                            *[{'id':f'small{i}','arrival':0,'duration':2,'size':2} for i in range(4)]],
        'long_successor_tradeoff': [{'id':'resident','arrival':0,'duration':10,'size':8},
                                   {'id':'head','arrival':0,'duration':5,'size':10},
                                   *[{'id':f'long{i}','arrival':0,'duration':20,'size':2} for i in range(2)]],
        'finite_arrival_stream': [{'id':'resident','arrival':0,'duration':10,'size':8},
                                 {'id':'head','arrival':0,'duration':4,'size':12},
                                 *[{'id':f'new{i}','arrival':i,'duration':3,'size':2} for i in range(20)]],
        'two_blocked_predecessors': [{'id':'resident','arrival':0,'duration':10,'size':8},
                                    {'id':'head','arrival':0,'duration':4,'size':8},
                                    {'id':'second','arrival':0,'duration':2,'size':10},
                                    *[{'id':f'small{i}','arrival':0,'duration':20,'size':2} for i in range(4)]],
    }
    results = {name: {'capacity':12, 'jobs':jobs,
                     'fifo': simulate(jobs, 12, QueuePolicy(mode='fifo')),
                     'bounded_backfill': simulate(jobs, 12, QueuePolicy())} for name, jobs in cases.items()}
    return {'schema_version':2, 'time_unit':'model_tick', 'cases':results,
            'source_sha256': {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                              for name in ['runtime/minimax_h3/queue_policy.py','research/study_queue_policy.py']},
            'global_wait_time_upper_bound':None,
            'scope':'Finite admission working-set capacity model using the production policy predicate. Clock values and age cutoff are model ticks, not measured seconds. Not a GPU/stage simulation, speedup, global optimum or arbitrary arrival starvation proof. Each successful later admission is charged to every still-queued predecessor, including before that predecessor becomes the head. Already admitted long jobs can delay the head.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = study()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({name:{mode:{k:row[mode][k] for k in ('max_wait_ticks','mean_wait_ticks','runnable_successor_blocked_job_ticks','completed','unfinished')}
                           for mode in ('fifo','bounded_backfill')} for name,row in result['cases'].items()}))


if __name__ == '__main__':
    main()
