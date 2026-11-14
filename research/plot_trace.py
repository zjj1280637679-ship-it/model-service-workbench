"""Export a measured host-stage timeline; matplotlib is an optional report dependency."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def union(intervals):
    ordered = []
    for start, end in intervals:
        if any(type(x) not in (int, float) or not math.isfinite(x) for x in (start, end)) or end < start:
            raise ValueError('Expected finite ordered half-open intervals')
        if end > start:
            ordered.append((start, end))
    merged = []
    for start, end in sorted(ordered):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def overlap_seconds(left, right):
    # Each time point contributes once even when several CPU stages overlap.
    a, b = union(left), union(right)
    i = j = 0
    total = 0.0
    while i < len(a) and j < len(b):
        total += max(0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return total


def build_intervals(trace):
    starts, result = {}, []
    events = trace['model_events']
    for event in sorted(events, key=lambda e:e['sequence']):
        if event['type'] not in {'stage.started', 'stage.finished'}:
            continue
        key = (event['attempt_id'], event['stage'])
        if type(event.get('monotonic_ns')) is not int:
            raise ValueError('No trustworthy monotonic stage clock')
        if event['type'] == 'stage.started':
            if key in starts:
                raise ValueError('Duplicate stage start')
            starts[key] = event
        else:
            start = starts.pop(key)
            if start['job_id'] != event['job_id'] or event['monotonic_ns'] < start['monotonic_ns']:
                raise ValueError('Invalid paired stage')
            result.append({'job_id':event['job_id'], 'stage':event['stage'],
                           'start_ns':start['monotonic_ns'], 'end_ns':event['monotonic_ns']})
    if starts or not result:
        raise ValueError('Incomplete measured stages')
    origin = min(row['start_ns'] for row in result)
    for row in result:
        row['start_seconds'] = (row.pop('start_ns')-origin)/1e9
        row['end_seconds'] = (row.pop('end_ns')-origin)/1e9
    return result


def summarize(rows):
    cpu = [(r['start_seconds'],r['end_seconds']) for r in rows if r['stage'] != 'gpu']
    gpu = [(r['start_seconds'],r['end_seconds']) for r in rows if r['stage'] == 'gpu']
    return {'schema_version':1, 'clock':'one_model_instance_monotonic_ns',
            'stage_makespan_seconds':max(r['end_seconds'] for r in rows)-min(r['start_seconds'] for r in rows),
            'stage_duration_sum_seconds':sum(r['end_seconds']-r['start_seconds'] for r in rows),
            'cpu_gpu_union_overlap_seconds':overlap_seconds(cpu,gpu),
            'gpu_scope_union_seconds':sum(e-s for s,e in union(gpu)), 'intervals':rows,
            'scope':'Observed host stage scopes, not GPU kernel busy time. Sum of concurrent stage durations is not a measured serial control; no speedup claim.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace',type=Path)
    parser.add_argument('--output',type=Path,required=True,help='Output stem for PNG, SVG and JSON')
    args=parser.parse_args()
    from analyze_trace import analyze
    trace=json.loads(args.trace.read_text(encoding='utf8'))
    analysis=analyze(trace)
    if any(analysis['checks'][key]['status'] != 'pass' for key in ('stage_causality','stage_durations')):
        raise SystemExit('Complete causal measured stages are required for this figure')
    rows=build_intervals(trace)
    summary=summarize(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    identities=list(dict.fromkeys(r['job_id'] for r in sorted(rows,key=lambda r:r['start_seconds'])))
    labels={identity:'Job '+chr(65+i) for i,identity in enumerate(identities)}
    colors={'prepare':'#218e81','gpu':'#4169bd','encode_validate':'#d28535'}
    fig,axes=plt.subplots(2,1,figsize=(11,6.4),height_ratios=(1.2,1),layout='constrained')
    for ax in axes:
        for row in rows:
            y=identities.index(row['job_id'])
            ax.broken_barh([(row['start_seconds'],row['end_seconds']-row['start_seconds'])],
                           (y-.23,.46),facecolors=colors[row['stage']])
        ax.set_yticks(range(len(identities)),[labels[x] for x in identities])
        ax.invert_yaxis()
        ax.grid(axis='x',alpha=.18)
        ax.spines[['top','right']].set_visible(False)
        ax.set_xlabel('Seconds since first observed prepare stage')
    axes[0].set_title('A100 native BF16: observed task stages',loc='left',fontweight='bold')
    axes[0].set_xlim(-.3,summary['stage_makespan_seconds']*1.02)
    ends=sorted(row['end_seconds'] for row in rows if row['stage']=='gpu')
    handoff=ends[0]
    axes[1].set_xlim(max(0,handoff-2),min(summary['stage_makespan_seconds']+.2,handoff+5))
    axes[1].set_title('Detail around the first GPU handoff',loc='left')
    axes[1].axvline(handoff,color='#555555',lw=.8,ls='--')
    axes[0].legend(handles=[Patch(color=colors[k],label=v) for k,v in
        [('prepare','Prepare'),('gpu','GPU host scope'),('encode_validate','Encode + validate')]],loc='upper right',ncols=3)
    fig.suptitle('Two retained jobs; cancelled queued job has no execution stages',fontsize=10,color='#555555')
    fig.supxlabel('Single finite run. Host scope is not kernel utilization; no serial-control or speedup inference.',fontsize=9,color='#555555')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output.with_suffix('.png'),dpi=170)
    fig.savefig(args.output.with_suffix('.svg'))
    plt.close(fig)
    args.output.with_suffix('.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf8',newline='\n')
    print(json.dumps({k:v for k,v in summary.items() if k != 'intervals'}))


if __name__ == '__main__':
    main()
