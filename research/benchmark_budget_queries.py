"""CPU-only control-query cost experiment; never imports a model or uses CUDA."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime/minimax_h3'))
from pipeline_scheduler import PipelineBudget


class MeasuredCondition:
    def __init__(self, condition):
        self.condition, self.elapsed_ns = condition, 0

    def __enter__(self):
        self.condition.__enter__()
        self.started = time.perf_counter_ns()
        return self

    def __exit__(self, *args):
        self.elapsed_ns += time.perf_counter_ns() - self.started
        return self.condition.__exit__(*args)


def percentile(values, fraction=.95):
    return sorted(values)[max(0, int(len(values) * fraction + .999999) - 1)] if values else None


def benchmark(iterations=300):
    records = []
    with tempfile.TemporaryDirectory(prefix='budget-query-bench-') as directory:
        for label, endpoints in [('empty', 0), ('short', 12), ('full_tail', 516)]:
            budget = PipelineBudget(Path(directory), limits={'cpu_threads': 1, 'ram_bytes': 1000, 'disk_bytes': 1000})
            budget.reserve('bench', {'cpu_threads': 1, 'ram_bytes': 1, 'disk_bytes': 1})
            for index in range(endpoints // 2):
                with budget.stage('bench', 'cpu'):
                    pass
            budget.finish('bench')
            budget._condition = MeasuredCondition(budget._condition)

            def light_status():
                snapshot = budget.snapshot(include_history=False)
                return {'accepting': snapshot['lease_journal']['complete'] and all(value > 0 for value in snapshot['free'].values()),
                        'budget': snapshot}

            def old_capacity():
                snapshot = budget.snapshot()
                return all(value > 0 for value in snapshot['free'].values())

            def old_status_equivalent():
                return copy.deepcopy({'accepting': old_capacity(), 'budget': budget.snapshot()})

            for name, operation in [('capacity_light', budget.has_capacity), ('capacity_full_equivalent', old_capacity),
                                    ('status_light', light_status), ('status_repeated_full_equivalent', old_status_equivalent)]:
                latencies, holds, sizes, errors, timeouts = [], [], [], 0, 0
                for _ in range(iterations):
                    lock_before = budget._condition.elapsed_ns
                    started = time.perf_counter_ns()
                    try:
                        payload = json.dumps(operation(), separators=(',', ':'), allow_nan=False).encode()
                        sizes.append(len(payload))
                    except Exception:
                        errors += 1
                    elapsed = (time.perf_counter_ns() - started) / 1e6
                    latencies.append(elapsed)
                    holds.append((budget._condition.elapsed_ns - lock_before) / 1e6)
                    timeouts += elapsed > 2000
                records.append({'history': label, 'durable_endpoints': endpoints, 'query': name, 'samples': iterations,
                    'errors': errors, 'over_2s': timeouts, 'response_bytes_median': statistics.median(sizes) if sizes else None,
                    'request_p95_ms_including_errors': percentile(latencies), 'budget_lock_hold_p95_ms': percentile(holds)})
    return {'schema_version': 1, 'scope': 'Local CPU control-query microbenchmark; no GPU/model inference; not A100 throughput evidence.',
            'comparison': 'Repeated full snapshot/deepcopy equivalent versus current lightweight path; both use current payload schema.',
            'methodology': {'execution': 'Sequential single-process warmed filesystem; durable journal is populated before timed queries.',
                'request_timing': 'perf_counter_ns covers query, instrumentation and JSON encoding, including failed requests; no HTTP/network/model state lock.',
                'lock_timing': 'Sum of measured budget condition critical sections per query; excludes lock wait and JSON encoding.',
                'status_scope': 'Budget portion and accepting flag only, not complete model status or API latency.',
                'p95': 'Nearest-rank 95th percentile over all attempts, not success-only.',
                'over_2s': 'Measured duration above 2000ms, not enforced timeout; all exceptions counted as errors.',
                'history': '0, 12 and 516 durable endpoints, with at most 512 retained in memory.',
                'reproduce': 'python research/benchmark_budget_queries.py --iterations 300 --output <result.json>',
                'limits': 'One local CPU process, no competing inference/HTTP requests; timings vary by host/cache/load. No claim of A100 speedup.'},
            'source_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                for name in ('research/benchmark_budget_queries.py', 'runtime/minimax_h3/pipeline_scheduler.py')},
            'python': platform.python_version(), 'platform': platform.platform(), 'records': records}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=300)
    args = parser.parse_args()
    if not 10 <= args.iterations <= 10000:
        parser.error('iterations must be between 10 and 10000')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(benchmark(args.iterations), indent=2) + '\n', encoding='utf-8')
