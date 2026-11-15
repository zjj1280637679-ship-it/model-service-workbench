import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('queue_policy_study', ROOT/'research/study_queue_policy.py')
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def test_report_measures_known_runnable_successor_delay_and_retains_head_tradeoff():
    result = study.study()
    short = result['cases']['short_successors']
    assert short['fifo']['runnable_successor_blocked_job_ticks'] > short['bounded_backfill']['runnable_successor_blocked_job_ticks']
    long = result['cases']['long_successor_tradeoff']
    assert long['bounded_backfill']['wait_ticks_by_job']['head'] > long['fifo']['wait_ticks_by_job']['head']
    assert result['global_wait_time_upper_bound'] is None and result['time_unit'] == 'model_tick'


def test_finite_arrival_stream_does_not_exceed_bypass_budget_or_lose_jobs():
    result = study.study()
    for case in result['cases'].values():
        for policy in ('fifo','bounded_backfill'):
            row = case[policy]
            assert row['completed'] == len(case['jobs']) and row['unfinished'] == 0
            assert all(count <= row['policy']['max_bypasses'] for count in row['successful_overtakes_by_job'].values())


def test_age_budget_changes_admission_only_when_observed_virtual_age_exceeds_limit():
    jobs=[{'id':'resident','arrival':0,'duration':40,'size':8},
          {'id':'head','arrival':0,'duration':1,'size':10},
          {'id':'late','arrival':31,'duration':1,'size':2}]
    row=study.simulate(jobs,12,study.QueuePolicy(max_head_age_seconds=30))
    assert row['wait_ticks_by_job']['late'] == 9
    assert row['successful_overtakes_by_job'] == {}


def test_overtake_counts_match_actual_admission_order_for_every_queued_predecessor():
    result = study.study()
    for case in result['cases'].values():
        fifo_order = {job['id']: i for i, job in enumerate(sorted(case['jobs'], key=lambda job: (job['arrival'], case['jobs'].index(job))))}
        for mode in ('fifo', 'bounded_backfill'):
            row = case[mode]
            for index, job in enumerate(row['admissions']):
                actual = sum(other['start'] >= job['arrival'] and fifo_order[other['id']] > fifo_order[job['id']]
                             for other in row['admissions'][:index])
                assert row['successful_overtakes_by_job'].get(job['id'], 0) == actual
                assert actual <= row['policy']['max_bypasses']
    both = result['cases']['two_blocked_predecessors']['bounded_backfill']['successful_overtakes_by_job']
    assert both['head'] == both['second'] == 2


def test_unsorted_input_obeys_arrival_order_for_fifo():
    jobs=[{'id':'late','arrival':5,'duration':1,'size':10},
          {'id':'resident','arrival':0,'duration':10,'size':8},
          {'id':'early','arrival':1,'duration':1,'size':10}]
    row=study.simulate(jobs,12,study.QueuePolicy(mode='fifo'))
    assert row['wait_ticks_by_job'] == {'resident':0, 'early':9, 'late':6}
