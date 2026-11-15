"""Finite counterexamples for local four-valued gates, without GPU execution."""
import json

import pytest

from condition_guard import evaluate_gate, negate_observation, observation_state


@pytest.mark.parametrize('value,state', [
    (True, 'true'), (False, 'false'), (None, 'unknown'), (1, 'unknown'),
    (0, 'unknown'), ('true', 'unknown'), ('false', 'unknown'),
    ('unknown', 'unknown'), ('conflict', 'conflict'), ({}, 'unknown'),
    ([], 'unknown'), ([True, None], 'true'), ([False, None], 'false'),
    ([True, False], 'conflict'), ([True, 'conflict'], 'conflict'),
    ([[True]], 'unknown'), ([1, 'true', None], 'unknown'),
    ([True] * 65, 'unknown'),
])
def test_native_boolean_evidence_and_conflict(value, state):
    receipt = evaluate_gate('local.gate', ['needed'], {'needed': value}, {'instance_id': 'current'})
    assert receipt['conditions'] == {'needed': state}
    assert receipt['allowed'] is (state == 'true')
    assert receipt['blocked'] == ([] if state == 'true' else ['needed'])
    assert observation_state(value) == state


@pytest.mark.parametrize('required', [[], (), 'condition', ['a', 'a'], [''], [1], ['a'] * 65])
def test_empty_or_ambiguous_requirements_cannot_vacuously_pass(required):
    with pytest.raises(ValueError):
        evaluate_gate('gate', required, {'a': True}, {})


def test_missing_conditions_and_unrequested_facts_cannot_enable_gate():
    result = evaluate_gate('gate', ['missing', 'known_false'], {'known_false': False, 'irrelevant': True}, {})
    assert not result['allowed']
    assert result['conditions'] == {'missing': 'unknown', 'known_false': 'false'}


def test_receipt_has_independent_json_identity_and_copies_context():
    context = {'model_instance_id': 'one', 'nested': {'revision': 1}}
    first = evaluate_gate('gate', ['ok'], {'ok': True}, context)
    second = evaluate_gate('gate', ['ok'], {'ok': True}, context)
    context['nested']['revision'] = 2
    assert first['context']['nested']['revision'] == 1
    assert first['decision_id'] != second['decision_id']
    assert first['monotonic_ns'] <= second['monotonic_ns']
    assert first['schema_version'] == first['gate_version'] == '1.0'
    assert json.loads(json.dumps(first)) == first


@pytest.mark.parametrize('context', [[], {'bad': float('nan')}, {'bad': object()},
                                    {'large': 'x' * 8193}, {1: 'not-json-key'}])
def test_context_is_bounded_finite_json(context):
    with pytest.raises(ValueError):
        evaluate_gate('gate', ['ok'], {'ok': True}, context)


@pytest.mark.parametrize('value,expected', [(True, False), (False, True),
                                          (None, None), (1, None),
                                          ([True, False], 'conflict')])
def test_negation_preserves_unknown_and_conflict(value, expected):
    assert negate_observation(value) == expected
