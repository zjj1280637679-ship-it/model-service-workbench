"""Small independent countermodels and source-bound evidence rejection cases."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


spec = importlib.util.spec_from_file_location("condition_sets_under_test", Path(__file__).resolve().parents[1] / "research/condition_sets.py")
conditions = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = conditions
spec.loader.exec_module(conditions)


def literal(atom, value=True):
    return {"atom": atom, "value": value}


def test_world_budget_does_not_turn_unknown_entailment_into_false():
    item = {"id": "budget_unknown", "scope": "finite counterexample", "atoms": ["a", "q"],
            "target": "q", "constraints": ["q"], "candidate_literals": [literal("a")]}
    limited = conditions.search(item, max_worlds=1)
    complete = conditions.search(item, max_worlds=4)
    assert limited["status"] == "truncated"
    assert limited["model_already_implies_target"] is None
    assert limited["necessary_literals_status"] == "unknown"
    assert complete["model_already_implies_target"] is True
    item["constraints"].append({"not": "q"})
    inconsistent = conditions.search(item)
    assert inconsistent["status"] == "inconsistent_model"
    assert inconsistent["model_already_implies_target"] is None


def model(atoms, target, candidates, constraints=(), **extra):
    return {"id": "countermodel", "scope": "Finite declared Boolean scenario, no live authorization",
            "atoms": atoms, "target": target, "constraints": list(constraints),
            "candidate_literals": candidates, **extra}


def sets(result):
    return {frozenset((item["atom"], item["value"]) for item in subset) for subset in result["minimal_sets"]}


def test_inclusion_minimal_solutions_include_different_cardinalities():
    # a alone suffices, but it must not hide the longer independent alternative.
    m = model(["a", "b", "c"], {"any": ["a", {"all": ["b", "c"]}]},
              [literal("a"), literal("b"), literal("c")])
    result = conditions.search(m)
    assert result["status"] == "complete"
    assert sets(result) == {frozenset({("a", True)}), frozenset({("b", True), ("c", True)})}
    assert result["necessary_literals"] == []
    assert result["scope"]


def test_full_counterexample_satisfies_premises_and_refutes_target():
    m = model(["a", "b", "unrelated"], {"all": ["a", "b"]}, [literal("a")])
    result = conditions.verify(m, [literal("a")])
    assert result["status"] == "refuted"
    world = result["counterexample"]
    assert set(world) == {"a", "b", "unrelated"}
    assert all(type(value) is bool for value in world.values())
    assert world["a"] is True and world["b"] is False


def test_constraints_are_respected_by_counterexample():
    m = model(["a", "b", "c"], "c", [literal("a")], constraints=[{"all": ["a", "b"]}])
    result = conditions.verify(m, [literal("a")])
    assert result["status"] == "refuted"
    assert result["counterexample"] == {"a": True, "b": True, "c": False}


def test_unsatisfiable_premise_is_not_vacuously_sufficient():
    m = model(["a", "b"], "b", [literal("a")], constraints=[{"not": "a"}])
    result = conditions.verify(m, [literal("a")])
    assert result["status"] == "unsatisfiable_premise"
    assert result["applicable_worlds"] == 0
    assert conditions.search(m)["minimal_sets"] == []


def test_opposite_literal_premises_are_unsatisfiable_not_a_proof():
    m = model(["a", "b"], "b", [literal("a"), literal("a", False)])
    assert conditions.verify(m, [literal("a"), literal("a", False)])["status"] == "unsatisfiable_premise"


def test_inconsistent_world_model_is_distinct_from_unsatisfiable_premise():
    m = model(["a", "b"], "b", [literal("a")], constraints=["a", {"not": "a"}])
    assert conditions.verify(m, [])["status"] == "inconsistent_model"
    result = conditions.search(m)
    assert result["status"] == "inconsistent_model"
    assert result["minimal_sets"] == []
    assert result["necessary_literals"] == []


def test_empty_premise_is_the_only_minimal_set_when_constraints_imply_target():
    m = model(["a", "q"], "q", [literal("a"), literal("a", False)], constraints=["q"])
    result = conditions.search(m, max_subsets=1)
    assert result["status"] == "complete"
    assert result["minimal_sets"] == [[]]
    assert result["model_already_implies_target"] is True
    assert conditions.verify(m, [])["status"] == "sufficient"


def test_literal_true_target_has_empty_minimal_set_without_any_candidates():
    result = conditions.search(model(["irrelevant"], True, []))
    assert result["status"] == "complete"
    assert result["minimal_sets"] == [[]]


def test_unreachable_target_does_not_make_every_literal_necessary():
    m = model(["a", "q"], "q", [literal("a"), literal("a", False)], constraints=[{"not": "q"}])
    result = conditions.search(m)
    assert result["status"] == "complete"
    assert result["target_reachable"] is False
    assert result["minimal_sets"] == result["necessary_literals"] == []
    assert result["model_already_implies_target"] is False


def test_necessary_literals_are_consequences_of_reachable_target_not_set_intersection():
    # Under C, all three atoms have equal values. Both a and b are necessary
    # for q, even though either singleton is a sufficient premise set.
    constraints = [{"any": [{"not": "q"}, "a"]}, {"any": [{"not": "a"}, "q"]},
                   {"any": [{"not": "q"}, "b"]}, {"any": [{"not": "b"}, "q"]}]
    m = model(["a", "b", "q"], "q", [literal("a"), literal("b")], constraints)
    result = conditions.search(m)
    assert sets(result) == {frozenset({("a", True)}), frozenset({("b", True)})}
    assert {(x["atom"], x["value"]) for x in result["necessary_literals"]} == {("a", True), ("b", True)}


def test_negative_literals_can_be_real_sufficient_conditions():
    m = model(["a", "b"], {"any": [{"not": "a"}, "b"]},
              [literal("a"), literal("a", False), literal("b")])
    assert sets(conditions.search(m)) == {frozenset({("a", False)}), frozenset({("b", True)})}


def test_world_budget_truncation_never_becomes_a_partial_world_proof():
    m = model(["a", "b"], "b", [literal("a")])
    assert conditions.verify(m, [literal("a")], max_worlds=1)["status"] == "unknown"
    result = conditions.search(m, max_worlds=1)
    assert result["status"] == "truncated"
    assert result["minimal_sets"] == result["necessary_literals"] == []
    assert result["model_already_implies_target"] is not True


def test_subset_budget_may_retain_only_fully_proven_minimal_sets():
    m = model(["a", "b", "c"], {"any": ["a", {"all": ["b", "c"]}]},
              [literal("a"), literal("b"), literal("c")])
    result = conditions.search(m, max_subsets=2)
    assert result["status"] == "truncated"
    assert result["examined_subsets"] <= 2
    assert sets(result) <= {frozenset({("a", True)}), frozenset({("b", True), ("c", True)})}
    for subset in result["minimal_sets"]:
        assert conditions.verify(m, subset)["status"] == "sufficient"


@pytest.mark.parametrize("value", [True, False])
def test_bare_target_cannot_be_inserted_as_a_premise(value):
    m = model(["a", "q"], "q", [literal("q", value)])
    with pytest.raises(ValueError):
        conditions.search(m)


def test_alias_cannot_smuggle_target_into_candidates():
    m = model(["a", "q"], "conclusion", [literal("another_name")],
              aliases={"conclusion": "q", "another_name": "q"})
    with pytest.raises(ValueError):
        conditions.search(m)


def test_explicit_forbidden_atom_is_rejected_even_in_composite_target():
    m = model(["a", "b"], {"any": ["a", "b"]}, [literal("a")], forbidden_atoms=["a"])
    with pytest.raises(ValueError):
        conditions.search(m)


def test_valid_aliases_use_canonical_world_identity():
    m = model(["a", "b"], {"any": ["alpha", "b"]}, [literal("alpha")], aliases={"alpha": "a"})
    assert conditions.verify(m, [literal("alpha")])["status"] == "sufficient"
    counterexample = conditions.verify(m, [])["counterexample"]
    assert counterexample == {"a": False, "b": False}


@pytest.mark.parametrize("changes", [
    {"atoms": ["a", "a"]},
    {"atoms": [f"v{i}" for i in range(17)]},
    {"target": "undeclared"},
    {"target": {"any": [True, "undeclared"]}},
    {"target": {"all": [False, "undeclared"]}},
    {"target": {"all": ["a"], "not": "a"}},
    {"target": {"unknown_operator": "a"}},
    {"candidate_literals": [literal("missing")]},
    {"candidate_literals": [{"atom": "a", "value": 0}]},
    {"candidate_literals": [literal("a"), literal("a")]},
    {"aliases": {"alias_a": "alias_b", "alias_b": "a"}},
    {"forbidden_atoms": [[]]},
])
def test_schema_errors_are_value_errors_not_defaulted_worlds(changes):
    m = model(["a", "b"], True, [literal("a")])
    m.update(changes)
    with pytest.raises(ValueError):
        conditions.search(m)


def test_candidate_count_limit_is_enforced():
    atoms = [f"v{i}" for i in range(11)]
    m = model(atoms, True, [literal(atom, sign) for atom in atoms for sign in (True, False)])
    with pytest.raises(ValueError):
        conditions.search(m)


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_invalid_world_and_subset_budgets_are_rejected(budget):
    m = model(["a"], True, [])
    with pytest.raises(ValueError):
        conditions.search(m, max_worlds=budget)
    with pytest.raises(ValueError):
        conditions.search(m, max_subsets=budget)


BINDINGS = {"subject": "model-7", "policy": "policy-bf16", "run": "run-9"}


def evidence(tmp_path, *, filename="observation.json", value=True, source_bindings=None, **changes):
    document = {"identity": deepcopy(source_bindings if source_bindings is not None else BINDINGS), "ready": value}
    path = tmp_path / filename
    path.write_text(json.dumps(document, allow_nan=False), encoding="utf-8")
    observation = {"atom": "ready", "bindings": deepcopy(BINDINGS), "valid_from": 10, "valid_until": 20,
        "validity_basis": "Declared observation validity for this finite historical example", "method": "observed",
        "source": {"path": filename, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
        "binding_selectors": {key: "/identity/" + key for key in BINDINGS}, "pointer": "/ready",
        "value_map": [{"equals": True, "state": "true"}, {"equals": False, "state": "false"}]}
    observation.update(changes)
    return observation


def audit(tmp_path, observations, *, as_of=15):
    return conditions.audit_evidence(["ready", "unobserved"], observations, BINDINGS, as_of, tmp_path)


def test_observed_false_and_missing_fact_are_distinct(tmp_path):
    result = audit(tmp_path, [evidence(tmp_path, value=False)])
    assert result["states"] == {"ready": "false", "unobserved": "unknown"}
    assert len(result["accepted"]) == 1 and result["excluded"] == []


def test_overlapping_opposite_observations_produce_conflict_not_latest_wins(tmp_path):
    observations = [evidence(tmp_path, filename="true.json", value=True), evidence(tmp_path, filename="false.json", value=False)]
    for order in (observations, list(reversed(observations))):
        result = audit(tmp_path, order)
        assert result["states"]["ready"] == "conflict"
        assert len(result["accepted"]) == 2


@pytest.mark.parametrize("as_of,expected", [(10, "true"), (19.999, "true"), (20, "unknown"), (9.999, "unknown")])
def test_validity_window_is_half_open(tmp_path, as_of, expected):
    result = audit(tmp_path, [evidence(tmp_path)], as_of=as_of)
    assert result["states"]["ready"] == expected


def test_expired_disagreement_does_not_conflict_with_current_false(tmp_path):
    observations = [evidence(tmp_path, filename="old.json", value=True, valid_from=1, valid_until=10),
                    evidence(tmp_path, filename="current.json", value=False)]
    result = audit(tmp_path, observations)
    assert result["states"]["ready"] == "false"
    assert len(result["accepted"]) == len(result["excluded"]) == 1


@pytest.mark.parametrize("key", ["subject", "policy", "run"])
def test_claimed_identity_mismatch_excludes_both_true_and_false(tmp_path, key):
    for value in (True, False):
        observation = evidence(tmp_path, value=value)
        observation["bindings"][key] = "old-identity"
        result = audit(tmp_path, [observation])
        assert result["states"]["ready"] == "unknown"
        assert not result["accepted"] and len(result["excluded"]) == 1


@pytest.mark.parametrize("key", ["subject", "policy", "run"])
def test_source_identity_must_match_even_when_claimed_binding_is_correct(tmp_path, key):
    old = {**BINDINGS, key: "old-source-identity"}
    result = audit(tmp_path, [evidence(tmp_path, source_bindings=old)])
    assert result["states"]["ready"] == "unknown"
    assert not result["accepted"] and len(result["excluded"]) == 1


@pytest.mark.parametrize("value", [0, 1, None, "false"])
def test_boolean_mapping_does_not_treat_numeric_or_text_values_as_booleans(tmp_path, value):
    result = audit(tmp_path, [evidence(tmp_path, value=value)])
    assert result["states"]["ready"] == "unknown"
    assert result["accepted"][0]["state"] == "unknown"


@pytest.mark.parametrize("changes", [
    {"method": "policy"}, {"method": "strategy"}, {"method": None},
    {"validity_basis": ""}, {"valid_from": 20, "valid_until": 10},
    {"valid_from": False}, {"valid_until": float("inf")},
    {"pointer": "/missing"}, {"binding_selectors": {"run": "/identity/run"}},
])
def test_non_observations_or_incomplete_provenance_remain_unknown(tmp_path, changes):
    result = audit(tmp_path, [evidence(tmp_path, **changes)])
    assert result["states"]["ready"] == "unknown"
    assert not result["accepted"] and len(result["excluded"]) == 1


def test_source_modified_after_hash_capture_is_excluded_not_negated(tmp_path):
    observation = evidence(tmp_path)
    (tmp_path / "observation.json").write_text('{"ready": false}', encoding="utf-8")
    result = audit(tmp_path, [observation])
    assert result["states"]["ready"] == "unknown"
    assert not result["accepted"] and len(result["excluded"]) == 1


def test_wrong_digest_cannot_authenticate_an_existing_source(tmp_path):
    observation = evidence(tmp_path)
    observation["source"]["sha256"] = "0" * 64
    assert audit(tmp_path, [observation])["states"]["ready"] == "unknown"


@pytest.mark.parametrize("path", ["../outside.json", "/absolute.json", "C:/outside.json", "nested\\outside.json"])
def test_evidence_paths_cannot_escape_declared_root(tmp_path, path):
    observation = evidence(tmp_path)
    observation["source"]["path"] = path
    result = audit(tmp_path, [observation])
    assert result["states"]["ready"] == "unknown"
    assert not result["accepted"]


def test_explicit_derived_observation_is_distinct_from_strategy_choice(tmp_path):
    observation = evidence(tmp_path, method="derived")
    result = audit(tmp_path, [observation])
    assert result["states"]["ready"] == "true"
    assert result["accepted"][0]["method"] == "derived"


def test_ambiguous_value_mapping_is_excluded(tmp_path):
    observation = evidence(tmp_path)
    observation["value_map"].append({"equals": True, "state": "false"})
    result = audit(tmp_path, [observation])
    assert result["states"]["ready"] == "unknown"
    assert not result["accepted"]


@pytest.mark.parametrize("as_of", [True, float("nan"), float("inf")])
def test_nonfinite_or_boolean_query_time_is_schema_error(tmp_path, as_of):
    with pytest.raises(ValueError):
        audit(tmp_path, [evidence(tmp_path)], as_of=as_of)
