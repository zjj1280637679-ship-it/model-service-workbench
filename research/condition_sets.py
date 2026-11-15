"""Bounded sufficient-condition search and evidence checks; never dispatches work.

Logical proofs are conditional on the supplied finite Boolean model. Source
hashes establish file identity, not the truth or completeness of observations.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import re

NAME = re.compile(r"^[a-z][a-z0-9_]{0,79}$")
STATES = {"true", "false", "unknown", "conflict"}
SCOPE = "Conditional proof in the declared finite Boolean model; not empirical causation, live authorization or eventual completion."


def require(test, message):
    if not test:
        raise ValueError(message)


def _budget(value, name):
    require(type(value) is int and value > 0, name + " must be a positive integer")


def _expression(expr, atoms, aliases, depth=0, counter=None):
    counter = [0] if counter is None else counter
    counter[0] += 1
    require(depth <= 32 and counter[0] <= 4096, "Expression too large")
    if type(expr) is bool:
        return expr
    if isinstance(expr, str):
        expr = aliases.get(expr, expr)
        require(expr in atoms, "Undeclared atom: " + expr)
        return expr
    require(isinstance(expr, dict) and len(expr) == 1, "Invalid expression")
    key, value = next(iter(expr.items()))
    require(key in {"all", "any", "not"}, "Unknown operator")
    if key == "not":
        return {key: _expression(value, atoms, aliases, depth + 1, counter)}
    require(isinstance(value, list) and bool(value), "all/any require nonempty arrays")
    return {key: [_expression(x, atoms, aliases, depth + 1, counter) for x in value]}


def _eval(expr, world):
    if type(expr) is bool:
        return expr
    if isinstance(expr, str):
        return world[expr]
    if "not" in expr:
        return not _eval(expr["not"], world)
    if "all" in expr:
        return all(_eval(x, world) for x in expr["all"])
    return any(_eval(x, world) for x in expr["any"])


def _model(model):
    require(isinstance(model, dict), "Model must be an object")
    require(isinstance(model.get("id"), str) and bool(model["id"]), "Model identity required")
    require(isinstance(model.get("scope"), str) and bool(model["scope"]), "Model scope required")
    atoms = model.get("atoms")
    require(isinstance(atoms, list) and 0 < len(atoms) <= 16, "Expected 1..16 atoms")
    require(all(isinstance(x, str) and NAME.fullmatch(x) for x in atoms), "Invalid atom name")
    require(len(set(atoms)) == len(atoms), "Duplicate atoms")
    aliases = model.get("aliases", {})
    require(isinstance(aliases, dict), "Aliases must be an object")
    require(all(isinstance(k, str) and NAME.fullmatch(k) and k not in atoms and isinstance(v, str) and v in atoms
                for k, v in aliases.items()), "Aliases must map fresh names directly to canonical atoms")
    target = _expression(model.get("target"), atoms, aliases)
    constraints = model.get("constraints", [])
    require(isinstance(constraints, list) and len(constraints) <= 64, "Expected at most 64 constraints")
    constraints = [_expression(expr, atoms, aliases) for expr in constraints]
    forbidden = model.get("forbidden_atoms", [])
    require(isinstance(forbidden, list), "Forbidden atoms must be a list")
    require(all(isinstance(x, str) for x in forbidden), "Forbidden atoms must be names")
    forbidden = [aliases.get(x, x) for x in forbidden]
    require(all(x in atoms for x in forbidden), "Undeclared forbidden atom")
    forbidden = set(forbidden)
    conclusion = target
    while isinstance(conclusion, dict):
        if "not" in conclusion:
            conclusion = conclusion["not"]
        elif any(k in conclusion and len(conclusion[k]) == 1 for k in ("all", "any")):
            conclusion = next(iter(conclusion.values()))[0]
        else:
            break
    if isinstance(conclusion, str):
        forbidden.add(conclusion)
    candidates = model.get("candidate_literals")
    require(isinstance(candidates, list) and len(candidates) <= 20, "Expected at most 20 candidate literals")
    normalized = []
    for literal in candidates:
        require(isinstance(literal, dict) and set(literal) == {"atom", "value"}, "Invalid literal")
        atom = aliases.get(literal["atom"], literal["atom"]) if isinstance(literal["atom"], str) else None
        require(atom in atoms and type(literal["value"]) is bool, "Invalid atom or Boolean literal")
        require(atom not in forbidden, "Conclusion/forbidden atom cannot be a premise")
        item = (atom, literal["value"])
        require(item not in normalized, "Duplicate canonical candidate literal")
        normalized.append(item)
    return atoms, aliases, target, constraints, normalized


def _worlds(model, max_worlds):
    _budget(max_worlds, "max_worlds")
    atoms, aliases, target, constraints, candidates = _model(model)
    if 2 ** len(atoms) > max_worlds:
        return atoms, aliases, candidates, None, None
    worlds, target_mask = [], 0
    for values in itertools.product((False, True), repeat=len(atoms)):
        world = dict(zip(atoms, values))
        if all(_eval(expr, world) for expr in constraints):
            if _eval(target, world):
                target_mask |= 1 << len(worlds)
            worlds.append(world)
    return atoms, aliases, candidates, worlds, target_mask


def _literal_json(literals):
    return [{"atom": atom, "value": value} for atom, value in literals]


def _masks(candidates, worlds):
    return [sum(1 << i for i, world in enumerate(worlds) if world[atom] is value)
            for atom, value in candidates]


def verify(model, premises, max_worlds=65536):
    atoms, aliases, candidates, worlds, target_mask = _worlds(model, max_worlds)
    require(isinstance(premises, list), "Premises must be a list")
    normalized = []
    for item in premises:
        require(isinstance(item, dict) and set(item) == {"atom", "value"}
                and isinstance(item["atom"], str) and type(item["value"]) is bool, "Invalid premise")
        literal = (aliases.get(item["atom"], item["atom"]), item["value"])
        require(literal in candidates, "Premise is not an allowed candidate")
        require(literal not in normalized, "Duplicate premise")
        normalized.append(literal)
    result = {"model_id": model["id"], "scope": SCOPE, "premises": _literal_json(normalized)}
    if worlds is None:
        return {**result, "status": "unknown", "reason": "world_budget_exhausted"}
    if not worlds:
        return {**result, "status": "inconsistent_model", "applicable_worlds": 0}
    applicable = [(i, w) for i, w in enumerate(worlds) if all(w[a] is v for a, v in normalized)]
    if not applicable:
        return {**result, "status": "unsatisfiable_premise", "applicable_worlds": 0}
    counterexample = next((w for i, w in applicable if not (target_mask & (1 << i))), None)
    return {**result, "status": "refuted" if counterexample is not None else "sufficient",
            "applicable_worlds": len(applicable), "counterexample": counterexample,
            "model_already_implies_target": target_mask == (1 << len(worlds)) - 1}


def search(model, max_worlds=65536, max_subsets=65536):
    _budget(max_subsets, "max_subsets")
    atoms, aliases, candidates, worlds, target_mask = _worlds(model, max_worlds)
    result = {"model_id": model["id"], "scope": SCOPE, "minimal_sets": [], "necessary_literals": [],
              "examined_subsets": 0, "model_already_implies_target": None,
              "necessary_literals_status": "unknown",
              "minimality": "inclusion-minimal, not merely minimum cardinality"}
    if worlds is None:
        return {**result, "status": "truncated", "reason": "world_budget_exhausted"}
    result["valid_worlds"] = len(worlds)
    if not worlds:
        return {**result, "status": "inconsistent_model"}
    full = (1 << len(worlds)) - 1
    result["model_already_implies_target"] = target_mask == full
    masks = _masks(candidates, worlds)
    result["target_reachable"] = bool(target_mask)
    result["necessary_literals_status"] = "evaluated" if target_mask else "target_unreachable"
    if target_mask:
        result["necessary_literals"] = _literal_json([literal for literal, mask in zip(candidates, masks)
                                                       if not (target_mask & (full ^ mask))])
    known_sets = []
    for size in range(len(candidates) + 1):
        for indices in itertools.combinations(range(len(candidates)), size):
            index_set = frozenset(indices)
            # A previously proven subset makes this candidate non-minimal.
            if any(found <= index_set for found in known_sets):
                continue
            if result["examined_subsets"] >= max_subsets:
                return {**result, "status": "truncated", "reason": "subset_budget_exhausted"}
            result["examined_subsets"] += 1
            mask = full
            for index in indices:
                mask &= masks[index]
            if mask and not (mask & (full ^ target_mask)):
                known_sets.append(index_set)
                result["minimal_sets"].append(_literal_json([candidates[i] for i in indices]))
    return {**result, "status": "complete"}


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _pointer(document, pointer):
    require(isinstance(pointer, str) and (pointer == "" or pointer.startswith("/")), "Invalid JSON pointer")
    value = document
    for key in pointer.split("/")[1:] if pointer else []:
        key = key.replace("~1", "/").replace("~0", "~")
        if isinstance(value, dict):
            value = value[key]
        elif isinstance(value, list) and key.isdigit():
            value = value[int(key)]
        else:
            raise KeyError("Pointer has no matching field")
    return value


def audit_evidence(atoms, observations, bindings, as_of, root):
    """Read-only, source-hashed historical facts; validity intervals are [from,until).

Each accepted source must independently expose every required binding through
explicit JSON pointers. Inconsistent overlapping records become conflict; no
latest-wins rule. A failed provenance check excludes evidence, not its negation.
"""
    require(isinstance(atoms, list) and all(isinstance(a, str) and NAME.fullmatch(a) for a in atoms)
            and len(atoms) == len(set(atoms)), "Invalid evidence atom set")
    require(isinstance(observations, list), "Observations must be a list")
    require(isinstance(bindings, dict) and bool(bindings)
            and all(isinstance(k, str) and isinstance(v, str) and v for k, v in bindings.items()), "Explicit identity bindings required")
    require(_finite(as_of), "Finite query time required")
    root = Path(root).resolve()
    values = {atom: set() for atom in atoms}
    accepted, excluded = [], []
    cache = {}
    for index, observation in enumerate(observations):
        atom = observation.get("atom") if isinstance(observation, dict) else None
        require(atom in values, "Observation refers to undeclared atom")
        reason = None
        if observation.get("bindings") != bindings:
            reason = "identity_mismatch"
        elif not (_finite(observation.get("valid_from")) and _finite(observation.get("valid_until"))
                  and observation["valid_from"] < observation["valid_until"]):
            reason = "invalid_validity_interval"
        elif not observation["valid_from"] <= as_of < observation["valid_until"]:
            reason = "outside_validity_interval"
        elif observation.get("method") not in {"observed", "derived"}:
            reason = "not_observational_evidence"
        elif not isinstance(observation.get("validity_basis"), str) or not observation["validity_basis"].strip():
            reason = "validity_basis_missing"
        if reason:
            excluded.append({"index": index, "atom": atom, "reason": reason})
            continue
        try:
            source = observation["source"]
            path = source["path"]
            require(isinstance(path, str) and not PurePosixPath(path).is_absolute()
                    and not PureWindowsPath(path).drive and "\\" not in path
                    and ".." not in PurePosixPath(path).parts, "Unsafe evidence path")
            file = (root / path).resolve()
            require(file.is_relative_to(root) and file.is_file(), "Missing or escaped evidence file")
            require(isinstance(source.get("sha256"), str) and re.fullmatch(r"[a-f0-9]{64}", source["sha256"]), "Invalid source digest")
            if path not in cache:
                raw = file.read_bytes()
                cache[path] = hashlib.sha256(raw).hexdigest(), json.loads(raw)
            digest, document = cache[path]
            require(digest == source["sha256"], "Source digest mismatch")
            selectors = observation.get("binding_selectors", {})
            require(set(selectors) == set(bindings), "Binding selectors incomplete")
            require(all(_pointer(document, selectors[key]) == value for key, value in bindings.items()), "Source binding mismatch")
            observed = _pointer(document, observation["pointer"])
            mapping = observation["value_map"]
            require(isinstance(mapping, list) and mapping, "Explicit value map required")
            mapped = []
            for rule in mapping:
                require(isinstance(rule, dict) and set(rule) == {"equals", "state"}
                        and rule["state"] in STATES, "Invalid value map")
                # bool is not interchangeable with numeric 0/1.
                if type(observed) is type(rule["equals"]) and observed == rule["equals"]:
                    mapped.append(rule["state"])
            require(len(mapped) <= 1, "Ambiguous value map")
            state = mapped[0] if mapped else "unknown"
            values[atom].update({"true", "false"} if state == "conflict" else {state} if state != "unknown" else set())
            accepted.append({"index": index, "atom": atom, "state": state, "source": path, "sha256": digest,
                             "pointer": observation["pointer"], "method": observation["method"]})
        except (KeyError, IndexError, ValueError, TypeError, OSError, UnicodeError):
            excluded.append({"index": index, "atom": atom, "reason": "source_selector_or_integrity_invalid"})
    states = {atom: "conflict" if len(found) == 2 else next(iter(found)) if found else "unknown" for atom, found in values.items()}
    return {"as_of": as_of, "bindings": bindings, "states": states, "accepted": accepted, "excluded": excluded,
            "scope": "Source-bound assertions at the stated historical query time. Hash and bindings do not authenticate reality; no live capability or atomic resource reservation is granted."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--premises", type=Path, help="Verify this literal array instead of searching")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-worlds", type=int, default=65536)
    parser.add_argument("--max-subsets", type=int, default=65536)
    args = parser.parse_args()
    model = json.loads(args.model.read_text(encoding="utf8"))
    result = (verify(model, json.loads(args.premises.read_text(encoding="utf8")), args.max_worlds) if args.premises
              else search(model, args.max_worlds, args.max_subsets))
    rendered = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf8", newline="\n")
    else:
        print(rendered, end="")
    return 0 if result["status"] in {"complete", "sufficient"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
