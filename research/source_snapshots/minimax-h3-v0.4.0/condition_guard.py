"""Bounded, side-effect-free four-valued admission decisions.

The caller owns observation identity, freshness and the lock that makes a decision
actionable. This helper performs no I/O and acquires no locks. Its timestamp is
the decision time, not a claim that an external observation was refreshed then.
"""
from __future__ import annotations

import json
import math
import time
import uuid

SCHEMA_VERSION = "1.0"
GATE_VERSION = "1.0"
MAX_CONDITIONS = 64
MAX_WITNESSES = 64
MAX_CONTEXT_BYTES = 8192


def observation_state(value):
    """Only native booleans assert facts; missing witnesses do not negate facts."""
    if type(value) is bool:
        return "true" if value else "false"
    if type(value) is str and value in {"unknown", "conflict"}:
        return value
    if type(value) is list:
        if len(value) > MAX_WITNESSES:
            return "unknown"
        # Lists are one level only: nested structures cannot introduce assertions.
        states = {observation_state(item) if type(item) is not list else "unknown"
                  for item in value}
        if "conflict" in states or {"true", "false"} <= states:
            return "conflict"
        if "true" in states:
            return "true"
        if "false" in states:
            return "false"
    return "unknown"


def negate_observation(value):
    """Negate known booleans without turning unknown/conflict into a true fact."""
    state = observation_state(value)
    return {"true": False, "false": True, "unknown": None, "conflict": "conflict"}[state]


def _context_copy(context):
    if type(context) is not dict:
        raise ValueError("Gate context must be a JSON object")
    def check(value, depth=0):
        if depth > 5:
            raise ValueError("Gate context is too deeply nested")
        if value is None or type(value) in {bool, int, str}:
            return
        if type(value) is float and math.isfinite(value):
            return
        if type(value) in {list, dict} and len(value) <= 64:
            if type(value) is dict:
                if any(type(key) is not str for key in value):
                    raise ValueError("Gate context keys must be strings")
                value = value.values()
            for item in value:
                check(item, depth + 1)
            return
        raise ValueError("Gate context must contain bounded finite JSON values")
    check(context)
    encoded = json.dumps(context, allow_nan=False, separators=(",", ":"), ensure_ascii=True)
    if len(encoded.encode("utf-8")) > MAX_CONTEXT_BYTES:
        raise ValueError("Gate context exceeds the receipt budget")
    return json.loads(encoded)


def evaluate_gate(gate_id, required, observations, context):
    """Return a decision receipt; malformed declarations fail before admission.

Explicit false facts differ from unknown and contradictory facts. In a witness
list, a native boolean supplies evidence even if another witness is missing;
opposite boolean witnesses produce conflict. No string 'true' or numeric 1 is a
true assertion. Inputs outside the declared required set cannot enable a gate.
"""
    if type(gate_id) is not str or not 1 <= len(gate_id) <= 128:
        raise ValueError("Gate identity must be a bounded nonempty string")
    if (type(required) not in {list, tuple} or not 1 <= len(required) <= MAX_CONDITIONS
            or any(type(name) is not str or not 1 <= len(name) <= 128 for name in required)
            or len(required) != len(set(required))):
        raise ValueError("Gate requires a bounded, nonempty set of unique condition names")
    if type(observations) is not dict:
        raise ValueError("Gate observations must be a mapping")
    context = _context_copy(context)
    conditions = {name: observation_state(observations.get(name)) for name in required}
    blocked = [name for name, state in conditions.items() if state != "true"]
    return {"schema_version": SCHEMA_VERSION, "gate_version": GATE_VERSION,
            "gate_id": gate_id, "decision_id": uuid.uuid4().hex,
            "observed_at": time.time(), "monotonic_ns": time.monotonic_ns(),
            "context": context, "conditions": conditions,
            "allowed": not blocked, "blocked": blocked}
