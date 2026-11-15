"""Bounded backfill policy; admission remains the model owner's decision."""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class QueuePolicy:
    mode: str = "bounded_backfill"
    max_bypasses: int = 2
    lookahead: int = 8
    max_head_age_seconds: float = 30.0

    def __post_init__(self):
        if self.mode not in {"fifo", "bounded_backfill"}:
            raise ValueError("Queue policy must be fifo or bounded_backfill")
        if type(self.max_bypasses) is not int or not 1 <= self.max_bypasses <= 16:
            raise ValueError("Queue bypass budget must be 1..16")
        if type(self.lookahead) is not int or not 2 <= self.lookahead <= 64:
            raise ValueError("Queue candidate window must be 2..64; it is not a queue size limit")
        if type(self.max_head_age_seconds) not in {int, float} or not math.isfinite(self.max_head_age_seconds) or self.max_head_age_seconds <= 0:
            raise ValueError("Queue head age budget must be finite and positive")

    @classmethod
    def from_environment(cls, env):
        return cls(mode=env.get("H3_QUEUE_POLICY", "bounded_backfill"),
                   max_bypasses=int(env.get("H3_QUEUE_MAX_BYPASSES", "2")),
                   lookahead=int(env.get("H3_QUEUE_LOOKAHEAD", "8")),
                   max_head_age_seconds=float(env.get("H3_QUEUE_HEAD_AGE_SECONDS", "30")))

    def as_dict(self):
        return {"mode": self.mode, "max_bypasses": self.max_bypasses,
                "lookahead": self.lookahead, "max_head_age_seconds": self.max_head_age_seconds,
                "scope": "Successful later admissions are charged to every still-queued predecessor; no finite completion-time guarantee or queue length limit."}

    def may_bypass(self, *, bypasses, first_deferred_at, now):
        if self.mode != "bounded_backfill" or type(bypasses) is not int or bypasses < 0 or bypasses >= self.max_bypasses:
            return False
        if any(type(value) not in {int, float} or not math.isfinite(value) for value in (first_deferred_at, now)):
            return False
        return 0 <= now - first_deferred_at < self.max_head_age_seconds

    def may_bypass_all(self, predecessors, *, now):
        """Every skipped queued owner must have known remaining fairness budget.

        Callers provide only the bounded candidate prefix. Missing intermediate
        evidence is not permission to reuse the oldest job's remaining budget.
        """
        return (isinstance(predecessors, (list, tuple)) and 0 < len(predecessors) < self.lookahead
                and all(isinstance(state, dict) and self.may_bypass(
                    bypasses=state.get("bypasses"), first_deferred_at=state.get("first_deferred_at"), now=now)
                    for state in predecessors))


def guaranteed_resource_deferral(status, rejection):
    return (status == 429 and isinstance(rejection, dict)
            and rejection.get("guarantees_not_accepted") is True
            and rejection.get("reason") == "resource_budget_exhausted")
