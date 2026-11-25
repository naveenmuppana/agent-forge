# Author: Naveen Muppana
"""6. Guardian: every proposed side effect -> allow / escalate / deny, per policy.yaml.

Deliberately deterministic (no LLM): the safety gate must not be talked out of its rules.
"""
from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import yaml

from ..contracts import GuardDecision, ProposedAction, Risk

DEFAULT_POLICY_PATH = Path(__file__).resolve().parent.parent / "policy.yaml"


def _cond_ok(op: str, expected: Any, params: dict[str, Any]) -> bool:
    field, _, kind = op.rpartition("_")
    value = params.get(field)
    vals = value if isinstance(value, list) else [value]
    exp = expected if isinstance(expected, list) else [expected]
    if value is None:
        return False
    if kind == "endswith":
        return all(any(str(v).lower().endswith(str(e).lower()) for e in exp) for v in vals)
    if kind == "in":
        return all(v in exp for v in vals)
    if kind == "max":
        return all(float(v) <= float(expected) for v in vals)
    return False


class Guardian:
    def __init__(self, policy: dict[str, Any] | None = None, path: Path = DEFAULT_POLICY_PATH) -> None:
        self.policy = policy if policy is not None else yaml.safe_load(path.read_text(encoding="utf-8"))

    def decide(self, action: ProposedAction, approved_ids: set[str] = frozenset()) -> GuardDecision:
        p = self.policy
        if any(fnmatch(action.kind, g) for g in p.get("deny", [])):
            return GuardDecision(action_id=action.action_id, decision="deny", reason="denied by policy")
        if action.action_id in approved_ids:
            return GuardDecision(action_id=action.action_id, decision="allow", reason="approved by user")
        if any(fnmatch(action.kind, g) for g in p.get("always_escalate", [])):
            return GuardDecision(action_id=action.action_id, decision="escalate", reason="always requires your approval")
        if action.risk == Risk.READ:
            return GuardDecision(action_id=action.action_id, decision="allow", reason="read-only")
        if action.risk == Risk.REVERSIBLE and p.get("auto_reversible", True):
            return GuardDecision(action_id=action.action_id, decision="allow", reason="reversible")
        for rule in p.get("standing_rules", []) or []:
            if fnmatch(action.kind, rule["kind"]) and all(
                _cond_ok(op, exp, action.params) for op, exp in (rule.get("when") or {}).items()
            ):
                return GuardDecision(action_id=action.action_id, decision="allow",
                                     reason=f"standing rule: {rule.get('name', rule['kind'])}")
        return GuardDecision(action_id=action.action_id, decision="escalate",
                             reason="external/irreversible action not covered by a standing rule")
