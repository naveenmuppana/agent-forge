# Author: Naveen Muppana
"""4. Router: picks ready steps, a specialist + model for each, and tracks reliability.

Reliability stats (fed back by the orchestrator) drive two decisions:
  * model escalation - a specialist that keeps failing on the cheap model gets the strong one;
  * circuit breaker  - after N consecutive failures a specialist is skipped for a cool-down,
                       which forces a replan around it instead of burning budget.
"""
from __future__ import annotations

import time
from typing import Optional

from ..contracts import Plan, PlanStep, StepResult, StepStatus
from ..llm import PRIMARY_MODEL, STRONG_MODEL
from ..state import Store

BREAKER_THRESHOLD = 3
BREAKER_COOLDOWN_S = 600


class Router:
    def __init__(self, store: Store, specialists: set[str]) -> None:
        self.store, self.specialists = store, specialists

    def ready_steps(self, plan: Plan, results: dict[str, StepResult]) -> list[PlanStep]:
        def st(sid: str) -> StepStatus:
            return results[sid].status if sid in results else StepStatus.PENDING

        return [s for s in plan.steps
                if st(s.step_id) == StepStatus.PENDING
                and all(st(d) == StepStatus.DONE for d in s.depends_on)]

    def _stats(self, name: str) -> dict:
        got = self.store.find("routing_stats", {"_id": name})
        return got[0] if got else {"_id": name, "ok": 0, "fail": 0, "consecutive_fail": 0, "opened_at": 0}

    def is_open(self, name: str) -> bool:
        s = self._stats(name)
        return s["consecutive_fail"] >= BREAKER_THRESHOLD and time.time() - s["opened_at"] < BREAKER_COOLDOWN_S

    def choose_model(self, name: str) -> str:
        s = self._stats(name)
        total = s["ok"] + s["fail"]
        if total >= 5 and s["ok"] / total < 0.7:
            return STRONG_MODEL
        return PRIMARY_MODEL

    def route(self, step: PlanStep) -> Optional[str]:
        """Returns the specialist name, or None if it is unavailable (breaker open)."""
        if step.capability not in self.specialists or self.is_open(step.capability):
            return None
        return step.capability

    def record(self, name: str, ok: bool) -> None:
        s = self._stats(name)
        if ok:
            s["ok"] += 1
            s["consecutive_fail"] = 0
        else:
            s["fail"] += 1
            s["consecutive_fail"] += 1
            if s["consecutive_fail"] >= BREAKER_THRESHOLD:
                s["opened_at"] = time.time()
        self.store.put("routing_stats", s)
