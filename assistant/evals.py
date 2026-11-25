# Author: Naveen Muppana
"""Eval gate for prompt changes proposed by the Learner.

A candidate planner prompt is scored on eval cases (seed cases in evals/cases.yaml
plus regression cases the Learner records from failed episodes). It is promoted
only if it scores at least as well as the active prompt; otherwise rejected.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from .agents.planner import PlanError, PlannerAgent
from .contracts import ContextPack, TaskEnvelope
from .prompts import PromptRegistry

if TYPE_CHECKING:
    from .orchestrator import Assistant

SEED_CASES = Path(__file__).resolve().parent.parent / "evals" / "cases.yaml"


def load_cases(store) -> list[dict[str, Any]]:
    seed = yaml.safe_load(SEED_CASES.read_text(encoding="utf-8")) if SEED_CASES.exists() else []
    return list(seed or []) + store.find("eval_cases")


class _FixedPrompt(PromptRegistry):
    def __init__(self, store, name: str, text: str) -> None:
        super().__init__(store)
        self._name, self._text = name, text

    def get(self, name: str) -> str:
        return self._text if name == self._name else super().get(name)


def score_planner_prompt(a: "Assistant", text: str, cases: list[dict]) -> float:
    planner = PlannerAgent(a.llm, _FixedPrompt(a.store, "planner", text), a.planner.capabilities)
    hits = 0
    for c in cases:
        try:
            plan = planner.run(TaskEnvelope(raw_input=c["raw_input"], intent=c.get("intent", "")), ContextPack())
        except PlanError:
            continue
        caps = {s.capability for s in plan.steps}
        expected = set(c.get("expect_capabilities", []))
        max_steps = c.get("max_steps", 6)
        if expected <= caps and len(plan.steps) <= max_steps:
            hits += 1
    return hits / len(cases) if cases else 0.0


def evaluate_candidates(a: "Assistant") -> list[dict[str, Any]]:
    cases = load_cases(a.store)
    out = []
    for cand in a.prompts.candidates():
        if cand["name"] != "planner":
            out.append({"id": cand["_id"], "result": "skipped (no automatic eval for this agent yet)"})
            continue
        base = score_planner_prompt(a, a.prompts.get("planner"), cases)
        new = score_planner_prompt(a, cand["text"], cases)
        if new >= base:
            a.prompts.promote(cand["_id"])
            out.append({"id": cand["_id"], "result": "promoted", "baseline": base, "candidate": new})
        else:
            a.prompts.reject(cand["_id"], f"regressed {base:.2f} -> {new:.2f}")
            out.append({"id": cand["_id"], "result": "rejected", "baseline": base, "candidate": new})
    return out
