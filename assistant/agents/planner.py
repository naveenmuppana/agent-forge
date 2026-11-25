# Author: Naveen Muppana
"""3. Planner: envelope + context -> validated Plan (a DAG of capability steps)."""
from __future__ import annotations

import json
from typing import Optional

from ..contracts import ContextPack, Plan, TaskEnvelope
from ..llm import LLM, STRONG_MODEL, LLMError
from ..prompts import PromptRegistry


class PlanError(Exception):
    pass


def validate_plan(plan: Plan, capabilities: set[str], done_ids: frozenset[str] = frozenset()) -> list[str]:
    """`done_ids` are steps completed before a replan; new steps may depend on them."""
    problems: list[str] = []
    ids = [s.step_id for s in plan.steps]
    if not plan.steps:
        problems.append("plan has no steps")
    if len(ids) != len(set(ids)):
        problems.append("duplicate step_id")
    for s in plan.steps:
        if s.capability not in capabilities:
            problems.append(f"step {s.step_id}: unknown capability {s.capability!r}; use one of {sorted(capabilities)}")
        for d in s.depends_on:
            if d not in ids and d not in done_ids:
                problems.append(f"step {s.step_id}: depends on missing step {d!r}")
    # cycle check (Kahn)
    deps = {s.step_id: set(d for d in s.depends_on if d in ids) for s in plan.steps}
    while deps:
        free = [k for k, v in deps.items() if not v]
        if not free:
            problems.append(f"dependency cycle among {sorted(deps)}")
            break
        for k in free:
            deps.pop(k)
        for v in deps.values():
            v.difference_update(free)
    return problems


class PlannerAgent:
    def __init__(self, llm: LLM, prompts: PromptRegistry, capabilities: dict[str, str]) -> None:
        self.llm, self.prompts, self.capabilities = llm, prompts, capabilities

    def run(self, env: TaskEnvelope, ctx: ContextPack, *, failure: Optional[str] = None,
            completed: Optional[dict[str, str]] = None, id_prefix: str = "") -> Plan:
        caps = "\n".join(f"- {k}: {v}" for k, v in self.capabilities.items())
        user = [
            f"Goal: {env.raw_input}",
            f"Intent: {env.intent}",
            f"Capabilities:\n{caps}",
            f"User preferences: {json.dumps(ctx.preferences)}",
            f"Lessons learned: {ctx.lessons}",
            f"Similar past tasks: {ctx.relevant_memories}",
        ]
        if ctx.matching_playbook:
            user.append(f"A playbook that worked before for this kind of task (adapt it): {json.dumps(ctx.matching_playbook)}")
        if failure:
            user.append(f"REPLAN. The previous plan failed: {failure}\n"
                        f"Already completed (do not repeat; you may reference their outputs): {json.dumps(completed or {})}")
        prompt = "\n".join(user)

        last_problems: list[str] = []
        for _ in range(3):
            try:
                plan = self.llm.structured(self.prompts.get("planner"), prompt, Plan, model=STRONG_MODEL)
            except LLMError as exc:
                raise PlanError(str(exc)) from exc
            done_ids = frozenset(completed or {})
            last_problems = validate_plan(plan, set(self.capabilities), done_ids)
            if any(s.step_id in done_ids for s in plan.steps):
                last_problems.append("new steps must not reuse ids of completed steps")
            if not last_problems:
                if id_prefix:
                    remap = {s.step_id: f"{id_prefix}{s.step_id}" for s in plan.steps}
                    for s in plan.steps:
                        s.step_id = remap[s.step_id]
                        s.depends_on = [remap.get(d, d) for d in s.depends_on]
                return plan
            prompt += f"\n\nYour last plan was invalid: {last_problems}. Fix it."
        raise PlanError(f"invalid plan after retries: {last_problems}")
