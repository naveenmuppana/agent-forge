# Author: Naveen Muppana
"""Generic tool-using specialist, generalised from weather_agent.run_agent.

Read tools run immediately. Write tools never run inside the loop: they return a
ProposedAction which the Guardian later allows, escalates or denies, and only
then does `execute()` perform it (idempotently, recording an undo handle).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

from ..contracts import ContextPack, ExecutedEffect, PlanStep, ProposedAction, Risk, StepResult, StepStatus
from ..llm import LLM, PRIMARY_MODEL, LLMError
from ..state import Store

logger = logging.getLogger(__name__)
MAX_TOOL_STEPS = 6


@dataclass
class Tool:
    name: str
    description: str
    params: dict[str, str]                       # arg name -> type/description
    fn: Callable[..., Any]                       # read: returns result; write: executes the effect
    risk: Risk = Risk.READ
    undo: Optional[Callable[[dict], None]] = None   # write tools: compensate using effect.result
    check: Optional[Callable[[dict], bool]] = None  # deterministic post-condition for the Verifier


class AgentTurn(BaseModel):
    thought: str = ""
    tool: Optional[str] = None
    args: dict[str, Any] = Field(default_factory=dict)
    final: Optional[str] = None


@dataclass
class Specialist:
    name: str
    description: str
    tools: list[Tool]
    store: Store
    llm: LLM
    persona: str = ""
    model: str = PRIMARY_MODEL                   # set per step by the Router
    _by_name: dict[str, Tool] = field(init=False)

    def __post_init__(self) -> None:
        self._by_name = {t.name: t for t in self.tools}

    # ---------------- the loop ----------------
    def run(self, step: PlanStep, ctx: ContextPack, prior: dict[str, str], fix_hint: str = "") -> StepResult:
        result = StepResult(step_id=step.step_id)
        tool_doc = "\n".join(
            f"- {t.name}({', '.join(f'{k}: {v}' for k, v in t.params.items())}) [{t.risk.value}]: {t.description}"
            for t in self.tools
        )
        system = (
            f"You are the {self.name} specialist of a personal assistant. {self.persona}\n"
            f"Tools:\n{tool_doc}\n"
            "Each turn either call ONE tool (set `tool` and `args`) or finish (set `final`).\n"
            "Write tools are not executed immediately; they are queued for approval. Never invent tool results."
        )
        transcript = [
            f"Task: {step.instruction}",
            f"Success criteria: {step.success_criteria}",
            f"User preferences: {json.dumps(ctx.preferences)}",
            f"Lessons from past tasks: {ctx.lessons}",
            f"Outputs of earlier steps: {json.dumps(prior)}",
        ]
        if fix_hint:
            transcript.append(f"A reviewer rejected your previous attempt. Fix this: {fix_hint}")

        for _ in range(MAX_TOOL_STEPS):
            try:
                turn = self.llm.structured(system, "\n".join(transcript), AgentTurn, model=self.model)
            except LLMError as exc:
                result.status, result.error = StepStatus.FAILED, f"llm: {exc}"
                return result

            if turn.final is not None:
                result.output = turn.final
                result.status = StepStatus.DONE
                return result

            tool = self._by_name.get(turn.tool or "")
            if tool is None:
                transcript.append(f"Observation: unknown tool {turn.tool!r}. Choose from {list(self._by_name)}.")
                continue

            if tool.risk == Risk.READ:
                try:
                    obs = tool.fn(**turn.args)
                except Exception as exc:  # tool failure becomes an observation, not a crash
                    logger.warning("%s.%s failed: %s", self.name, tool.name, exc)
                    obs = {"error": str(exc)}
            else:
                action = ProposedAction(
                    kind=f"{self.name}.{tool.name}", risk=tool.risk, params=turn.args,
                    summary=turn.thought or f"{tool.name} {turn.args}",
                )
                same = next((a for a in result.proposed_actions if a.idempotency_key == action.idempotency_key), None)
                if same:   # the model repeated an identical call: don't queue (or ask about) it twice
                    obs = {"status": "already_queued", "action_id": same.action_id,
                           "note": "This exact action is already queued. Finish now."}
                else:
                    result.proposed_actions.append(action)
                    obs = {"status": "queued_for_approval", "action_id": action.action_id}
            transcript.append(f"Call {tool.name}({json.dumps(turn.args)}) -> {json.dumps(obs, default=str)[:2000]}")

        result.status, result.error = StepStatus.FAILED, f"exceeded {MAX_TOOL_STEPS} tool steps"
        return result

    # ---------------- side effects ----------------
    def _tool_for(self, action: ProposedAction) -> Tool:
        return self._by_name[action.kind.split(".", 1)[1]]

    def execute(self, action: ProposedAction, scope: str) -> ExecutedEffect:
        """Perform an approved action. `scope` (the task id) keeps keys per-task."""
        key = f"{scope}:{action.idempotency_key}"
        prior = self.store.find("effects", {"_id": key})
        if prior:  # crash-resume or retry: never perform the same side effect twice
            logger.info("Idempotent replay of %s (%s)", action.kind, key)
            return ExecutedEffect.model_validate(prior[0]["effect"])
        tool = self._tool_for(action)
        res = tool.fn(**action.params) or {}
        effect = ExecutedEffect(action=action, idempotency_key=key, result=res,
                                undo={"kind": action.kind, "result": res} if tool.undo else None)
        self.store.put("effects", {"_id": key, "effect": effect.model_dump(mode="json")})
        return effect

    def compensate(self, effect: ExecutedEffect) -> None:
        tool = self._tool_for(effect.action)
        if tool.undo and effect.undo:
            tool.undo(effect.undo["result"])
            self.store.delete("effects", effect.idempotency_key)

    def check(self, effect: ExecutedEffect) -> bool:
        tool = self._tool_for(effect.action)
        return tool.check(effect.result) if tool.check else True
