# Author: Naveen Muppana
"""1. Intake: raw trigger -> TaskEnvelope (intent, urgency, actionable?, duplicate?)."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel

from ..contracts import TaskEnvelope, TaskState
from ..llm import LLM, LLMError
from ..prompts import PromptRegistry
from ..state import Store


class IntakeResult(BaseModel):
    intent: str
    urgency: Literal["low", "normal", "high"] = "normal"
    actionable: bool = True
    clarification_question: str = ""


class IntakeAgent:
    def __init__(self, llm: LLM, store: Store, prompts: PromptRegistry) -> None:
        self.llm, self.store, self.prompts = llm, store, prompts

    def run(self, env: TaskEnvelope) -> tuple[TaskEnvelope, Optional[TaskState], IntakeResult]:
        """Returns (envelope, existing_duplicate_or_None, triage)."""
        dup = self.store.find_active_by_dedupe(env.dedupe_key)
        if dup:
            return env, dup, IntakeResult(intent=dup.envelope.intent)
        try:
            tri = self.llm.structured(self.prompts.get("intake"), f"Source: {env.source}\nRequest: {env.raw_input}",
                                      IntakeResult)
        except LLMError:
            tri = IntakeResult(intent=env.raw_input[:60])   # degrade: let the planner try anyway
        env.intent, env.urgency = tri.intent, tri.urgency
        return env, None, tri
