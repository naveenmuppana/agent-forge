# Author: Naveen Muppana
"""8. Reporter: final TaskState -> short message to the user."""
from __future__ import annotations

from ..contracts import TaskState, TaskStatus
from ..llm import LLM, LLMError
from ..prompts import PromptRegistry


class ReporterAgent:
    def __init__(self, llm: LLM, prompts: PromptRegistry) -> None:
        self.llm, self.prompts = llm, prompts

    @staticmethod
    def facts(state: TaskState) -> str:
        done = [e.action.summary or e.action.kind for r in state.results.values() for e in r.effects]
        lines = [f"Request: {state.envelope.raw_input}", f"Status: {state.status.value}"]
        lines += [f"Step {sid} ({r.status.value}): {r.output[:400]}" for sid, r in state.results.items()]
        if done:
            lines.append("Actions taken: " + "; ".join(done))
        if state.pending_approvals:
            lines.append("Waiting for approval: " + "; ".join(f"[{a.action_id}] {a.summary}" for a in state.pending_approvals))
        if state.errors:
            lines.append("Problems: " + "; ".join(state.errors[-3:]))
        return "\n".join(lines)

    def run(self, state: TaskState) -> str:
        facts = self.facts(state)
        if state.status == TaskStatus.FAILED:
            return f"I couldn't finish this task.\n{facts}"   # no LLM: failures must always reach the user
        try:
            return self.llm.complete(self.prompts.get("reporter"), facts)
        except LLMError:
            return facts
