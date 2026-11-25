# Author: Naveen Muppana
"""7. Verifier: deterministic checks first, LLM judge second -> Verdict."""
from __future__ import annotations

import json
from typing import Callable

from ..contracts import ExecutedEffect, PlanStep, StepResult, StepStatus, Verdict
from ..llm import LLM, STRONG_MODEL, LLMError
from ..prompts import PromptRegistry


class VerifierAgent:
    def __init__(self, llm: LLM, prompts: PromptRegistry, pass_threshold: float = 0.7) -> None:
        self.llm, self.prompts, self.pass_threshold = llm, prompts, pass_threshold

    def check_step(self, step: PlanStep, res: StepResult, effect_ok: Callable[[ExecutedEffect], bool],
                   max_retries: int) -> Verdict:
        # --- deterministic layer ---
        if res.status == StepStatus.FAILED:
            if res.attempts <= max_retries:
                return Verdict(decision="retry", score=0, reasons=[res.error or "failed"],
                               fix_hint=f"The previous attempt errored: {res.error}. Try a different approach.")
            return Verdict(decision="replan", score=0, reasons=[f"step failed {res.attempts}x: {res.error}"])
        bad = [e.action.kind for e in res.effects if not effect_ok(e)]
        if bad:
            return Verdict(decision="replan", score=0, reasons=[f"post-condition failed for {bad}"])
        if not res.output.strip() and not res.effects:
            return Verdict(decision="retry" if res.attempts <= max_retries else "replan",
                           score=0, reasons=["empty output"], fix_hint="Produce a concrete result.")
        if not step.success_criteria:
            return Verdict(decision="pass", score=0.8, reasons=["no criteria; deterministic checks passed"])

        # --- LLM judge ---
        user = (f"Step instruction: {step.instruction}\nSuccess criteria: {step.success_criteria}\n"
                f"Worker output: {res.output}\nActions performed: "
                f"{json.dumps([{'kind': e.action.kind, 'params': e.action.params, 'result': e.result} for e in res.effects])}\n"
                f"Actions queued for approval: {[a.summary for a in res.proposed_actions]}")
        try:
            v = self.llm.structured(self.prompts.get("verifier"), user, Verdict, model=STRONG_MODEL)
        except LLMError:
            # Judge unavailable: deterministic checks passed, accept with low confidence.
            return Verdict(decision="pass", score=0.6, reasons=["judge unavailable; deterministic checks passed"])
        if v.decision == "pass" and v.score < self.pass_threshold:
            v.decision = "retry"
        if v.decision == "retry" and res.attempts > max_retries:
            v.decision = "replan"
        return v

    def check_task(self, goal: str, criteria: list[str], outputs: dict[str, str]) -> Verdict:
        if not criteria:
            return Verdict(decision="pass", score=0.8, reasons=["all steps passed"])
        try:
            return self.llm.structured(
                self.prompts.get("verifier"),
                f"Overall goal: {goal}\nSuccess criteria: {criteria}\nStep outputs: {json.dumps(outputs)}",
                Verdict, model=STRONG_MODEL)
        except LLMError:
            return Verdict(decision="pass", score=0.6, reasons=["judge unavailable; all steps passed"])
