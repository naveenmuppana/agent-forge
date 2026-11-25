# Author: Naveen Muppana
"""9. Learner: turns finished episodes into lessons, playbooks, preferences and prompt candidates.

It may write: lessons, playbooks, preferences, prompt *candidates*, eval cases.
It may NOT write: policy.yaml or anything the Guardian reads. That boundary is the
guardrail that stops self-improvement from widening its own permissions.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel

from ..contracts import TaskState, TaskStatus
from ..llm import LLM, LLMError
from ..prompts import PromptRegistry
from ..state import Store
from .context import similarity

logger = logging.getLogger(__name__)
PLAYBOOK_MIN_SUCCESSES = 3
LESSON_DECAY = 0.5
LESSON_BOOST = 0.1
MIN_LESSON_WEIGHT = 0.2      # below this a lesson is no longer retrieved, so it is pruned


class Lesson(BaseModel):
    lesson: str


def score_episode(state: TaskState) -> float:
    """0..1 outcome score from automatic signals plus explicit feedback."""
    if state.status == TaskStatus.FAILED:
        base = 0.0
    elif state.status == TaskStatus.WAITING_APPROVAL:
        base = 0.6
    else:
        base = 1.0
    base -= 0.1 * state.replans
    base -= 0.05 * sum(max(r.attempts - 1, 0) for r in state.results.values())
    fb = state.feedback or {}
    if fb.get("rating") == "down" or fb.get("undone"):
        base = min(base, 0.2)
    elif fb.get("rating") == "up":
        base = max(base, 0.9)
    return round(max(0.0, min(1.0, base)), 3)


class LearnerAgent:
    def __init__(self, llm: LLM, store: Store, prompts: PromptRegistry) -> None:
        self.llm, self.store, self.prompts = llm, store, prompts

    # ---------- called by the orchestrator when a task closes ----------
    def on_episode(self, state: TaskState) -> dict[str, Any]:
        score = score_episode(state)
        env = state.envelope
        prev = self.store.find("episodes", {"_id": state.task_id})
        prev_dir = prev[0].get("lesson_signal") if prev else None
        ep = {
            "_id": state.task_id, "user_id": env.user_id, "raw_input": env.raw_input, "intent": env.intent,
            "outcome": state.status.value, "score": score, "summary": state.report[:500],
            "plan": state.plan.model_dump(mode="json") if state.plan else None,
            "errors": state.errors, "feedback": state.feedback, "replans": state.replans,
            "lessons_used": state.context.lessons, "cost_usd": state.cost_usd,
            "closed_at": datetime.now(timezone.utc).isoformat(),
            "lesson_signal": "good" if score >= 0.7 else "bad",
        }
        self.store.put("episodes", ep)
        learned: dict[str, Any] = {"score": score}

        # Lessons that were in context for a bad episode lose weight; good episodes reinforce them.
        # An episode counts once: later feedback replaces (not adds to) the earlier signal.
        if prev_dir != ep["lesson_signal"]:
            self._reweight_lessons(env.user_id, state.context.lessons, good=score >= 0.7, undo=prev_dir)

        if score < 0.7 or state.errors or (state.feedback or {}).get("correction"):
            lesson = self._reflect(state)
            if lesson:
                learned["lesson"] = lesson
        if score >= 0.9 and state.plan:
            learned["playbook"] = self._update_playbook(env.user_id, env.intent, ep)
        if score < 0.5:
            # Failed episodes become regression cases for prompt evals.
            self.store.put("eval_cases", {"_id": state.task_id, "raw_input": env.raw_input,
                                          "intent": env.intent, "expect_capabilities": self._caps(ep),
                                          "note": "; ".join(state.errors[-2:])})
        return learned

    def on_feedback(self, state: TaskState) -> dict[str, Any]:
        """Explicit feedback after the fact: update preferences and re-learn from the episode."""
        fb = state.feedback or {}
        if fb.get("preferences"):
            doc = (self.store.find("preferences", {"_id": state.envelope.user_id}) or
                   [{"_id": state.envelope.user_id, "prefs": {}}])[0]
            doc["prefs"].update(fb["preferences"])
            self.store.put("preferences", doc)
        return self.on_episode(state)

    # ---------- helpers ----------
    @staticmethod
    def _caps(ep: dict) -> list[str]:
        return sorted({s["capability"] for s in (ep.get("plan") or {}).get("steps", [])})

    def _reflect(self, state: TaskState) -> Optional[str]:
        fb = state.feedback or {}
        trace = (f"Request: {state.envelope.raw_input}\nOutcome: {state.status.value}\n"
                 f"Plan: {[(s.capability, s.instruction) for s in (state.plan.steps if state.plan else [])]}\n"
                 f"Errors: {state.errors}\nUser feedback: {fb}")
        try:
            text = self.llm.structured(self.prompts.get("reflect"), trace, Lesson).lesson.strip()
        except LLMError:
            return None
        if not text:
            return None
        uid = state.envelope.user_id
        for existing in self.store.find("lessons", {"user_id": uid}):
            if similarity(existing["text"], text) > 0.6:      # merge near-duplicates
                existing["weight"] = min(existing.get("weight", 1) + 0.25, 2.0)
                self.store.put("lessons", existing)
                return existing["text"]
        self.store.put("lessons", {"_id": uuid.uuid4().hex[:12], "user_id": uid, "text": text,
                                   "intent": state.envelope.intent, "weight": 1.0,
                                   "source_task": state.task_id,
                                   "created_at": datetime.now(timezone.utc).isoformat()})
        return text

    def _reweight_lessons(self, user_id: str, used: list[str], good: bool, undo: Optional[str] = None) -> None:
        for l in self.store.find("lessons", {"user_id": user_id}):
            if l["text"] in used:
                w = l.get("weight", 1)
                if undo == "good":
                    w -= LESSON_BOOST
                elif undo == "bad":
                    w /= LESSON_DECAY
                l["weight"] = min(w + LESSON_BOOST, 2.0) if good else w * LESSON_DECAY
                if l["weight"] < MIN_LESSON_WEIGHT:
                    self.store.delete("lessons", l["_id"])      # pruned: kept proving wrong
                else:
                    self.store.put("lessons", l)

    def _update_playbook(self, user_id: str, intent: str, ep: dict) -> Optional[str]:
        similar = [e for e in self.store.find("episodes", {"user_id": user_id})
                   if e["score"] >= 0.9 and e.get("plan") and similarity(e["intent"], intent) >= 0.5]
        if len(similar) < PLAYBOOK_MIN_SUCCESSES:
            return None
        best = max(similar, key=lambda e: (e["score"], -e.get("cost_usd", 0)))
        steps = [{k: s[k] for k in ("step_id", "capability", "instruction", "depends_on", "risk", "success_criteria")}
                 for s in best["plan"]["steps"]]
        pid = f"{user_id}:{intent.lower().strip()[:60]}"
        self.store.put("playbooks", {"_id": pid, "user_id": user_id, "intent": intent, "steps": steps,
                                     "support": len(similar), "from_task": best["_id"]})
        return pid

    # ---------- weekly: propose prompt changes from failure patterns ----------
    def propose_prompt_improvements(self, agent: str = "planner", min_failures: int = 3) -> Optional[str]:
        fails = [e for e in self.store.find("episodes") if e["score"] < 0.5]
        if len(fails) < min_failures:
            return None
        digest = "\n".join(f"- {e['raw_input']} | errors: {e['errors'][-2:]} | feedback: {e.get('feedback')}"
                           for e in fails[-15:])
        try:
            new = self.llm.complete(
                "You improve system prompts for an AI agent. Return ONLY the full revised prompt text.",
                f"Current {agent} prompt:\n{self.prompts.get(agent)}\n\nRecent failures:\n{digest}\n"
                "Revise the prompt to prevent these failures without making it task-specific.")
        except LLMError:
            return None
        return self.prompts.propose(agent, new.strip(), reason=f"{len(fails)} failed episodes")

    def digest(self, user_id: str = "default") -> dict[str, Any]:
        """What the assistant learned; shown to the user weekly. Any lesson can be deleted."""
        return {
            "lessons": [{"id": l["_id"], "text": l["text"], "weight": round(l.get("weight", 1), 2)}
                        for l in self.store.find("lessons", {"user_id": user_id})],
            "playbooks": [{"id": p["_id"], "intent": p["intent"], "support": p["support"]}
                          for p in self.store.find("playbooks", {"user_id": user_id})],
            "prompt_candidates": [{"id": c["_id"], "agent": c["name"], "reason": c["reason"]}
                                  for c in self.prompts.candidates()],
        }
