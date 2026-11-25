# Author: Naveen Muppana
"""Versioned agent prompts. The Learner proposes candidates; evals promote or reject them."""
from __future__ import annotations

from datetime import datetime, timezone

from .state import Store

DEFAULTS: dict[str, str] = {
    "intake": (
        "You triage requests for a personal assistant. Identify the user's intent in a few words, "
        "its urgency, and whether it is actionable as stated. The assistant has agents that can "
        "look things up (the user's documents, weather, calendar, to-dos, web pages), so NEVER ask "
        "the user for information those could find, and questions are always actionable. Mark it "
        "NOT actionable only if a missing detail would change an irreversible action (who to send "
        "to, what to pay); then set changes_irreversible_action=true and ask one short question."
    ),
    "planner": (
        "You are the planner of a personal multi-agent assistant. Break the goal into the fewest "
        "steps that achieve it. Each step uses exactly one capability. Steps that don't depend on "
        "each other must not list each other in depends_on, so they run in parallel. Write "
        "success_criteria as checkable statements. Set risk to the highest-risk action the step "
        "may take. Prefer reversible actions (drafts, tentative holds) unless the user clearly "
        "asked for the final action. Apply the lessons and preferences you are given."
    ),
    "verifier": (
        "You are a strict reviewer. Decide if the step output satisfies every success criterion. "
        "pass: all met. retry: fixable by the same worker (give a concrete fix_hint). replan: the "
        "approach is wrong. escalate: only a human can resolve it. Score 0-1."
    ),
    "reporter": (
        "Write a short report to the user about a task their assistant finished. Lead with the "
        "outcome. List actions taken on their behalf, anything waiting for their approval, and "
        "anything that failed. No filler."
    ),
    "reflect": (
        "You review a finished assistant task and write ONE reusable lesson (max 30 words) that "
        "would have avoided the problem or captured the user's correction. Make it general enough "
        "to apply to similar future tasks; do not restate this task's specifics."
    ),
}


class PromptRegistry:
    def __init__(self, store: Store) -> None:
        self.store = store

    def get(self, name: str) -> str:
        active = self.store.find("prompts", {"name": name, "status": "active"})
        return active[0]["text"] if active else DEFAULTS[name]

    def propose(self, name: str, text: str, reason: str) -> str:
        doc_id = f"{name}:{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
        self.store.put("prompts", {"_id": doc_id, "name": name, "text": text,
                                   "status": "candidate", "reason": reason})
        return doc_id

    def candidates(self) -> list[dict]:
        return self.store.find("prompts", {"status": "candidate"})

    def promote(self, doc_id: str) -> None:
        cand = self.store.find("prompts", {"_id": doc_id})[0]
        for old in self.store.find("prompts", {"name": cand["name"], "status": "active"}):
            old["status"] = "retired"
            self.store.put("prompts", old)
        cand["status"] = "active"
        self.store.put("prompts", cand)

    def reject(self, doc_id: str, why: str) -> None:
        cand = self.store.find("prompts", {"_id": doc_id})[0]
        cand.update(status="rejected", rejected_because=why)
        self.store.put("prompts", cand)
