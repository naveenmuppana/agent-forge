# Author: Naveen Muppana
"""Typed messages passed between agents (docs/ARCHITECTURE.md §2)."""
from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _id() -> str:
    return uuid.uuid4().hex[:12]


class Risk(str, Enum):
    READ = "read"
    REVERSIBLE = "reversible-write"
    EXTERNAL = "irreversible-or-external"


class TaskStatus(str, Enum):
    PLANNING = "planning"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"   # an action needs your yes/no
    NEEDS_INPUT = "needs_input"             # a question only you can answer
    DONE = "done"
    FAILED = "failed"          # dead-lettered


class StepStatus(str, Enum):
    PENDING = "pending"
    DONE = "done"
    FAILED = "failed"
    BLOCKED = "blocked"        # waiting on Guardian escalation


# ---------- 1. Intake ----------
class TaskEnvelope(BaseModel):
    task_id: str = Field(default_factory=_id)
    source: Literal["chat", "email", "cron", "webhook"] = "chat"
    raw_input: str
    user_id: str = "default"
    intent: str = ""
    urgency: Literal["low", "normal", "high"] = "normal"
    deadline: Optional[datetime] = None
    created_at: datetime = Field(default_factory=_now)

    @property
    def dedupe_key(self) -> str:
        return hashlib.sha256(f"{self.user_id}|{self.raw_input.strip().lower()}".encode()).hexdigest()[:16]


# ---------- 2. Context ----------
class ContextPack(BaseModel):
    preferences: dict[str, Any] = Field(default_factory=dict)
    relevant_memories: list[str] = Field(default_factory=list)
    lessons: list[str] = Field(default_factory=list)
    matching_playbook: Optional[dict[str, Any]] = None


# ---------- 3. Planner ----------
class PlanStep(BaseModel):
    step_id: str
    capability: str                         # specialist name
    instruction: str                        # what the specialist should achieve
    depends_on: list[str] = Field(default_factory=list)
    risk: Risk = Risk.READ
    success_criteria: list[str] = Field(default_factory=list)


class Budget(BaseModel):
    max_llm_calls: int = 40
    max_cost_usd: float = 0.50
    max_replans: int = 2
    max_retries_per_step: int = 2


class Plan(BaseModel):
    goal: str
    success_criteria: list[str] = Field(default_factory=list)
    steps: list[PlanStep]
    budget: Budget = Field(default_factory=Budget)


# ---------- 5/6. Specialist output + Guardian ----------
class ProposedAction(BaseModel):
    """A side effect a specialist wants to perform. Not executed until the Guardian allows it."""
    action_id: str = Field(default_factory=_id)
    kind: str                               # e.g. "email.send", "calendar.hold"
    risk: Risk
    params: dict[str, Any] = Field(default_factory=dict)
    summary: str = ""

    @property
    def idempotency_key(self) -> str:
        return hashlib.sha256(f"{self.kind}|{sorted(self.params.items())}".encode()).hexdigest()[:16]


class ExecutedEffect(BaseModel):
    action: ProposedAction
    idempotency_key: str
    result: dict[str, Any] = Field(default_factory=dict)
    undo: Optional[dict[str, Any]] = None   # how to compensate (saga)


class StepResult(BaseModel):
    step_id: str
    status: StepStatus = StepStatus.PENDING
    output: str = ""
    proposed_actions: list[ProposedAction] = Field(default_factory=list)
    effects: list[ExecutedEffect] = Field(default_factory=list)
    attempts: int = 0
    error: Optional[str] = None
    fix_hint: str = ""                      # set by the Verifier on retry


class GuardDecision(BaseModel):
    action_id: str
    decision: Literal["allow", "escalate", "deny"]
    reason: str


# ---------- 7. Verifier ----------
class Verdict(BaseModel):
    decision: Literal["pass", "retry", "replan", "escalate"]
    score: float = Field(ge=0, le=1)
    reasons: list[str] = Field(default_factory=list)
    fix_hint: str = ""


# ---------- Shared checkpointed state ----------
class TaskState(BaseModel):
    envelope: TaskEnvelope
    status: TaskStatus = TaskStatus.PLANNING
    context: ContextPack = Field(default_factory=ContextPack)
    plan: Optional[Plan] = None
    results: dict[str, StepResult] = Field(default_factory=dict)
    pending_approvals: list[ProposedAction] = Field(default_factory=list)
    approved_action_ids: list[str] = Field(default_factory=list)
    replans: int = 0
    llm_calls: int = 0
    cost_usd: float = 0.0
    errors: list[str] = Field(default_factory=list)
    report: str = ""
    feedback: Optional[dict[str, Any]] = None
    updated_at: datetime = Field(default_factory=_now)

    @property
    def task_id(self) -> str:
        return self.envelope.task_id
