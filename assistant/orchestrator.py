# Author: Naveen Muppana
"""The state machine that wires the agents together (docs/ARCHITECTURE.md §1).

Intake -> Context -> Planner -> loop[ Router -> Specialists (parallel) -> Guardian -> Verifier ]
       -> final Verifier -> Reporter -> Learner

TaskState is checkpointed after every transition, so `resume_running()` can pick
up any task after a crash; idempotency keys stop side effects running twice.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

from .agents.context import ContextAgent
from .agents.guardian import Guardian
from .agents.intake import IntakeAgent
from .agents.learner import LearnerAgent
from .agents.planner import PlanError, PlannerAgent
from .agents.reporter import ReporterAgent
from .agents.router import Router
from .agents.verifier import VerifierAgent
from .contracts import (Plan, PlanStep, StepResult, StepStatus, TaskEnvelope, TaskState, TaskStatus)
from .llm import LLM, ResilientLLM, openai_backend
from .prompts import PromptRegistry
from .specialists import Specialist, build_all
from .state import Store, default_store

logger = logging.getLogger(__name__)
MAX_LOOP_ITERATIONS = 50
MAX_PARALLEL = 4


class Replan(Exception):
    pass


class Assistant:
    def __init__(self, llm: Optional[LLM] = None, store: Optional[Store] = None,
                 specialists: Optional[dict[str, Specialist]] = None, guardian: Optional[Guardian] = None,
                 embed: Optional[Callable[[str], list[float]]] = None) -> None:
        self.llm = llm or ResilientLLM(openai_backend())
        self.store = store or default_store()
        self.specialists = specialists if specialists is not None else build_all(self.store, self.llm)
        self.prompts = PromptRegistry(self.store)
        self.intake = IntakeAgent(self.llm, self.store, self.prompts)
        self.context = ContextAgent(self.store, embed)
        self.planner = PlannerAgent(self.llm, self.prompts, {n: s.description for n, s in self.specialists.items()})
        self.router = Router(self.store, set(self.specialists))
        self.guardian = guardian or Guardian()
        self.verifier = VerifierAgent(self.llm, self.prompts)
        self.reporter = ReporterAgent(self.llm, self.prompts)
        self.learner = LearnerAgent(self.llm, self.store, self.prompts)

    # =============================== public API ===============================
    def submit(self, raw_input: str, source: str = "chat", user_id: str = "default",
               task_id: Optional[str] = None) -> TaskState:
        """`task_id` lets a caller (e.g. the UI) know the id up front and poll while it runs."""
        env = TaskEnvelope(raw_input=raw_input, source=source, user_id=user_id)
        if task_id:
            env.task_id = task_id
        calls0, cost0 = self.llm.usage.calls, self.llm.usage.cost_usd   # meter intake + planning too
        env, dup, triage = self.intake.run(env)
        if dup:
            logger.info("Duplicate of active task %s", dup.task_id)
            if task_id:   # the caller is polling `task_id`: point it at the existing task
                self._save(TaskState(envelope=env, status=TaskStatus.DONE,
                                     report=f"Already working on this as task {dup.task_id}."))
            return dup
        state = TaskState(envelope=env)
        self._save(state)                       # visible as "planning" while context + plan run
        state.context = self.context.run(env)
        if not triage.actionable:
            state.status = TaskStatus.NEEDS_INPUT
            state.report = triage.clarification_question or "Could you give me a bit more detail?"
            self._meter(state, calls0, cost0)
            self._save(state)
            return state
        try:
            state.plan = self.planner.run(env, state.context)
        except PlanError as exc:
            self._meter(state, calls0, cost0)
            return self._fail(state, f"planning failed: {exc}")
        self._meter(state, calls0, cost0)
        state.status = TaskStatus.RUNNING
        self._save(state)
        return self.run(state.task_id)

    def run(self, task_id: str) -> TaskState:
        state = self._load(task_id)
        if state.status != TaskStatus.RUNNING:
            return state
        start_calls, start_cost = self.llm.usage.calls, self.llm.usage.cost_usd

        for _ in range(MAX_LOOP_ITERATIONS):
            self._meter(state, start_calls, start_cost)
            start_calls, start_cost = self.llm.usage.calls, self.llm.usage.cost_usd
            b = state.plan.budget
            if state.llm_calls > b.max_llm_calls or state.cost_usd > b.max_cost_usd:
                return self._fail(state, f"budget exceeded ({state.llm_calls} calls, ${state.cost_usd:.3f})")

            ready = self.router.ready_steps(state.plan, state.results)
            if not ready:
                break
            failures = []
            for step, res in self._dispatch(state, ready):
                try:   # handle every parallel result before replanning so finished work is kept
                    self._handle(state, step, res)
                except Replan as why:
                    failures.append(str(why))
            self._save(state)
            if failures and not self._replan(state, " | ".join(failures)):
                return self._fail(state, f"gave up after {state.replans} replans: {' | '.join(failures)}")
            self._save(state)
        else:
            return self._fail(state, "orchestrator loop limit hit")

        self._meter(state, start_calls, start_cost)
        return self._finish(state)

    def approve(self, task_id: str, approve: list[str] = (), reject: list[str] = ()) -> TaskState:
        """Your one-tap answer to escalated actions. The task resumes automatically."""
        state = self._load(task_id)
        state.approved_action_ids += list(approve)
        decided = set(approve) | set(reject)
        state.pending_approvals = [a for a in state.pending_approvals if a.action_id not in decided]
        for sid, res in list(state.results.items()):
            if res.status != StepStatus.BLOCKED:
                continue
            declined = [a for a in res.proposed_actions if a.action_id in set(reject)]
            res.proposed_actions = [a for a in res.proposed_actions if a.action_id not in set(reject)]
            if declined:
                res.output += "\n(User declined: " + "; ".join(a.summary for a in declined) + ")"
            if any(a.action_id not in decided for a in res.proposed_actions):
                continue      # still waiting on another decision for this step
            step = self._step(state, sid)
            try:
                self._handle(state, step, res)
            except Replan as why:
                state.status = TaskStatus.RUNNING
                if not self._replan(state, str(why)):
                    return self._fail(state, f"gave up after replans: {why}")
        state.status = TaskStatus.RUNNING
        self._save(state)
        return self.run(task_id)

    def clarify(self, task_id: str, answer: str) -> TaskState:
        state = self._load(task_id)
        state.envelope.raw_input += f"\nUser clarification: {answer}"
        state.context = self.context.run(state.envelope)
        done = {sid: r.output for sid, r in state.results.items() if r.status == StepStatus.DONE}
        try:
            new = self.planner.run(state.envelope, state.context, completed=done, id_prefix=f"c{state.replans}_")
        except PlanError as exc:
            return self._fail(state, f"planning failed: {exc}")
        self._merge_plan(state, new)
        state.status = TaskStatus.RUNNING
        self._save(state)
        return self.run(task_id)

    def feedback(self, task_id: str, rating: Optional[str] = None, correction: str = "",
                 preferences: Optional[dict] = None, undo: bool = False) -> dict:
        state = self._load(task_id)
        if undo:
            self._compensate(state)
        state.feedback = {"rating": rating, "correction": correction, "preferences": preferences or {}, "undone": undo}
        self._save(state)
        return self.learner.on_feedback(state)

    def resume_running(self) -> list[str]:
        """Call on startup: continue every task that was mid-flight when the process died."""
        ids = self.store.task_ids_with_status(TaskStatus.RUNNING.value)
        for tid in ids:
            logger.info("Resuming task %s after restart", tid)
            self.run(tid)
        return ids

    def get(self, task_id: str) -> Optional[TaskState]:
        return self.store.load_task(task_id)

    def recent(self, limit: int = 20) -> list[TaskState]:
        return self.store.recent_tasks(limit)

    # =============================== internals ===============================
    def _dispatch(self, state: TaskState, ready: list[PlanStep]) -> list[tuple[PlanStep, StepResult]]:
        def work(step: PlanStep) -> tuple[PlanStep, StepResult]:
            prev = state.results.get(step.step_id)
            attempts = (prev.attempts if prev else 0) + 1
            name = self.router.route(step)
            if name is None:
                return step, StepResult(step_id=step.step_id, status=StepStatus.FAILED, attempts=99,
                                        error=f"specialist {step.capability!r} unavailable (circuit open)")
            sp = self.specialists[name]
            sp.model = self.router.choose_model(name)
            prior = {d: state.results[d].output for d in step.depends_on if d in state.results}
            try:
                res = sp.run(step, state.context, prior, fix_hint=prev.fix_hint if prev else "")
            except Exception as exc:   # a specialist bug must not take the whole task down
                logger.exception("Specialist %s crashed", name)
                res = StepResult(step_id=step.step_id, status=StepStatus.FAILED, error=f"crash: {exc}")
            res.attempts = attempts
            return step, res

        if len(ready) == 1:
            return [work(ready[0])]
        with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL, len(ready))) as pool:
            return list(pool.map(work, ready))

    def _handle(self, state: TaskState, step: PlanStep, res: StepResult) -> None:
        """Guardian gate, then Verifier verdict, for one step result."""
        sp = self.specialists.get(step.capability)
        state.results[step.step_id] = res
        still_pending = []
        if res.status != StepStatus.FAILED and sp is not None:
            for action in res.proposed_actions:
                d = self.guardian.decide(action, set(state.approved_action_ids))
                if d.decision == "allow":
                    try:
                        res.effects.append(sp.execute(action, scope=state.task_id))
                    except Exception as exc:
                        res.status, res.error = StepStatus.FAILED, f"{action.kind} failed: {exc}"
                        state.errors.append(res.error)
                        break
                elif d.decision == "escalate":
                    still_pending.append(action)
                    if all(p.idempotency_key != action.idempotency_key for p in state.pending_approvals):
                        state.pending_approvals.append(action)
                else:
                    state.errors.append(f"blocked by policy: {action.kind} ({d.reason})")
                    res.output += f"\n(Not done, blocked by policy: {action.summary})"
        res.proposed_actions = still_pending
        if still_pending and res.status != StepStatus.FAILED:
            res.status = StepStatus.BLOCKED
            return

        verdict = self.verifier.check_step(step, res, sp.check if sp else (lambda e: True),
                                           state.plan.budget.max_retries_per_step)
        # Reliability counts once per step outcome: retries inside a step don't trip the
        # breaker, and a step skipped *because* the breaker is open isn't a new failure.
        if verdict.decision != "retry" and "circuit open" not in (res.error or ""):
            self.router.record(step.capability, verdict.decision == "pass")
        logger.info("Step %s -> %s (%.2f) %s", step.step_id, verdict.decision, verdict.score, verdict.reasons)
        if verdict.decision == "pass":
            res.status = StepStatus.DONE
        elif verdict.decision == "retry":
            self._compensate_step(res)      # undo this attempt's reversible effects before redoing it
            res.status, res.fix_hint = StepStatus.PENDING, verdict.fix_hint
        elif verdict.decision == "replan":
            self._compensate_step(res)
            res.status = StepStatus.FAILED
            state.errors.append(f"{step.step_id}: {'; '.join(verdict.reasons)}")
            raise Replan(f"step {step.step_id} ({step.capability}) failed: {'; '.join(verdict.reasons)}")
        else:  # escalate: only a human can resolve it
            res.status = StepStatus.BLOCKED
            state.errors.append(f"{step.step_id} needs you: {'; '.join(verdict.reasons)}")

    def _replan(self, state: TaskState, why: str) -> bool:
        if state.replans >= state.plan.budget.max_replans:
            return False
        state.replans += 1
        done = {sid: r.output for sid, r in state.results.items() if r.status == StepStatus.DONE}
        try:
            new = self.planner.run(state.envelope, state.context, failure=why, completed=done,
                                   id_prefix=f"r{state.replans}_")
        except PlanError as exc:
            state.errors.append(f"replan failed: {exc}")
            return False
        self._merge_plan(state, new)
        return True

    @staticmethod
    def _merge_plan(state: TaskState, new: Plan) -> None:
        """Keep completed steps (and their outputs); replace everything else with the new plan."""
        if state.plan is None:          # e.g. clarifying a task that stopped before planning
            state.plan = new
            return
        keep = [s for s in state.plan.steps if state.results.get(s.step_id, StepResult(step_id="")).status == StepStatus.DONE]
        state.results = {sid: r for sid, r in state.results.items() if r.status == StepStatus.DONE}
        state.pending_approvals = []
        state.plan = Plan(goal=new.goal, success_criteria=new.success_criteria or state.plan.success_criteria,
                          steps=keep + new.steps, budget=state.plan.budget)

    def _finish(self, state: TaskState) -> TaskState:
        statuses = {r.status for r in state.results.values()}
        all_ids = {s.step_id for s in state.plan.steps}
        if StepStatus.BLOCKED in statuses:
            state.status = (TaskStatus.WAITING_APPROVAL if state.pending_approvals else TaskStatus.NEEDS_INPUT)
        elif all(state.results.get(i) and state.results[i].status == StepStatus.DONE for i in all_ids):
            final = self.verifier.check_task(state.plan.goal, state.plan.success_criteria,
                                             {sid: r.output for sid, r in state.results.items()})
            if final.decision != "pass" and self._replan(state, "final check: " + "; ".join(final.reasons)):
                state.status = TaskStatus.RUNNING
                self._save(state)
                return self.run(state.task_id)
            if final.decision != "pass":
                return self._fail(state, "final check failed: " + "; ".join(final.reasons))
            state.status = TaskStatus.DONE
        else:
            return self._fail(state, "some steps could not run")
        state.report = self.reporter.run(state)
        self._save(state)
        if state.status == TaskStatus.DONE:
            self._learn(state)
        return state

    def _fail(self, state: TaskState, reason: str) -> TaskState:
        logger.error("Task %s failed: %s", state.task_id, reason)
        state.errors.append(reason)
        self._compensate(state)           # saga: roll back reversible side effects
        state.status = TaskStatus.FAILED
        state.report = self.reporter.run(state)
        self.store.put("dead_letter", {"_id": state.task_id, "reason": reason,
                                       "state": state.model_dump(mode="json")})
        self._save(state)
        self._learn(state)
        return state

    def _compensate_step(self, res: StepResult) -> None:
        for eff in reversed(res.effects):
            sp = self.specialists.get(eff.action.kind.split(".")[0])
            if sp and eff.undo:
                try:
                    sp.compensate(eff)
                except Exception:
                    logger.exception("Compensation failed for %s", eff.action.kind)
        res.effects = [e for e in res.effects if not e.undo]   # irreversible ones remain on record

    def _compensate(self, state: TaskState) -> None:
        for res in reversed(list(state.results.values())):
            self._compensate_step(res)

    def _learn(self, state: TaskState) -> None:
        try:
            self.learner.on_episode(state)
        except Exception:   # learning is best-effort; never fail a task because of it
            logger.exception("Learner failed on %s", state.task_id)

    def _meter(self, state: TaskState, calls0: int, cost0: float) -> None:
        state.llm_calls += self.llm.usage.calls - calls0
        state.cost_usd += self.llm.usage.cost_usd - cost0

    @staticmethod
    def _step(state: TaskState, sid: str) -> PlanStep:
        return next(s for s in state.plan.steps if s.step_id == sid)

    def _save(self, state: TaskState) -> None:
        self.store.save_task(state)

    def _load(self, task_id: str) -> TaskState:
        st = self.store.load_task(task_id)
        if st is None:
            raise KeyError(task_id)
        return st
