# Personal AI Multi-Agent Assistant — Architecture

## Context
`agent-forge` today has three separate capabilities with no coordination between them: a direct LLM chat ([chat_client.py](../chat_client.py)), a tool-calling weather agent ([weather_agent.py](../weather_agent.py)), and RAG over MongoDB Atlas ([agent_service.py](../agent_service.py)). Shared clients live in [core.py](../core.py).
The goal is a **personal assistant made of cooperating agents**. You give it a goal once (or a trigger fires) and it plans, acts, checks its work, recovers from failures, and reports back, without you stepping in. It also gets better with every task it runs.

Stack stays the same: OpenAI Responses API (`gpt-4.1-mini`, plus a stronger model for planning and checking), MongoDB Atlas for state, memory and vectors, and FastAPI for the entry point.

> Existing bug found: `agent_service.py:15-16` imports `basic_openai_weather` and `openai_mcp_weather`, but neither module exists. The real functions are `chat_client.get_basic_llm_response` and `weather_agent.run_agent`. This gets fixed in step 0.

---

## 1. The pipeline at a glance

```
Trigger ─► [1 Intake] ─► [2 Context/Memory] ─► [3 Planner] ─► ┌─────────── per step ───────────┐
                                                              │ [4 Router] ─► [5 Specialist]    │
                                                              │       ▲            │            │
                                                              │       │      [6 Guardian] ──(blocked)──► escalate
                                                              │       │            │            │
                                                              │   retry/replan ◄ [7 Verifier]   │
                                                              └────────────────────┬────────────┘
                                                                                   ▼
                                                          [8 Reporter] ─► user    [9 Learner] (async, after)
```

Every agent reads and writes one shared, checkpointed **TaskState** document in Mongo, so a crash at any point resumes from the last completed step.

## 2. Shared contracts (Pydantic, in `assistant/contracts.py`)

| Object | Key fields |
|---|---|
| `TaskEnvelope` | `task_id`, `source` (chat/email/cron/webhook), `raw_input`, `user_id`, `deadline`, `autonomy_level` |
| `ContextPack` | `preferences`, `relevant_memories[]`, `relevant_docs[]`, `lessons[]` (from Learner), `matching_playbook?` |
| `Plan` | `goal`, `success_criteria[]`, `steps[]` (DAG: `step_id`, `capability`, `inputs`, `depends_on`, `risk`), `budget` (tokens/$/time) |
| `StepResult` | `step_id`, `status`, `output`, `side_effects[]` (with idempotency key + undo handle), `cost`, `attempts` |
| `Verdict` | `pass` / `retry` / `replan` / `escalate`, `score 0-1`, `reasons[]`, `fix_hint` |
| `Episode` | full trace: envelope, plan, results, verdicts, errors, final outcome, user feedback |

## 3. Agent by agent

### 1. Intake Agent
- **Receives:** raw trigger (chat message, forwarded email, cron tick, webhook).
- **Decides:** the intent, whether the request is actionable or ambiguous, its urgency, and whether it duplicates a task already running (via dedupe hash). If it is ambiguous, it first tries to resolve it from memory. It asks you only if the ambiguity would change an irreversible action.
- **Passes on:** a `TaskEnvelope` sent to Context/Memory.

### 2. Context / Memory Agent
- **Receives:** a `TaskEnvelope`.
- **Decides:** what you already know that matters here. It runs RAG over your docs (reusing the existing Atlas vector search), episodic memory (past similar tasks), your preference profile ("prefers morning meetings", "signs emails as Naveen"), the Learner's lessons, and any stored **playbook** that matches this kind of task.
- **Passes on:** `TaskEnvelope + ContextPack` sent to the Planner.

### 3. Planner (Orchestrator)
- **Receives:** the envelope and the context pack.
- **Decides:** how to split the goal into a DAG of steps, which capability handles each step, the success criteria (testable statements), a risk level per step (`read` / `reversible-write` / `irreversible-or-external`), and the budget. If a playbook matches, it adapts that playbook instead of planning from scratch.
- **Passes on:** a `Plan` sent to the Router. The plan is saved before any step runs.

### 4. Router
- **Receives:** the next steps whose dependencies are met.
- **Decides:** which specialist and which model handle each step, based on capability and the specialist's live reliability and cost stats. It runs independent steps in parallel.
- **Passes on:** a step request sent to a Specialist.

### 5. Specialists (tool-using workers, each a small loop like `weather_agent.run_agent`)
| Specialist | Tools | Example |
|---|---|---|
| Research | web search, fetch, summarise | "compare 3 flights" |
| Knowledge | existing RAG upload/query | "what did my lease say about notice?" |
| Calendar | read/create/move events | "book 30 min with X next week" |
| Comms | draft/send email & messages | "reply to the landlord" |
| Tasks/Notes | todo list, notes | "remind me Friday" |
| Utility | weather (existing), maps, time zones | "will it rain at the venue?" |

- **Receives:** a step and its slice of the context.
- **Decides:** which tool calls to make, with a capped step count (the existing `MAX_STEPS` pattern).
- **Passes on:** a *proposed* `StepResult`. A write is not executed yet; it is a proposed action that goes to the Guardian.

### 6. Guardian (policy and safety gate)
- **Receives:** every proposed side effect.
- **Decides:** whether to allow, block, or escalate, using your **autonomy policy**:
  - `read` actions are always allowed.
  - `reversible-write` actions (a draft, a tentative calendar hold) are allowed automatically.
  - `irreversible-or-external` actions (sending an email, paying, deleting) are allowed only if a **standing rule** you set covers them (for example, "may auto-reply to my calendar invites" or "may send emails to contacts tagged *family*"). Otherwise they are escalated with a one-tap approve.
  - Money moves, credentials and permission changes are always escalated. The Learner can never widen these rules.
- **Passes on:** an allowed action goes to execution, with an idempotency key and an undo handle recorded. A blocked action goes to the Reporter or escalation queue.

"Without me touching it" holds for everything your policy pre-authorises. The Guardian is the one place where you decide what that covers.

### 7. Verifier (critic)
- **Receives:** the `StepResult` and the plan's success criteria. At the end, it receives the whole task output.
- **Decides:** whether the step met its criteria, using deterministic checks first (schema valid, event actually exists on the calendar, email actually in Sent) and an LLM judge second. It outputs a `Verdict`:
  - `pass` goes to the next step.
  - `retry` (with a `fix_hint`) goes back to the same specialist.
  - `replan` goes back to the Planner with the failure attached.
  - `escalate` goes to you.
- **Passes on:** a `Verdict` sent to the Router or Planner. On the final pass, it goes to the Reporter.

### 8. Reporter
- **Receives:** the final TaskState.
- **Decides:** how much to tell you, based on your verbosity preference. It sends a short summary, links to anything it created, any actions it took on its own authority, and undo links.
- **Passes on:** a message to you (chat, push, or email digest), then closes the `Episode` and sends it to the Learner.

### 9. Learner (asynchronous, off the critical path)
Described in §5.

## 4. Error handling

**Error taxonomy, and how each type is handled automatically:**

| Class | Example | Handling |
|---|---|---|
| Transient | 429, timeout, Mongo blip | Exponential backoff with jitter, 3 tries. Then a **fallback model or provider**. |
| Malformed model output | Bad JSON or tool arguments (as at `weather_agent.py:128`) | Validate with Pydantic, then a "repair" re-prompt containing the validation error, at most 2 times. |
| Tool failure | API 4xx, city not found | Returned to the specialist as an observation so it can choose an alternative. If it fails repeatedly, the **circuit breaker** marks the tool unhealthy and the Router avoids it. |
| Verification failure | Output doesn't meet criteria | `retry` with a `fix_hint`, then `replan`. A **replan budget of 2** stops loops. |
| Budget/loop | Step or token or $ cap hit | Hard stop, partial result, escalate. |
| Policy block | Action not covered by a rule | Park the task and ask you once. It resumes automatically when you approve. |
| Crash mid-task | Process dies | Resume from the TaskState checkpoint. Idempotency keys prevent duplicate sends or bookings. |
| Partial side effects | Booked flight, hotel failed | **Saga compensation:** run the recorded undo handles in reverse order, then replan or escalate. |

Tasks that are still unresolved after all of this go to a **dead-letter queue** with the full trace, and you get one clear message about each.

## 5. Self-improvement loop

Every `Episode` is scored with these signals: Verifier score, retries and replans used, cost and latency, whether you **edited, undid or complained**, and explicit 👍/👎. The Learner turns those scores into five kinds of improvement:

1. **Lessons (reflection memory):** after a failure or user correction, an LLM writes a short, general rule, for example "Landlord emails: user prefers formal tone and CC's spouse". The lesson is embedded and stored, and the Context agent retrieves it next time. Lessons that keep proving wrong get their weight reduced, then pruned.
2. **Playbooks:** plans that scored high and recur (≥3 similar successes) are distilled into reusable templates. Later tasks of that kind are planned faster and cheaper, with fewer errors.
3. **Preference learning:** consistent edits you make, such as always shortening drafts or moving meetings to mornings, update your preference profile.
4. **Routing stats:** success rate, latency and cost per (specialist, tool, model) feed the Router and the circuit breakers. For example, it can use the cheaper model where it already passes.
5. **Prompt evolution (gated):** the Learner proposes changes to agent prompts. These run against an **eval set** built from past episodes, failures included. A change is promoted only if it doesn't regress, and it is versioned so it can be rolled back.

**Guardrails:** the Learner can change prompts, memories, playbooks and routing. It can never change the Guardian policy or permissions. A weekly digest shows you what it learned, and you can delete any lesson.

## 6. Build plan (files)

```
assistant/
  contracts.py        # Pydantic models from §2
  state.py            # TaskState checkpoint store (Mongo, reuses core.get_mongo_collection)
  llm.py              # call wrapper: retry/backoff, fallback model, JSON-repair, cost tracking
  agents/intake.py  context.py  planner.py  router.py  guardian.py  verifier.py  reporter.py  learner.py
  specialists/base.py # generic tool loop extracted from weather_agent.run_agent
  specialists/utility.py (weather)  knowledge.py (RAG)  research.py  comms.py  calendar.py  tasks.py
  orchestrator.py     # the state machine wiring §1
  policy.yaml         # your autonomy rules
evals/                # episode-derived eval cases + runner
```
- Step 0: fix the broken imports in `agent_service.py`.
- Step 1: contracts, state store, LLM wrapper.
- Step 2: generalise `weather_agent.run_agent` into `specialists/base.py`, then port weather and RAG as the first two specialists.
- Step 3: the Intake, Context, Planner, Router, Verifier and Reporter loop, end to end, with read-only specialists.
- Step 4: the Guardian, plus Comms/Calendar specialists using drafts and holds first.
- Step 5: the Learner: episode logging, lessons, playbooks, eval gate.
- Add `POST /assistant/task` and `GET /assistant/task/{id}` to `agent_service.py`.
- Reuse `core.get_openai_client`, `core.get_mongo_collection` and `core.get_embeddings`, plus the existing RAG and vector-store setup.

## 7. Verification
- Unit: contracts validate, Guardian decisions match `policy.yaml` cases, and the repair loop fixes malformed JSON.
- Fault injection: force a 429, a tool exception, a malformed output and a mid-task crash, then confirm retry, fallback, resume and no duplicate side effects.
- End to end: `uvicorn agent_service:app`, then POST "Will it rain in Delhi tomorrow, and what does my uploaded lease say about notice period?". Confirm a 2-step parallel plan, both verified, one report.
- Learning: run a task, give 👎 with a correction, rerun a similar task, and confirm the lesson is retrieved and the behaviour changes.
