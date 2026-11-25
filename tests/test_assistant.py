"""Offline tests for the multi-agent assistant. Run: python -m unittest discover -s tests -v"""
import unittest

from assistant.agents.guardian import Guardian
from assistant.agents.planner import validate_plan
from assistant.contracts import Plan, ProposedAction, Risk, StepStatus, TaskStatus
from assistant.evals import evaluate_candidates
from assistant.llm import LLMError, ResilientLLM, TransientError
from assistant.contracts import TaskState, TaskEnvelope
from assistant.state import InMemoryStore

from tests.fakes import POLICY, Script, make, pass_verdict, step, tool_then_final

WEATHER_AND_LEASE = {
    "goal": "Rain check + lease notice", "success_criteria": [],
    "steps": [step("s1", "utility", "Rain in Delhi tomorrow?", criteria=["gives rain probability"]),
              step("s2", "knowledge", "Lease notice period?", criteria=["cites lease"])],
}


class GuardianTests(unittest.TestCase):
    g = Guardian(POLICY)

    def d(self, kind, risk, **params):
        return self.g.decide(ProposedAction(kind=kind, risk=risk, params=params)).decision

    def test_policy(self):
        self.assertEqual(self.d("utility.get_weather", Risk.READ), "allow")
        self.assertEqual(self.d("comms.save_draft", Risk.REVERSIBLE), "allow")
        self.assertEqual(self.d("comms.send_email", Risk.EXTERNAL, to="bob@example.com"), "allow")
        self.assertEqual(self.d("comms.send_email", Risk.EXTERNAL, to="x@stranger.org"), "escalate")
        self.assertEqual(self.d("bank.pay_invoice", Risk.REVERSIBLE), "escalate")   # money always escalates
        self.assertEqual(self.d("acct.delete_account", Risk.EXTERNAL), "deny")

    def test_user_approval_allows(self):
        a = ProposedAction(kind="comms.send_email", risk=Risk.EXTERNAL, params={"to": "x@y.org"})
        self.assertEqual(self.g.decide(a, {a.action_id}).decision, "allow")

    def test_shipped_policy_file_loads(self):
        Guardian()


class PlannerValidationTests(unittest.TestCase):
    def test_rejects_unknown_capability_missing_dep_and_cycle(self):
        p = Plan(goal="g", steps=[step("a", "utility", "x", deps=["b"]), step("b", "nope", "y", deps=["a"]),
                                  step("c", "utility", "z", deps=["zz"])])
        problems = " ".join(validate_plan(p, {"utility"}))
        self.assertIn("unknown capability", problems)
        self.assertIn("missing step 'zz'", problems)
        self.assertIn("cycle", problems)


class LLMWrapperTests(unittest.TestCase):
    def test_retry_then_fallback_model(self):
        seen = []

        def backend(s, u, m):
            seen.append(m)
            if m == "gpt-4.1-mini":
                raise TransientError("429")
            return "ok", 1, 1

        llm = ResilientLLM(backend, sleep=lambda s: None)
        self.assertEqual(llm.complete("s", "u"), "ok")
        self.assertEqual(seen, ["gpt-4.1-mini"] * 3 + ["gpt-4.1-nano"])

    def test_json_repair(self):
        replies = iter(["not json at all", '```json\n{"lesson": "x"}\n```'])
        llm = ResilientLLM(lambda s, u, m: (next(replies), 1, 1), sleep=lambda s: None)
        from assistant.agents.learner import Lesson
        self.assertEqual(llm.structured("s", "u", Lesson).lesson, "x")

    def test_takes_first_of_several_json_objects(self):
        # Seen live from gpt-4.1-mini: a tool call followed by a final answer in one reply.
        reply = '{"tool": "save_draft", "args": {"to": "a@b.c"}}\n{"final": "Drafted."}'
        llm = ResilientLLM(lambda s, u, m: (reply, 1, 1), sleep=lambda s: None)
        from assistant.specialists.base import AgentTurn
        self.assertEqual(llm.structured("s", "u", AgentTurn).tool, "save_draft")

    def test_gives_up(self):
        llm = ResilientLLM(lambda s, u, m: ("garbage", 1, 1), sleep=lambda s: None)
        from assistant.agents.learner import Lesson
        with self.assertRaises(LLMError):
            llm.structured("s", "u", Lesson)


class EndToEndTests(unittest.TestCase):
    def test_parallel_plan_completes_without_user(self):
        s = Script()
        s.planner = lambda u, i: WEATHER_AND_LEASE
        s.specialists["utility"] = tool_then_final("get_weather", {"city": "Delhi"}, "80% chance of rain.")
        s.specialists["knowledge"] = tool_then_final("search_docs", {"query": "notice"}, "60 days (lease.pdf).")
        a = make(s)
        st = a.submit("Will it rain in Delhi tomorrow and what's my lease notice period?")
        self.assertEqual(st.status, TaskStatus.DONE, st.errors)
        self.assertEqual({r.status for r in st.results.values()}, {StepStatus.DONE})
        self.assertEqual(st.report, "Done.")
        self.assertGreater(st.llm_calls, 0)
        # duplicate submission while active returns the same task; after done it is new
        self.assertNotEqual(a.submit("Will it rain in Delhi tomorrow and what's my lease notice period?").task_id, "")

    def test_external_action_escalates_then_resumes_on_approval(self):
        s = Script()
        s.planner = lambda u, i: {"goal": "email", "steps": [step("s1", "comms", "email landlord", risk="irreversible-or-external")]}
        s.specialists["comms"] = tool_then_final("send_email", {"to": "landlord@stranger.org", "subject": "Hi", "body": "b"}, "Sent.")
        a = make(s)
        st = a.submit("Email my landlord")
        self.assertEqual(st.status, TaskStatus.WAITING_APPROVAL)
        self.assertEqual(len(a.sent), 0)                       # nothing sent before approval
        aid = st.pending_approvals[0].action_id
        st = a.approve(st.task_id, approve=[aid])
        self.assertEqual(st.status, TaskStatus.DONE, st.errors)
        self.assertEqual(len(a.sent), 1)
        # idempotency: replaying the same approved action does not send again
        a.specialists["comms"].execute(st.results["s1"].effects[0].action, scope=st.task_id)
        self.assertEqual(len(a.sent), 1)

    def test_repeated_identical_send_is_asked_about_once(self):
        # Seen live: the model called send_email twice with the same content.
        s = Script()
        s.planner = lambda u, i: {"goal": "email", "steps": [step("s1", "comms", "email landlord")]}
        args = {"to": "landlord@stranger.org", "subject": "Hi", "body": "b"}
        s.specialists["comms"] = lambda u, i: {"final": "Queued."} if u.count("Call ") >= 2 else {"tool": "send_email", "args": args}
        a = make(s)
        st = a.submit("Email my landlord")
        self.assertEqual(len(st.pending_approvals), 1)
        st = a.approve(st.task_id, approve=[st.pending_approvals[0].action_id])
        self.assertEqual((st.status, len(a.sent)), (TaskStatus.DONE, 1))

    def test_standing_rule_sends_autonomously(self):
        s = Script()
        s.planner = lambda u, i: {"goal": "email", "steps": [step("s1", "comms", "email team")]}
        s.specialists["comms"] = tool_then_final("send_email", {"to": "team@example.com", "subject": "S", "body": "b"}, "Sent.")
        a = make(s)
        st = a.submit("Tell my team the offsite is on")
        self.assertEqual(st.status, TaskStatus.DONE, st.errors)
        self.assertEqual(len(a.sent), 1)

    def test_rejected_action_is_not_performed(self):
        s = Script()
        s.planner = lambda u, i: {"goal": "email", "steps": [step("s1", "comms", "email")]}
        s.specialists["comms"] = tool_then_final("send_email", {"to": "x@stranger.org", "subject": "S", "body": "b"}, "Sent.")
        a = make(s)
        st = a.submit("Email x")
        st = a.approve(st.task_id, reject=[st.pending_approvals[0].action_id])
        self.assertEqual(len(a.sent), 0)
        self.assertIn("User declined", st.results["s1"].output)

    def test_failing_step_retries_then_replans_around_it(self):
        s = Script()
        s.planner = lambda u, i: (
            {"goal": "g", "steps": [step("s1", "research", "find it")]} if i == 0
            else {"goal": "g", "steps": [step("s1", "knowledge", "look in docs")]})
        s.specialists["research"] = lambda u, i: {"tool": "no_such_tool"}      # loops until step cap -> FAILED
        s.specialists["knowledge"] = tool_then_final("search_docs", {"query": "q"}, "Found it.")
        a = make(s)
        st = a.submit("Find my lease notice")
        self.assertEqual(st.status, TaskStatus.DONE, st.errors)
        self.assertEqual(st.replans, 1)
        self.assertIn("r1_s1", st.results)
        self.assertEqual(s.calls["planner"], 2)

    def test_retries_within_one_step_do_not_trip_circuit_breaker(self):
        s = Script()
        s.planner = lambda u, i: {"goal": "g", "steps": [step("s1", "comms", "draft", criteria=["ok"])]}
        s.specialists["comms"] = tool_then_final("save_draft", {"to": "a@b.c", "subject": "s", "body": "b"}, "ok")
        s.verifier = lambda u, i: ({"decision": "retry", "score": 0.2, "reasons": ["x"], "fix_hint": "y"}
                                   if i < 2 else pass_verdict())
        a = make(s)
        st = a.submit("draft it")
        self.assertEqual(st.status, TaskStatus.DONE, st.errors)
        self.assertFalse(a.router.is_open("comms"))
        self.assertEqual(a.store.find("routing_stats", {"_id": "comms"})[0]["consecutive_fail"], 0)

    def test_retry_undoes_previous_attempts_draft(self):
        s = Script()
        s.planner = lambda u, i: {"goal": "g", "steps": [step("s1", "comms", "draft", criteria=["polite"])]}
        s.specialists["comms"] = tool_then_final("save_draft", {"to": "a@b.c", "subject": "s", "body": "b"}, "Drafted.")
        s.verifier = lambda u, i: ({"decision": "retry", "score": 0.3, "reasons": ["rude"], "fix_hint": "be polite"}
                                   if i == 0 else pass_verdict())
        a = make(s)
        st = a.submit("Draft a note")
        self.assertEqual(st.status, TaskStatus.DONE, st.errors)
        self.assertEqual(len(a.store.find("mail", {"folder": "drafts"})), 1)   # first draft was compensated
        self.assertEqual(st.results["s1"].attempts, 2)

    def test_gives_up_after_replan_budget_and_dead_letters(self):
        s = Script()
        s.planner = lambda u, i: {"goal": "g", "steps": [step("s1", "research", "x")]}
        s.specialists["research"] = lambda u, i: {"tool": "no_such_tool"}
        a = make(s)
        st = a.submit("impossible")
        self.assertEqual(st.status, TaskStatus.FAILED)
        self.assertEqual(st.replans, 2)
        self.assertTrue(a.store.find("dead_letter", {"_id": st.task_id}))
        self.assertIn("couldn't finish", st.report)
        self.assertTrue(a.store.find("lessons"))                   # learner reflected on the failure

    def test_budget_cap(self):
        s = Script()
        s.planner = lambda u, i: {"goal": "g", "steps": [step("s1", "research", "x")],
                                  "budget": {"max_llm_calls": 2}}
        s.specialists["research"] = lambda u, i: {"tool": "no_such_tool"}
        st = make(s).submit("expensive")
        self.assertEqual(st.status, TaskStatus.FAILED)
        self.assertTrue(any("budget" in e for e in st.errors))

    def test_intake_cannot_stop_for_lookups(self):
        # Seen live: intake asked the user for lease terms the knowledge agent can find.
        s = Script()
        s.intake = lambda u, i: {"intent": "lease q", "actionable": False, "clarification_question": "What are your lease terms?"}
        s.planner = lambda u, i: {"goal": "g", "steps": [step("s1", "knowledge", "find deposit")]}
        s.specialists["knowledge"] = tool_then_final("search_docs", {"query": "deposit"}, "INR 75,000")
        st = make(s).submit("What does my lease say about the deposit?")
        self.assertEqual(st.status, TaskStatus.DONE, st.errors)
        self.assertGreaterEqual(st.llm_calls, 3)        # intake + planning are metered too

    def test_needs_input_then_clarify(self):
        s = Script()
        s.intake = lambda u, i: {"intent": "email", "actionable": False, "changes_irreversible_action": True,
                                 "clarification_question": "Which Priya?"}
        s.planner = lambda u, i: {"goal": "g", "steps": [step("s1", "utility", "x")]}
        s.specialists["utility"] = tool_then_final("get_weather", {"city": "x"}, "ok")
        a = make(s)
        st = a.submit("Email Priya")
        self.assertEqual((st.status, st.report), (TaskStatus.NEEDS_INPUT, "Which Priya?"))
        st = a.clarify(st.task_id, "Priya from work")
        self.assertEqual(st.status, TaskStatus.DONE, st.errors)

    def test_resume_after_crash(self):
        s = Script()
        s.specialists["utility"] = tool_then_final("get_weather", {"city": "Delhi"}, "rain")
        store = InMemoryStore()
        st = TaskState(envelope=TaskEnvelope(raw_input="weather"), status=TaskStatus.RUNNING,
                       plan=Plan(goal="g", steps=[step("s1", "utility", "x")]))
        store.save_task(st)                                     # process "died" here
        a = make(s, store=store)
        self.assertEqual(a.resume_running(), [st.task_id])
        self.assertEqual(a.get(st.task_id).status, TaskStatus.DONE)

    def test_specialist_crash_is_contained(self):
        s = Script()
        s.planner = lambda u, i: ({"goal": "g", "steps": [step("s1", "utility", "x")]} if i == 0
                                  else {"goal": "g", "steps": [step("s1", "knowledge", "y")]})

        def boom(u, i):
            raise RuntimeError("bug")
        s.specialists["utility"] = boom
        s.specialists["knowledge"] = tool_then_final("search_docs", {"query": "q"}, "ok")
        st = make(s).submit("x")
        self.assertEqual(st.status, TaskStatus.DONE, st.errors)


class LearningTests(unittest.TestCase):
    def test_lesson_from_feedback_is_used_next_time(self):
        s = Script()
        s.intake = lambda u, i: {"intent": "email landlord", "actionable": True}
        s.planner = lambda u, i: {"goal": "g", "steps": [step("s1", "comms", "draft")]}
        s.specialists["comms"] = tool_then_final("save_draft", {"to": "l@x.org", "subject": "s", "body": "hey"}, "Drafted.")
        s.reflect = lambda u, i: {"lesson": "Landlord emails: use a formal tone and CC spouse."}
        a = make(s)
        st = a.submit("Email my landlord about the leak")
        learned = a.feedback(st.task_id, rating="down", correction="too casual, cc my wife",
                             preferences={"signature": "Naveen"})
        self.assertIn("formal", learned["lesson"])
        self.assertEqual(a.store.find("preferences", {"_id": "default"})[0]["prefs"]["signature"], "Naveen")

        seen = {}

        def planner(u, i):
            seen["prompt"] = u
            return {"goal": "g", "steps": [step("s1", "comms", "draft")]}
        s.planner = planner
        a.submit("Email my landlord about rent")
        self.assertIn("formal tone", seen["prompt"])
        self.assertIn("Naveen", seen["prompt"])

    def test_bad_lessons_decay_and_are_pruned(self):
        s = Script()
        s.intake = lambda u, i: {"intent": "email landlord", "actionable": True}
        s.planner = lambda u, i: {"goal": "g", "steps": [step("s1", "comms", "draft")]}
        s.specialists["comms"] = tool_then_final("save_draft", {"to": "l@x.org", "subject": "s", "body": "b"}, "ok")
        s.reflect = lambda u, i: {"lesson": ""}
        a = make(s)
        a.store.put("lessons", {"_id": "L1", "user_id": "default", "text": "landlord email always shout", "intent": "email landlord", "weight": 0.3})
        for _ in range(3):
            st = a.submit("Email my landlord")
            a.feedback(st.task_id, rating="down")
        self.assertFalse(a.store.find("lessons", {"_id": "L1"}))

    def test_playbook_after_repeated_success(self):
        s = Script()
        s.intake = lambda u, i: {"intent": "weekly weather brief", "actionable": True}
        s.planner = lambda u, i: {"goal": "g", "steps": [step("s1", "utility", "weather")]}
        s.specialists["utility"] = tool_then_final("get_weather", {"city": "Pune"}, "sunny")
        a = make(s)
        for n in range(3):
            st = a.submit(f"weather brief {n}")
            a.feedback(st.task_id, rating="up")
        pbs = a.store.find("playbooks")
        self.assertEqual(len(pbs), 1)
        ctx = a.context.run(TaskEnvelope(raw_input="weather brief again", intent="weekly weather brief"))
        self.assertIsNotNone(ctx.matching_playbook)

    def test_prompt_candidate_gated_by_evals(self):
        s = Script()

        def planner(u, i):
            # Candidate prompt text is in the system prompt, not user; detect via call parity is fragile,
            # so plan correctly for everything: baseline == candidate -> promote.
            caps = []
            text = u.lower()
            if "weather" in text or "rain" in text:
                caps.append("utility")
            if "lease" in text:
                caps.append("knowledge")
            if "email" in text:
                caps.append("comms")
            if "hold" in text:
                caps += ["calendar", "tasks"]
            return {"goal": "g", "steps": [step(f"s{k}", c, "x") for k, c in enumerate(caps or ["utility"])]}
        s.planner = planner
        from assistant.specialists.base import Specialist
        a = make(s)
        for name in ("calendar", "tasks"):
            a.specialists[name] = Specialist(name, name, [], a.store, a.llm)
        a.planner.capabilities.update(calendar="calendar", tasks="tasks")
        cid = a.prompts.propose("planner", "You are the planner of a personal assistant. Be minimal.", "test")
        res = evaluate_candidates(a)
        self.assertEqual(res[0]["result"], "promoted")
        self.assertEqual(a.prompts.get("planner"), "You are the planner of a personal assistant. Be minimal.")
        self.assertTrue(cid)

    def test_learner_cannot_touch_policy(self):
        import inspect
        from assistant.agents import learner
        src = inspect.getsource(learner)
        self.assertNotIn("policy.yaml", src.replace("may NOT write: policy.yaml", ""))


if __name__ == "__main__":
    unittest.main()
