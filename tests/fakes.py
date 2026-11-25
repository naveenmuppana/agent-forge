"""Offline fakes: a scripted LLM backend routed by which agent is calling, and fake tools."""
from __future__ import annotations

import json
import re
from typing import Callable

from assistant.contracts import Risk
from assistant.llm import ResilientLLM
from assistant.orchestrator import Assistant
from assistant.agents.guardian import Guardian
from assistant.specialists.base import Specialist, Tool
from assistant.state import InMemoryStore

POLICY = {
    "auto_reversible": True,
    "deny": ["*.delete_account"],
    "always_escalate": ["*.pay*"],
    "standing_rules": [{"name": "own domain", "kind": "comms.send_email", "when": {"to_endswith": ["@example.com"]}}],
}

Handler = Callable[[str, str], object]


def j(obj) -> str:
    return json.dumps(obj)


def step(sid, cap, instr, deps=(), risk="read", criteria=()):
    return {"step_id": sid, "capability": cap, "instruction": instr, "depends_on": list(deps),
            "risk": risk, "success_criteria": list(criteria)}


def pass_verdict(score=0.95):
    return {"decision": "pass", "score": score, "reasons": ["ok"], "fix_hint": ""}


class Script:
    """Holds per-agent handlers: fn(user_prompt, call_index) -> dict | str."""

    def __init__(self) -> None:
        self.intake: Handler = lambda u, i: {"intent": "do the thing", "urgency": "normal", "actionable": True}
        self.planner: Handler = lambda u, i: {"goal": "g", "success_criteria": [], "steps": []}
        self.verifier: Handler = lambda u, i: pass_verdict()
        self.reporter: Handler = lambda u, i: "Done."
        self.reflect: Handler = lambda u, i: {"lesson": "Always confirm the recipient before sending."}
        self.specialists: dict[str, Handler] = {}
        self.calls: dict[str, int] = {}
        self.models: list[str] = []

    def backend(self, system: str, user: str, model: str):
        self.models.append(model)
        m = re.search(r"You are the (\w+) specialist", system)
        if m:
            agent = m.group(1)
            fn = self.specialists[agent]
        elif "You triage" in system:
            agent, fn = "intake", self.intake
        elif "planner of a personal" in system:
            agent, fn = "planner", self.planner
        elif "strict reviewer" in system:
            agent, fn = "verifier", self.verifier
        elif "short report" in system:
            agent, fn = "reporter", self.reporter
        elif "ONE reusable lesson" in system:
            agent, fn = "reflect", self.reflect
        else:
            agent, fn = "other", lambda u, i: "revised prompt"
        i = self.calls.get(agent, 0)
        self.calls[agent] = i + 1
        out = fn(user, i)
        return (out if isinstance(out, str) else j(out)), 100, 50


def tool_then_final(tool: str, args: dict, final: str) -> Handler:
    """Specialist that calls one tool, then answers."""
    return lambda u, i: ({"final": final} if "Call " in u else {"tool": tool, "args": args})


def make(script: Script, store: InMemoryStore | None = None, extra: dict | None = None) -> Assistant:
    store = store or InMemoryStore()
    llm = ResilientLLM(script.backend, sleep=lambda s: None)
    sent: list = []

    def send_email(to, subject, body):
        m = {"_id": f"m{len(sent)}", "to": to, "subject": subject, "body": body, "folder": "sent"}
        sent.append(m)
        store.put("mail", m)
        return {"message_id": m["_id"]}

    def save_draft(to, subject, body):
        m = {"_id": f"d{len(store.find('mail'))}", "to": to, "folder": "drafts"}
        store.put("mail", m)
        return {"message_id": m["_id"]}

    specs = {
        "utility": Specialist("utility", "Weather lookups", [
            Tool("get_weather", "weather", {"city": "str"}, lambda city: {"city": city, "rain_prob": 0.8})], store, llm),
        "knowledge": Specialist("knowledge", "Answers from user docs", [
            Tool("search_docs", "rag", {"query": "str"},
                 lambda query: [{"source": "lease.pdf", "text": "Notice period is 60 days."}])], store, llm),
        "comms": Specialist("comms", "Email", [
            Tool("save_draft", "draft", {"to": "str", "subject": "str", "body": "str"}, save_draft, Risk.REVERSIBLE,
                 undo=lambda r: store.delete("mail", r["message_id"]),
                 check=lambda r: bool(store.find("mail", {"_id": r["message_id"]}))),
            Tool("send_email", "send", {"to": "str", "subject": "str", "body": "str"}, send_email, Risk.EXTERNAL,
                 check=lambda r: bool(store.find("mail", {"_id": r["message_id"], "folder": "sent"}))),
        ], store, llm),
        "research": Specialist("research", "Web research", [
            Tool("fetch_url", "get", {"url": "str"}, lambda url: {"text": "page"})], store, llm),
    }
    specs.update(extra or {})
    a = Assistant(llm=llm, store=store, specialists=specs, guardian=Guardian(POLICY))
    a.sent = sent  # type: ignore[attr-defined]
    return a
