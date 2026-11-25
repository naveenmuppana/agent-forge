# Author: Naveen Muppana
"""Built-in specialists.

Utility (weather) and Knowledge (RAG) reuse the existing project code. Calendar,
Comms and Tasks are backed by the assistant's own store as *local adapters*:
swap each tool's `fn`/`undo`/`check` for a Google/Microsoft/Gmail client to go live.
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Any

import requests

from ..contracts import Risk
from ..llm import LLM
from ..state import Store
from .base import Specialist, Tool


def _uid() -> str:
    return uuid.uuid4().hex[:10]


# ---------------- Utility: weather (from weather_agent.py) ----------------
def utility(store: Store, llm: LLM) -> Specialist:
    from weather_agent import get_coordinates, get_weather

    return Specialist(
        name="utility", store=store, llm=llm,
        description="Weather, geocoding and other quick lookups.",
        persona="Always use tools for facts; never guess.",
        tools=[
            Tool("get_coordinates", "Latitude/longitude of a city", {"city": "str"}, get_coordinates),
            Tool("get_weather", "Current weather at coordinates", {"latitude": "float", "longitude": "float"}, get_weather),
        ],
    )


# ---------------- Knowledge: RAG (same index as /rag/query) ----------------
def _search_docs(query: str, k: int = 4) -> list[dict[str, Any]]:
    from langchain_mongodb import MongoDBAtlasVectorSearch

    from core import get_embeddings, get_mongo_collection

    vs = MongoDBAtlasVectorSearch(
        collection=get_mongo_collection(), embedding=get_embeddings(),
        index_name=os.getenv("MONGODB_VECTOR_INDEX", "vector_index"), relevance_score_fn="cosine",
    )
    return [
        {"score": round(s, 4), "source": d.metadata.get("source", "unknown"), "text": d.page_content}
        for d, s in vs.similarity_search_with_score(query, k=k)
    ]


def knowledge(store: Store, llm: LLM) -> Specialist:
    return Specialist(
        name="knowledge", store=store, llm=llm,
        description="Answers questions from the user's uploaded documents.",
        persona="Answer only from retrieved passages and cite the source file. If nothing relevant, say so.",
        tools=[Tool("search_docs", "Semantic search over uploaded docs", {"query": "str", "k": "int"}, _search_docs)],
    )


# ---------------- Research: fetch public pages ----------------
def _fetch_url(url: str) -> dict[str, Any]:
    r = requests.get(url, timeout=15, headers={"User-Agent": "agent-forge-assistant"})
    r.raise_for_status()
    return {"status": r.status_code, "text": r.text[:6000]}


def research(store: Store, llm: LLM) -> Specialist:
    return Specialist(
        name="research", store=store, llm=llm,
        description="Reads public web pages and summarises/compares them.",
        tools=[Tool("fetch_url", "GET a public URL and return its text", {"url": "str"}, _fetch_url)],
    )


# ---------------- Calendar (local adapter) ----------------
def calendar(store: Store, llm: LLM) -> Specialist:
    def list_events(date: str = "") -> list[dict]:
        evs = store.find("calendar")
        return [e for e in evs if not date or e["start"].startswith(date)]

    def create_hold(title: str, start: str, end: str, attendees: list[str] | None = None) -> dict:
        ev = {"_id": _uid(), "title": title, "start": start, "end": end,
              "attendees": attendees or [], "status": "tentative"}
        store.put("calendar", ev)
        return {"event_id": ev["_id"]}

    def confirm_event(event_id: str) -> dict:
        ev = store.find("calendar", {"_id": event_id})
        if not ev:
            raise ValueError(f"no event {event_id}")
        ev[0]["status"] = "confirmed"
        store.put("calendar", ev[0])
        return {"event_id": event_id, "invites_sent_to": ev[0]["attendees"]}

    return Specialist(
        name="calendar", store=store, llm=llm,
        description="Reads the calendar, places tentative holds, confirms events (sends invites).",
        persona="Respect the user's preferred meeting hours. Check for conflicts before holding.",
        tools=[
            Tool("list_events", "Events, optionally for a YYYY-MM-DD date", {"date": "str"}, list_events),
            Tool("create_hold", "Create a tentative event", {"title": "str", "start": "ISO", "end": "ISO", "attendees": "list[str]"},
                 create_hold, Risk.REVERSIBLE,
                 undo=lambda r: store.delete("calendar", r["event_id"]),
                 check=lambda r: bool(store.find("calendar", {"_id": r["event_id"]}))),
            Tool("confirm_event", "Confirm a hold and send invites", {"event_id": "str"},
                 confirm_event, Risk.EXTERNAL,
                 check=lambda r: bool(store.find("calendar", {"_id": r["event_id"], "status": "confirmed"}))),
        ],
    )


# ---------------- Comms (local outbox adapter) ----------------
def comms(store: Store, llm: LLM) -> Specialist:
    def save_draft(to: str, subject: str, body: str) -> dict:
        d = {"_id": _uid(), "to": to, "subject": subject, "body": body, "folder": "drafts"}
        store.put("mail", d)
        return {"message_id": d["_id"]}

    def send_email(to: str, subject: str, body: str) -> dict:
        m = {"_id": _uid(), "to": to, "subject": subject, "body": body, "folder": "sent",
             "sent_at": datetime.now(timezone.utc).isoformat()}
        store.put("mail", m)
        return {"message_id": m["_id"]}

    return Specialist(
        name="comms", store=store, llm=llm,
        description="Drafts and sends emails/messages in the user's voice.",
        persona="Match the user's tone and signature from preferences. Prefer drafts unless told to send.",
        tools=[
            Tool("save_draft", "Save an email draft", {"to": "str", "subject": "str", "body": "str"},
                 save_draft, Risk.REVERSIBLE, undo=lambda r: store.delete("mail", r["message_id"]),
                 check=lambda r: bool(store.find("mail", {"_id": r["message_id"]}))),
            Tool("send_email", "Send an email now", {"to": "str", "subject": "str", "body": "str"},
                 send_email, Risk.EXTERNAL,
                 check=lambda r: bool(store.find("mail", {"_id": r["message_id"], "folder": "sent"}))),
        ],
    )


# ---------------- Tasks / notes ----------------
def tasks(store: Store, llm: LLM) -> Specialist:
    def add_task(title: str, due: str = "") -> dict:
        t = {"_id": _uid(), "title": title, "due": due, "done": False}
        store.put("todos", t)
        return {"task_id": t["_id"]}

    return Specialist(
        name="tasks", store=store, llm=llm,
        description="Manages the user's to-do list and reminders.",
        tools=[
            Tool("list_tasks", "List open to-dos", {}, lambda: store.find("todos", {"done": False})),
            Tool("add_task", "Add a to-do", {"title": "str", "due": "ISO date"}, add_task, Risk.REVERSIBLE,
                 undo=lambda r: store.delete("todos", r["task_id"]),
                 check=lambda r: bool(store.find("todos", {"_id": r["task_id"]}))),
        ],
    )


FACTORIES = {"utility": utility, "knowledge": knowledge, "research": research,
             "calendar": calendar, "comms": comms, "tasks": tasks}


def build_all(store: Store, llm: LLM) -> dict[str, Specialist]:
    return {name: f(store, llm) for name, f in FACTORIES.items()}
