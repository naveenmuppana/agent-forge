# Author: Naveen Muppana
"""2. Context/Memory: envelope -> ContextPack (preferences, lessons, similar episodes, playbook).

Ranking uses token overlap so it works offline; set `embed` to an embedding
function (e.g. core.get_embeddings().embed_query) for semantic ranking.
"""
from __future__ import annotations

import math
import re
from typing import Callable, Optional

from ..contracts import ContextPack, TaskEnvelope
from ..state import Store

_WORD = re.compile(r"[a-z0-9]+")
STOP = {"the", "a", "an", "to", "of", "and", "or", "for", "in", "on", "my", "me", "i", "is", "it", "with", "at", "what"}


def tokens(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in STOP}


def similarity(a: str, b: str, embed: Optional[Callable[[str], list[float]]] = None) -> float:
    if embed:
        va, vb = embed(a), embed(b)
        dot = sum(x * y for x, y in zip(va, vb))
        return dot / (math.sqrt(sum(x * x for x in va)) * math.sqrt(sum(y * y for y in vb)) or 1)
    ta, tb = tokens(a), tokens(b)
    return len(ta & tb) / (len(ta | tb) or 1)


class ContextAgent:
    def __init__(self, store: Store, embed: Optional[Callable[[str], list[float]]] = None,
                 max_lessons: int = 5, min_sim: float = 0.08) -> None:
        self.store, self.embed, self.max_lessons, self.min_sim = store, embed, max_lessons, min_sim

    def run(self, env: TaskEnvelope) -> ContextPack:
        query = f"{env.intent} {env.raw_input}"
        prefs = self.store.find("preferences", {"_id": env.user_id})
        lessons = [l for l in self.store.find("lessons", {"user_id": env.user_id}) if l.get("weight", 1) >= 0.2]
        lessons.sort(key=lambda l: similarity(query, l["text"] + " " + l.get("intent", ""), self.embed) * l.get("weight", 1),
                     reverse=True)
        top_lessons = [l["text"] for l in lessons[: self.max_lessons]
                       if similarity(query, l["text"] + " " + l.get("intent", ""), self.embed) >= self.min_sim]

        episodes = self.store.find("episodes", {"user_id": env.user_id})
        episodes.sort(key=lambda e: similarity(query, e["raw_input"], self.embed), reverse=True)
        memories = [f"{e['raw_input']} -> {e['outcome']} ({e['summary'][:160]})" for e in episodes[:3]
                    if similarity(query, e["raw_input"], self.embed) >= 0.2]

        playbook = None
        best = 0.0
        for pb in self.store.find("playbooks", {"user_id": env.user_id}):
            s = similarity(env.intent or env.raw_input, pb["intent"], self.embed)
            if s > best and s >= 0.5:
                best, playbook = s, pb

        if top_lessons:
            for l in lessons:
                if l["text"] in top_lessons:      # track usage so the Learner can judge lessons
                    l["times_used"] = l.get("times_used", 0) + 1
                    self.store.put("lessons", l)

        return ContextPack(
            preferences=(prefs[0].get("prefs", {}) if prefs else {}),
            relevant_memories=memories,
            lessons=top_lessons,
            matching_playbook=({"intent": playbook["intent"], "steps": playbook["steps"]} if playbook else None),
        )
