# Author: Naveen Muppana
"""LLM call wrapper: retry with backoff, model fallback, JSON repair, usage tracking.

Agents depend on the `LLM` protocol, so tests can pass a scripted fake.
"""
from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol, Type, TypeVar

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

PRIMARY_MODEL = "gpt-4.1-mini"
STRONG_MODEL = "gpt-4.1"          # planner / verifier
FALLBACK_MODEL = "gpt-4.1-nano"
# Rough $ per 1M tokens (input, output) for budget enforcement only.
PRICES = {"gpt-4.1": (2.0, 8.0), "gpt-4.1-mini": (0.4, 1.6), "gpt-4.1-nano": (0.1, 0.4)}


class LLMError(Exception):
    """Raised when every retry and fallback is exhausted."""


class TransientError(Exception):
    """Raised by a backend for errors worth retrying (429, 5xx, timeouts)."""


@dataclass
class Usage:
    calls: int = 0
    cost_usd: float = 0.0


class LLM(Protocol):
    usage: Usage

    def complete(self, system: str, user: str, *, model: str = PRIMARY_MODEL) -> str: ...

    def structured(self, system: str, user: str, schema: Type[T], *, model: str = PRIMARY_MODEL) -> T: ...


def _extract_json(text: str) -> Any:
    """First complete JSON value in `text`, ignoring code fences, prose, and anything after it.

    Models sometimes emit several objects back to back (e.g. a tool call then a final
    answer); we take the first, which is the one the agent loop asked for.
    """
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if not starts:
        return json.loads(text)          # raises a clear error for the repair prompt
    value, _ = json.JSONDecoder().raw_decode(text, min(starts))
    return value


@dataclass
class ResilientLLM:
    """Wraps a raw `backend(system, user, model) -> (text, in_tokens, out_tokens)` callable."""

    backend: Callable[[str, str, str], tuple[str, int, int]]
    max_retries: int = 3
    max_repairs: int = 2
    base_delay: float = 0.5
    fallback_model: Optional[str] = FALLBACK_MODEL
    usage: Usage = field(default_factory=Usage)
    sleep: Callable[[float], None] = time.sleep

    def _call(self, system: str, user: str, model: str) -> str:
        models = [model] + ([self.fallback_model] if self.fallback_model and self.fallback_model != model else [])
        last: Exception | None = None
        for m in models:
            for attempt in range(self.max_retries):
                try:
                    text, tin, tout = self.backend(system, user, m)
                    self.usage.calls += 1
                    pin, pout = PRICES.get(m, (1.0, 4.0))
                    self.usage.cost_usd += (tin * pin + tout * pout) / 1_000_000
                    return text
                except TransientError as exc:
                    last = exc
                    delay = self.base_delay * (2 ** attempt) + random.uniform(0, self.base_delay)
                    logger.warning("Transient LLM error on %s (attempt %s): %s; sleeping %.2fs", m, attempt + 1, exc, delay)
                    self.sleep(delay)
            logger.error("Model %s exhausted retries; trying fallback.", m)
        raise LLMError(f"All models failed: {last}")

    def complete(self, system: str, user: str, *, model: str = PRIMARY_MODEL) -> str:
        return self._call(system, user, model)

    def structured(self, system: str, user: str, schema: Type[T], *, model: str = PRIMARY_MODEL) -> T:
        sys_prompt = (
            f"{system}\n\nRespond with ONLY a JSON object matching this JSON schema:\n"
            f"{json.dumps(schema.model_json_schema())}"
        )
        prompt = user
        for attempt in range(self.max_repairs + 1):
            raw = self._call(sys_prompt, prompt, model)
            try:
                return schema.model_validate(_extract_json(raw))
            except (ValueError, ValidationError) as exc:
                logger.warning("Malformed %s output (repair %s): %s", schema.__name__, attempt, exc)
                prompt = (
                    f"{user}\n\nYour previous reply was invalid:\n{raw}\n\nError:\n{exc}\n"
                    "Return corrected JSON only."
                )
        raise LLMError(f"Could not get valid {schema.__name__} after {self.max_repairs} repairs")


def openai_backend() -> Callable[[str, str, str], tuple[str, int, int]]:
    """Real backend using the existing OpenAI client from core.py."""
    import openai

    from core import get_openai_client

    client = get_openai_client()

    def call(system: str, user: str, model: str) -> tuple[str, int, int]:
        try:
            resp = client.responses.create(
                model=model,
                input=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
        except (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError) as exc:
            raise TransientError(str(exc)) from exc
        u = getattr(resp, "usage", None)
        return resp.output_text or "", getattr(u, "input_tokens", 0), getattr(u, "output_tokens", 0)

    return call
