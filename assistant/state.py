# Author: Naveen Muppana
"""Checkpoint store for TaskState plus the memory collections the Learner writes.

`InMemoryStore` is used for tests and when MONGODB_URI is unset; `MongoStore`
reuses core.get_mongo_collection for production.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Iterable, Optional, Protocol

from .contracts import TaskState

logger = logging.getLogger(__name__)
DB_NAME = os.getenv("ASSISTANT_DB", "assistant_db")


class Store(Protocol):
    def save_task(self, state: TaskState) -> None: ...
    def load_task(self, task_id: str) -> Optional[TaskState]: ...
    def find_active_by_dedupe(self, key: str) -> Optional[TaskState]: ...
    def task_ids_with_status(self, status: str) -> list[str]: ...
    def put(self, collection: str, doc: dict[str, Any]) -> None: ...
    def find(self, collection: str, filt: dict[str, Any] | None = None) -> list[dict[str, Any]]: ...
    def delete(self, collection: str, doc_id: str) -> None: ...


ACTIVE = {"planning", "running", "waiting_approval"}


class InMemoryStore:
    def __init__(self) -> None:
        self._tasks: dict[str, dict] = {}
        self._cols: dict[str, dict[str, dict]] = {}

    def save_task(self, state: TaskState) -> None:
        state.updated_at = datetime.now(timezone.utc)
        self._tasks[state.task_id] = state.model_dump(mode="json")

    def load_task(self, task_id: str) -> Optional[TaskState]:
        raw = self._tasks.get(task_id)
        return TaskState.model_validate(raw) if raw else None

    def find_active_by_dedupe(self, key: str) -> Optional[TaskState]:
        for raw in self._tasks.values():
            st = TaskState.model_validate(raw)
            if st.envelope.dedupe_key == key and st.status.value in ACTIVE:
                return st
        return None

    def task_ids_with_status(self, status: str) -> list[str]:
        return [tid for tid, raw in self._tasks.items() if raw["status"] == status]

    def put(self, collection: str, doc: dict[str, Any]) -> None:
        self._cols.setdefault(collection, {})[doc["_id"]] = dict(doc)

    def find(self, collection: str, filt: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        docs: Iterable[dict] = self._cols.get(collection, {}).values()
        filt = filt or {}
        return [dict(d) for d in docs if all(d.get(k) == v for k, v in filt.items())]

    def delete(self, collection: str, doc_id: str) -> None:
        self._cols.get(collection, {}).pop(doc_id, None)


class MongoStore:
    def __init__(self) -> None:
        from core import get_mongo_collection  # lazy: tests never need Mongo

        self._col = lambda name: get_mongo_collection(DB_NAME, name)
        self._tasks = self._col("tasks")

    def save_task(self, state: TaskState) -> None:
        state.updated_at = datetime.now(timezone.utc)
        doc = state.model_dump(mode="json")
        doc["_id"] = state.task_id
        doc["dedupe_key"] = state.envelope.dedupe_key
        self._tasks.replace_one({"_id": state.task_id}, doc, upsert=True)

    def load_task(self, task_id: str) -> Optional[TaskState]:
        doc = self._tasks.find_one({"_id": task_id})
        return TaskState.model_validate(doc) if doc else None

    def find_active_by_dedupe(self, key: str) -> Optional[TaskState]:
        doc = self._tasks.find_one({"dedupe_key": key, "status": {"$in": list(ACTIVE)}})
        return TaskState.model_validate(doc) if doc else None

    def task_ids_with_status(self, status: str) -> list[str]:
        return [d["_id"] for d in self._tasks.find({"status": status}, {"_id": 1})]

    def put(self, collection: str, doc: dict[str, Any]) -> None:
        self._col(collection).replace_one({"_id": doc["_id"]}, doc, upsert=True)

    def find(self, collection: str, filt: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        return list(self._col(collection).find(filt or {}))

    def delete(self, collection: str, doc_id: str) -> None:
        self._col(collection).delete_one({"_id": doc_id})


def default_store() -> Store:
    if os.getenv("MONGODB_URI"):
        try:
            return MongoStore()
        except Exception:  # pragma: no cover - fall back rather than refuse to start
            logger.exception("MongoStore unavailable; falling back to in-memory store.")
    logger.warning("Using InMemoryStore: task state will not survive a restart.")
    return InMemoryStore()
