"""In-memory repository, for tests, offline demos and development.

``InMemoryRepository`` is the real :class:`BaseRepository` running on an
:class:`InMemoryCollection`, so it applies exactly the same rules — string
ids, generated ids, automatic timestamps, duplicate detection, input
validation, the ``delete_many`` guard — without a server.  Code written
against the :class:`Repository` interface (the BDI agent's state store,
future catalogue repositories) can be exercised with it unchanged.

``InMemoryCollection`` implements only the PyMongo collection calls the
repository uses, with **equality-only filters** and ``$set`` /
``$setOnInsert`` updates.  It is not a MongoDB emulator: richer query
semantics are verified by the optional live Atlas tests.  It is
thread-safe.
"""

from __future__ import annotations

import copy
import threading
from types import SimpleNamespace
from typing import Any, Iterable

from pymongo.errors import BulkWriteError, DuplicateKeyError

from smartnest.database.repositories.base_repository import BaseRepository


def _matches(doc: dict, flt: dict) -> bool:
    return all(doc.get(key) == value for key, value in flt.items())


class InMemoryCursor:
    def __init__(self, docs: list[dict]) -> None:
        self._docs = docs

    def sort(self, spec: Iterable[tuple[str, int]]) -> "InMemoryCursor":
        for key, direction in reversed(list(spec)):  # stable multi-key sort
            self._docs.sort(key=lambda d: (d.get(key) is None, d.get(key)), reverse=direction < 0)
        return self

    def skip(self, n: int) -> "InMemoryCursor":
        self._docs = self._docs[n:]
        return self

    def limit(self, n: int) -> "InMemoryCursor":
        self._docs = self._docs[:n]
        return self

    def __iter__(self):
        return iter(self._docs)


class InMemoryCollection:
    """A dictionary pretending to be a PyMongo collection (the subset BaseRepository uses)."""

    def __init__(self, name: str = "in_memory") -> None:
        self.name = name
        self.docs: dict[Any, dict] = {}
        self.calls: list[str] = []
        self._lock = threading.RLock()

    # ── writes ───────────────────────────────────────────────────────────
    def insert_one(self, doc: dict) -> SimpleNamespace:
        with self._lock:
            self.calls.append("insert_one")
            if doc["_id"] in self.docs:
                raise DuplicateKeyError("E11000 duplicate key error")
            self.docs[doc["_id"]] = copy.deepcopy(doc)
            return SimpleNamespace(inserted_id=doc["_id"])

    def insert_many(self, docs: list[dict], ordered: bool = True) -> SimpleNamespace:
        with self._lock:
            self.calls.append("insert_many")
            ids = []
            for doc in docs:
                if doc["_id"] in self.docs:
                    raise BulkWriteError({"writeErrors": [{"code": 11000, "errmsg": "duplicate key"}]})
                self.docs[doc["_id"]] = copy.deepcopy(doc)
                ids.append(doc["_id"])
            return SimpleNamespace(inserted_ids=ids)

    @staticmethod
    def _apply(doc: dict, update: dict) -> None:
        doc.update(copy.deepcopy(update.get("$set", {})))

    def update_one(self, flt: dict, update: dict, upsert: bool = False) -> SimpleNamespace:
        with self._lock:
            self.calls.append("update_one")
            for doc in self.docs.values():
                if _matches(doc, flt):
                    self._apply(doc, update)
                    return SimpleNamespace(matched_count=1, modified_count=1, upserted_id=None)
            if upsert:
                new = {**flt, **copy.deepcopy(update.get("$set", {})),
                       **copy.deepcopy(update.get("$setOnInsert", {}))}
                self.docs[new["_id"]] = new
                return SimpleNamespace(matched_count=0, modified_count=0, upserted_id=new["_id"])
            return SimpleNamespace(matched_count=0, modified_count=0, upserted_id=None)

    def update_many(self, flt: dict, update: dict) -> SimpleNamespace:
        with self._lock:
            self.calls.append("update_many")
            matched = [d for d in self.docs.values() if _matches(d, flt)]
            for doc in matched:
                self._apply(doc, update)
            return SimpleNamespace(matched_count=len(matched), modified_count=len(matched))

    def delete_one(self, flt: dict) -> SimpleNamespace:
        with self._lock:
            self.calls.append("delete_one")
            for key, doc in list(self.docs.items()):
                if _matches(doc, flt):
                    del self.docs[key]
                    return SimpleNamespace(deleted_count=1)
            return SimpleNamespace(deleted_count=0)

    def delete_many(self, flt: dict) -> SimpleNamespace:
        with self._lock:
            self.calls.append("delete_many")
            gone = [k for k, d in self.docs.items() if _matches(d, flt)]
            for key in gone:
                del self.docs[key]
            return SimpleNamespace(deleted_count=len(gone))

    # ── reads ────────────────────────────────────────────────────────────
    def find_one(self, flt: dict | None = None) -> dict | None:
        with self._lock:
            self.calls.append("find_one")
            for doc in self.docs.values():
                if _matches(doc, flt or {}):
                    return copy.deepcopy(doc)
            return None

    def find(self, flt: dict | None = None, projection: dict | None = None) -> InMemoryCursor:
        with self._lock:
            self.calls.append("find")
            return InMemoryCursor([copy.deepcopy(d) for d in self.docs.values() if _matches(d, flt or {})])

    def count_documents(self, flt: dict, limit: int = 0) -> int:
        with self._lock:
            self.calls.append("count_documents")
            n = sum(1 for d in self.docs.values() if _matches(d, flt))
            return min(n, limit) if limit else n


class InMemoryRepository(BaseRepository):
    """:class:`BaseRepository` backed by memory instead of MongoDB."""

    def __init__(self, name: str = "in_memory", id_prefix: str = "doc") -> None:
        super().__init__(InMemoryCollection(name), id_prefix=id_prefix)
