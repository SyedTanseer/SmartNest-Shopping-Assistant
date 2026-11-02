"""Repository interface and reusable MongoDB CRUD implementation.

Agents, workers and services should depend on the :class:`Repository`
interface, never on PyMongo.  That keeps database access out of agent
logic: an agent can be tested against any in-memory implementation, and
the storage technology can change without touching it.

:class:`BaseRepository` implements the interface on one MongoDB
collection.  Conventions:

    - Document ``_id`` values are **strings**.  If a document has no
      ``_id`` one is generated as ``<id_prefix>_<16 hex chars>`` (the same
      style as the models' ``evt_…`` / ``msg_…`` IDs).  Pass your own ID
      (e.g. a model's ``event_id``) as ``_id`` to keep it.
    - ``created_at`` / ``updated_at`` (timezone-aware UTC) are maintained
      automatically.
    - Input documents are never mutated.  Pydantic models are accepted
      and converted with ``model_dump()``.
    - Documents are returned as plain ``dict`` objects.
    - Every failure is raised as a :class:`~smartnest.database.errors.DatabaseError`
      subclass, never as a raw PyMongo exception.
    - ``delete_many`` refuses an empty filter unless explicitly told to
      wipe the collection.

Repositories hold no connection state: pass them a PyMongo collection
(``connection.collection(Collections.PRODUCTS)``).  Because they share
the process-wide ``MongoClient``, they are safe to use from background
worker threads.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator, Mapping, Sequence

from pydantic import BaseModel
from pymongo.errors import PyMongoError

from smartnest.database.errors import (
    DatabaseError,
    InvalidDocumentError,
    translate_error,
)

Document = dict[str, Any]
Filter = Mapping[str, Any]
SortSpec = Sequence[tuple[str, int]]

ID_FIELD = "_id"
CREATED_AT = "created_at"
UPDATED_AT = "updated_at"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Repository(ABC):
    """Storage-agnostic CRUD interface used by the rest of SmartNest."""

    @abstractmethod
    def create(self, document: Document | BaseModel) -> str:
        """Insert a document and return its ``_id``."""

    @abstractmethod
    def create_many(self, documents: Sequence[Document | BaseModel]) -> list[str]:
        """Insert several documents and return their ``_id`` values."""

    @abstractmethod
    def get(self, document_id: str) -> Document | None:
        """Return the document with this ``_id``, or ``None``."""

    @abstractmethod
    def find_one(self, filter: Filter | None = None) -> Document | None:
        """Return the first document matching ``filter``, or ``None``."""

    @abstractmethod
    def find(
        self,
        filter: Filter | None = None,
        *,
        sort: SortSpec | None = None,
        limit: int = 0,
        skip: int = 0,
        projection: Mapping[str, Any] | None = None,
    ) -> list[Document]:
        """Return matching documents (``limit=0`` means no limit)."""

    @abstractmethod
    def count(self, filter: Filter | None = None) -> int:
        """Count documents matching ``filter``."""

    @abstractmethod
    def exists(self, document_id: str) -> bool:
        """Whether a document with this ``_id`` exists."""

    @abstractmethod
    def update(self, document_id: str, changes: Document | BaseModel) -> bool:
        """Set fields on one document.  Returns ``False`` if it does not exist."""

    @abstractmethod
    def update_many(self, filter: Filter, changes: Document | BaseModel) -> int:
        """Set fields on every matching document.  Returns how many matched."""

    @abstractmethod
    def upsert(self, document_id: str, document: Document | BaseModel) -> bool:
        """Create the document or merge ``document`` into it.  ``True`` if created."""

    @abstractmethod
    def delete(self, document_id: str) -> bool:
        """Delete one document.  Returns ``False`` if it did not exist."""

    @abstractmethod
    def delete_many(self, filter: Filter, *, allow_all: bool = False) -> int:
        """Delete matching documents and return how many were deleted."""


class BaseRepository(Repository):
    """MongoDB implementation of :class:`Repository` for one collection.

    Args:
        collection: A PyMongo collection, e.g.
            ``connection.collection(Collections.PRODUCTS)``.
        id_prefix: Prefix for generated IDs (``"prd"`` → ``prd_3f2a…``).
    """

    def __init__(self, collection: Any, id_prefix: str = "doc") -> None:
        if not isinstance(id_prefix, str) or not id_prefix.strip():
            raise ValueError("id_prefix must be a non-empty string")
        self._collection = collection
        self._id_prefix = id_prefix.strip()

    @property
    def collection(self) -> Any:
        """The underlying PyMongo collection (for operations not covered here)."""
        return self._collection

    @property
    def name(self) -> str:
        return self._collection.name

    # ── Helpers ──────────────────────────────────────────────────────────

    @contextmanager
    def _guard(self, action: str) -> Iterator[None]:
        """Translate driver errors into DatabaseError subclasses."""
        try:
            yield
        except DatabaseError:
            raise
        except PyMongoError as exc:
            raise translate_error(exc, f"{action} on '{self.name}'") from exc

    @staticmethod
    def _as_dict(value: Document | BaseModel, what: str = "document") -> Document:
        if isinstance(value, BaseModel):
            return value.model_dump()
        if isinstance(value, Mapping):
            return dict(value)
        raise InvalidDocumentError(f"{what} must be a mapping or Pydantic model, got {type(value).__name__}")

    @staticmethod
    def _check_id(document_id: Any) -> str:
        if not isinstance(document_id, str) or not document_id.strip():
            raise InvalidDocumentError(f"document id must be a non-empty string, got {document_id!r}")
        return document_id

    def _new_id(self) -> str:
        return f"{self._id_prefix}_{uuid.uuid4().hex[:16]}"

    def _prepare_new(self, document: Document | BaseModel) -> Document:
        doc = self._as_dict(document)
        if ID_FIELD in doc and doc[ID_FIELD] is not None:
            self._check_id(doc[ID_FIELD])
        else:
            doc[ID_FIELD] = self._new_id()
        now = _utc_now()
        doc.setdefault(CREATED_AT, now)
        doc[UPDATED_AT] = now
        return doc

    @staticmethod
    def _check_filter(filter: Filter | None) -> Filter:
        if filter is None:
            return {}
        if not isinstance(filter, Mapping):
            raise InvalidDocumentError(f"filter must be a mapping, got {type(filter).__name__}")
        return filter

    @staticmethod
    def _check_changes(changes: Document | BaseModel) -> Document:
        data = BaseRepository._as_dict(changes, "changes")
        if not data:
            raise InvalidDocumentError("changes must not be empty")
        bad = [k for k in data if k == ID_FIELD or str(k).startswith("$")]
        if bad:
            raise InvalidDocumentError(
                f"changes may not contain _id or operator keys: {sorted(map(str, bad))}"
            )
        data.pop(CREATED_AT, None)  # creation time is immutable
        data.pop(UPDATED_AT, None)  # always set by the repository
        return data

    # ── Create ───────────────────────────────────────────────────────────

    def create(self, document: Document | BaseModel) -> str:
        doc = self._prepare_new(document)
        with self._guard("create"):
            self._collection.insert_one(doc)
        return doc[ID_FIELD]

    def create_many(self, documents: Sequence[Document | BaseModel]) -> list[str]:
        prepared = [self._prepare_new(d) for d in documents]
        if not prepared:
            return []
        with self._guard("create_many"):
            self._collection.insert_many(prepared, ordered=True)
        return [d[ID_FIELD] for d in prepared]

    # ── Read ─────────────────────────────────────────────────────────────

    def get(self, document_id: str) -> Document | None:
        self._check_id(document_id)
        with self._guard("get"):
            return self._collection.find_one({ID_FIELD: document_id})

    def find_one(self, filter: Filter | None = None) -> Document | None:
        flt = self._check_filter(filter)
        with self._guard("find_one"):
            return self._collection.find_one(dict(flt))

    def find(
        self,
        filter: Filter | None = None,
        *,
        sort: SortSpec | None = None,
        limit: int = 0,
        skip: int = 0,
        projection: Mapping[str, Any] | None = None,
    ) -> list[Document]:
        flt = self._check_filter(filter)
        # PyMongo treats a negative limit as "abs(limit), single batch", which
        # would silently hide the mistake — refuse it instead.
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise InvalidDocumentError("limit must be an integer >= 0")
        if isinstance(skip, bool) or not isinstance(skip, int) or skip < 0:
            raise InvalidDocumentError("skip must be an integer >= 0")
        with self._guard("find"):
            cursor = self._collection.find(dict(flt), dict(projection) if projection else None)
            if sort:
                cursor = cursor.sort(list(sort))
            if skip:
                cursor = cursor.skip(skip)
            if limit:
                cursor = cursor.limit(limit)
            return list(cursor)

    def count(self, filter: Filter | None = None) -> int:
        flt = self._check_filter(filter)
        with self._guard("count"):
            return int(self._collection.count_documents(dict(flt)))

    def exists(self, document_id: str) -> bool:
        self._check_id(document_id)
        with self._guard("exists"):
            return int(self._collection.count_documents({ID_FIELD: document_id}, limit=1)) > 0

    # ── Update ───────────────────────────────────────────────────────────

    def update(self, document_id: str, changes: Document | BaseModel) -> bool:
        self._check_id(document_id)
        data = self._check_changes(changes)
        data[UPDATED_AT] = _utc_now()
        with self._guard("update"):
            result = self._collection.update_one({ID_FIELD: document_id}, {"$set": data})
        return result.matched_count > 0

    def update_many(self, filter: Filter, changes: Document | BaseModel) -> int:
        flt = self._check_filter(filter)
        data = self._check_changes(changes)
        data[UPDATED_AT] = _utc_now()
        with self._guard("update_many"):
            result = self._collection.update_many(dict(flt), {"$set": data})
        return int(result.matched_count)

    def upsert(self, document_id: str, document: Document | BaseModel) -> bool:
        self._check_id(document_id)
        data = self._as_dict(document)
        data.pop(ID_FIELD, None)
        data.pop(CREATED_AT, None)
        now = _utc_now()
        data[UPDATED_AT] = now
        with self._guard("upsert"):
            result = self._collection.update_one(
                {ID_FIELD: document_id},
                {"$set": data, "$setOnInsert": {CREATED_AT: now}},
                upsert=True,
            )
        return result.upserted_id is not None

    # ── Delete ───────────────────────────────────────────────────────────

    def delete(self, document_id: str) -> bool:
        self._check_id(document_id)
        with self._guard("delete"):
            result = self._collection.delete_one({ID_FIELD: document_id})
        return result.deleted_count > 0

    def delete_many(self, filter: Filter, *, allow_all: bool = False) -> int:
        flt = self._check_filter(filter)
        if not flt and not allow_all:
            raise InvalidDocumentError(
                "delete_many with an empty filter would delete every document; "
                "pass allow_all=True if that is intended"
            )
        with self._guard("delete_many"):
            result = self._collection.delete_many(dict(flt))
        return int(result.deleted_count)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} collection={self.name!r} id_prefix={self._id_prefix!r}>"
