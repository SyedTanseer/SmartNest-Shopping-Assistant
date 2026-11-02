"""Persistence for knowledge records, behind a repository interface.

``KnowledgeStore`` keeps records in any document repository exposing
``upsert`` / ``get`` / ``find`` / ``delete`` / ``count`` — in practice the
MongoDB ``BaseRepository`` (collection ``knowledge``) or the
``InMemoryRepository`` in tests.  The repository is injected, so this
module imports no database driver and no database package.

Each record is stored as one document whose ``_id`` is the
``knowledge_id``.  Queries use equality filters only (``kind``,
``entity``, ``valid``), which every repository supports.

Failures from the repository are re-raised as :class:`KnowledgeStoreError`
with the original message (already credential-redacted by the database
layer).  Stored documents that no longer validate are skipped on list
queries and reported through :attr:`KnowledgeStore.skipped`, never
silently trusted.
"""

from __future__ import annotations

from typing import Any, Protocol

from smartnest.knowledge.models import (
    KnowledgeError,
    KnowledgeKind,
    KnowledgeRecord,
    Provenance,
    Relationship,
    make_record,
)

#: Default collection name for knowledge records.
KNOWLEDGE_COLLECTION = "knowledge"


class KnowledgeStoreError(Exception):
    """Reading or writing knowledge failed."""


class DocumentRepository(Protocol):
    """The slice of a repository that the store uses (structural typing)."""

    def upsert(self, document_id: str, document: dict[str, Any]) -> bool: ...
    def get(self, document_id: str) -> dict[str, Any] | None: ...
    def find(self, filter: dict[str, Any] | None = None, *, sort: Any = None, limit: int = 0,
             skip: int = 0, projection: Any = None) -> list[dict[str, Any]]: ...
    def delete(self, document_id: str) -> bool: ...
    def count(self, filter: dict[str, Any] | None = None) -> int: ...


class KnowledgeStore:
    """Saves, loads, queries, updates and invalidates :class:`KnowledgeRecord` objects."""

    def __init__(self, repository: DocumentRepository) -> None:
        self._repository = repository
        self.skipped: list[str] = []  # ids of stored documents that failed validation

    # ── Writing ──────────────────────────────────────────────────────────

    def save(self, record: KnowledgeRecord) -> KnowledgeRecord:
        """Insert or replace the record (keyed by ``knowledge_id``)."""
        try:
            self._repository.upsert(record.knowledge_id, record.to_document())
        except Exception as exc:
            raise KnowledgeStoreError(f"could not save {record.knowledge_id}: {exc}") from None
        return record

    def create(self, kind: KnowledgeKind, entity: str, attributes: dict[str, Any], provenance: Provenance,
               confidence: float = 1.0, relationships: list[Relationship] | None = None) -> KnowledgeRecord:
        """Validate, build and save a new record."""
        return self.save(make_record(kind, entity, attributes, provenance, confidence, relationships))

    def update(self, knowledge_id: str, changes: dict[str, Any], provenance: Provenance,
               confidence: float | None = None) -> dict[str, tuple[Any, Any]]:
        """Update a stored record.  Returns what changed (``{}`` = nothing, and nothing is written)."""
        record = self._require(knowledge_id)
        diff = record.update(changes, provenance, confidence)
        if diff:
            self.save(record)
        return diff

    def invalidate(self, knowledge_id: str, reason: str) -> bool:
        record = self._require(knowledge_id)
        changed = record.invalidate(reason)
        if changed:
            self.save(record)
        return changed

    def delete(self, knowledge_id: str) -> bool:
        """Physically remove a record (prefer :meth:`invalidate`; this is for cleanup and tests)."""
        try:
            return bool(self._repository.delete(knowledge_id))
        except Exception as exc:
            raise KnowledgeStoreError(f"could not delete {knowledge_id}: {exc}") from None

    # ── Reading ──────────────────────────────────────────────────────────

    def get(self, knowledge_id: str) -> KnowledgeRecord | None:
        try:
            document = self._repository.get(knowledge_id)
        except Exception as exc:
            raise KnowledgeStoreError(f"could not read {knowledge_id}: {exc}") from None
        if document is None:
            return None
        try:
            return KnowledgeRecord.from_document(document)
        except KnowledgeError as exc:
            raise KnowledgeStoreError(f"{knowledge_id}: {exc}") from None

    def find(self, kind: KnowledgeKind | None = None, entity: str | None = None,
             valid_only: bool = True, limit: int = 0) -> list[KnowledgeRecord]:
        """Records matching the filters, most recently modified first."""
        flt: dict[str, Any] = {}
        if kind is not None:
            flt["kind"] = KnowledgeKind(kind).value
        if entity is not None:
            flt["entity"] = entity
        if valid_only:
            flt["valid"] = True
        try:
            documents = self._repository.find(flt, sort=[("modified_at", -1)], limit=limit)
        except Exception as exc:
            raise KnowledgeStoreError(f"could not query knowledge: {exc}") from None
        records = []
        for document in documents:
            try:
                records.append(KnowledgeRecord.from_document(document))
            except KnowledgeError:
                self.skipped.append(str(document.get("_id", "?")))
        return records

    def latest(self, kind: KnowledgeKind, entity: str) -> KnowledgeRecord | None:
        """The most recently modified valid record for ``entity``."""
        found = self.find(kind, entity, valid_only=True, limit=1)
        return found[0] if found else None

    def count(self, kind: KnowledgeKind | None = None, valid_only: bool = True) -> int:
        flt: dict[str, Any] = {}
        if kind is not None:
            flt["kind"] = KnowledgeKind(kind).value
        if valid_only:
            flt["valid"] = True
        try:
            return int(self._repository.count(flt))
        except Exception as exc:
            raise KnowledgeStoreError(f"could not count knowledge: {exc}") from None

    def related_to(self, knowledge_id: str, relation: str | None = None) -> list[KnowledgeRecord]:
        """Records that ``knowledge_id`` points to through its relationships."""
        record = self._require(knowledge_id)
        found = []
        for rel in record.related(relation):
            target = self.get(rel.target_id)
            if target is not None:
                found.append(target)
        return found

    def _require(self, knowledge_id: str) -> KnowledgeRecord:
        record = self.get(knowledge_id)
        if record is None:
            raise KnowledgeStoreError(f"unknown knowledge record {knowledge_id}")
        return record
