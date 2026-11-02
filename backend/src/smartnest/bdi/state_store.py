"""Persistence for BDI agent state, behind an interface.

The agent depends only on :class:`AgentStateStore` (``load`` / ``save``).
Two implementations are provided:

``InMemoryAgentStateStore``
    A dictionary.  For tests and offline demos.

``RepositoryAgentStateStore``
    Stores the snapshot as one document per agent in any repository with
    ``get`` / ``upsert`` / ``delete`` — in practice the MongoDB
    ``BaseRepository`` on the ``agent_states`` collection, or the
    ``InMemoryRepository`` in tests.  The repository is **injected**; this
    module imports nothing from the database package and no database
    driver, so BDI stays independent of storage technology.

A snapshot holds the agent's identity, its beliefs, desires, intentions
and its transition history.  Everything is JSON-compatible (timestamps are
ISO-8601 strings), so it round-trips through any document store.

Failures surface as :class:`StateStoreError` — an unreachable database, a
missing-permission error or a corrupted document alike — and the agent
turns them into recorded transitions instead of crashing.
"""

from __future__ import annotations

import copy
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from smartnest.bdi.common import utc_now

SCHEMA_VERSION = 1


class StateStoreError(Exception):
    """Saving or loading agent state failed."""


class AgentSnapshot(BaseModel):
    """Everything needed to restore a BDI agent."""

    model_config = ConfigDict(extra="ignore")  # tolerate repository bookkeeping fields

    agent_id: str
    agent_type: str = "bdi"
    schema_version: int = SCHEMA_VERSION
    beliefs: list[dict[str, Any]] = Field(default_factory=list)
    desires: list[dict[str, Any]] = Field(default_factory=list)
    intentions: list[dict[str, Any]] = Field(default_factory=list)
    transitions: list[dict[str, Any]] = Field(default_factory=list)
    saved_at: datetime = Field(default_factory=utc_now)

    def to_document(self) -> dict[str, Any]:
        """A JSON-compatible dict (no ``_id``; the repository keys it by agent id)."""
        return self.model_dump(mode="json")

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> "AgentSnapshot":
        snapshot = cls.model_validate(document)
        if snapshot.schema_version > SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version {snapshot.schema_version}")
        return snapshot


@runtime_checkable
class AgentStateStore(Protocol):
    """What the agent needs from persistence."""

    def load(self, agent_id: str) -> AgentSnapshot | None:
        """The saved snapshot, or ``None`` if there is no record.  Raises :class:`StateStoreError`."""

    def save(self, snapshot: AgentSnapshot) -> None:
        """Persist the snapshot.  Raises :class:`StateStoreError`."""


class InMemoryAgentStateStore:
    """Dictionary-backed store for tests and offline use."""

    def __init__(self) -> None:
        self._documents: dict[str, dict[str, Any]] = {}
        self.save_count = 0

    def load(self, agent_id: str) -> AgentSnapshot | None:
        document = self._documents.get(agent_id)
        if document is None:
            return None
        try:
            return AgentSnapshot.from_document(copy.deepcopy(document))
        except (ValidationError, ValueError) as exc:
            raise StateStoreError(f"stored state for '{agent_id}' is invalid: {exc}") from None

    def save(self, snapshot: AgentSnapshot) -> None:
        self._documents[snapshot.agent_id] = copy.deepcopy(snapshot.to_document())
        self.save_count += 1

    def delete(self, agent_id: str) -> bool:
        return self._documents.pop(agent_id, None) is not None


class StateRepository(Protocol):
    """The slice of a document repository this store uses (structural typing)."""

    def get(self, document_id: str) -> dict[str, Any] | None: ...

    def upsert(self, document_id: str, document: dict[str, Any]) -> bool: ...

    def delete(self, document_id: str) -> bool: ...


class RepositoryAgentStateStore:
    """Stores agent snapshots through an injected document repository.

    The document ``_id`` is the agent id, so each agent has exactly one
    record, updated in place.
    """

    def __init__(self, repository: StateRepository) -> None:
        self._repository = repository

    def load(self, agent_id: str) -> AgentSnapshot | None:
        try:
            document = self._repository.get(agent_id)
        except Exception as exc:
            raise StateStoreError(f"could not read state for '{agent_id}': {exc}") from None
        if document is None:
            return None
        try:
            return AgentSnapshot.from_document(document)
        except (ValidationError, ValueError) as exc:
            raise StateStoreError(f"stored state for '{agent_id}' is invalid: {exc}") from None

    def save(self, snapshot: AgentSnapshot) -> None:
        try:
            self._repository.upsert(snapshot.agent_id, snapshot.to_document())
        except Exception as exc:
            raise StateStoreError(f"could not save state for '{snapshot.agent_id}': {exc}") from None

    def delete(self, agent_id: str) -> bool:
        try:
            return bool(self._repository.delete(agent_id))
        except Exception as exc:
            raise StateStoreError(f"could not delete state for '{agent_id}': {exc}") from None
