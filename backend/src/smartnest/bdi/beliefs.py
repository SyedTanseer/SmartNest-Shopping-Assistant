"""Beliefs: what the agent currently holds to be true.

A belief is a fact about a *subject* (``"user"``, ``"SOFA_A2"``) with a
*predicate* (``"sofa_budget"``, ``"availability"``), a value, where it
came from, and when.  There is one belief per (subject, predicate); a new
value replaces the old one.

The belief base reports *what actually changed*:

    - updating a belief to the value it already has changes nothing and
      returns ``None`` — so repeated, identical observations can never
      cause replanning;
    - a real change returns a :class:`BeliefChange` carrying the old and
      new value, which is what the agent uses to decide whether anything
      it cares about was affected.

Beliefs can be **invalidated** (the agent no longer trusts them).  An
invalid belief behaves as *unknown*: lookups return nothing, and
conditions that need it fail.  A later update makes it valid again.

Values must be JSON-compatible (numbers, strings, booleans, lists, dicts)
so beliefs persist with the agent state.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable, Literal

from pydantic import BaseModel, Field

from smartnest.bdi.common import utc_now


class Belief(BaseModel):
    """One fact held by the agent."""

    subject: str
    predicate: str
    value: Any = None
    source: str = "unknown"
    timestamp: datetime = Field(default_factory=utc_now)
    valid: bool = True
    invalidated_at: datetime | None = None
    invalidation_reason: str = ""

    @property
    def key(self) -> tuple[str, str]:
        return (self.subject, self.predicate)


class BeliefChange(BaseModel):
    """A real change to the belief base (never produced by a no-op update)."""

    subject: str
    predicate: str
    kind: Literal["added", "updated", "invalidated"]
    old_value: Any = None
    new_value: Any = None
    source: str = "unknown"
    timestamp: datetime = Field(default_factory=utc_now)

    @property
    def key(self) -> tuple[str, str]:
        return (self.subject, self.predicate)

    def describe(self) -> str:
        label = f"{self.subject}.{self.predicate}"
        if self.kind == "added":
            return f"{label} = {self.new_value!r} (new, from {self.source})"
        if self.kind == "invalidated":
            return f"{label} = {self.old_value!r} invalidated"
        return f"{label}: {self.old_value!r} -> {self.new_value!r} (from {self.source})"


def _clean(name: Any, label: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"belief {label} must be a non-empty string, got {name!r}")
    return name.strip()


class BeliefBase:
    """Stores, updates, invalidates and queries beliefs."""

    def __init__(self) -> None:
        self._beliefs: dict[tuple[str, str], Belief] = {}

    # ── Writing ──────────────────────────────────────────────────────────

    def update(self, subject: str, predicate: str, value: Any, source: str = "unknown") -> BeliefChange | None:
        """Record that ``subject.predicate`` is ``value``.

        Returns:
            The :class:`BeliefChange`, or ``None`` if the belief already
            held exactly this value (nothing changed).
        """
        subject, predicate = _clean(subject, "subject"), _clean(predicate, "predicate")
        key = (subject, predicate)
        existing = self._beliefs.get(key)
        now = utc_now()

        if existing is not None and existing.valid and existing.value == value:
            return None

        previous = existing.value if (existing is not None and existing.valid) else None
        kind: Literal["added", "updated"] = "updated" if previous is not None else "added"
        self._beliefs[key] = Belief(subject=subject, predicate=predicate, value=value,
                                    source=source, timestamp=now)
        return BeliefChange(subject=subject, predicate=predicate, kind=kind,
                            old_value=previous, new_value=value, source=source, timestamp=now)

    def ensure(self, subject: str, predicate: str, value: Any, source: str) -> BeliefChange | None:
        """Set a belief only if it is missing, invalid, or was set by the same ``source``.

        Used for background knowledge (e.g. catalogue attributes) so it
        never overwrites something learned from a more specific source
        such as an environmental event.
        """
        existing = self._beliefs.get((subject, predicate))
        if existing is None or not existing.valid or existing.source == source:
            return self.update(subject, predicate, value, source)
        return None

    def invalidate(self, subject: str, predicate: str, reason: str = "", source: str = "unknown") -> BeliefChange | None:
        """Mark a belief as no longer trusted.  ``None`` if unknown or already invalid."""
        existing = self._beliefs.get((subject, predicate))
        if existing is None or not existing.valid:
            return None
        now = utc_now()
        existing.valid = False
        existing.invalidated_at = now
        existing.invalidation_reason = reason
        return BeliefChange(subject=subject, predicate=predicate, kind="invalidated",
                            old_value=existing.value, new_value=None, source=source, timestamp=now)

    # ── Reading ──────────────────────────────────────────────────────────

    def get(self, subject: str, predicate: str, include_invalid: bool = False) -> Belief | None:
        """A copy of the belief, or ``None`` if unknown (or invalid, by default)."""
        belief = self._beliefs.get((subject, predicate))
        if belief is None or (not belief.valid and not include_invalid):
            return None
        return belief.model_copy(deep=True)

    def value(self, subject: str, predicate: str, default: Any = None) -> Any:
        """The current value, or ``default`` if unknown or invalid."""
        belief = self._beliefs.get((subject, predicate))
        if belief is None or not belief.valid:
            return default
        return belief.value

    def has(self, subject: str, predicate: str) -> bool:
        return self.get(subject, predicate) is not None

    def query(self, subject: str | None = None, predicate: str | None = None,
              include_invalid: bool = False) -> list[Belief]:
        return [
            b.model_copy(deep=True) for b in self._beliefs.values()
            if (subject is None or b.subject == subject)
            and (predicate is None or b.predicate == predicate)
            and (b.valid or include_invalid)
        ]

    def as_dict(self) -> dict[str, Any]:
        """``{"subject.predicate": value}`` for every valid belief."""
        return {f"{b.subject}.{b.predicate}": b.value for b in self._beliefs.values() if b.valid}

    def __len__(self) -> int:
        return len(self._beliefs)

    def __contains__(self, key: object) -> bool:
        return key in self._beliefs

    # ── Persistence ──────────────────────────────────────────────────────

    def to_list(self) -> list[dict[str, Any]]:
        return [b.model_dump(mode="json") for b in self._beliefs.values()]

    def load(self, items: Iterable[Belief | dict[str, Any]]) -> None:
        """Replace all beliefs with ``items`` (used when restoring state)."""
        restored: dict[tuple[str, str], Belief] = {}
        for item in items:
            belief = item if isinstance(item, Belief) else Belief.model_validate(item)
            restored[belief.key] = belief
        self._beliefs = restored

    def clear(self) -> None:
        self._beliefs.clear()
