"""Desires: the goals the agent wants to achieve or keep true.

A desire has a description, a **priority** (0–100, higher = more
important), optional **activation conditions** over beliefs, and a
**status**:

    ACTIVE      wanted and currently pursuable
    BLOCKED     wanted, but no way to satisfy it exists right now
                (e.g. no feasible product); retried when beliefs change
    SUSPENDED   set aside — by the user, or because a higher-priority
                desire it conflicts with won (see ``conflicts_with``)
    ACHIEVED    done
    DROPPED     abandoned

Several desires can be active at once.  :meth:`DesireManager.pursuable`
returns the ones whose conditions hold, highest priority first, and
:meth:`DesireManager.resolve_conflicts` makes sure desires that exclude
each other are not pursued together: the higher priority wins and the
other is suspended, then automatically reactivated if the winner goes
away.

Two kinds of goal:

    achievement   reached by carrying out an intention (select a sofa)
    maintenance   a state to keep true, monitored rather than "done"
                  (stay within the total budget)
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Iterable, Literal

from pydantic import BaseModel, Field

from smartnest.bdi.common import new_id, utc_now
from smartnest.bdi.conditions import BeliefReader, Condition, ConditionResult


class DesireStatus(str, Enum):
    ACTIVE = "active"
    BLOCKED = "blocked"
    SUSPENDED = "suspended"
    ACHIEVED = "achieved"
    DROPPED = "dropped"


# Statuses from which a desire can still be pursued.
_PURSUABLE = (DesireStatus.ACTIVE, DesireStatus.BLOCKED)
_TERMINAL = (DesireStatus.ACHIEVED, DesireStatus.DROPPED)


class Desire(BaseModel):
    """A goal."""

    desire_id: str = Field(default_factory=lambda: new_id("des"))
    description: str
    goal_type: str = "select_product"
    kind: Literal["achievement", "maintenance"] = "achievement"
    priority: int = Field(default=50, strict=True, ge=0, le=100)
    conditions: list[Condition] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)
    conflicts_with: list[str] = Field(default_factory=list)
    status: DesireStatus = DesireStatus.ACTIVE
    status_reason: str = ""
    suspended_by: str | None = None  # desire_id that won a conflict, if any
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    def model_post_init(self, __context: Any) -> None:
        if not self.description.strip():
            raise ValueError("description must not be empty")


def select_sofa_desire(priority: int = 80, desire_id: str = "des_select_sofa", **params: Any) -> Desire:
    """The standard goal: choose the best sofa for the user's requirements."""
    return Desire(
        desire_id=desire_id,
        description="Select the highest-utility sofa that satisfies the user's requirements",
        goal_type="select_product",
        kind="achievement",
        priority=priority,
        params={"category": "sofa", **params},
    )


def stay_within_budget_desire(priority: int = 90, desire_id: str = "des_total_budget") -> Desire:
    """A maintenance goal: committed and recorded spending stay within the total budget."""
    return Desire(
        desire_id=desire_id,
        description="Keep furnishing spend within the total budget",
        goal_type="maintain_budget",
        kind="maintenance",
        priority=priority,
    )


class DesireManager:
    """Holds the agent's desires and decides which are pursued, and in what order."""

    def __init__(self) -> None:
        self._desires: dict[str, Desire] = {}

    # ── Adding / reading ─────────────────────────────────────────────────

    def add(self, desire: Desire) -> Desire:
        if desire.desire_id in self._desires:
            raise ValueError(f"desire {desire.desire_id} already exists")
        self._desires[desire.desire_id] = desire
        return desire

    def get(self, desire_id: str) -> Desire | None:
        return self._desires.get(desire_id)

    def all(self) -> list[Desire]:
        """Every desire, highest priority first (creation order breaks ties)."""
        return sorted(self._desires.values(), key=lambda d: (-d.priority, d.created_at))

    def with_status(self, *statuses: DesireStatus) -> list[Desire]:
        return [d for d in self.all() if d.status in statuses]

    def __len__(self) -> int:
        return len(self._desires)

    # ── Conditions and prioritisation ────────────────────────────────────

    @staticmethod
    def unmet_conditions(desire: Desire, beliefs: BeliefReader) -> list[ConditionResult]:
        results = [c.evaluate(beliefs) for c in desire.conditions]
        return [r for r in results if not r.ok]

    def pursuable(self, beliefs: BeliefReader) -> list[Desire]:
        """Desires that are wanted now: active or blocked, conditions met — by priority."""
        return [
            d for d in self.all()
            if d.status in _PURSUABLE and not self.unmet_conditions(d, beliefs)
        ]

    def waiting_for_conditions(self, beliefs: BeliefReader) -> list[tuple[Desire, list[ConditionResult]]]:
        """Wanted desires that cannot be pursued yet, with the conditions that fail."""
        waiting = []
        for d in self.all():
            if d.status in _PURSUABLE:
                unmet = self.unmet_conditions(d, beliefs)
                if unmet:
                    waiting.append((d, unmet))
        return waiting

    def set_priority(self, desire_id: str, priority: int) -> bool:
        desire = self._require(desire_id)
        if isinstance(priority, bool) or not isinstance(priority, int) or not 0 <= priority <= 100:
            raise ValueError("priority must be an integer between 0 and 100")
        if desire.priority == priority:
            return False
        desire.priority = priority
        desire.updated_at = utc_now()
        return True

    # ── Status changes ───────────────────────────────────────────────────

    def set_status(self, desire_id: str, status: DesireStatus, reason: str = "") -> bool:
        """Change status.  Returns ``False`` if nothing changed.  Terminal desires stay terminal."""
        desire = self._require(desire_id)
        if desire.status in _TERMINAL and status != desire.status:
            return False
        if desire.status == status and desire.status_reason == reason:
            return False
        desire.status = status
        desire.status_reason = reason
        if status != DesireStatus.SUSPENDED:
            desire.suspended_by = None
        desire.updated_at = utc_now()
        return True

    def achieve(self, desire_id: str, reason: str = "") -> bool:
        return self.set_status(desire_id, DesireStatus.ACHIEVED, reason)

    def drop(self, desire_id: str, reason: str = "") -> bool:
        return self.set_status(desire_id, DesireStatus.DROPPED, reason)

    def suspend(self, desire_id: str, reason: str = "") -> bool:
        return self.set_status(desire_id, DesireStatus.SUSPENDED, reason)

    def reactivate(self, desire_id: str, reason: str = "") -> bool:
        desire = self._require(desire_id)
        if desire.status != DesireStatus.SUSPENDED:
            return False
        return self.set_status(desire_id, DesireStatus.ACTIVE, reason)

    def resolve_conflicts(self) -> list[tuple[Desire, str]]:
        """Suspend desires that lose to a higher-priority conflicting desire.

        Walks desires from highest to lowest priority.  A desire wins
        unless a desire it conflicts with (in either direction) has
        already won.  Losers are suspended; desires previously suspended
        *because of a conflict* whose winner is gone are reactivated.

        Returns:
            ``(desire, message)`` for every status change made, for logging.
        """
        changes: list[tuple[Desire, str]] = []
        winners: set[str] = set()
        for desire in self.all():
            contender = desire.status in _PURSUABLE or (
                desire.status == DesireStatus.SUSPENDED and desire.suspended_by is not None
            )
            if not contender:
                continue
            rival = next(
                (w for w in winners
                 if w in desire.conflicts_with or desire.desire_id in self._desires[w].conflicts_with),
                None,
            )
            if rival is not None:
                if desire.status != DesireStatus.SUSPENDED:
                    desire.status = DesireStatus.SUSPENDED
                    desire.suspended_by = rival
                    desire.status_reason = f"conflicts with higher-priority desire {rival}"
                    desire.updated_at = utc_now()
                    changes.append((desire, f"suspended: {desire.status_reason}"))
                continue
            winners.add(desire.desire_id)
            if desire.status == DesireStatus.SUSPENDED:  # its winner is gone
                desire.status = DesireStatus.ACTIVE
                desire.suspended_by = None
                desire.status_reason = "conflicting desire no longer holds"
                desire.updated_at = utc_now()
                changes.append((desire, f"reactivated: {desire.status_reason}"))
        return changes

    # ── Persistence ──────────────────────────────────────────────────────

    def to_list(self) -> list[dict[str, Any]]:
        return [d.model_dump(mode="json") for d in self._desires.values()]

    def load(self, items: Iterable[Desire | dict[str, Any]]) -> None:
        restored = {}
        for item in items:
            desire = item if isinstance(item, Desire) else Desire.model_validate(item)
            restored[desire.desire_id] = desire
        self._desires = restored

    def clear(self) -> None:
        self._desires.clear()

    def _require(self, desire_id: str) -> Desire:
        desire = self._desires.get(desire_id)
        if desire is None:
            raise KeyError(f"unknown desire {desire_id}")
        return desire
