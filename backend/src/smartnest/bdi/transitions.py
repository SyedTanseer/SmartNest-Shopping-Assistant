"""Transition records: the agent's explained history.

Every meaningful step the agent takes — a belief changing, an update
being ignored, an intention being adopted, invalidated, replaced or
retained, a goal being blocked — is recorded as a :class:`Transition`
with the state **before** and **after** and a human-readable
**explanation** of why.  The history is persisted with the agent, so
after a restart it can still explain how it got where it is.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from smartnest.bdi.common import new_id, utc_now

# Transition kinds ------------------------------------------------------------
BELIEF_UPDATED = "belief_updated"
UPDATE_REJECTED = "update_rejected"
UPDATE_IGNORED = "update_ignored"
DESIRE_ADDED = "desire_added"
DESIRES_RECONSIDERED = "desires_reconsidered"
DESIRE_WAITING = "desire_waiting"
DESIRE_BLOCKED = "desire_blocked"
DESIRE_UNBLOCKED = "desire_unblocked"
DESIRE_ACHIEVED = "desire_achieved"
DESIRE_STATUS = "desire_status"
INTENTION_ADOPTED = "intention_adopted"
INTENTION_INVALIDATED = "intention_invalidated"
INTENTION_REPLACED = "intention_replaced"
INTENTION_RETAINED = "intention_retained"
INTENTION_CANCELLED = "intention_cancelled"
INTENTION_PROGRESS = "intention_progress"
STATE_RESTORED = "state_restored"
STATE_NOT_FOUND = "state_not_found"
PERSISTENCE_ERROR = "persistence_error"


class Transition(BaseModel):
    """One explained step in the agent's history."""

    transition_id: str = Field(default_factory=lambda: new_id("trn"))
    timestamp: datetime = Field(default_factory=utc_now)
    kind: str
    trigger: str = ""
    summary: str
    before: dict[str, Any] | None = None
    after: dict[str, Any] | None = None
    explanation: str = ""
    details: dict[str, Any] = Field(default_factory=dict)

    def format(self) -> str:
        """A multi-line, human-readable rendering."""
        lines = [f"[{self.kind}] {self.summary}"]
        if self.trigger:
            lines.append(f"  trigger    : {self.trigger}")
        if self.before is not None:
            lines.append(f"  before     : {self._brief(self.before)}")
        if self.after is not None:
            lines.append(f"  after      : {self._brief(self.after)}")
        if self.explanation:
            lines.append(f"  explanation: {self.explanation}")
        return "\n".join(lines)

    @staticmethod
    def _brief(state: dict[str, Any]) -> str:
        return ", ".join(f"{k}={v}" for k, v in state.items() if v not in (None, ""))
