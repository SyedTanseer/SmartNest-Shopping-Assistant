"""Negotiation state machine for Contract Net Protocol.

Provides explicit states and validated transitions so the coordinator
and bidding agents share a common lifecycle vocabulary.

States:
    CREATED → CFP_SENT → COLLECTING_PROPOSALS → EVALUATING
           → NEGOTIATING → AWARDED → EXECUTING → COMPLETED → TERMINATED
                    ↘ FAILED → TERMINATED

    EVALUATING → NEGOTIATING → CFP_SENT   (another round)
    AWARDED → NEGOTIATING                 (winner refused the award; the
                                           coordinator tries the runner-up)
    TERMINATED is the only terminal state and is reachable from any
    non-terminal state.

Invalid transitions raise ``InvalidStateTransition``.
Each transition is timestamped and recorded in ``history``.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class NegotiationState(str, Enum):
    """Lifecycle states for a CNP negotiation."""

    CREATED = "created"
    CFP_SENT = "cfp_sent"
    COLLECTING_PROPOSALS = "collecting_proposals"
    EVALUATING = "evaluating"
    NEGOTIATING = "negotiating"
    AWARDED = "awarded"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    TERMINATED = "terminated"


class InvalidStateTransition(Exception):
    """Raised when a state transition is not allowed."""


# Valid transitions: from_state → {allowed next states}
_VALID_TRANSITIONS: dict[NegotiationState, set[NegotiationState]] = {
    NegotiationState.CREATED: {
        NegotiationState.CFP_SENT,
        NegotiationState.FAILED,
        NegotiationState.TERMINATED,
    },
    NegotiationState.CFP_SENT: {
        NegotiationState.COLLECTING_PROPOSALS,
        NegotiationState.FAILED,
        NegotiationState.TERMINATED,
    },
    NegotiationState.COLLECTING_PROPOSALS: {
        NegotiationState.EVALUATING,
        NegotiationState.FAILED,
        NegotiationState.TERMINATED,
    },
    NegotiationState.EVALUATING: {
        NegotiationState.NEGOTIATING,
        NegotiationState.AWARDED,
        NegotiationState.FAILED,
        NegotiationState.TERMINATED,
    },
    NegotiationState.NEGOTIATING: {
        NegotiationState.CFP_SENT,
        NegotiationState.EVALUATING,
        NegotiationState.AWARDED,
        NegotiationState.FAILED,
        NegotiationState.TERMINATED,
    },
    NegotiationState.AWARDED: {
        NegotiationState.EXECUTING,
        NegotiationState.NEGOTIATING,  # award refused → try next candidate
        NegotiationState.FAILED,
        NegotiationState.TERMINATED,
    },
    NegotiationState.EXECUTING: {
        NegotiationState.COMPLETED,
        NegotiationState.FAILED,
        NegotiationState.TERMINATED,
    },
    NegotiationState.COMPLETED: {
        NegotiationState.TERMINATED,
    },
    NegotiationState.FAILED: {
        NegotiationState.TERMINATED,
    },
    NegotiationState.TERMINATED: set(),
}


class NegotiationStateMachine:
    """A state machine tracking one negotiation's lifecycle.

    Attributes:
        negotiation_id: Unique identifier for this negotiation.
        current_state: The current ``NegotiationState``.
        history: List of ``(state, timestamp)`` transition records.
    """

    def __init__(self, negotiation_id: str) -> None:
        self.negotiation_id = negotiation_id
        self._state: NegotiationState = NegotiationState.CREATED
        self._history: list[dict[str, Any]] = []
        self._record_transition(NegotiationState.CREATED)

    @property
    def current_state(self) -> NegotiationState:
        return self._state

    @property
    def history(self) -> list[dict[str, Any]]:
        """Return a copy of the transition history."""
        return list(self._history)

    def transition(self, new_state: NegotiationState) -> None:
        """Attempt to move to ``new_state``.

        Raises:
            InvalidStateTransition: If the transition is not allowed.
        """
        if new_state not in _VALID_TRANSITIONS.get(self._state, set()):
            raise InvalidStateTransition(
                f"Cannot transition from {self._state.value} to {new_state.value}"
            )
        old = self._state
        self._state = new_state
        self._record_transition(new_state)
        logger.info(
            "Negotiation %s: %s → %s",
            self.negotiation_id,
            old.value,
            new_state.value,
        )

    def can_transition(self, new_state: NegotiationState) -> bool:
        """Check if a transition is allowed without performing it."""
        return new_state in _VALID_TRANSITIONS.get(self._state, set())

    def is_terminal(self) -> bool:
        """Whether the machine is in a terminal state."""
        return self._state in (NegotiationState.TERMINATED,)

    def _record_transition(self, state: NegotiationState) -> None:
        self._history.append({
            "state": state.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "negotiation_id": self.negotiation_id,
        })
