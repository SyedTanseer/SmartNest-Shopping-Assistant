"""Core data models for SmartNest.

These models use Pydantic v2 for validation and serialization.
They are designed to be JSON-serializable so they can later be stored
in MongoDB, transmitted over sockets, or consumed by other agents.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def _utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


def _generate_id(prefix: str = "evt") -> str:
    """Generate a short unique identifier with a human-readable prefix."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# ── Enums ────────────────────────────────────────────────────────────────────

class EventType(str, Enum):
    """Categories of environmental events the simulation can emit."""

    BUDGET_UPDATE = "budget_update"
    FURNITURE_COMPATIBILITY = "furniture_compatibility"
    PRODUCT_AVAILABILITY = "product_availability"
    MISSING_INFORMATION = "missing_information"
    PRODUCT_VALIDATION = "product_validation"
    PRICE_CHANGE = "price_change"
    # Assignment 4.1: changes the user (not the market) makes to the request.
    ROOM_UPDATE = "room_update"
    PREFERENCE_UPDATE = "preference_update"


class ActionResultType(str, Enum):
    """Outcome categories for executed actions."""

    WARNING = "warning"
    REJECTION = "rejection"
    FLAG = "flag"
    REQUEST = "request"
    APPROVAL = "approval"
    INFO = "info"
    ERROR = "error"


# ── Environmental Event ───────────────────────────────────────────────────────

class EnvironmentalEvent(BaseModel):
    """A structured observation from the environment.

    Environmental events describe what the simulated world observes.
    They are *not* inter-agent messages.  Future inter-agent communication
    will use a separate MessageEnvelope (see below).
    """

    event_id: str = Field(default_factory=lambda: _generate_id("evt"))
    event_type: EventType
    source: str = Field(default="environment", description="Origin of the event")
    timestamp: datetime = Field(default_factory=_utc_now)
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured data specific to the event type",
    )

    model_config = {
        "use_enum_values": True,
    }


# ── Action Execution Log ─────────────────────────────────────────────────────

class ActionLog(BaseModel):
    """A log entry produced when an agent executes an action.

    Captures the full chain: event → rule → action → result.
    """

    log_id: str = Field(default_factory=lambda: _generate_id("log"))
    event_id: str
    rule_id: str
    rule_name: str
    action_description: str
    result_type: ActionResultType
    result_message: str
    result_details: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=_utc_now)

    model_config = {
        "use_enum_values": True,
    }


# ── Message Envelope (future inter-agent communication) ──────────────────────
# Defined here so future assignments can use it without refactoring core models.
# Not used in Assignment 1.2 but kept for forward compatibility.

class MessageType(str, Enum):
    """Message types for inter-agent communication.

    Includes types needed for Contract Net Protocol (cfp, proposal, award,
    etc.) so the schema is ready for future assignments.
    """

    # Basic messaging
    INFORM = "inform"
    REQUEST = "request"
    QUERY = "query"

    # State synchronisation & task dispatch (Assignment 2.2)
    STATE_SYNC = "state_sync"
    TASK_REQUEST = "task_request"
    ACK = "ack"

    # Contract Net Protocol
    CFP = "call_for_proposals"
    PROPOSAL = "proposal"
    REJECT_PROPOSAL = "reject_proposal"
    ACCEPT_PROPOSAL = "accept_proposal"
    AWARD = "award"
    INFORM_DONE = "inform_done"
    INFORM_RESULT = "inform_result"

    # Failure / cancellation
    FAILURE = "failure"
    CANCEL = "cancel"


class MessageEnvelope(BaseModel):
    """Envelope for future inter-agent messages.

    Separate from EnvironmentalEvent.  Compatible serialization format
    so both can be transmitted over sockets and stored in MongoDB.
    """

    message_id: str = Field(default_factory=lambda: _generate_id("msg"))
    sender: str
    receiver: str
    message_type: MessageType
    task_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=_utc_now)
    schema_version: str = "1.0"
    signature: str | None = Field(
        default=None,
        description="Ed25519 signature placeholder — not used in Assignment 1.2",
    )

    model_config = {
        "use_enum_values": True,
    }
