"""Mock environment that emits simulated environmental events.

The ``MockEnvironment`` builds a pre-configured list of events for the
Assignment 1.2 scenario and emits them onto an ``EventStream``.

Events are defined declaratively so they are easy to inspect, modify and
extend.  The environment does *not* call agent functions directly — it
only pushes events onto the stream, maintaining clean separation between
the world and the agent.
"""

from __future__ import annotations

import logging

from smartnest.core.events import EventStream
from smartnest.core.models import EnvironmentalEvent, EventType

logger = logging.getLogger(__name__)


class MockEnvironment:
    """A simulated environment that emits a predefined set of events.

    Args:
        event_stream: The ``EventStream`` to emit events onto.
    """

    def __init__(self, event_stream: EventStream) -> None:
        self.stream = event_stream

    # ── Event definitions ──────────────────────────────────────────────────

    @staticmethod
    def _assignment_1_2_events() -> list[EnvironmentalEvent]:
        """Return the five events for Assignment 1.2."""

        events: list[EnvironmentalEvent] = []

        # EVENT 1 — Budget exceeded
        events.append(EnvironmentalEvent(
            event_id="evt_001",
            event_type=EventType.BUDGET_UPDATE,
            source="budget_tracker",
            payload={
                "total_budget": 200000,
                "estimated_expenditure": 218000,
            },
        ))

        # EVENT 2 — Furniture incompatibility
        events.append(EnvironmentalEvent(
            event_id="evt_002",
            event_type=EventType.FURNITURE_COMPATIBILITY,
            source="room_analyzer",
            payload={
                "product_name": "Three-seater sofa",
                "product_width": 230,
                "room_width": 210,
            },
        ))

        # EVENT 3 — Product unavailable
        events.append(EnvironmentalEvent(
            event_id="evt_003",
            event_type=EventType.PRODUCT_AVAILABILITY,
            source="product_catalog",
            payload={
                "product_name": "Refrigerator REF_001",
                "availability": "out_of_stock",
            },
        ))

        # EVENT 4 — Missing information
        events.append(EnvironmentalEvent(
            event_id="evt_004",
            event_type=EventType.MISSING_INFORMATION,
            source="room_analyzer",
            payload={
                "missing_field": "room_dimensions",
                "context": "Cannot evaluate furniture fit without room measurements.",
            },
        ))

        # EVENT 5 — Valid product candidate
        events.append(EnvironmentalEvent(
            event_id="evt_005",
            event_type=EventType.PRODUCT_VALIDATION,
            source="product_catalog",
            payload={
                "product_name": "Dining table",
                "product_price": 24999,
                "available_budget": 30000,
                "fits_room": True,
            },
        ))

        return events

    # ── Emission ──────────────────────────────────────────────────────────

    def emit_all(self) -> None:
        """Emit all assignment events onto the stream in order."""
        events = self._assignment_1_2_events()
        for event in events:
            self.stream.emit(event)
            logger.info(
                "Environment emitted event %s (%s)",
                event.event_id,
                event.event_type,
            )
        logger.info("Environment emitted %d events.", len(events))

    def emit(self, event: EnvironmentalEvent) -> None:
        """Emit a single custom event onto the stream."""
        self.stream.emit(event)
        logger.info(
            "Environment emitted event %s (%s)",
            event.event_id,
            event.event_type,
        )

    @property
    def events(self) -> list[EnvironmentalEvent]:
        """Return the list of assignment events (does not emit)."""
        return self._assignment_1_2_events()
