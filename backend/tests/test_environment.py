"""Tests for the MockEnvironment and EventStream.

Verify the environment emits the correct five events and the stream
delivers them in order.
"""

from __future__ import annotations

import pytest

from smartnest.core.events import EventStream
from smartnest.core.models import EventType
from smartnest.simulation.environment import MockEnvironment


class TestMockEnvironment:
    def test_emits_five_events(self):
        stream = EventStream()
        env = MockEnvironment(stream)
        env.emit_all()
        events = list(stream.iterate(timeout=0.5))
        assert len(events) == 5

    def test_events_in_correct_order(self):
        stream = EventStream()
        env = MockEnvironment(stream)
        env.emit_all()

        types = [e.event_type for e in stream.iterate(timeout=0.5)]
        assert types == [
            EventType.BUDGET_UPDATE,
            EventType.FURNITURE_COMPATIBILITY,
            EventType.PRODUCT_AVAILABILITY,
            EventType.MISSING_INFORMATION,
            EventType.PRODUCT_VALIDATION,
        ]

    def test_event_ids_are_sequential(self):
        stream = EventStream()
        env = MockEnvironment(stream)
        env.emit_all()
        ids = [e.event_id for e in stream.iterate(timeout=0.5)]
        assert ids == ["evt_001", "evt_002", "evt_003", "evt_004", "evt_005"]

    def test_event_payloads_are_populated(self):
        stream = EventStream()
        env = MockEnvironment(stream)
        env.emit_all()
        for event in stream.iterate(timeout=0.5):
            assert event.payload, f"Event {event.event_id} has empty payload"

    def test_budget_event_payload(self):
        stream = EventStream()
        env = MockEnvironment(stream)
        env.emit_all()
        events = list(stream.iterate(timeout=0.5))
        budget_event = events[0]
        assert budget_event.payload["total_budget"] == 200_000
        assert budget_event.payload["estimated_expenditure"] == 218_000


class TestEventStream:
    def test_fifo_order(self):
        stream = EventStream()
        e1 = MockEnvironment._assignment_1_2_events()[0]
        e2 = MockEnvironment._assignment_1_2_events()[1]
        stream.emit(e1)
        stream.emit(e2)
        assert stream.next_event().event_id == e1.event_id
        assert stream.next_event().event_id == e2.event_id

    def test_next_event_empty_returns_none(self):
        stream = EventStream()
        assert stream.next_event(timeout=0.1) is None

    def test_clear(self):
        stream = EventStream()
        env = MockEnvironment(stream)
        env.emit_all()
        assert stream.size > 0
        stream.clear()
        assert stream.size == 0
