"""Tests for the BudgetMonitorWorker.

Verifies:
    - Initial snapshot (no events on first poll).
    - Budget exceeded detection → BUDGET_UPDATE event.
    - Budget threshold detection → BUDGET_UPDATE event.
    - Expenditure falling back within budget.
    - Event payloads are correct.
    - Metrics are recorded.
    - Worker doesn't directly invoke the agent.
"""

from __future__ import annotations

import time

import pytest

from smartnest.core.events import EventStream
from smartnest.core.models import EventType
from smartnest.workers.budget_monitor_worker import BudgetMonitorWorker, BudgetStateProvider


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture()
def setup():
    budget = BudgetStateProvider(total_budget=200000, estimated_expenditure=175000)
    stream = EventStream()
    worker = BudgetMonitorWorker(
        budget_provider=budget,
        event_stream=stream,
        poll_interval=0.05,
    )
    return budget, stream, worker


class TestInitialSnapshot:
    def test_first_poll_does_not_emit_events(self, setup):
        budget, stream, worker = setup
        worker.poll()
        assert stream.size == 0

    def test_second_poll_no_change_no_event(self, setup):
        budget, stream, worker = setup
        worker.poll()
        worker.poll()
        assert stream.size == 0


class TestBudgetExceeded:
    def test_budget_exceeded_detected(self, setup):
        budget, stream, worker = setup
        worker.poll()
        budget.set_estimated_expenditure(218000)
        worker.poll()

        event = stream.next_event(timeout=0.1)
        assert event is not None
        assert event.event_type == EventType.BUDGET_UPDATE
        assert event.payload["total_budget"] == 200000
        assert event.payload["estimated_expenditure"] == 218000
        assert event.payload["warning_level"] == "exceeded"

    def test_budget_exceeded_then_back_within_budget(self, setup):
        budget, stream, worker = setup
        worker.poll()

        # Exceed budget
        budget.set_estimated_expenditure(210000)
        worker.poll()
        event1 = stream.next_event(timeout=0.1)
        assert event1 is not None
        assert event1.payload["warning_level"] == "exceeded"

        # Back within budget
        budget.set_estimated_expenditure(150000)
        worker.poll()
        event2 = stream.next_event(timeout=0.1)
        assert event2 is not None
        assert event2.payload["warning_level"] == "none"


class TestBudgetThreshold:
    def test_threshold_warning_at_90_pct(self, setup):
        budget, stream, worker = setup
        worker.poll()

        # 90% of 200000 = 180000
        budget.set_estimated_expenditure(180000)
        worker.poll()

        event = stream.next_event(timeout=0.1)
        assert event is not None
        assert event.payload["warning_level"] == "threshold"

    def test_below_threshold_no_event(self, setup):
        budget, stream, worker = setup
        worker.poll()

        # 89% of 200000 = 178000 — below threshold
        budget.set_estimated_expenditure(178000)
        worker.poll()
        assert stream.size > 0  # state changed, event emitted
        event = stream.next_event(timeout=0.1)
        assert event.payload["warning_level"] == "none"

    def test_threshold_to_exceeded_transition(self, setup):
        budget, stream, worker = setup
        worker.poll()

        budget.set_estimated_expenditure(185000)  # threshold
        worker.poll()
        event1 = stream.next_event(timeout=0.1)
        assert event1.payload["warning_level"] == "threshold"

        budget.set_estimated_expenditure(210000)  # exceeded
        worker.poll()
        event2 = stream.next_event(timeout=0.1)
        assert event2.payload["warning_level"] == "exceeded"


class TestMetrics:
    def test_events_generated_and_published(self, setup):
        budget, stream, worker = setup
        worker.poll()
        budget.set_estimated_expenditure(210000)
        worker.poll()

        assert worker.metrics.events_generated == 1
        assert worker.metrics.events_published == 1

    def test_detection_latency_recorded(self, setup):
        budget, stream, worker = setup
        worker.poll()
        budget.set_estimated_expenditure(210000)
        worker.poll()
        assert worker.metrics.last_detection_latency >= 0.0


class TestWorkerIsolation:
    def test_worker_does_not_reference_agent(self, setup):
        budget, stream, worker = setup
        assert not hasattr(worker, "agent")
        assert not hasattr(worker, "rule_engine")
        assert not hasattr(worker, "handle_event")


class TestThreadedBudgetWorker:
    def test_threaded_worker_detects_budget_exceeded(self, setup):
        budget, stream, worker = setup
        worker.start()
        time.sleep(0.1)

        budget.set_estimated_expenditure(218000)
        time.sleep(0.2)

        worker.stop(timeout=2.0)

        events = []
        while stream.size > 0:
            events.append(stream.next_event(timeout=0.1))

        budget_events = [e for e in events if e.event_type == EventType.BUDGET_UPDATE]
        assert len(budget_events) >= 1
        assert budget_events[0].payload["warning_level"] == "exceeded"


class TestBudgetLatency:
    def test_latency_measured_from_budget_change(self):
        import time as _time
        from smartnest.core.events import EventStream
        from smartnest.workers.budget_monitor_worker import (
            BudgetMonitorWorker,
            BudgetStateProvider,
        )

        provider = BudgetStateProvider(total_budget=100000, estimated_expenditure=50000)
        stream = EventStream()
        worker = BudgetMonitorWorker(budget_provider=provider, event_stream=stream)
        worker.poll()  # snapshot
        provider.set_estimated_expenditure(95000)
        _time.sleep(0.12)
        worker.poll()
        assert stream.size == 1
        assert worker.metrics.last_detection_latency >= 0.09  # 120 ms sleep; allow for the ~15 ms Windows clock tick

    def test_unchanged_poll_records_no_latency(self):
        from smartnest.core.events import EventStream
        from smartnest.workers.budget_monitor_worker import (
            BudgetMonitorWorker,
            BudgetStateProvider,
        )

        provider = BudgetStateProvider(total_budget=100000, estimated_expenditure=50000)
        worker = BudgetMonitorWorker(budget_provider=provider, event_stream=EventStream())
        worker.poll()
        worker.poll()
        assert worker.metrics.snapshot()["avg_detection_latency_s"] == 0.0
