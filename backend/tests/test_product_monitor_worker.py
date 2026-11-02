"""Tests for the ProductMonitorWorker.

Verifies:
    - Initial snapshot (no events on first poll).
    - Price increase detection → PRICE_CHANGE event.
    - Price decrease detection → PRICE_CHANGE event.
    - Availability change → PRODUCT_AVAILABILITY event.
    - Multiple concurrent changes.
    - Event payloads are correct.
    - Metrics are recorded.
    - Worker doesn't directly invoke the agent.
"""

from __future__ import annotations

import time

import pytest

from smartnest.core.events import EventStream
from smartnest.core.models import EventType
from smartnest.workers.catalog_provider import SimulatedCatalogProvider
from smartnest.workers.product_monitor_worker import ProductMonitorWorker


# ── Fixtures ────────────────────────────────────────────────────────────────

INITIAL_PRODUCTS = [
    {"product_name": "Sofa",          "price": 24999, "availability": "in_stock"},
    {"product_name": "Refrigerator",   "price": 32000, "availability": "in_stock"},
    {"product_name": "Washing machine","price": 18500, "availability": "in_stock"},
    {"product_name": "Study table",    "price":  7000, "availability": "in_stock"},
    {"product_name": "Office chair",   "price":  5500, "availability": "in_stock"},
]


@pytest.fixture()
def setup():
    """Create a catalog, stream and worker for each test."""
    catalog = SimulatedCatalogProvider(INITIAL_PRODUCTS)
    stream = EventStream()
    worker = ProductMonitorWorker(
        catalog=catalog,
        event_stream=stream,
        poll_interval=0.05,
    )
    return catalog, stream, worker


class TestInitialSnapshot:
    def test_first_poll_does_not_emit_events(self, setup):
        catalog, stream, worker = setup
        worker.poll()  # First poll = snapshot
        assert stream.size == 0
        assert len(worker.last_known_state) == 5

    def test_second_poll_with_no_changes_emits_nothing(self, setup):
        catalog, stream, worker = setup
        worker.poll()  # snapshot
        worker.poll()  # no changes
        assert stream.size == 0


class TestPriceChange:
    def test_price_increase_detected(self, setup):
        catalog, stream, worker = setup
        worker.poll()  # snapshot
        catalog.update_price("Refrigerator", 34500)
        worker.poll()

        events = []
        while stream.size > 0:
            events.append(stream.next_event(timeout=0.1))

        assert len(events) == 1
        assert events[0].event_type == EventType.PRICE_CHANGE
        assert events[0].payload["product_name"] == "Refrigerator"
        assert events[0].payload["old_price"] == 32000
        assert events[0].payload["new_price"] == 34500
        assert events[0].payload["change"] == 2500

    def test_price_decrease_detected(self, setup):
        catalog, stream, worker = setup
        worker.poll()
        catalog.update_price("Sofa", 22000)
        worker.poll()

        event = stream.next_event(timeout=0.1)
        assert event is not None
        assert event.event_type == EventType.PRICE_CHANGE
        assert event.payload["product_name"] == "Sofa"
        assert event.payload["old_price"] == 24999
        assert event.payload["new_price"] == 22000
        assert event.payload["change"] == -2999

    def test_no_price_change_no_event(self, setup):
        catalog, stream, worker = setup
        worker.poll()
        worker.poll()
        assert stream.size == 0

    def test_multiple_price_changes(self, setup):
        catalog, stream, worker = setup
        worker.poll()
        catalog.update_price("Sofa", 23000)
        catalog.update_price("Refrigerator", 35000)
        worker.poll()

        events = []
        while stream.size > 0:
            events.append(stream.next_event(timeout=0.1))
        assert len(events) == 2
        names = {e.payload["product_name"] for e in events}
        assert names == {"Sofa", "Refrigerator"}


class TestAvailabilityChange:
    def test_product_goes_out_of_stock(self, setup):
        catalog, stream, worker = setup
        worker.poll()
        catalog.update_availability("Washing machine", "out_of_stock")
        worker.poll()

        event = stream.next_event(timeout=0.1)
        assert event is not None
        assert event.event_type == EventType.PRODUCT_AVAILABILITY
        assert event.payload["product_name"] == "Washing machine"
        assert event.payload["availability"] == "out_of_stock"
        assert event.payload["previous_availability"] == "in_stock"

    def test_product_back_in_stock(self, setup):
        catalog, stream, worker = setup
        catalog.update_availability("Office chair", "out_of_stock")
        worker.poll()  # snapshot with out-of-stock item

        catalog.update_availability("Office chair", "in_stock")
        worker.poll()

        event = stream.next_event(timeout=0.1)
        assert event is not None
        assert event.event_type == EventType.PRODUCT_AVAILABILITY
        assert event.payload["availability"] == "in_stock"
        assert event.payload["previous_availability"] == "out_of_stock"


class TestCombinedChanges:
    def test_price_and_availability_change_same_poll(self, setup):
        catalog, stream, worker = setup
        worker.poll()
        catalog.update_price("Refrigerator", 30000)
        catalog.update_availability("Refrigerator", "out_of_stock")
        worker.poll()

        events = []
        while stream.size > 0:
            events.append(stream.next_event(timeout=0.1))
        assert len(events) == 2
        types = {e.event_type for e in events}
        assert EventType.PRICE_CHANGE in types
        assert EventType.PRODUCT_AVAILABILITY in types


class TestMetrics:
    def test_events_generated_and_published_match(self, setup):
        catalog, stream, worker = setup
        worker.poll()
        catalog.update_price("Sofa", 23000)
        worker.poll()

        assert worker.metrics.events_generated == 1
        assert worker.metrics.events_published == 1
        assert worker.metrics.events_lost == 0

    def test_detection_latency_recorded(self, setup):
        catalog, stream, worker = setup
        worker.poll()
        catalog.update_price("Sofa", 23000)
        worker.poll()
        assert worker.metrics.last_detection_latency >= 0.0

    def test_execution_count_increments_when_threaded(self, setup):
        """Execution count is tracked by the _run loop, not by direct poll()."""
        catalog, stream, worker = setup
        worker.start()
        import time
        time.sleep(0.15)
        worker.stop(timeout=2.0)
        assert worker.metrics.execution_count >= 2


class TestWorkerIsolation:
    def test_worker_does_not_reference_agent(self, setup):
        """The worker should only have catalog, stream, source — no agent."""
        catalog, stream, worker = setup
        # Check that the worker has no agent-related attributes
        assert not hasattr(worker, "agent")
        assert not hasattr(worker, "rule_engine")
        assert not hasattr(worker, "handle_event")


class TestThreadedProductWorker:
    def test_threaded_worker_detects_change(self, setup):
        catalog, stream, worker = setup
        worker.start()
        time.sleep(0.1)  # Let first poll run (snapshot)

        catalog.update_price("Refrigerator", 40000)
        time.sleep(0.2)  # Let second poll detect change

        worker.stop(timeout=2.0)

        events = []
        while stream.size > 0:
            events.append(stream.next_event(timeout=0.1))

        price_events = [e for e in events if e.event_type == EventType.PRICE_CHANGE]
        assert len(price_events) >= 1
        assert price_events[0].payload["product_name"] == "Refrigerator"


# ── Removal, real latency and lost events ───────────────────────────────────

class TestRemovalAndLatency:
    def test_removed_product_emits_unavailable_event(self, setup):
        catalog, stream, worker = setup
        worker.poll()  # snapshot
        catalog.remove_product("Study table")
        worker.poll()
        event = stream.next_event(timeout=0.1)
        assert event is not None
        assert event.event_type == EventType.PRODUCT_AVAILABILITY
        assert event.payload["product_name"] == "Study table"
        assert event.payload["availability"] == "unavailable"
        assert event.payload["removed_from_catalog"] is True
        assert "Study table" not in worker.last_known_state

    def test_empty_catalog_then_products_are_baselined_not_reported(self):
        catalog = SimulatedCatalogProvider([])
        stream = EventStream()
        worker = ProductMonitorWorker(catalog=catalog, event_stream=stream)
        worker.poll()  # baseline of an empty catalogue
        catalog.add_product({"product_name": "Lamp", "price": 900, "availability": "in_stock"})
        worker.poll()  # new product → baseline only
        assert stream.size == 0
        catalog.update_price("Lamp", 800)
        worker.poll()
        assert stream.next_event(timeout=0.1).payload["new_price"] == 800

    def test_detection_latency_measures_change_to_publish(self, setup):
        """Latency is measured from the catalogue change, not the poll start."""
        catalog, stream, worker = setup
        worker.poll()  # snapshot
        catalog.update_price("Sofa", 20000)
        time.sleep(0.15)  # change sits undetected for ~150 ms
        worker.poll()
        assert worker.metrics.last_detection_latency >= 0.12  # 150 ms sleep; allow for the ~15 ms Windows clock tick
        assert worker.metrics.max_detection_latency >= 0.12

    def test_full_bounded_stream_counts_lost_event(self):
        catalog = SimulatedCatalogProvider(INITIAL_PRODUCTS)
        stream = EventStream(maxsize=1)
        worker = ProductMonitorWorker(catalog=catalog, event_stream=stream)
        worker.poll()  # snapshot
        catalog.update_price("Sofa", 1)
        catalog.update_price("Refrigerator", 2)
        # Shorten the publish timeout so the test stays fast.
        original = worker._publish
        worker._publish = lambda s, e, timeout=1.0: original(s, e, timeout=0.05)
        worker.poll()
        assert worker.metrics.events_generated == 2
        assert worker.metrics.events_published == 1
        assert worker.metrics.events_lost == 1
