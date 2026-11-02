"""Tests for the WorkerManager.

Verifies registration, start/stop of multiple workers, status tracking,
metrics collection and CSV export.
"""

from __future__ import annotations

import os
import time

import pytest

from smartnest.core.events import EventStream
from smartnest.workers.background_worker import BackgroundWorker, WorkerStatus
from smartnest.workers.budget_monitor_worker import BudgetMonitorWorker, BudgetStateProvider
from smartnest.workers.catalog_provider import SimulatedCatalogProvider
from smartnest.workers.product_monitor_worker import ProductMonitorWorker
from smartnest.workers.worker_manager import WorkerManager


# ── Fixtures ────────────────────────────────────────────────────────────────

class SimpleWorker(BackgroundWorker):
    """Simple worker for manager tests."""
    def poll(self):
        pass


@pytest.fixture()
def manager():
    return WorkerManager()


# ── Registration ─────────────────────────────────────────────────────────────

class TestRegistration:
    def test_register_worker(self, manager):
        w = SimpleWorker(worker_id="w1")
        manager.register(w)
        assert manager.worker_count == 1
        assert "w1" in manager.workers

    def test_duplicate_id_raises(self, manager):
        manager.register(SimpleWorker(worker_id="w1"))
        with pytest.raises(ValueError, match="Duplicate worker_id"):
            manager.register(SimpleWorker(worker_id="w1"))

    def test_unregister(self, manager):
        manager.register(SimpleWorker(worker_id="w1"))
        manager.unregister("w1")
        assert manager.worker_count == 0

    def test_unregister_nonexistent_is_noop(self, manager):
        manager.unregister("nonexistent")
        assert manager.worker_count == 0


# ── Start / Stop ────────────────────────────────────────────────────────────

class TestStartStopAll:
    def test_start_all_starts_all_workers(self, manager):
        w1 = SimpleWorker(worker_id="w1", poll_interval=0.05)
        w2 = SimpleWorker(worker_id="w2", poll_interval=0.05)
        manager.register(w1)
        manager.register(w2)
        manager.start_all()

        try:
            assert w1.status == WorkerStatus.RUNNING
            assert w2.status == WorkerStatus.RUNNING
        finally:
            manager.stop_all(timeout=2.0)

    def test_stop_all_stops_all_workers(self, manager):
        w1 = SimpleWorker(worker_id="w1", poll_interval=0.05)
        w2 = SimpleWorker(worker_id="w2", poll_interval=0.05)
        manager.register(w1)
        manager.register(w2)
        manager.start_all()
        time.sleep(0.1)

        result = manager.stop_all(timeout=2.0)
        assert result is True
        assert w1.status == WorkerStatus.STOPPED
        assert w2.status == WorkerStatus.STOPPED

    def test_get_status(self, manager):
        w1 = SimpleWorker(worker_id="w1", poll_interval=0.05)
        manager.register(w1)
        status = manager.get_status()
        assert status == {"w1": "idle"}

    def test_get_status_after_start(self, manager):
        w1 = SimpleWorker(worker_id="w1", poll_interval=0.05)
        manager.register(w1)
        manager.start_all()
        try:
            assert manager.get_status()["w1"] == "running"
        finally:
            manager.stop_all(timeout=2.0)


# ── Metrics ─────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_get_metrics_returns_all(self, manager):
        w1 = SimpleWorker(worker_id="w1")
        w2 = SimpleWorker(worker_id="w2")
        manager.register(w1)
        manager.register(w2)
        metrics = manager.get_metrics()
        assert len(metrics) == 2

    def test_get_metrics_snapshot(self, manager):
        w1 = SimpleWorker(worker_id="w1")
        manager.register(w1)
        snapshots = manager.get_metrics_snapshot()
        assert len(snapshots) == 1
        assert snapshots[0]["worker_id"] == "w1"

    def test_export_metrics_csv(self, manager, tmp_path):
        w1 = SimpleWorker(worker_id="w1")
        manager.register(w1)
        csv_file = tmp_path / "metrics.csv"
        path = manager.export_metrics_csv(str(csv_file))
        assert os.path.exists(path)
        content = open(path).read()
        assert "worker_id" in content
        assert "w1" in content


# ── Concurrent workers ──────────────────────────────────────────────────────

class TestConcurrentWorkers:
    def test_product_and_budget_workers_concurrently(self):
        """Both workers run concurrently and publish events to the same stream."""
        catalog = SimulatedCatalogProvider([
            {"product_name": "Sofa", "price": 24999, "availability": "in_stock"},
            {"product_name": "Refrigerator", "price": 32000, "availability": "in_stock"},
        ])
        budget = BudgetStateProvider(total_budget=200000, estimated_expenditure=175000)
        stream = EventStream()

        product_worker = ProductMonitorWorker(
            catalog=catalog, event_stream=stream, poll_interval=0.05
        )
        budget_worker = BudgetMonitorWorker(
            budget_provider=budget, event_stream=stream, poll_interval=0.05
        )

        manager = WorkerManager()
        manager.register(product_worker)
        manager.register(budget_worker)
        manager.start_all()

        time.sleep(0.1)  # Initial snapshots

        # Trigger changes
        catalog.update_price("Refrigerator", 35000)
        budget.set_estimated_expenditure(210000)

        time.sleep(0.2)  # Let workers detect

        manager.stop_all(timeout=2.0)

        # Collect events
        events = []
        while stream.size > 0:
            events.append(stream.next_event(timeout=0.1))

        # Both workers should have published events
        event_types = {e.event_type for e in events}
        from smartnest.core.models import EventType
        assert EventType.PRICE_CHANGE in event_types
        assert EventType.BUDGET_UPDATE in event_types

    def test_workers_continue_after_one_exception(self):
        """If one worker hits an exception, the other should keep running."""
        class FlakyWorker(BackgroundWorker):
            def __init__(self):
                super().__init__(worker_id="flaky", poll_interval=0.02)
                self._call = 0

            def poll(self):
                self._call += 1
                if self._call == 2:
                    raise RuntimeError("Flaky failure")

        flaky = FlakyWorker()
        stable = SimpleWorker(worker_id="stable", poll_interval=0.02)
        manager = WorkerManager()
        manager.register(flaky)
        manager.register(stable)

        manager.start_all()
        time.sleep(0.15)

        # Stable worker should still be running despite flaky worker's exception
        assert stable.status == WorkerStatus.RUNNING

        manager.stop_all(timeout=2.0)
