"""Tests for the reusable BackgroundWorker abstraction.

Uses a dummy worker to test the generic infrastructure: start/stop,
restart, status tracking, metrics collection, exception handling,
and graceful shutdown.
"""

from __future__ import annotations

import threading
import time

import pytest

from smartnest.workers.background_worker import BackgroundWorker, WorkerStatus


class DummyWorker(BackgroundWorker):
    """A minimal worker for testing infrastructure."""

    def __init__(self, worker_id="dummy", name=None, poll_interval=0.05):
        super().__init__(worker_id=worker_id, name=name, poll_interval=poll_interval)
        self.poll_count = 0
        self.poll_lock = threading.Lock()
        self._raise_on_next = False

    def poll(self):
        with self.poll_lock:
            self.poll_count += 1
        if self._raise_on_next:
            self._raise_on_next = False
            raise RuntimeError("Simulated poll failure")


# ── Tests: Status lifecycle ─────────────────────────────────────────────────

class TestWorkerStatus:
    def test_initial_status_is_idle(self):
        w = DummyWorker()
        assert w.status == WorkerStatus.IDLE

    def test_status_running_after_start(self):
        w = DummyWorker(poll_interval=0.05)
        w.start()
        try:
            assert w.status == WorkerStatus.RUNNING
        finally:
            w.stop(timeout=2.0)

    def test_status_stopped_after_stop(self):
        w = DummyWorker(poll_interval=0.05)
        w.start()
        time.sleep(0.1)
        w.stop(timeout=2.0)
        assert w.status == WorkerStatus.STOPPED


# ── Tests: Start / Stop ─────────────────────────────────────────────────────

class TestWorkerStartStop:
    def test_start_runs_poll(self):
        w = DummyWorker(poll_interval=0.02)
        w.start()
        time.sleep(0.15)
        w.stop(timeout=2.0)
        assert w.poll_count >= 1

    def test_stop_is_idempotent(self):
        w = DummyWorker(poll_interval=0.02)
        w.start()
        time.sleep(0.05)
        assert w.stop(timeout=2.0) is True
        # Calling stop again should be safe
        assert w.stop(timeout=2.0) is True

    def test_double_start_is_noop(self):
        w = DummyWorker(poll_interval=0.02)
        w.start()
        w.start()  # Should not create a second thread
        time.sleep(0.05)
        assert w.stop(timeout=2.0) is True

    def test_restart_after_stop(self):
        w = DummyWorker(poll_interval=0.02)
        w.start()
        time.sleep(0.05)
        w.stop(timeout=2.0)
        assert w.status == WorkerStatus.STOPPED

        # Restart
        w.start()
        time.sleep(0.05)
        assert w.status == WorkerStatus.RUNNING
        w.stop(timeout=2.0)
        assert w.status == WorkerStatus.STOPPED

    def test_worker_thread_is_daemon(self):
        w = DummyWorker(poll_interval=0.02)
        w.start()
        try:
            assert w._thread is not None
            assert w._thread.daemon is True
        finally:
            w.stop(timeout=2.0)


# ── Tests: Exception handling ──────────────────────────────────────────────

class TestWorkerExceptionHandling:
    def test_poll_exception_does_not_crash_worker(self):
        w = DummyWorker(poll_interval=0.02)
        w.start()
        time.sleep(0.05)

        # Inject a failure
        w._raise_on_next = True
        time.sleep(0.1)

        # Worker should still be running (exception caught, not fatal)
        assert w.status == WorkerStatus.RUNNING
        w.stop(timeout=2.0)

    def test_poll_exception_is_recorded(self):
        """A failed poll is visible via poll_errors and last_error."""
        w = DummyWorker(poll_interval=0.02)
        w._raise_on_next = True
        w.start()
        time.sleep(0.1)
        w.stop(timeout=2.0)
        assert w.metrics.poll_errors == 1
        assert isinstance(w.last_error, RuntimeError)
        assert w.metrics.snapshot()["poll_errors"] == 1


class BlockingWorker(BackgroundWorker):
    """A worker whose poll blocks until released (ignores stop requests)."""

    def __init__(self):
        super().__init__(worker_id="blocking", poll_interval=0.01)
        self.release = threading.Event()
        self.polls = 0

    def poll(self):
        self.polls += 1
        self.release.wait(5.0)


class TestStuckWorker:
    def test_stop_timeout_leaves_status_stopping(self):
        w = BlockingWorker()
        w.start()
        time.sleep(0.05)
        assert w.stop(timeout=0.1) is False
        assert w.status == WorkerStatus.STOPPING
        w.release.set()
        assert w.join(timeout=2.0) is True
        assert w.status == WorkerStatus.STOPPED

    def test_restart_refused_while_old_thread_alive(self):
        w = BlockingWorker()
        w.start()
        time.sleep(0.05)
        w.stop(timeout=0.1)
        old_thread = w._thread
        w.start()  # must not spawn a second polling thread
        assert w._thread is old_thread
        w.release.set()
        w.join(timeout=2.0)


# ── Tests: Metrics ──────────────────────────────────────────────────────────

class TestWorkerMetrics:
    def test_execution_count_increments(self):
        w = DummyWorker(poll_interval=0.02)
        w.start()
        time.sleep(0.15)
        w.stop(timeout=2.0)
        assert w.metrics.execution_count >= 1

    def test_poll_durations_recorded(self):
        w = DummyWorker(poll_interval=0.02)
        w.start()
        time.sleep(0.15)
        w.stop(timeout=2.0)
        snap = w.metrics.snapshot()
        assert snap["average_poll_duration_s"] >= 0.0
        assert snap["max_poll_duration_s"] >= snap["average_poll_duration_s"]

    def test_uptime_recorded(self):
        w = DummyWorker(poll_interval=0.02)
        w.start()
        time.sleep(0.1)
        w.stop(timeout=2.0)
        snap = w.metrics.snapshot()
        assert snap["uptime_s"] > 0.0

    def test_metrics_snapshot_keys(self):
        w = DummyWorker(poll_interval=0.02)
        w.start()
        time.sleep(0.05)
        w.stop(timeout=2.0)
        snap = w.metrics.snapshot()
        expected_keys = {
            "worker_id", "worker_name", "execution_count", "poll_errors",
            "average_poll_duration_s", "max_poll_duration_s",
            "events_generated", "events_published", "events_lost",
            "last_detection_latency_s", "avg_detection_latency_s",
            "max_detection_latency_s", "uptime_s",
        }
        assert set(snap.keys()) == expected_keys


# ── Tests: Poll interval ────────────────────────────────────────────────────

class TestWorkerPollInterval:
    def test_poll_interval_setter(self):
        w = DummyWorker(poll_interval=1.0)
        assert w.poll_interval == 1.0
        w.poll_interval = 0.5
        assert w.poll_interval == 0.5

    def test_poll_interval_minimum(self):
        w = DummyWorker(poll_interval=0.001)
        assert w.poll_interval >= 0.01

    def test_stop_during_sleep_wakes_quickly(self):
        """Worker should wake from sleep quickly when stop is called."""
        w = DummyWorker(poll_interval=5.0)  # Long sleep
        w.start()
        time.sleep(0.1)
        t0 = time.monotonic()
        w.stop(timeout=3.0)
        elapsed = time.monotonic() - t0
        # Should stop much faster than the 5s poll interval
        assert elapsed < 2.0
