"""Centralized worker manager.

The :class:`WorkerManager` provides a single point of control for
starting, stopping and monitoring all background workers in SmartNest.

Responsibilities:
    - Register workers by their ``worker_id``.
    - Start all workers concurrently.
    - Stop all workers gracefully with configurable join timeout.
    - Track worker status.
    - Collect performance metrics from all workers.
    - Export metrics to CSV.

The manager does **not** know about the agent, rule engine or event
stream — it only manages worker lifecycles, keeping responsibilities
cleanly separated.
"""

from __future__ import annotations

import logging
from typing import Any

from smartnest.workers.background_worker import BackgroundWorker
from smartnest.workers.metrics import WorkerMetrics, export_metrics_csv

logger = logging.getLogger(__name__)


class WorkerManager:
    """Manages the lifecycle of multiple background workers.

    Workers are registered by their ``worker_id``.  Duplicate IDs are
    rejected.  The manager tracks all registered workers and can start,
    stop and collect metrics from them as a group.
    """

    def __init__(self) -> None:
        self._workers: dict[str, BackgroundWorker] = {}

    # ── Registration ──────────────────────────────────────────────────────

    def register(self, worker: BackgroundWorker) -> None:
        """Add a worker to the manager.

        Args:
            worker: The worker to register.

        Raises:
            ValueError: If a worker with the same ID is already registered.
        """
        if worker.worker_id in self._workers:
            raise ValueError(f"Duplicate worker_id: {worker.worker_id}")
        self._workers[worker.worker_id] = worker
        logger.debug("Registered worker %s", worker.worker_id)

    def unregister(self, worker_id: str) -> None:
        """Remove a worker by ID."""
        self._workers.pop(worker_id, None)

    @property
    def workers(self) -> dict[str, BackgroundWorker]:
        """Return a copy of the registered workers."""
        return dict(self._workers)

    @property
    def worker_count(self) -> int:
        """Number of registered workers."""
        return len(self._workers)

    # ── Lifecycle ────────────────────────────────────────────────────────

    def start_all(self) -> None:
        """Start all registered workers concurrently."""
        for worker in self._workers.values():
            worker.start()

    def stop_all(self, timeout: float = 5.0) -> bool:
        """Stop all workers gracefully.

        Signals all workers to stop, then joins each one.  Workers are
        stopped in parallel (signaled first, then joined) to avoid
        serial delays.

        Args:
            timeout: Maximum seconds to wait for each worker to join.

        Returns:
            ``True`` if all workers stopped within the timeout.
        """
        # Signal all workers first (through the public worker API) ...
        for worker in self._workers.values():
            worker.request_stop()

        # ... then join each, so total time ≈ slowest worker, not the sum.
        all_stopped = True
        for worker in self._workers.values():
            if not worker.join(timeout=timeout):
                all_stopped = False

        logger.info("All workers stopped (success=%s)", all_stopped)
        return all_stopped

    # ── Status & metrics ─────────────────────────────────────────────────

    def get_status(self) -> dict[str, str]:
        """Return a mapping of worker_id → status string."""
        return {
            wid: worker.status.value for wid, worker in self._workers.items()
        }

    def get_metrics(self) -> list[WorkerMetrics]:
        """Return the metrics objects for all registered workers."""
        return [w.metrics for w in self._workers.values()]

    def get_metrics_snapshot(self) -> list[dict[str, Any]]:
        """Return a point-in-time metrics snapshot for all workers."""
        return [w.metrics.snapshot() for w in self._workers.values()]

    def export_metrics_csv(self, filepath: str) -> str:
        """Export all worker metrics to a CSV file.

        Args:
            filepath: Output CSV path.

        Returns:
            The path of the written file as a string.
        """
        path = export_metrics_csv(self.get_metrics(), filepath)
        logger.info("Metrics exported to %s", path)
        return str(path)
