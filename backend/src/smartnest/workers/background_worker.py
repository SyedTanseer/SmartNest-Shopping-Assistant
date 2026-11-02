"""Reusable background worker abstraction.

``BackgroundWorker`` provides the infrastructure for running an
independent daemon thread that polls on a configurable interval.  It
handles start/stop, graceful shutdown via ``threading.Event``, exception
handling, execution status tracking and performance metrics.

Subclasses implement :meth:`poll` with the domain-specific monitoring
logic.  The base class is intentionally generic — it does not depend on
SmartNest models, the rule engine, or any specific event type.

Thread-safety:
    - ``_stop_event`` (``threading.Event``) coordinates shutdown.
    - ``_status_lock`` protects the ``_status`` field.
    - The worker thread is a daemon so it won't block process exit, but
      :meth:`stop` also explicitly joins for deterministic shutdown.

Status semantics:
    - ``RUNNING`` while the loop is alive, even if individual polls fail.
      Poll failures are counted (``metrics.poll_errors``) and the latest
      exception is exposed via :attr:`last_error`.
    - ``ERROR`` only when the loop itself crashes and the thread exits.
    - ``STOPPING`` if :meth:`stop` timed out and the thread is still alive.

Future compatibility:
    - Workers are independent components (Assignment 3.1 readiness).
    - They do not block the main application (Assignment 5.2 — FastAPI).
    - They do not implement BDI, utility optimization, or CNP.
"""

from __future__ import annotations

import logging
import threading
from enum import Enum
from time import monotonic
from typing import Any

from smartnest.workers.metrics import WorkerMetrics

logger = logging.getLogger(__name__)


class WorkerStatus(str, Enum):
    """Lifecycle states for a background worker."""

    IDLE = "idle"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class BackgroundWorker:
    """A reusable background daemon worker that polls on a configurable interval.

    Subclasses must implement :meth:`poll`, which is called repeatedly
    until the worker is stopped.  The :meth:`poll` method should be
    short-lived and must check :meth:`should_stop` periodically if it
    performs long operations.

    Args:
        worker_id: Unique identifier for this worker.
        name: Human-readable name.
        poll_interval: Seconds between poll cycles (default 2.0).
    """

    def __init__(
        self,
        worker_id: str,
        name: str | None = None,
        poll_interval: float = 2.0,
    ) -> None:
        self.worker_id = worker_id
        self.name = name or worker_id

        self._poll_interval = max(0.01, poll_interval)
        self._stop_event = threading.Event()
        self._status_lock = threading.Lock()
        self._status: WorkerStatus = WorkerStatus.IDLE
        self._thread: threading.Thread | None = None
        self._metrics = WorkerMetrics(worker_id=worker_id, worker_name=self.name)
        self._last_error: BaseException | None = None
        self.logger = logging.getLogger(f"smartnest.worker.{self.name}")

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def status(self) -> WorkerStatus:
        """Current worker status (thread-safe read)."""
        with self._status_lock:
            return self._status

    @property
    def metrics(self) -> WorkerMetrics:
        """The worker's performance metrics."""
        return self._metrics

    @property
    def last_error(self) -> BaseException | None:
        """The most recent exception raised by :meth:`poll` (or ``None``)."""
        return self._last_error

    @property
    def is_alive(self) -> bool:
        """Whether the worker thread is currently alive."""
        return self._thread is not None and self._thread.is_alive()

    @property
    def poll_interval(self) -> float:
        return self._poll_interval

    @poll_interval.setter
    def poll_interval(self, value: float) -> None:
        self._poll_interval = max(0.01, value)

    # ── Status helpers ──────────────────────────────────────────────────────

    def _set_status(self, status: WorkerStatus) -> None:
        with self._status_lock:
            self._status = status
        self.logger.debug("Status → %s", status.value)

    def should_stop(self) -> bool:
        """Check if the worker has been asked to stop."""
        return self._stop_event.is_set()

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the worker thread.

        Calling ``start`` on an already-running worker is a no-op.
        Calling it after stop is supported (restart).
        """
        if self.status == WorkerStatus.RUNNING:
            self.logger.warning("Worker %s is already running", self.worker_id)
            return
        if self.is_alive:
            # A previous stop() timed out and the old thread is still
            # running.  Starting another would leave two loops polling.
            self.logger.warning(
                "Worker %s cannot restart: previous thread is still alive",
                self.worker_id,
            )
            return

        # Reset for a fresh start / restart
        self._stop_event.clear()
        self._last_error = None
        self._metrics = WorkerMetrics(
            worker_id=self.worker_id, worker_name=self.name
        )
        self._metrics.mark_started()
        self._set_status(WorkerStatus.RUNNING)

        self._thread = threading.Thread(
            target=self._run,
            name=f"worker-{self.worker_id}",
            daemon=True,
        )
        self._thread.start()
        self.logger.info("Worker %s started (interval=%.2fs)", self.worker_id, self._poll_interval)

    def request_stop(self) -> None:
        """Signal the worker to stop without waiting for it.

        Used by :class:`WorkerManager` to signal all workers before joining
        any of them, so shutdown time is the slowest worker, not the sum.
        """
        if self.status in (WorkerStatus.IDLE, WorkerStatus.STOPPED):
            return
        self._set_status(WorkerStatus.STOPPING)
        self._stop_event.set()

    def join(self, timeout: float = 5.0) -> bool:
        """Wait for a stopping worker's thread to exit.

        Returns:
            ``True`` if the thread has exited (status becomes ``STOPPED``),
            ``False`` if it is still alive after ``timeout`` (status stays
            ``STOPPING``).
        """
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                self.logger.warning(
                    "Worker %s did not stop within %.1fs", self.worker_id, timeout
                )
                self._set_status(WorkerStatus.STOPPING)
                return False

        if self.status != WorkerStatus.IDLE:
            self._set_status(WorkerStatus.STOPPED)
        return True

    def stop(self, timeout: float = 5.0) -> bool:
        """Signal the worker to stop and wait for it to finish.

        Args:
            timeout: Maximum seconds to wait for the thread to join.

        Returns:
            ``True`` if the thread joined within the timeout, ``False`` if it
            timed out.
        """
        if self.status in (WorkerStatus.IDLE, WorkerStatus.STOPPED) and not self.is_alive:
            self.logger.debug("Worker %s is not running (status=%s)", self.worker_id, self.status)
            return True

        self.request_stop()
        stopped = self.join(timeout=timeout)
        if stopped:
            self.logger.info("Worker %s stopped", self.worker_id)
        return stopped

    # ── Thread entry point ─────────────────────────────────────────────────

    def _run(self) -> None:
        """Main loop executed in the worker thread.

        Calls :meth:`poll` repeatedly until :meth:`should_stop` returns
        ``True``.  Catches exceptions so a single poll failure doesn't
        crash the worker.
        """
        try:
            while not self.should_stop():
                t0 = monotonic()
                failed = False
                try:
                    self.poll()
                except Exception as exc:
                    # A single failed poll must not kill the worker.  Record
                    # it and keep going; status stays RUNNING.
                    failed = True
                    self._last_error = exc
                    self.logger.exception(
                        "Worker %s poll raised an exception", self.worker_id
                    )
                elapsed = monotonic() - t0
                self._metrics.record_poll(elapsed, failed=failed)

                # Sleep for the remaining interval, but wake early if stopped
                remaining = self._poll_interval - elapsed
                if remaining > 0:
                    self._stop_event.wait(remaining)
        except Exception as exc:
            self._last_error = exc
            self.logger.exception("Worker %s crashed", self.worker_id)
            self._set_status(WorkerStatus.ERROR)

    # ── Publishing helper ──────────────────────────────────────────────────

    def _publish(self, stream: Any, event: Any, timeout: float = 1.0) -> bool:
        """Publish ``event`` via ``stream.emit`` and update event metrics.

        Counts the event as generated, and as published only if ``emit``
        succeeds.  A full bounded stream (or any emit error) is logged and
        shows up as ``events_lost`` instead of blocking the worker forever.

        Returns:
            ``True`` if the event was published.
        """
        self._metrics.record_event_generated()
        try:
            stream.emit(event, timeout=timeout)
        except Exception:
            self.logger.exception("Worker %s failed to publish event", self.worker_id)
            return False
        self._metrics.record_event_published()
        return True

    # ── Domain logic (override in subclasses) ──────────────────────────────

    def poll(self) -> None:
        """Perform one monitoring cycle.

        Subclasses must implement this.  It should be short-lived and
        must not block indefinitely.
        """
        raise NotImplementedError("Subclasses must implement poll()")
