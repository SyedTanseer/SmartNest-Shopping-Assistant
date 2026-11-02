"""Performance metrics for background workers.

Thread-safe collection of execution statistics using ``threading.Lock``
to protect mutable counters.  Uses ``time.monotonic`` for all duration
measurements to avoid wall-clock adjustments affecting results.

Metric definitions:
    - ``execution_count``: poll cycles run (including ones that raised).
    - ``poll_errors``: poll cycles that raised an exception.
    - ``events_generated``: changes detected that should produce an event.
    - ``events_published``: events successfully placed on the stream.
    - ``events_lost``: generated − published (non-zero only if publishing
      failed, e.g. a bounded stream stayed full).
    - ``detection latency``: time from the moment the monitored source
      changed to the moment the corresponding event was published.  When
      a source cannot report change times, the poll-cycle detection time
      is used instead (a lower bound).
"""

from __future__ import annotations

import csv
import threading
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic
from typing import Any


@dataclass(eq=False)
class WorkerMetrics:
    """Thread-safe performance metrics for a single worker.

    All counters are protected by an internal lock.  Duration
    measurements use ``time.monotonic`` for accuracy.  ``eq=False`` keeps
    identity-based equality (comparing locks/counters is meaningless).
    """

    worker_id: str
    worker_name: str

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # Counters
    _execution_count: int = 0
    _poll_errors: int = 0
    _events_generated: int = 0
    _events_published: int = 0

    # Duration tracking (monotonic seconds)
    _total_poll_duration: float = 0.0
    _max_poll_duration: float = 0.0

    # Timing
    _start_time: float = 0.0
    _last_detection_latency: float = 0.0
    _total_detection_latency: float = 0.0
    _max_detection_latency: float = 0.0
    _detection_count: int = 0

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def mark_started(self) -> None:
        """Record the worker's start time."""
        with self._lock:
            self._start_time = monotonic()

    @property
    def uptime(self) -> float:
        """Seconds since the worker started (0 if not started)."""
        with self._lock:
            if self._start_time == 0.0:
                return 0.0
            return monotonic() - self._start_time

    # ── Poll tracking ─────────────────────────────────────────────────────

    def record_poll(self, duration: float, failed: bool = False) -> None:
        """Record the duration of a single poll cycle.

        Args:
            duration: Poll duration in seconds (monotonic).
            failed: Whether the poll raised an exception.
        """
        with self._lock:
            self._execution_count += 1
            if failed:
                self._poll_errors += 1
            self._total_poll_duration += duration
            if duration > self._max_poll_duration:
                self._max_poll_duration = duration

    @property
    def execution_count(self) -> int:
        with self._lock:
            return self._execution_count

    @property
    def poll_errors(self) -> int:
        with self._lock:
            return self._poll_errors

    @property
    def average_poll_duration(self) -> float:
        """Average poll duration in seconds (0 if no polls yet)."""
        with self._lock:
            if self._execution_count == 0:
                return 0.0
            return self._total_poll_duration / self._execution_count

    @property
    def max_poll_duration(self) -> float:
        with self._lock:
            return self._max_poll_duration

    # ── Event tracking ────────────────────────────────────────────────────

    def record_event_generated(self) -> None:
        """Increment the count of events detected/generated."""
        with self._lock:
            self._events_generated += 1

    def record_event_published(self) -> None:
        """Increment the count of events successfully published."""
        with self._lock:
            self._events_published += 1

    def record_detection_latency(self, latency: float) -> None:
        """Record change-to-publish latency for one detected change (seconds)."""
        latency = max(0.0, latency)
        with self._lock:
            self._last_detection_latency = latency
            self._total_detection_latency += latency
            self._detection_count += 1
            if latency > self._max_detection_latency:
                self._max_detection_latency = latency

    @property
    def events_generated(self) -> int:
        with self._lock:
            return self._events_generated

    @property
    def events_published(self) -> int:
        with self._lock:
            return self._events_published

    @property
    def events_lost(self) -> int:
        """Number of events generated but not published (should be 0)."""
        with self._lock:
            return self._events_generated - self._events_published

    @property
    def last_detection_latency(self) -> float:
        with self._lock:
            return self._last_detection_latency

    @property
    def average_detection_latency(self) -> float:
        with self._lock:
            if self._detection_count == 0:
                return 0.0
            return self._total_detection_latency / self._detection_count

    @property
    def max_detection_latency(self) -> float:
        with self._lock:
            return self._max_detection_latency

    # ── Snapshot ─────────────────────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        """Return a point-in-time copy of all metrics as a dictionary."""
        with self._lock:
            return {
                "worker_id": self.worker_id,
                "worker_name": self.worker_name,
                "execution_count": self._execution_count,
                "poll_errors": self._poll_errors,
                "average_poll_duration_s": round(
                    self._total_poll_duration / self._execution_count
                    if self._execution_count
                    else 0.0,
                    6,
                ),
                "max_poll_duration_s": round(self._max_poll_duration, 6),
                "events_generated": self._events_generated,
                "events_published": self._events_published,
                "events_lost": self._events_generated - self._events_published,
                "last_detection_latency_s": round(self._last_detection_latency, 6),
                "avg_detection_latency_s": round(
                    self._total_detection_latency / self._detection_count
                    if self._detection_count
                    else 0.0,
                    6,
                ),
                "max_detection_latency_s": round(self._max_detection_latency, 6),
                "uptime_s": round(
                    monotonic() - self._start_time if self._start_time else 0.0, 6
                ),
            }


# ── CSV export ──────────────────────────────────────────────────────────────

def export_metrics_csv(
    metrics_list: list[WorkerMetrics],
    filepath: str | Path,
) -> Path:
    """Export worker metrics snapshots to a CSV file.

    Args:
        metrics_list: Worker metrics to export.
        filepath: Output CSV path.

    Returns:
        The resolved path of the written file.
    """
    filepath = Path(filepath)
    rows = [m.snapshot() for m in metrics_list]
    if not rows:
        filepath.write_text("")
        return filepath

    fieldnames = list(rows[0].keys())
    with filepath.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return filepath
