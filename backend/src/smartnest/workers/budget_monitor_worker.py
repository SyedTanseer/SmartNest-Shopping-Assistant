"""Budget monitoring background worker.

Monitors the overall furnishing budget by reading from a budget state
provider.  Detects:
    - Expenditure exceeding the budget (budget exceeded)
    - Expenditure crossing a threshold (e.g. 90% of budget)
    - Expenditure falling back within budget

When a change is detected, the worker publishes an
:class:`EnvironmentalEvent` to the shared :class:`EventStream`.

The worker does **not** directly invoke the agent or the rule engine.

Future compatibility:
    The ``BudgetStateProvider`` abstraction means the budget source
    can later be replaced with a real backend (MongoDB, API) without
    rewriting the worker (Assignment 4.2 pattern).
"""

from __future__ import annotations

import logging
import threading
from time import monotonic

from smartnest.core.events import EventStream
from smartnest.core.models import EnvironmentalEvent, EventType
from smartnest.workers.background_worker import BackgroundWorker

logger = logging.getLogger(__name__)


class BudgetStateProvider:
    """Thread-safe provider for budget state.

    In Assignment 2.1 this is a simple in-memory store.  The interface
    allows future replacement with a real backend without changing
    :class:`BudgetMonitorWorker`.

    Args:
        total_budget: Total budget in INR.
        estimated_expenditure: Current estimated expenditure in INR.
    """

    def __init__(
        self,
        total_budget: float,
        estimated_expenditure: float = 0.0,
    ) -> None:
        self._lock = threading.RLock()
        self._total_budget = total_budget
        self._estimated_expenditure = estimated_expenditure
        self._changed_at: float | None = None

    def get_state(self) -> dict[str, float]:
        """Return a thread-safe snapshot of the budget state."""
        with self._lock:
            return {
                "total_budget": self._total_budget,
                "estimated_expenditure": self._estimated_expenditure,
            }

    def get_change_time(self) -> float | None:
        """``time.monotonic()`` instant of the last change, or ``None``."""
        with self._lock:
            return self._changed_at

    def set_total_budget(self, total: float) -> None:
        """Update the total budget."""
        with self._lock:
            if total != self._total_budget:
                self._total_budget = total
                self._changed_at = monotonic()

    def set_estimated_expenditure(self, expenditure: float) -> None:
        """Update the estimated expenditure."""
        with self._lock:
            if expenditure != self._estimated_expenditure:
                self._estimated_expenditure = expenditure
                self._changed_at = monotonic()


class BudgetMonitorWorker(BackgroundWorker):
    """Background worker that monitors the overall furnishing budget.

    Args:
        budget_provider: Source of budget state.
        event_stream: The shared event stream to publish events to.
        worker_id: Unique worker identifier.
        name: Human-readable name.
        poll_interval: Seconds between monitoring cycles (default 2.0).
        source: Source name for emitted events (default "budget_monitor").
        threshold_pct: Budget percentage that triggers a threshold warning (default 0.9).
    """

    def __init__(
        self,
        budget_provider: BudgetStateProvider,
        event_stream: EventStream,
        worker_id: str = "budget_monitor",
        name: str | None = None,
        poll_interval: float = 2.0,
        source: str = "budget_monitor",
        threshold_pct: float = 0.9,
    ) -> None:
        super().__init__(
            worker_id=worker_id,
            name=name or "BudgetMonitorWorker",
            poll_interval=poll_interval,
        )
        self._budget_provider = budget_provider
        self._stream = event_stream
        self._source = source
        self._threshold_pct = threshold_pct

        # Last-known state for diffing — only accessed from worker thread
        self._last_state: dict[str, float] | None = None
        self._last_warning_level: str = "none"  # "none", "threshold", "exceeded"

    # ── BackgroundWorker implementation ─────────────────────────────────

    def poll(self) -> None:
        """Perform one monitoring cycle.

        Compares current budget state to the last-known state and publishes
        events for transitions (e.g. from "within budget" to "exceeded").
        """
        cycle_t0 = monotonic()

        current = self._budget_provider.get_state()
        total = current["total_budget"]
        estimated = current["estimated_expenditure"]

        # First poll: snapshot only, no events
        if self._last_state is None:
            self._last_state = dict(current)
            self._last_warning_level = self._compute_warning_level(total, estimated)
            self.logger.info(
                "Initial budget snapshot: budget ₹%s, expenditure ₹%s",
                total,
                estimated,
            )
            return

        new_level = self._compute_warning_level(total, estimated)

        # Publish only when the budget numbers changed.  (The warning level
        # is derived from the numbers, so it can't change on its own.)
        if current != self._last_state:
            self._publish_budget_event(total, estimated, new_level)
            changed_at = self._budget_provider.get_change_time()
            reference = changed_at if changed_at is not None else cycle_t0
            self._metrics.record_detection_latency(monotonic() - reference)

        self._last_state = dict(current)
        self._last_warning_level = new_level

    # ── Helpers ──────────────────────────────────────────────────────────

    def _compute_warning_level(self, total: float, estimated: float) -> str:
        """Determine the current warning level.

        Returns one of: ``"none"``, ``"threshold"``, ``"exceeded"``.
        """
        if estimated > total:
            return "exceeded"
        if estimated >= self._threshold_pct * total:
            return "threshold"
        return "none"

    def _publish_budget_event(
        self, total: float, estimated: float, warning_level: str
    ) -> None:
        """Publish a BUDGET_UPDATE event to the stream.

        The event payload always contains ``total_budget`` and
        ``estimated_expenditure``.  The existing rules in the rule engine
        will evaluate which action to take (budget exceeded, threshold
        warning, etc.).
        """
        event = EnvironmentalEvent(
            event_type=EventType.BUDGET_UPDATE,
            source=self._source,
            payload={
                "total_budget": total,
                "estimated_expenditure": estimated,
                "warning_level": warning_level,
            },
        )
        if not self._publish(self._stream, event):
            return

        self.logger.info(
            "Budget update: budget ₹%s, expenditure ₹%s (level=%s)",
            total,
            estimated,
            warning_level,
        )

    # ── Introspection ─────────────────────────────────────────────────────

    @property
    def last_known_state(self) -> dict[str, float] | None:
        """Return a copy of the last-known budget state."""
        return dict(self._last_state) if self._last_state is not None else None
