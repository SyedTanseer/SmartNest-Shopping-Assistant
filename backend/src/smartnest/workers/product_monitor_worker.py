"""Product monitoring background worker.

Monitors a product catalogue via a :class:`CatalogProvider` and detects:
    - Price increases
    - Price decreases
    - Products going out of stock
    - Products becoming available again
    - Products disappearing from the catalogue (reported as "unavailable")

When a change is detected, the worker publishes an
:class:`EnvironmentalEvent` to the shared :class:`EventStream`.

Thread safety:
    - The worker reads product state from the provider (which is
      thread-safe via its own lock).
    - The worker keeps a *local snapshot* of the last-known state for
      comparison.  This snapshot is only accessed from the worker's own
      thread, so it doesn't need its own lock.
    - Events are published via ``EventStream.emit`` which is thread-safe
      (``queue.Queue.put``).

The worker does **not** directly invoke the agent or the rule engine.
It only publishes events to the stream, maintaining the decoupled
architecture from Assignment 1.2.
"""

from __future__ import annotations

import logging
from time import monotonic

from smartnest.core.events import EventStream
from smartnest.core.models import EnvironmentalEvent, EventType
from smartnest.workers.background_worker import BackgroundWorker
from smartnest.workers.catalog_provider import CatalogProvider

logger = logging.getLogger(__name__)


class ProductMonitorWorker(BackgroundWorker):
    """Background worker that monitors product prices and availability.

    Args:
        catalog: The product catalogue provider to read from.
        event_stream: The shared event stream to publish events to.
        worker_id: Unique worker identifier.
        name: Human-readable name.
        poll_interval: Seconds between monitoring cycles (default 2.0).
        source: Source name for emitted events (default "product_monitor").
    """

    def __init__(
        self,
        catalog: CatalogProvider,
        event_stream: EventStream,
        worker_id: str = "product_monitor",
        name: str | None = None,
        poll_interval: float = 2.0,
        source: str = "product_monitor",
    ) -> None:
        super().__init__(
            worker_id=worker_id,
            name=name or "ProductMonitorWorker",
            poll_interval=poll_interval,
        )
        self._catalog = catalog
        self._stream = event_stream
        self._source = source
        # Last-known state for diffing — only accessed from worker thread
        self._last_known: dict[str, dict] = {}
        self._initialized = False

    # ── BackgroundWorker implementation ─────────────────────────────────

    def poll(self) -> None:
        """Perform one monitoring cycle.

        Compares current catalogue state to the last-known snapshot and
        publishes events for any detected changes.  The first cycle only
        records a baseline so the initial state isn't reported as changes.
        """
        cycle_t0 = monotonic()
        current_products = self._catalog.get_all_products()

        if not self._initialized:
            for p in current_products:
                self._last_known[p["product_name"]] = dict(p)
            self._initialized = True
            self.logger.info("Initial snapshot: %d products", len(self._last_known))
            return

        seen: set[str] = set()
        for current in current_products:
            name = current["product_name"]
            seen.add(name)
            previous = self._last_known.get(name)

            if previous is None:
                # New product appeared — baseline it, no change event.
                self._last_known[name] = dict(current)
                continue

            changed = False

            # ── Price changes ─────────────────────────────────────────
            old_price = previous.get("price")
            new_price = current.get("price")
            if old_price is not None and new_price is not None and old_price != new_price:
                self._publish_price_change(name, old_price, new_price)
                changed = True

            # ── Availability changes ──────────────────────────────────
            old_avail = previous.get("availability", "")
            new_avail = current.get("availability", "")
            if old_avail != new_avail:
                self._publish_availability_change(name, old_avail, new_avail)
                changed = True

            if changed:
                self._record_latency(name, cycle_t0)
            self._last_known[name] = dict(current)

        # ── Removed products ──────────────────────────────────────────
        for name in [n for n in self._last_known if n not in seen]:
            previous = self._last_known.pop(name)
            self._publish_availability_change(
                name, previous.get("availability", ""), "unavailable", removed=True
            )
            self._record_latency(name, cycle_t0)

    def _record_latency(self, product_name: str, cycle_t0: float) -> None:
        """Record change → publish latency for one detected change.

        Uses the provider's change timestamp when available; otherwise falls
        back to the time spent in this poll cycle (a lower bound).
        """
        changed_at = self._catalog.get_change_time(product_name)
        reference = changed_at if changed_at is not None else cycle_t0
        self._metrics.record_detection_latency(monotonic() - reference)

    # ── Event publishing ──────────────────────────────────────────────────

    def _publish_price_change(
        self, product_name: str, old_price: float, new_price: float
    ) -> None:
        """Publish a PRICE_CHANGE event to the stream."""
        event = EnvironmentalEvent(
            event_type=EventType.PRICE_CHANGE,
            source=self._source,
            payload={
                "product_name": product_name,
                "old_price": old_price,
                "new_price": new_price,
                "change": new_price - old_price,
            },
        )
        if self._publish(self._stream, event):
            self.logger.info(
                "Price change detected: %s ₹%s → ₹%s", product_name, old_price, new_price
            )

    def _publish_availability_change(
        self,
        product_name: str,
        old_availability: str,
        new_availability: str,
        removed: bool = False,
    ) -> None:
        """Publish a PRODUCT_AVAILABILITY event to the stream."""
        payload = {
            "product_name": product_name,
            "availability": new_availability,
            "previous_availability": old_availability,
        }
        if removed:
            payload["removed_from_catalog"] = True
        event = EnvironmentalEvent(
            event_type=EventType.PRODUCT_AVAILABILITY,
            source=self._source,
            payload=payload,
        )
        if self._publish(self._stream, event):
            self.logger.info(
                "Availability change: %s %s → %s%s",
                product_name,
                old_availability,
                new_availability,
                " (removed)" if removed else "",
            )

    # ── Introspection ─────────────────────────────────────────────────────

    @property
    def last_known_state(self) -> dict[str, dict]:
        """Return a copy of the last-known product states."""
        return {k: dict(v) for k, v in self._last_known.items()}
