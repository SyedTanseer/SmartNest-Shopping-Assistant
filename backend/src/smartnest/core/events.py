"""Event stream and event bus utilities.

Provides a thread-safe event stream that the environment writes to and
agents read from.  The implementation uses Python's ``queue.Queue`` so it
is ready for background-worker concurrency in future assignments without
changing the interface.
"""

from __future__ import annotations

import logging
from queue import Empty, Full, Queue  # noqa: F401  (Full re-exported for producers)
from typing import Iterator

from smartnest.core.models import EnvironmentalEvent

logger = logging.getLogger(__name__)


class EventStream:
    """A thread-safe FIFO stream of environmental events.

    Wraps ``queue.Queue`` so the same interface works in both synchronous
    single-threaded (Assignment 1.2) and future threaded scenarios.
    """

    def __init__(self, maxsize: int = 0) -> None:
        """Create a new EventStream.

        Args:
            maxsize: Maximum queue size.  0 means unbounded.
        """
        self._queue: Queue[EnvironmentalEvent] = Queue(maxsize=maxsize)

    # ── Producer side ──────────────────────────────────────────────────────

    def emit(self, event: EnvironmentalEvent, timeout: float | None = None) -> None:
        """Push an event onto the stream.

        Args:
            event: The event to publish.
            timeout: For a bounded stream, the maximum seconds to wait for
                free space.  ``None`` blocks until space is available.

        Raises:
            queue.Full: If the stream is bounded, full, and ``timeout``
                elapsed.  Producers (e.g. workers) catch this and count the
                event as lost instead of blocking forever.
        """
        logger.debug("EventStream emitting %s: %s", event.event_id, event.event_type)
        self._queue.put(event, timeout=timeout)

    # ── Consumer side ─────────────────────────────────────────────────────

    def next_event(self, timeout: float | None = None) -> EnvironmentalEvent | None:
        """Return the next event or ``None`` if the stream is empty."""
        try:
            return self._queue.get(timeout=timeout)
        except Empty:
            return None

    def iterate(self, timeout: float = 0.5) -> Iterator[EnvironmentalEvent]:
        """Iterate over events until the stream is drained.

        Blocks briefly waiting for new events, then stops when no more
        arrive within the timeout window.
        """
        while True:
            event = self.next_event(timeout=timeout)
            if event is None:
                break
            yield event

    @property
    def size(self) -> int:
        """Approximate number of events currently in the stream."""
        return self._queue.qsize()

    def clear(self) -> None:
        """Remove all pending events."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except Empty:
                break
