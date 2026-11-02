"""Candidate sources backed by live product discovery.

The BDI planner depends on one narrow interface, ``CandidateSource``:
``get_candidates() -> list[dict]`` (records in the optimizer's schema).
It calls that method often — once per planning pass and again for each
relevance check — so a live source must **never hit the network from
``get_candidates``**.  :class:`LiveCandidateSource` therefore keeps a
snapshot: it is filled by an explicit :meth:`refresh`, and
``get_candidates`` serves the snapshot, minus listings that are too old
to trust.

Because it satisfies the same interface, it can replace — or be combined
with, via :class:`CompositeCandidateSource` — the existing
``CatalogCandidateSource``; the planner, agent and optimizer are unchanged.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta, timezone
from typing import Callable, Sequence

from smartnest.integrations.products.models import DiscoveryResult, ProductDiscoveryError, ProductQuery, utc_now
from smartnest.integrations.products.provider import ProductProvider

logger = logging.getLogger(__name__)


def _parse_time(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return None


class LiveCandidateSource:
    """A ``CandidateSource`` fed by a :class:`ProductProvider`.

    Args:
        provider: Where listings come from.
        query: Default query for :meth:`refresh`.
        max_age: Listings verified longer ago than this are treated as stale and not offered.
        clock: Injectable time source.
    """

    def __init__(self, provider: ProductProvider, query: ProductQuery | None = None, *,
                 max_age: timedelta = timedelta(hours=6), clock: Callable[[], datetime] = utc_now) -> None:
        if max_age <= timedelta(0):
            raise ValueError("max_age must be positive")
        self._provider = provider
        self._query = query
        self._max_age = max_age
        self._clock = clock
        self._lock = threading.RLock()
        self._listings: list[dict] = []
        self._previous: list[dict] = []
        self._result: DiscoveryResult | None = None
        self._refreshed_at: datetime | None = None
        self._last_error: str | None = None

    # ── Refreshing ───────────────────────────────────────────────────────

    def set_query(self, query: ProductQuery) -> None:
        with self._lock:
            self._query = query

    @property
    def query(self) -> ProductQuery | None:
        with self._lock:
            return self._query

    def refresh(self, query: ProductQuery | None = None) -> DiscoveryResult:
        """Run discovery now and replace the snapshot.

        If the provider fails, the previous snapshot is kept (it will
        age out through ``max_age``), the error is remembered, and
        :class:`ProductDiscoveryError` is raised.
        """
        with self._lock:
            if query is not None:
                self._query = query
            effective = self._query
        if effective is None:
            raise ValueError("no query: pass one to refresh() or set_query() first")
        try:
            result = self._provider.discover(effective)
        except ProductDiscoveryError as exc:
            with self._lock:
                self._last_error = str(exc)
            logger.warning("Product refresh failed: %s", exc)
            raise
        with self._lock:
            self._previous = self._listings
            self._listings = [dict(r) for r in result.listings]
            self._result = result
            self._refreshed_at = self._clock()
            self._last_error = None
        return result

    # ── CandidateSource ─────────────────────────────────────────────────

    def get_candidates(self) -> list[dict]:
        """Fresh listings from the snapshot (no network access)."""
        fresh, _ = self._split()
        return fresh

    # ── Introspection ────────────────────────────────────────────────────

    def _split(self) -> tuple[list[dict], list[dict]]:
        with self._lock:
            listings = [dict(r) for r in self._listings]
        now = self._clock()
        fresh, stale = [], []
        for record in listings:
            verified = _parse_time(record.get("verified_at"))
            (fresh if verified is not None and now - verified <= self._max_age else stale).append(record)
        return fresh, stale

    @property
    def stale_listings(self) -> list[dict]:
        """Listings withheld because their verification time is missing or older than ``max_age``."""
        return self._split()[1]

    @property
    def previous_listings(self) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._previous]

    @property
    def current_listings(self) -> list[dict]:
        """The whole snapshot, including stale listings."""
        with self._lock:
            return [dict(r) for r in self._listings]

    @property
    def last_result(self) -> DiscoveryResult | None:
        with self._lock:
            return self._result

    @property
    def refreshed_at(self) -> datetime | None:
        with self._lock:
            return self._refreshed_at

    @property
    def last_error(self) -> str | None:
        with self._lock:
            return self._last_error


class CompositeCandidateSource:
    """Combines several candidate sources (e.g. live listings plus a local catalogue).

    A source that raises is skipped for that call and recorded in
    :attr:`errors`; the others still contribute.  Duplicate
    ``(source_agent_id, product_id)`` pairs appear once.
    """

    def __init__(self, sources: Sequence[object]) -> None:
        self._sources = list(sources)
        self.errors: list[str] = []

    def get_candidates(self) -> list[dict]:
        self.errors = []
        merged: list[dict] = []
        seen: set[tuple] = set()
        for source in self._sources:
            try:
                records = source.get_candidates()
            except Exception as exc:
                self.errors.append(f"{type(source).__name__}: {type(exc).__name__}")
                logger.warning("Candidate source %s failed: %s", type(source).__name__, type(exc).__name__)
                continue
            for record in records:
                key = (record.get("source_agent_id"), record.get("product_id"))
                if key not in seen:
                    seen.add(key)
                    merged.append(record)
        return merged
