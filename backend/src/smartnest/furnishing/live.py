"""On-demand live product search for categories SmartNest has no listings for yet.

Real data is captured once per category (see ``src/capture_catalog.py``).  A request can need a category nobody has
searched, for example a study table for a child's room.  ``LiveSearcher`` runs ONE Google Shopping search for it through
SerpApi, saves the result as a snapshot, and the plan is re-made with it.

Safety (it spends a limited, paid allowance, so it is deliberately hard to trigger by accident):
    * off unless ``SMARTNEST_ALLOW_LIVE_SEARCH=1`` and a SerpApi key is configured;
    * the user must ask for it explicitly (a button / API call); the planner never searches by itself;
    * only categories that are in the current plan can be searched;
    * the usage ledger enforces the lifetime cap (``SERPAPI_LIFETIME_LIMIT``) and refuses when it is used up;
    * each search costs exactly one request (no product-details lookups) and is cached.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any

from smartnest.furnishing.catalog import ORIGIN_REAL, CategoryListings, SnapshotCatalog
from smartnest.furnishing.categories import category
from smartnest.integrations.products import ProductDiscoveryError, ProductQuery
from smartnest.integrations.products.serpapi import (
    SearchCache,
    SerpApiClient,
    SerpApiProvider,
    SerpApiSettings,
)
from smartnest.integrations.products.serpapi.client import SerpApiError, UsageLedger

logger = logging.getLogger(__name__)


class LiveSearchError(Exception):
    """A live search could not be done.  ``code`` is ``quota``, ``disabled`` or ``failed``."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class LiveSearcher:
    def __init__(self, settings: SerpApiSettings, snapshots: SnapshotCatalog, *, cache: SearchCache | None = None,
                 transport: Any = None) -> None:
        self._settings = dataclasses.replace(settings, max_detail_calls=0)
        self._snapshots = snapshots
        self._cache = cache
        self._transport = transport

    # ── Allowance ────────────────────────────────────────────────────────

    def _ledger(self) -> UsageLedger:
        return UsageLedger(self._settings.usage_file, self._settings.lifetime_limit or 10 ** 9)

    @property
    def limit(self) -> int | None:
        return self._settings.lifetime_limit

    @property
    def used(self) -> int:
        return self._ledger().used

    @property
    def remaining(self) -> int | None:
        return None if self.limit is None else max(0, self.limit - self.used)

    def status(self) -> dict[str, Any]:
        return {"enabled": True, "remaining": self.remaining, "limit": self.limit, "used": self.used}

    # ── Search ───────────────────────────────────────────────────────────

    def search(self, category_id: str, themes: tuple[str, ...] = ()) -> CategoryListings:
        profile = category(category_id)
        if self.remaining is not None and self.remaining < 1:
            raise LiveSearchError("quota", f"The SerpApi allowance is used up ({self.used} of {self.limit} requests). "
                                           "Raise SERPAPI_LIFETIME_LIMIT in backend/.env to search again.")
        text = profile.search_query(themes)
        client = SerpApiClient(self._settings, transport=self._transport)
        provider = SerpApiProvider(client, self._settings, cache=self._cache)
        try:
            result = provider.discover(ProductQuery(category=text))
        except (ProductDiscoveryError, SerpApiError) as exc:
            raise LiveSearchError("failed", f"The search did not work: {exc}") from None
        records = []
        for record in result.listings:
            record = dict(record, category=category_id, data_origin=ORIGIN_REAL)
            records.append(record)
        listings = CategoryListings(category_id, records, ORIGIN_REAL, result.retrieved_at.isoformat(), text,
                                    [{"name": r.name, "reasons": r.reasons} for r in result.rejected], list(result.notes))
        if records:                                    # an empty result must not overwrite an existing snapshot
            self._snapshots.save(listings)
        logger.info("live search %r: %d listing(s), %d request(s)", text, len(records), result.api_calls)
        return listings
