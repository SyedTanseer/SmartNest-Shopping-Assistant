"""SerpApi Google Shopping as a product provider and a ``CandidateSource``.

Flow of one discovery run::

    ProductQuery -> search text -> [cache] -> google_shopping search
        -> shortlist (within budget, first ``max_detail_calls``) -> [cache] -> product details
        -> parsing.build_listing -> complete records | rejected-with-reasons

``SerpApiProvider`` is a :class:`ProductProvider`.  :class:`SerpApiCandidateSource` is a
:class:`LiveCandidateSource` wired to it, so it satisfies the planner's ``CandidateSource``
interface unchanged: ``get_candidates()`` serves the last snapshot and **never** touches the
network; only an explicit ``refresh()`` calls SerpApi (through the cache).

Cache behaviour: fresh entry -> no API call; expired or missing -> call SerpApi; if that call
fails and an expired entry exists, the expired data is used but flagged
(``DiscoveryResult.stale``).  Each record's ``verified_at`` is the time the data was *fetched*,
so a cached response is never presented as newly verified.

Gemini is not involved: it neither searches nor extracts products here.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Callable

from smartnest.integrations.products.models import (
    DiscoveryResult,
    ProductDiscoveryError,
    ProductQuery,
    RejectedListing,
    utc_now,
)
from smartnest.integrations.products.serpapi import parsing
from smartnest.integrations.products.serpapi.cache import SearchCache
from smartnest.integrations.products.serpapi.client import SerpApiClient, SerpApiError, SerpApiResponseError
from smartnest.integrations.products.serpapi.config import SerpApiSettings
from smartnest.integrations.products.source import LiveCandidateSource

logger = logging.getLogger(__name__)

PROVIDER_ID = "serpapi_google_shopping"


def shopping_query(query: ProductQuery) -> str:
    """The Google Shopping search text for a query, e.g. ``"3 seater minimalist grey sofa"``."""
    words = []
    if query.min_seating:
        words.append(f"{query.min_seating} seater")
    if query.style:
        words.append(query.style)
    if query.colour:
        words.append(query.colour)
    words.append(query.category)
    return " ".join(words)


class SerpApiProvider:
    """Live product discovery through SerpApi's Google Shopping engine.

    Args:
        client: The HTTP client.
        settings: Result limits, country/language, cache TTL.
        cache: Optional :class:`SearchCache`; without it every discovery calls the API.
        clock: Injectable time source.
    """

    provider_id = PROVIDER_ID

    def __init__(self, client: SerpApiClient, settings: SerpApiSettings, *, cache: SearchCache | None = None,
                 clock: Callable[[], datetime] = utc_now) -> None:
        self._client = client
        self._settings = settings
        self._cache = cache
        self._clock = clock

    # ── Cached fetch ─────────────────────────────────────────────────────

    def _fetch(self, kind: str, params: dict[str, Any], call: Callable[[], dict], result: DiscoveryResult
               ) -> tuple[dict, datetime] | None:
        """Payload and its fetch time, via cache; ``None`` for a failed *optional* lookup is handled by the caller."""
        entry = self._cache.get(kind, params) if self._cache else None
        if entry is not None and entry.fresh:
            result.cache_hits += 1
            return entry.payload, entry.fetched_at
        try:
            payload = call()
        except SerpApiError as exc:
            if entry is not None:
                result.stale = True
                result.notes.append(f"{kind}: SerpApi failed ({exc}); using cached data from "
                                    f"{entry.fetched_at.isoformat()} which is STALE")
                return entry.payload, entry.fetched_at
            raise
        result.api_calls += 1
        fetched = self._cache.put(kind, params, payload) if self._cache else self._clock()
        return payload, fetched

    # ── Discovery ─────────────────────────────────────────────────────────

    def discover(self, query: ProductQuery) -> DiscoveryResult:
        s = self._settings
        text = shopping_query(query)
        result = DiscoveryResult(provider_id=self.provider_id, query=query, retrieved_at=self._clock())
        result.search_queries = [text]
        params = {"engine": "google_shopping", "q": text, "gl": s.country, "hl": s.language}

        try:
            search, fetched_at = self._fetch("search", params, lambda: self._client.search(text), result)
        except SerpApiError as exc:
            raise ProductDiscoveryError(self.provider_id, str(exc)) from None
        result.retrieved_at = fetched_at

        if search.get("_empty"):
            result.notes.append("Google Shopping returned no results for this search")
            return result
        items = search.get("shopping_results")
        if not isinstance(items, list) or not all(isinstance(i, dict) for i in items):
            raise ProductDiscoveryError(self.provider_id, str(SerpApiResponseError(
                "response has no valid 'shopping_results' list")))
        items = items[: s.max_results]
        result.extracted_count = len(items)
        if not items:
            result.notes.append("the search returned an empty shopping_results list")
            return result

        # Which products get a (paid) product-details lookup: within budget, in result order.
        shortlist: set[int] = set()
        for index, item in enumerate(items):
            if len(shortlist) >= s.max_detail_calls:
                break
            price, currency, problems = parsing.parse_price(item.get("price"), item.get("extracted_price"))
            token = item.get("immersive_product_page_token")
            if problems or currency != "INR" or not isinstance(token, str) or not token.strip():
                continue
            if query.max_price is not None and price > query.max_price:
                continue
            shortlist.add(index)

        seen: set[str] = set()
        details_failed = 0
        for index, item in enumerate(items):
            details, detail_time = None, None
            if index in shortlist:
                token = item["immersive_product_page_token"]
                try:
                    fetched = self._fetch("details", {"engine": "google_immersive_product", "page_token": token,
                                                      "gl": s.country, "hl": s.language},
                                          lambda t=token: self._client.product_details(t), result)
                    details, detail_time = fetched
                except SerpApiError as exc:
                    details_failed += 1
                    result.notes.append(f"product details unavailable for {item.get('title')!r}: {exc}")
            built = parsing.build_listing(item, details, retrieved_at=min(fetched_at, detail_time or fetched_at),
                                          provider_id=self.provider_id, category=query.category)
            if built.record is None:
                result.rejected.append(RejectedListing(built.name, built.summary.get("url"), built.reasons,
                                                       details=built.summary))
                continue
            pid = built.record["product_id"]
            if pid in seen:
                result.rejected.append(RejectedListing(built.name, built.record["url"],
                                                       ["duplicate of a listing already accepted"], details=built.summary))
                continue
            seen.add(pid)
            if built.record["url"]:
                result.sources.append(built.record["url"])
            result.listings.append(built.record)
        if details_failed and details_failed == len(shortlist):
            result.notes.append("no product-details lookup succeeded, so retailer offers and specifications are missing")
        logger.info("SerpApi discovery: %d results -> %d accepted, %d rejected (%d API call(s), %d cache hit(s))",
                    len(items), len(result.listings), len(result.rejected), result.api_calls, result.cache_hits)
        return result


class SerpApiCandidateSource(LiveCandidateSource):
    """A ``CandidateSource`` backed by SerpApi Google Shopping.

    Construct directly with a provider, or use :meth:`from_settings`.  ``get_candidates()``
    never calls the network; call :meth:`refresh` to search.
    """

    def __init__(self, provider: SerpApiProvider, query: ProductQuery | None = None, *,
                 max_age: timedelta = timedelta(hours=6), clock: Callable[[], datetime] = utc_now) -> None:
        super().__init__(provider, query, max_age=max_age, clock=clock)

    @classmethod
    def from_settings(cls, settings: SerpApiSettings, *, cache_repository: Any = None, transport: Any = None,
                      query: ProductQuery | None = None) -> "SerpApiCandidateSource":
        client = SerpApiClient(settings, transport=transport)
        ttl = timedelta(hours=settings.cache_ttl_hours)
        cache = SearchCache(cache_repository, ttl=ttl) if cache_repository is not None else None
        return cls(SerpApiProvider(client, settings, cache=cache), query, max_age=ttl)
