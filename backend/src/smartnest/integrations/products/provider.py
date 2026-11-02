"""Product providers.

A provider turns a :class:`ProductQuery` into a :class:`DiscoveryResult` of
optimizer-compatible records.  The :class:`ProductProvider` protocol is
the extension point: official retailer APIs (when credentials and access
exist) plug in here beside the Gemini-grounded provider.

``GeminiSearchProvider`` — live discovery in three stages
---------------------------------------------------------
1. **Search** — one Gemini call with Google Search grounding (and, when
   enabled, the URL-context tool so the model reads the product pages).
   The answer is free text plus *source metadata*.
2. **Extract** — a second, tool-free Gemini call turns that text into
   structured listings.  The model may only use URLs from the sources the
   search actually reported.
3. **Normalize** — each listing is validated by
   :func:`~smartnest.integrations.products.normalizer.normalize_listing`:
   incomplete or inconsistent listings are rejected with reasons; prices
   must appear in the retrieved text; URLs must be reported sources.

The result carries a retrieval timestamp on every record, so staleness can
be judged later.  If the search reports no sources, **no listings are
produced** — the provider never falls back to the model's own knowledge.

Honest limits: the prose a grounded model writes comes from the search
results but is still model output; checks 2 and 3 (URL provenance, price
present in the text, full-record validation, optimizer constraints)
bound what can slip through, they do not make a listing guaranteed
correct.  Every record keeps its source URL so a person can verify it.
"""

from __future__ import annotations

import logging
from typing import Callable, Protocol

from smartnest.integrations.gemini import prompts
from smartnest.integrations.gemini.errors import GeminiError
from smartnest.integrations.gemini.service import GeminiService
from smartnest.integrations.products.models import (
    DiscoveryResult,
    ProductDiscoveryError,
    ProductQuery,
    RejectedListing,
    utc_now,
)
from smartnest.integrations.products.normalizer import NormalizedListing, normalize_listing
from smartnest.integrations.products.url_validation import (
    AllowedUrls,
    ImageVerifier,
    RedirectResolver,
    UrlResolver,
)

logger = logging.getLogger(__name__)


class ProductProvider(Protocol):
    """Anything that can discover products for a query."""

    provider_id: str

    def discover(self, query: ProductQuery) -> DiscoveryResult: ...


class GeminiSearchProvider:
    """Live product discovery through Gemini with Google Search grounding.

    Args:
        service: The :class:`GeminiService` (its client makes the grounded call).
        resolver: Resolves Google grounding redirect links to real page URLs.
        image_verifier: Optional check that an image URL really serves an image.
        inspect_pages: Also enable the URL-context tool so pages are read, not just snippets.
        provider_id: Recorded as ``source_agent_id`` on every record.
        clock: Injectable time source.
    """

    def __init__(
        self,
        service: GeminiService,
        *,
        resolver: UrlResolver | None = None,
        image_verifier: ImageVerifier | None = None,
        inspect_pages: bool = True,
        provider_id: str = "gemini_search",
        clock: Callable = utc_now,
    ) -> None:
        self._service = service
        self._resolver = resolver or RedirectResolver()
        self._images = image_verifier
        self._inspect_pages = inspect_pages
        self.provider_id = provider_id
        self._clock = clock

    def discover(self, query: ProductQuery) -> DiscoveryResult:
        retrieved_at = self._clock()
        result = DiscoveryResult(provider_id=self.provider_id, query=query, retrieved_at=retrieved_at)

        # 1. Grounded search ------------------------------------------------
        try:
            response = self._service.client.generate(
                prompts.search_prompt(query.describe()),
                system=prompts.SYSTEM_SEARCH,
                grounding=True,
                url_context=self._inspect_pages,
                temperature=0.0,
            )
        except GeminiError as exc:
            raise ProductDiscoveryError(self.provider_id, f"search failed: {exc}") from None
        result.search_queries = list(response.search_queries)

        allowed = AllowedUrls()
        skipped = 0
        for source in response.sources:
            if not allowed.add(self._resolver.resolve(source.uri)):
                skipped += 1  # an unresolved Google redirect or an unusable link is not a product URL
        for retrieval in response.retrievals:
            if retrieval.succeeded:
                allowed.add(retrieval.url)
        result.sources = allowed.urls()
        if skipped:
            result.notes.append(f"{skipped} reported source link(s) could not be resolved to a usable page URL")

        if not len(allowed):
            result.notes.append("the search reported no usable sources, so no listings were produced")
            return result
        if not response.text.strip():
            result.notes.append("the search returned no text to extract products from")
            return result

        # 2. Extraction -------------------------------------------------------
        try:
            extraction = self._service.extract_products(response.text, allowed_urls=allowed.urls(),
                                                        limit=query.limit)
        except GeminiError as exc:
            raise ProductDiscoveryError(self.provider_id, f"product extraction failed: {exc}") from None
        result.extracted_count = len(extraction.products)
        result.notes.extend(extraction.notes)

        # 3. Normalization ----------------------------------------------------
        image_ok = self._images.verify if self._images is not None else None
        seen: set[str] = set()
        for item in extraction.products:
            outcome = normalize_listing(
                item, allowed=allowed, provider_id=self.provider_id, retrieved_at=retrieved_at,
                category=query.category, source_text=response.text, image_ok=image_ok)
            if isinstance(outcome, RejectedListing):
                result.rejected.append(outcome)
                continue
            assert isinstance(outcome, NormalizedListing)
            pid = outcome.record["product_id"]
            if pid in seen:
                result.rejected.append(RejectedListing(outcome.record["name"], outcome.record["url"],
                                                       ["duplicate of a listing already accepted"]))
                continue
            seen.add(pid)
            result.listings.append(outcome.record)
        logger.info("Product discovery: %s", result.summary())
        return result
