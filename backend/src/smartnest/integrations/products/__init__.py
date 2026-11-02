"""Live product discovery (Assignment 4.2).

    provider.py        ProductProvider protocol; GeminiSearchProvider (Google Search grounding + URL context)
    normalizer.py      listing -> optimizer-compatible record, or a rejection with reasons
    url_validation.py  URL checks, source provenance, grounding-link resolution
    source.py          LiveCandidateSource / CompositeCandidateSource  (the CandidateSource interface)
    changes.py         snapshot differences -> EnvironmentalEvents for the BDI agent
    models.py          ProductQuery, DiscoveryResult, RejectedListing

Official retailer APIs would be further ``ProductProvider`` implementations;
none is bundled because none could be verified without credentials (see README).
"""

from smartnest.integrations.products.changes import detect_changes
from smartnest.integrations.products.models import (
    DiscoveryResult,
    ProductDiscoveryError,
    ProductQuery,
    RejectedListing,
)
from smartnest.integrations.products.normalizer import NormalizedListing, normalize_listing, price_in_text
from smartnest.integrations.products.provider import GeminiSearchProvider, ProductProvider
from smartnest.integrations.products.source import CompositeCandidateSource, LiveCandidateSource
from smartnest.integrations.products.url_validation import (
    AllowedUrls,
    HttpImageVerifier,
    RedirectResolver,
    canonical_key,
    check_url,
)

from smartnest.integrations.products.serpapi import (  # noqa: E402  (after the modules it builds on)
    SerpApiCandidateSource,
    SerpApiProvider,
)

__all__ = [
    "SerpApiCandidateSource", "SerpApiProvider",
    "AllowedUrls", "CompositeCandidateSource", "DiscoveryResult", "GeminiSearchProvider", "HttpImageVerifier",
    "LiveCandidateSource", "NormalizedListing", "ProductDiscoveryError", "ProductProvider", "ProductQuery",
    "RedirectResolver", "RejectedListing", "canonical_key", "check_url", "detect_changes",
    "normalize_listing", "price_in_text",
]
