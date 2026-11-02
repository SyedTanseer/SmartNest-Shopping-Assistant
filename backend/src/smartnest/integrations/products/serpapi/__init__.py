"""SerpApi Google Shopping product discovery (primary live provider).

    config.py    SerpApiSettings / load_serpapi_settings (SERPAPI_API_KEY in backend/.env)
    client.py    httpx client, 401/429/timeout/response-shape handling
    parsing.py   missing-data policy, price/URL/spec parsing, records for the optimizer
    cache.py     MongoDB-backed response cache with fresh/stale flags
    provider.py  SerpApiProvider (ProductProvider) and SerpApiCandidateSource (CandidateSource)
"""

from smartnest.integrations.products.serpapi.cache import SEARCH_CACHE_COLLECTION, SearchCache, cache_key
from smartnest.integrations.products.serpapi.client import (
    SerpApiAuthError,
    SerpApiClient,
    SerpApiError,
    SerpApiQuotaError,
    SerpApiResponseError,
    SerpApiTimeoutError,
)
from smartnest.integrations.products.serpapi.config import (
    SerpApiConfigError,
    SerpApiSettings,
    load_serpapi_settings,
    serpapi_configured,
)
from smartnest.integrations.products.serpapi.parsing import classify_url, parse_price
from smartnest.integrations.products.serpapi.provider import (
    PROVIDER_ID,
    SerpApiCandidateSource,
    SerpApiProvider,
    shopping_query,
)

__all__ = [
    "PROVIDER_ID", "SEARCH_CACHE_COLLECTION", "SearchCache", "SerpApiAuthError", "SerpApiCandidateSource",
    "SerpApiClient", "SerpApiConfigError", "SerpApiError", "SerpApiProvider", "SerpApiQuotaError",
    "SerpApiResponseError", "SerpApiSettings", "SerpApiTimeoutError", "cache_key", "classify_url",
    "load_serpapi_settings", "parse_price", "serpapi_configured", "shopping_query",
]
