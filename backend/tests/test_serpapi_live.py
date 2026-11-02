"""Optional LIVE test: one real SerpApi Google Shopping search for a three-seater sofa in India.

Skipped by default and always skipped without a key.  To run (it consumes a few SerpApi searches):

    PowerShell:   $env:SMARTNEST_RUN_SERPAPI_TESTS = "1"; python -m pytest -m serpapi_live -s
    bash:         SMARTNEST_RUN_SERPAPI_TESTS=1 python -m pytest -m serpapi_live -s

Assertions are about properties (real URLs, INR prices, nothing invented), not about specific products.
"""

from __future__ import annotations

import os

import pytest

from smartnest.database.repositories.in_memory import InMemoryRepository
from smartnest.integrations.products import ProductQuery
from smartnest.integrations.products.serpapi import (
    SEARCH_CACHE_COLLECTION,
    SerpApiCandidateSource,
    SerpApiError,
    load_serpapi_settings,
    serpapi_configured,
)
from smartnest.optimization import UserRequirements, UtilityOptimizer, validate_record

pytestmark = [
    pytest.mark.serpapi_live,
    pytest.mark.skipif(os.environ.get("SMARTNEST_RUN_SERPAPI_TESTS") != "1",
                       reason="set SMARTNEST_RUN_SERPAPI_TESTS=1 to call the real SerpApi"),
    pytest.mark.skipif(not serpapi_configured(), reason="SERPAPI_API_KEY is not configured in backend/.env"),
]

QUERY = ProductQuery(category="sofa", min_seating=3, max_price=60000)
REQUIREMENTS = UserRequirements(budget=60000, max_width_cm=300, max_depth_cm=150, min_seating=3)


@pytest.fixture(scope="module")
def run():
    source = SerpApiCandidateSource.from_settings(
        load_serpapi_settings(), cache_repository=InMemoryRepository(SEARCH_CACHE_COLLECTION), query=QUERY)
    try:
        result = source.refresh()
    except SerpApiError as exc:  # pragma: no cover
        pytest.fail(f"live search failed: {exc}")
    return source, result


def test_a_real_search_returns_shopping_results(run):
    _, result = run
    assert result.api_calls >= 1
    assert result.extracted_count > 0, f"no shopping results: {result.notes}"
    print("\n" + result.summary())


def test_everything_accepted_is_complete_valid_inr_and_traceable(run):
    _, result = run
    for listing in result.listings:
        assert validate_record(listing)[1] == []
        assert listing["currency"] == "INR" and listing["price"] > 0
        assert listing["url_type"] in ("retailer", "google_shopping", "missing")
        if listing["url_type"] == "retailer":
            assert "google." not in listing["url"].split("/")[2]
        else:
            assert listing["url"] is None  # a Google page is never stored as the retailer URL


def test_everything_incomplete_says_what_is_missing(run):
    _, result = run
    for rejected in result.rejected:
        assert rejected.reasons


def test_a_second_refresh_is_served_from_the_cache(run):
    source, first = run
    second = source.refresh()
    assert second.api_calls == 0 and second.cache_hits >= 1


def test_accepted_listings_enter_the_optimizer(run):
    source, _ = run
    UtilityOptimizer().optimize(source.get_candidates(), REQUIREMENTS)  # must not raise
