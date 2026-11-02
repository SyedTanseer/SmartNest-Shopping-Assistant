"""Optional LIVE tests: they call the real Gemini API with your key.

Skipped by default (and always skipped without a key).  To run them:

    PowerShell:   $env:SMARTNEST_RUN_LIVE_TESTS = "1"; python -m pytest -m live -s
    bash:         SMARTNEST_RUN_LIVE_TESTS=1 python -m pytest -m live -s

They cost a few API calls.  Assertions are deliberately about *properties* (schema-valid output,
URLs that really came from sources, constraints respected), not exact wording or products, because
live answers vary.
"""

from __future__ import annotations

import os

import pytest

from smartnest.integrations.gemini import (
    GeminiService,
    GenaiClient,
    gemini_configured,
    load_gemini_settings,
)
from smartnest.integrations.products import (
    GeminiSearchProvider,
    HttpImageVerifier,
    ProductDiscoveryError,
    ProductQuery,
    RedirectResolver,
    check_url,
)
from smartnest.optimization import validate_record

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("SMARTNEST_RUN_LIVE_TESTS") != "1",
                       reason="set SMARTNEST_RUN_LIVE_TESTS=1 to call the real Gemini API"),
    pytest.mark.skipif(not gemini_configured(), reason="GEMINI_API_KEY is not configured in backend/.env"),
]


@pytest.fixture(scope="module")
def service() -> GeminiService:
    return GeminiService(GenaiClient(load_gemini_settings()))


class TestLiveStructuredOutputs:
    def test_requirements_are_extracted_into_the_schema(self, service):
        r = service.extract_requirements("I want a minimalist three-seater sofa under 30k rupees for a 2.1 metre wall")
        assert r.sofa_budget == 30000 and r.min_seating == 3 and r.preferred_style == "minimalist"
        assert r.max_width_cm in (None, 210.0)

    def test_a_follow_up_is_interpreted_as_a_style_change(self, service):
        change = service.interpret_change_request("I'd rather have a contemporary look", {"preferred_style": "minimalist"})
        assert change.intent == "modify_requirements"
        assert [(c.field, c.action, c.value) for c in change.changes] == [("preferred_style", "set", "contemporary")]

    def test_preferences_and_constraints(self, service):
        r = service.extract_preferences("I love leather and grey, hate red, and it must be under 25000 rupees")
        assert "leather" in r.preferred_materials and "red" in r.disliked_colours
        assert any(c.field == "price" and c.value == 25000 for c in r.constraints)


class TestLiveDiscovery:
    @pytest.fixture(scope="class")
    def result(self, service):
        provider = GeminiSearchProvider(service, resolver=RedirectResolver(), image_verifier=HttpImageVerifier())
        try:
            return provider.discover(ProductQuery(category="sofa", max_price=30000, min_seating=3, style="minimalist"))
        except ProductDiscoveryError as exc:
            if "429" in str(exc) or "404" in str(exc):  # a quota/availability limit of this key, not a code fault
                pytest.skip(f"Google Search grounding is not available to this key: {exc}")
            raise

    def test_the_search_reports_real_sources(self, result):
        assert result.sources, "Google Search grounding returned no sources"
        assert all(check_url(u) is None for u in result.sources)
        assert not any("vertexaisearch.cloud.google.com" in u for u in result.sources), "redirect links were not resolved"

    def test_every_accepted_listing_is_valid_and_traceable(self, result):
        for listing in result.listings:
            assert validate_record(listing)[1] == []
            assert listing["url"] in result.sources                       # never an invented link
            assert listing["currency"] == "INR" and listing["price"] > 0
            assert listing["verified_at"] == result.retrieved_at.isoformat()

    def test_rejections_are_explained(self, result):
        assert all(r.reasons for r in result.rejected)
        print("\n" + result.summary())
