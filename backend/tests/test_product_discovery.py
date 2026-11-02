"""Tests for live product discovery: URLs, normalization, the grounded provider, live sources, change events."""

from __future__ import annotations

import math
import urllib.error
from datetime import datetime, timedelta, timezone
from email.message import Message

import pytest

from smartnest.bdi import BDIAgent, BDIPlanner, CatalogCandidateSource, select_sofa_desire
from smartnest.cnp.product_catalog import catalog_agent_a
from smartnest.core.models import EventType
from smartnest.integrations.gemini import (
    GeminiAPIError,
    GeminiService,
    ProductExtraction,
)
from smartnest.integrations.gemini.testing import ScriptedLLMClient, grounded_response
from smartnest.integrations.products import (
    AllowedUrls,
    CompositeCandidateSource,
    DiscoveryResult,
    GeminiSearchProvider,
    HttpImageVerifier,
    LiveCandidateSource,
    ProductDiscoveryError,
    ProductQuery,
    RedirectResolver,
    RejectedListing,
    canonical_key,
    check_url,
    detect_changes,
    normalize_listing,
    price_in_text,
)
from smartnest.integrations.products.normalizer import NormalizedListing
from smartnest.integrations.products.url_validation import REDIRECT_HOSTS, strip_tracking
from smartnest.optimization import UserRequirements, UtilityOptimizer, validate_record

NOW = datetime(2026, 10, 31, 12, 0, tzinfo=timezone.utc)
URL = "https://shop-a.example.com/p/cozy-sofa"
REQUIREMENTS = UserRequirements(budget=30000, max_width_cm=210, max_depth_cm=110, min_seating=3,
                                preferred_style="minimalist", preferred_colour="neutral")


def allowed(*urls: str) -> AllowedUrls:
    a = AllowedUrls()
    for u in urls or (URL,):
        a.add(u)
    return a


def item(**over) -> ProductExtraction:
    data = dict(name="Cozy Sofa", price=24999, currency="INR", retailer="ShopA", source_url=URL,
                width_cm=200, depth_cm=90, height_cm=85, dimensions_unit="cm", seating_capacity=3,
                material="fabric", colour="grey", style="minimalist", availability="in_stock",
                rating=4.3, review_count=120)
    data.update(over)
    return ProductExtraction(**data)


def normalize(**over):
    kwargs = dict(allowed=allowed(), provider_id="test_provider", retrieved_at=NOW)
    text = over.pop("source_text", None)
    if text is not None:
        kwargs["source_text"] = text
    kwargs.update({k: over.pop(k) for k in list(over) if k in ("allowed", "category", "image_ok", "provider_id")})
    return normalize_listing(item(**over), **kwargs)


# ══ URLs ════════════════════════════════════════════════════════════════════

class TestCheckUrl:
    @pytest.mark.parametrize("url", ["https://shop.example.com/p/1", "https://www.shop.co.in/a/b?c=d#frag",
                                     "https://sub.domain.example.org:8443/x"])
    def test_acceptable(self, url):
        assert check_url(url) is None

    @pytest.mark.parametrize("url,fragment", [
        (None, "missing"), ("", "missing"), ("   ", "missing"), (5, "missing"),
        ("http://shop.example.com/p", "https"), ("ftp://shop.example.com", "https"), ("javascript:alert(1)", "https"),
        ("https://user:pw@shop.example.com/p", "credentials"), ("https://localhost/p", "host"),
        ("https://intranet/p", "host"), ("https://127.0.0.1/p", "IP"), ("https://192.168.0.1/p", "IP"),
        ("https://[::1]/p", "host"), ("https://printer.local/p", "non-public"), ("https://shop.example.com/a b", "whitespace"),
        (" https://shop.example.com/p", "whitespace"), ("https://shop.example.com:99999/p", "port"),
        ("https://" + "a" * 2100 + ".com", "long"),
    ])
    def test_rejected(self, url, fragment):
        problem = check_url(url)
        assert problem is not None and fragment in problem

    def test_tracking_parameters_and_cosmetics_do_not_change_identity(self):
        a = canonical_key("https://www.Shop.com/p/1/?utm_source=x&id=7&gclid=abc#top")
        b = canonical_key("https://shop.com/p/1?id=7")
        assert a == b
        assert canonical_key("https://shop.com/p/1?id=7") != canonical_key("https://shop.com/p/1?id=8")

    def test_strip_tracking(self):
        assert strip_tracking("https://shop.com/p?utm_medium=a&color=red&gclid=1#x") == "https://shop.com/p?color=red"


class TestAllowedUrls:
    def test_only_reported_urls_are_accepted(self):
        a = allowed("https://shop.com/p/1?utm_source=x")
        assert a.lookup("https://www.shop.com/p/1/") == "https://shop.com/p/1"
        assert a.lookup("https://shop.com/p/2") is None
        assert a.lookup("https://evil.example.org/p/1") is None

    def test_unusable_urls_are_not_added(self):
        a = AllowedUrls()
        assert a.add("http://shop.com/p") is False and a.add("https://10.0.0.1/p") is False and len(a) == 0

    def test_lookup_validates_its_input(self):
        assert allowed().lookup("javascript:alert(1)") is None and allowed().lookup(None) is None


class FakeRedirectOpener:
    def __init__(self, routes):
        self.routes = routes
        self.requested: list[str] = []

    def open(self, request, timeout=None):
        url = request.full_url
        self.requested.append(url)
        location = self.routes.get(url)
        if location is None:
            raise urllib.error.HTTPError(url, 404, "not found", Message(), None)
        headers = Message()
        headers["Location"] = location
        raise urllib.error.HTTPError(url, 302, "found", headers, None)


class TestRedirectResolver:
    REDIRECT = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbC123"

    def resolver(self, routes):
        r = RedirectResolver()
        r._opener = FakeRedirectOpener(routes)
        return r

    def test_grounding_links_are_resolved_to_the_real_page(self):
        r = self.resolver({self.REDIRECT: "https://shop.example.com/p/9"})
        assert r.resolve(self.REDIRECT) == "https://shop.example.com/p/9"
        assert r._opener.requested == [self.REDIRECT]  # the destination itself is never contacted

    def test_ordinary_urls_are_never_requested(self):
        r = self.resolver({})
        assert r.resolve("https://shop.example.com/p/9") == "https://shop.example.com/p/9"
        assert r._opener.requested == []

    def test_only_google_redirect_hosts_are_ever_contacted(self):
        assert REDIRECT_HOSTS == {"vertexaisearch.cloud.google.com"}
        r = self.resolver({"https://evil.example.org/x": "https://shop.example.com/p"})
        assert r.resolve("https://evil.example.org/x") == "https://evil.example.org/x" and r._opener.requested == []

    def test_chained_redirects_on_the_redirect_host(self):
        second = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/Second"
        r = self.resolver({self.REDIRECT: second, second: "https://shop.example.com/final"})
        assert r.resolve(self.REDIRECT) == "https://shop.example.com/final"

    def test_a_relative_location_is_joined_to_the_current_url(self):
        joined = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/Rel"
        r = self.resolver({self.REDIRECT: "/grounding-api-redirect/Rel", joined: "https://shop.example.com/p/rel"})
        assert r.resolve(self.REDIRECT) == "https://shop.example.com/p/rel"
        assert r._opener.requested == [self.REDIRECT, joined]

    def test_failures_return_the_original_link(self):
        assert self.resolver({}).resolve(self.REDIRECT) == self.REDIRECT

    def test_redirect_loops_are_bounded(self):
        r = self.resolver({self.REDIRECT: self.REDIRECT})
        assert r.resolve(self.REDIRECT) == self.REDIRECT and len(r._opener.requested) <= 4

    def test_network_errors_return_the_original_link(self):
        class Down:
            def open(self, *a, **k):
                raise OSError("unreachable")
        r = RedirectResolver()
        r._opener = Down()
        assert r.resolve(self.REDIRECT) == self.REDIRECT


class TestImageVerifier:
    def test_non_https_is_never_requested(self, monkeypatch):
        called = []
        monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: called.append(a))
        assert HttpImageVerifier().verify("http://cdn.example.com/a.jpg") is False and called == []

    def test_image_content_type_is_required(self, monkeypatch):
        class Resp:
            status = 200
            def __init__(self, ctype):
                self.headers = Message()
                self.headers["Content-Type"] = ctype
            def __enter__(self): return self
            def __exit__(self, *a): return False
        monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: Resp("image/jpeg"))
        assert HttpImageVerifier().verify("https://cdn.example.com/a.jpg") is True
        monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: Resp("text/html"))
        assert HttpImageVerifier().verify("https://cdn.example.com/a.jpg") is False

    def test_errors_mean_not_an_image(self, monkeypatch):
        def boom(*a, **k):
            raise OSError("nope")
        monkeypatch.setattr("urllib.request.urlopen", boom)
        assert HttpImageVerifier().verify("https://cdn.example.com/a.jpg") is False


# ══ Normalization ═══════════════════════════════════════════════════════════

class TestPriceInText:
    @pytest.mark.parametrize("price,text,expected", [
        (24999, "Rs 24,999, 200 x 90 cm", True), (24999, "₹24999.00 only", True), (24999, "(24,999)", True),
        (24999, "now 24 999 INR", True), (124999, "₹1,24,999 total", True),
        (24999, "price 1,24,999", False), (24999, "SKU 249990", False), (24999, "no price here", False),
        (500, "weight 1500 kg", False), (24999, "reviews 24999 and more", True),
    ])
    def test_forms(self, price, text, expected):
        assert price_in_text(price, text) is expected


class TestNormalizer:
    def test_a_complete_listing_becomes_an_optimizer_compatible_record(self):
        out = normalize()
        assert isinstance(out, NormalizedListing)
        record = out.record
        assert record["product_id"].startswith("WEB_SHOPAEXAMP_") and record["source_agent_id"] == "test_provider"
        assert (record["price"], record["width_cm"], record["depth_cm"], record["seating_capacity"]) == (24999.0, 200.0, 90.0, 3)
        assert record["url"] == URL and record["retailer"] == "ShopA" and record["verified_at"] == NOW.isoformat()
        candidate, problems = validate_record(record)
        assert problems == [] and candidate.price == 24999.0
        assert UtilityOptimizer().optimize([record], REQUIREMENTS).selected.product_id == record["product_id"]

    def test_product_ids_are_stable_across_runs_and_tracking_noise(self):
        a = normalize().record["product_id"]
        b = normalize(allowed=allowed(URL + "?utm_source=a"), source_url=URL + "?gclid=z").record["product_id"]
        assert a == b

    @pytest.mark.parametrize("field,value,reason", [
        ("name", None, "name is missing"), ("name", "  ", "name is missing"),
        ("price", None, "price is missing"), ("price", -5, "price is missing"), ("price", 0, "price is missing"),
        ("price", math.nan, "price is missing"), ("price", math.inf, "price is missing"),
        ("currency", None, "currency is not stated"), ("currency", "USD", "not INR"), ("currency", "EUR", "not INR"),
        ("width_cm", None, "width and depth"), ("depth_cm", None, "width and depth"),
        ("width_cm", -3, "positive"), ("depth_cm", 0, "positive"),
        ("dimensions_unit", None, "unit of the dimensions is not stated"),
        ("seating_capacity", None, "seating capacity"), ("seating_capacity", 0, "seating capacity"),
        ("material", None, "material is missing"), ("colour", "", "colour is missing"), ("style", None, "style is missing"),
        ("rating", None, "rating"), ("rating", 5.5, "rating"), ("rating", -1, "rating"),
        ("review_count", None, "review count"), ("review_count", -4, "review count"),
        ("source_url", None, "source URL is missing"),
    ])
    def test_incomplete_or_invalid_listings_are_rejected_with_a_reason(self, field, value, reason):
        out = normalize(**{field: value})
        assert isinstance(out, RejectedListing)
        assert any(reason in r for r in out.reasons), out.reasons

    def test_every_problem_is_listed_not_just_the_first(self):
        out = normalize(price=None, rating=None, colour=None, review_count=None)
        assert isinstance(out, RejectedListing) and len(out.reasons) >= 4

    def test_nothing_is_defaulted(self):
        out = normalize(rating=None)
        assert isinstance(out, RejectedListing)  # a missing rating is not replaced by 0 or an average

    @pytest.mark.parametrize("unit,w,d,expected", [("cm", 200, 90, (200, 90)), ("mm", 2000, 900, (200, 90)),
                                                     ("in", 80, 36, (203.2, 91.44))])
    def test_units_are_converted_to_centimetres(self, unit, w, d, expected):
        record = normalize(dimensions_unit=unit, width_cm=w, depth_cm=d).record
        assert (record["width_cm"], record["depth_cm"]) == expected

    @pytest.mark.parametrize("w,d", [(2.0, 90), (20000, 90), (200, 2.0), (200, 9000)])
    def test_implausible_dimensions_are_rejected(self, w, d):
        out = normalize(width_cm=w, depth_cm=d)
        assert isinstance(out, RejectedListing) and any("implausible" in r for r in out.reasons)

    def test_a_mislabelled_unit_that_makes_the_size_absurd_is_caught(self):
        assert isinstance(normalize(dimensions_unit="cm", width_cm=2000, depth_cm=900), RejectedListing)

    def test_the_price_must_appear_in_the_retrieved_text(self):
        assert isinstance(normalize(source_text="Cozy Sofa for Rs 24,999 in grey"), NormalizedListing)
        out = normalize(source_text="Cozy Sofa for Rs 19,999 in grey")
        assert isinstance(out, RejectedListing) and any("does not appear in the retrieved text" in r for r in out.reasons)

    # ── URL provenance ──────────────────────────────────────────────────────
    def test_an_invented_url_is_rejected(self):
        out = normalize(source_url="https://invented.example.org/p/1")
        assert isinstance(out, RejectedListing) and any("not among the sources" in r for r in out.reasons)

    @pytest.mark.parametrize("url", ["http://shop-a.example.com/p/cozy-sofa", "javascript:alert(1)",
                                     "https://user:pw@shop-a.example.com/p/cozy-sofa", "https://10.0.0.1/p"])
    def test_unsafe_urls_are_rejected(self, url):
        out = normalize(source_url=url)
        assert isinstance(out, RejectedListing) and any("source URL rejected" in r for r in out.reasons)

    def test_a_reported_url_matches_despite_tracking_and_www(self):
        out = normalize(source_url="https://www.shop-a.example.com/p/cozy-sofa/?utm_campaign=x")
        assert isinstance(out, NormalizedListing) and out.record["url"] == URL

    def test_the_stored_url_is_the_reported_one_not_the_models_spelling(self):
        out = normalize(allowed=allowed("https://shop-a.example.com/p/cozy-sofa"), source_url="https://SHOP-A.example.com/p/cozy-sofa/")
        assert out.record["url"] == "https://shop-a.example.com/p/cozy-sofa"

    # ── Optional parts ────────────────────────────────────────────────────────
    def test_inferred_style_is_recorded_in_provenance(self):
        assert normalize(style_inferred=True).record["inferred_fields"] == ["style"]
        assert normalize().record["inferred_fields"] == []

    def test_unknown_availability_is_kept_and_flagged_not_assumed(self):
        out = normalize(availability=None)
        assert out.record["availability"] == "unknown" and any("availability" in w for w in out.warnings)
        assert UtilityOptimizer().optimize([out.record], REQUIREMENTS).selected is None  # the optimizer refuses it

    def test_out_of_stock_is_kept_for_the_optimizer_to_judge(self):
        record = normalize(availability="out_of_stock").record
        assert record["availability"] == "out_of_stock"

    def test_images_must_be_verified_and_are_dropped_not_fatal(self):
        out = normalize(image_url="https://cdn.example.com/a.jpg", image_ok=lambda u: False)
        assert isinstance(out, NormalizedListing) and out.record["image_url"] is None and any("image" in w for w in out.warnings)
        kept = normalize(image_url="https://cdn.example.com/a.jpg", image_ok=lambda u: True)
        assert kept.record["image_url"] == "https://cdn.example.com/a.jpg"
        assert normalize(image_url="http://cdn.example.com/a.jpg", image_ok=lambda u: True).record["image_url"] is None

    def test_currency_aliases(self):
        for alias in ("inr", "₹", "Rs", "rupees"):
            assert isinstance(normalize(currency=alias), NormalizedListing)


# ══ The grounded provider ═══════════════════════════════════════════════════

TEXT = ("Cozy Sofa 3-seater by ShopA: Rs 24,999, 200 x 90 x 85 cm, fabric, grey, rated 4.3 (120 reviews), in stock. "
        "Plain Loveseat by ShopB: Rs 17,999, 150 x 80 cm, 2-seater.")
GOOD = {"name": "Cozy Sofa", "price": 24999, "currency": "INR", "retailer": "ShopA", "source_url": URL,
        "width_cm": 200, "depth_cm": 90, "dimensions_unit": "cm", "seating_capacity": 3, "material": "fabric",
        "colour": "grey", "style": "minimalist", "availability": "in_stock", "rating": 4.3, "review_count": 120}


class MapResolver:
    def __init__(self, mapping=None):
        self.mapping = mapping or {}
        self.resolved: list[str] = []

    def resolve(self, url):
        self.resolved.append(url)
        return self.mapping.get(url, url)


def provider(*responses, resolver=None, **kw):
    client = ScriptedLLMClient(*responses)
    return GeminiSearchProvider(GeminiService(client, sleep=lambda s: None), resolver=resolver or MapResolver(),
                                clock=lambda: NOW, **kw), client


class TestGeminiSearchProvider:
    def test_the_three_stages(self):
        p, client = provider(grounded_response(TEXT, [URL], queries=["sofa india"]), {"products": [GOOD]})
        result = p.discover(ProductQuery(max_price=30000, min_seating=3, style="minimalist"))
        search, extract = client.calls
        assert search["grounding"] is True and search["url_context"] is True and search.get("schema") is None
        assert extract["schema"] is not None and not extract.get("grounding") and not extract.get("url_context")
        assert "- " + URL in extract["prompt"] and TEXT in extract["prompt"]
        assert result.search_queries == ["sofa india"] and result.sources == [URL]
        assert len(result.listings) == 1 and result.listings[0]["url"] == URL
        assert result.retrieved_at == NOW and result.listings[0]["verified_at"] == NOW.isoformat()

    def test_the_search_prompt_describes_the_request(self):
        p, client = provider(grounded_response(TEXT, [URL]), {"products": []})
        p.discover(ProductQuery(max_price=30000, min_seating=3, style="minimalist", max_width_cm=210))
        prompt = client.calls[0]["prompt"]
        assert "3-seater minimalist sofa" in prompt and "₹30,000" in prompt and "210 cm" in prompt

    def test_page_inspection_can_be_switched_off(self):
        p, client = provider(grounded_response(TEXT, [URL]), {"products": []}, inspect_pages=False)
        p.discover(ProductQuery())
        assert client.calls[0]["url_context"] is False

    def test_no_sources_means_no_listings_and_no_extraction_call(self):
        p, client = provider(grounded_response("The model remembers sofas it saw in training.", []))
        result = p.discover(ProductQuery())
        assert result.listings == [] and len(client.calls) == 1
        assert any("no usable sources" in n for n in result.notes)

    def test_listings_with_invented_urls_are_rejected(self):
        invented = {**GOOD, "name": "Ghost Sofa", "source_url": "https://ghost.example.org/p/1"}
        p, _ = provider(grounded_response(TEXT, [URL]), {"products": [GOOD, invented]})
        result = p.discover(ProductQuery())
        assert [l["name"] for l in result.listings] == ["Cozy Sofa"]
        assert result.rejected[0].name == "Ghost Sofa" and any("not among the sources" in r for r in result.rejected[0].reasons)

    def test_a_price_not_in_the_retrieved_text_is_rejected(self):
        wrong = {**GOOD, "price": 18999}
        result = provider(grounded_response(TEXT, [URL]), {"products": [wrong]})[0].discover(ProductQuery())
        assert result.listings == [] and any("does not appear" in r for r in result.rejected[0].reasons)

    def test_grounding_redirect_links_are_resolved_to_real_pages(self):
        redirect = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/Zm9v"
        resolver = MapResolver({redirect: URL})
        p, client = provider(grounded_response(TEXT, [redirect]), {"products": [GOOD]}, resolver=resolver)
        result = p.discover(ProductQuery())
        assert resolver.resolved == [redirect] and result.sources == [URL] and result.listings[0]["url"] == URL
        assert "vertexaisearch" not in client.calls[1]["prompt"]  # the model only ever sees real page URLs

    def test_an_unresolved_redirect_is_not_accepted_as_a_product_link(self):
        redirect = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/Zm9v"
        p, _ = provider(grounded_response(TEXT, [redirect]), {"products": [GOOD]})
        result = p.discover(ProductQuery())
        # Google's redirect host is a valid https URL, but it must not become a listing's link.
        assert all("vertexaisearch" not in l["url"] for l in result.listings)

    def test_successful_url_context_retrievals_are_also_sources(self):
        page = "https://shop-b.example.com/p/9"
        response = grounded_response(TEXT, [URL], retrievals=[(page, "success"), ("https://paywalled.example.com/x", "paywall")])
        result = provider(response, {"products": []})[0].discover(ProductQuery())
        assert page in result.sources and "https://paywalled.example.com/x" not in result.sources

    def test_duplicates_are_dropped(self):
        result = provider(grounded_response(TEXT, [URL]), {"products": [GOOD, dict(GOOD)]})[0].discover(ProductQuery())
        assert len(result.listings) == 1 and "duplicate" in result.rejected[0].reasons[0]

    def test_funnel_summary(self):
        bad = {**GOOD, "name": "Bad", "rating": None}
        result = provider(grounded_response(TEXT, [URL]), {"products": [GOOD, bad]})[0].discover(ProductQuery())
        assert (len(result.sources), result.extracted_count, len(result.listings), len(result.rejected)) == (1, 2, 1, 1)
        assert "1 accepted, 1 rejected" in result.summary()

    def test_extraction_failure_surfaces_as_a_discovery_error(self):
        p, _ = provider(grounded_response(TEXT, [URL]), "garbage", "garbage")
        with pytest.raises(ProductDiscoveryError, match="extraction failed"):
            p.discover(ProductQuery())

    def test_api_failure_during_search_surfaces_as_a_discovery_error(self):
        p, _ = provider(GeminiAPIError("quota", 429, False))
        with pytest.raises(ProductDiscoveryError, match="search failed") as info:
            p.discover(ProductQuery())
        assert info.value.provider_id == "gemini_search"

    def test_images_are_verified_through_the_injected_verifier(self):
        class Images:
            checked: list[str] = []
            def verify(self, url):
                self.checked.append(url)
                return False
        images = Images()
        with_image = {**GOOD, "image_url": "https://cdn.example.com/a.jpg"}
        result = provider(grounded_response(TEXT, [URL]), {"products": [with_image]}, image_verifier=images)[0].discover(ProductQuery())
        assert images.checked == ["https://cdn.example.com/a.jpg"] and result.listings[0]["image_url"] is None


class TestProductQuery:
    def test_description(self):
        q = ProductQuery(category="sofa", max_price=30000, min_seating=3, style="minimalist", colour="neutral",
                         max_width_cm=210, max_depth_cm=110)
        d = q.describe()
        for fragment in ("3-seater", "minimalist", "neutral-coloured", "₹30,000", "210 cm", "110 cm", "India", "INR"):
            assert fragment in d

    def test_validation(self):
        with pytest.raises(ValueError):
            ProductQuery(category=" ")
        with pytest.raises(ValueError):
            ProductQuery(limit=0)


# ══ Live candidate sources ══════════════════════════════════════════════════

def listing(pid="WEB_A_1", price=20000.0, availability="in_stock", verified=NOW, **over):
    record = {"product_id": pid, "name": f"Sofa {pid}", "price": price, "width_cm": 200.0, "depth_cm": 90.0,
              "seating_capacity": 3, "material": "fabric", "colour": "neutral", "style": "minimalist",
              "availability": availability, "rating": 4.0, "review_count": 50, "source_agent_id": "fake_provider",
              "url": "https://shop.example.com/" + pid, "verified_at": verified.isoformat() if verified else None}
    record.update(over)
    return record


class FakeProvider:
    provider_id = "fake_provider"

    def __init__(self, *runs):
        self.runs = list(runs)
        self.calls: list[ProductQuery] = []

    def discover(self, query):
        self.calls.append(query)
        run = self.runs.pop(0)
        if isinstance(run, Exception):
            raise run
        return DiscoveryResult(provider_id=self.provider_id, query=query, retrieved_at=NOW, listings=run)


class TestLiveCandidateSource:
    def test_serves_a_snapshot_and_never_hits_the_network_on_read(self):
        provider_ = FakeProvider([listing()])
        source = LiveCandidateSource(provider_, ProductQuery(), clock=lambda: NOW)
        assert source.get_candidates() == [] and provider_.calls == []
        source.refresh()
        for _ in range(10):
            assert len(source.get_candidates()) == 1
        assert len(provider_.calls) == 1

    def test_refresh_replaces_the_snapshot_and_remembers_the_previous_one(self):
        source = LiveCandidateSource(FakeProvider([listing("A")], [listing("B")]), ProductQuery(), clock=lambda: NOW)
        source.refresh()
        source.refresh()
        assert [r["product_id"] for r in source.get_candidates()] == ["B"]
        assert [r["product_id"] for r in source.previous_listings] == ["A"] and source.refreshed_at == NOW

    def test_a_failed_refresh_keeps_the_last_good_snapshot(self):
        source = LiveCandidateSource(FakeProvider([listing("A")], ProductDiscoveryError("p", "quota exceeded")),
                                     ProductQuery(), clock=lambda: NOW)
        source.refresh()
        with pytest.raises(ProductDiscoveryError):
            source.refresh()
        assert [r["product_id"] for r in source.get_candidates()] == ["A"] and "quota exceeded" in source.last_error

    def test_a_successful_refresh_clears_the_error(self):
        source = LiveCandidateSource(FakeProvider(ProductDiscoveryError("p", "x"), [listing()]), ProductQuery(), clock=lambda: NOW)
        with pytest.raises(ProductDiscoveryError):
            source.refresh()
        source.refresh()
        assert source.last_error is None

    def test_a_query_is_required(self):
        with pytest.raises(ValueError, match="no query"):
            LiveCandidateSource(FakeProvider()).refresh()

    def test_the_query_can_change_between_refreshes(self):
        provider_ = FakeProvider([listing()], [listing()])
        source = LiveCandidateSource(provider_, ProductQuery(style="minimalist"), clock=lambda: NOW)
        source.refresh()
        source.refresh(ProductQuery(style="contemporary"))
        assert [q.style for q in provider_.calls] == ["minimalist", "contemporary"] and source.query.style == "contemporary"

    # ── staleness and missing data ──────────────────────────────────────────
    def test_stale_listings_are_withheld_and_reported(self):
        clock = {"now": NOW}
        source = LiveCandidateSource(FakeProvider([listing("OLD", verified=NOW - timedelta(hours=10)), listing("NEW")]),
                                     ProductQuery(), max_age=timedelta(hours=6), clock=lambda: clock["now"])
        source.refresh()
        assert [r["product_id"] for r in source.get_candidates()] == ["NEW"]
        assert [r["product_id"] for r in source.stale_listings] == ["OLD"]
        clock["now"] = NOW + timedelta(hours=7)  # time passes: everything goes stale
        assert source.get_candidates() == [] and len(source.stale_listings) == 2

    @pytest.mark.parametrize("verified", [None, "not a date", ""])
    def test_listings_without_a_usable_verification_time_are_treated_as_stale(self, verified):
        source = LiveCandidateSource(FakeProvider([listing(verified=None, verified_at=verified)]), ProductQuery(), clock=lambda: NOW)
        source.refresh()
        assert source.get_candidates() == [] and len(source.stale_listings) == 1

    def test_naive_timestamps_are_treated_as_utc(self):
        source = LiveCandidateSource(FakeProvider([listing(verified_at=NOW.replace(tzinfo=None).isoformat())]),
                                     ProductQuery(), clock=lambda: NOW)
        source.refresh()
        assert len(source.get_candidates()) == 1

    def test_max_age_must_be_positive(self):
        with pytest.raises(ValueError):
            LiveCandidateSource(FakeProvider(), max_age=timedelta(0))

    def test_returned_records_are_copies(self):
        source = LiveCandidateSource(FakeProvider([listing()]), ProductQuery(), clock=lambda: NOW)
        source.refresh()
        source.get_candidates()[0]["price"] = 1
        assert source.get_candidates()[0]["price"] == 20000.0


class TestCandidateSourceCompatibility:
    """The planner depends only on get_candidates(); live and catalogue sources are interchangeable."""

    def agent(self, source):
        agent = BDIAgent("compat", BDIPlanner.default(UtilityOptimizer(), source))
        agent.add_desire(select_sofa_desire())
        agent.set_requirements({"sofa_budget": 30000, "max_width_cm": 210, "max_depth_cm": 110, "min_seating": 3,
                                "preferred_style": "minimalist", "preferred_colour": "neutral"})
        agent.deliberate("start")
        return agent

    def test_the_bdi_planner_works_with_a_live_source(self):
        source = LiveCandidateSource(FakeProvider([listing("A", price=20000), listing("B", price=25000)]),
                                     ProductQuery(), clock=lambda: NOW)
        source.refresh()
        agent = self.agent(source)
        assert agent.current_intention().params["product_id"] == "A"

    def test_an_empty_live_source_means_waiting_not_blocked(self):
        agent = self.agent(LiveCandidateSource(FakeProvider(), ProductQuery(), clock=lambda: NOW))
        assert agent.current_intention() is None and agent.desires.get("des_select_sofa").status.value == "active"
        assert any(t.kind == "desire_waiting" and "candidates" in t.explanation for t in agent.transitions)

    def test_live_and_catalogue_sources_combine(self):
        live = LiveCandidateSource(FakeProvider([listing("LIVE", price=12000, rating=5.0, review_count=500)]),
                                   ProductQuery(), clock=lambda: NOW)
        live.refresh()
        combined = CompositeCandidateSource([CatalogCandidateSource([catalog_agent_a(True)]), live])
        records = combined.get_candidates()
        assert len(records) == 8 and any(r["product_id"] == "LIVE" for r in records) and any(r["product_id"] == "SOFA_A2" for r in records)
        assert self.agent(combined).current_intention().params["product_id"] == "LIVE"

    def test_the_existing_catalogue_source_still_works_unchanged(self):
        assert self.agent(CatalogCandidateSource([catalog_agent_a(True)])).current_intention().params["product_id"] == "SOFA_A2"

    def test_composite_skips_a_failing_source_and_deduplicates(self):
        class Broken:
            def get_candidates(self):
                raise RuntimeError("down")
        a = [listing("X")]
        combined = CompositeCandidateSource([Broken(), type("S", (), {"get_candidates": lambda s: a})(),
                                             type("S", (), {"get_candidates": lambda s: a})()])
        assert [r["product_id"] for r in combined.get_candidates()] == ["X"] and combined.errors == ["Broken: RuntimeError"]


# ══ Change detection ════════════════════════════════════════════════════════

class TestDetectChanges:
    def test_price_change(self):
        events = detect_changes([listing("A", price=20000)], [listing("A", price=18000)])
        assert len(events) == 1 and events[0].event_type == EventType.PRICE_CHANGE.value
        assert events[0].payload == {"product_id": "A", "product_name": "Sofa A", "old_price": 20000, "new_price": 18000, "change": -2000}

    def test_availability_change(self):
        events = detect_changes([listing("A")], [listing("A", availability="out_of_stock")])
        assert events[0].event_type == EventType.PRODUCT_AVAILABILITY.value
        assert events[0].payload["availability"] == "out_of_stock" and events[0].payload["previous_availability"] == "in_stock"

    def test_removed_products_become_unavailable(self):
        events = detect_changes([listing("A"), listing("B")], [listing("B")])
        assert events[0].payload == {"product_id": "A", "product_name": "Sofa A", "availability": "unavailable",
                                     "previous_availability": "in_stock", "removed_from_catalog": True}

    def test_new_products_and_no_changes_produce_no_events(self):
        assert detect_changes([listing("A")], [listing("A"), listing("NEW")]) == []
        assert detect_changes([listing("A")], [listing("A")]) == []
        assert detect_changes([], []) == []

    def test_tiny_price_noise_is_ignored(self):
        assert detect_changes([listing("A", price=100.0)], [listing("A", price=100.001)]) == []

    def test_several_changes_are_deterministically_ordered(self):
        before = [listing("B", price=10), listing("A", price=10)]
        after = [listing("A", price=12, availability="out_of_stock"), listing("B", price=9)]
        assert [(e.payload["product_id"], e.event_type) for e in detect_changes(before, after)] == [
            ("A", "price_change"), ("A", "product_availability"), ("B", "price_change")]

    def test_events_are_understood_by_the_bdi_agent(self):
        agent = TestCandidateSourceCompatibility().agent(
            _refreshed([listing("A", price=20000), listing("B", price=25000)]))
        assert agent.current_intention().params["product_id"] == "A"
        for event in detect_changes([listing("A")], [listing("A", availability="out_of_stock")]):
            agent.handle_event(event)
        assert agent.beliefs.value("A", "availability") == "out_of_stock"
        assert agent.current_intention().params["product_id"] != "A"


def _refreshed(listings):
    source = LiveCandidateSource(FakeProvider(listings), ProductQuery(), clock=lambda: NOW)
    source.refresh()
    return source
