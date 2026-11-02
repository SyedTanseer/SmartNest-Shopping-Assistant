"""Tests for SerpApi Google Shopping discovery.

Everything here uses a mocked HTTP transport.  The response bodies follow SerpApi's *documented*
field names (title, product_id, product_link, immersive_product_page_token, source, price,
extracted_price, rating, reviews, snippet, extensions, thumbnail; stores[] and
about_the_product.features for product details).  They are not recordings of real responses —
see test_serpapi_live.py for the optional check against the real API.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from smartnest.bdi import BDIAgent, BDIPlanner, select_sofa_desire
from smartnest.core.models import EventType
from smartnest.database.repositories.in_memory import InMemoryRepository
from smartnest.integrations.products import (
    DiscoveryResult,
    LiveCandidateSource,
    ProductDiscoveryError,
    ProductQuery,
    detect_changes,
)
from smartnest.integrations.products.serpapi import (
    SEARCH_CACHE_COLLECTION,
    SearchCache,
    SerpApiAuthError,
    SerpApiCandidateSource,
    SerpApiClient,
    SerpApiConfigError,
    SerpApiError,
    SerpApiProvider,
    SerpApiQuotaError,
    SerpApiResponseError,
    SerpApiSettings,
    SerpApiTimeoutError,
    cache_key,
    classify_url,
    load_serpapi_settings,
    parse_price,
    serpapi_configured,
    shopping_query,
)
from smartnest.integrations.products.serpapi import parsing
from smartnest.optimization import UserRequirements, UtilityOptimizer, validate_record

KEY = "test-key-0123456789abcdef"
NOW = datetime(2026, 10, 31, 12, 0, tzinfo=timezone.utc)
REQUIREMENTS = UserRequirements(budget=30000, max_width_cm=210, max_depth_cm=110, min_seating=3,
                                preferred_style="minimalist", preferred_colour="neutral")
QUERY = ProductQuery(category="sofa", max_price=30000, min_seating=3, style="minimalist")

# ── Documented-shape fixtures ────────────────────────────────────────────

def shopping_item(**over):
    item = {
        "position": 1, "title": "Nordic Lite 3 Seater Sofa", "product_id": "19637827",
        "product_link": "https://www.google.com/shopping/product/19637827?gl=in",
        "immersive_product_page_token": "TOKEN_A", "source": "Pepperfry",
        "price": "₹24,999.00", "extracted_price": 24999.0, "rating": 4.4, "reviews": 812,
        "snippet": "Minimalist fabric sofa", "extensions": ["Free delivery"],
        "thumbnail": "https://encrypted-tbn0.gstatic.com/images?q=tbn:abc",
    }
    item.update(over)
    return item


def search_body(*items):
    return {"search_metadata": {"status": "Success"}, "shopping_results": list(items) or [shopping_item()]}


def details_body(**over):
    """A google_immersive_product response (shape as returned by the real API: everything under product_results)."""
    inner = {
        "title": "Nordic Lite 3 Seater Sofa",
        "stores": [
            {"name": "Pepperfry", "link": "https://www.pepperfry.com/product/nordic-lite-3-seater-1234",
             "price": "₹24,999.00", "extracted_price": 24999.0, "details_and_offers": ["In stock online"]},
            {"name": "Amazon.in", "link": "https://www.google.com/aclk?sa=l&ai=xyz",
             "price": "₹23,999.00", "extracted_price": 23999.0},
            {"name": "Relative", "link": "/goto?url=CAESabc", "price": "₹20,000", "extracted_price": 20000.0},
            {"name": "GlobalMart", "link": "https://globalmart.example.com/p/1", "price": "$300.00",
             "extracted_price": 300.0},
        ],
        "about_the_product": {"features": [
            {"title": "Width", "value": "210 cm"}, {"title": "Depth", "value": "90 cm"},
            {"title": "Material", "value": "Fabric"}, {"title": "Color", "value": "Grey"},
            {"title": "Style", "value": "Minimalist"}]},
    }
    inner.update(over)
    return {"search_metadata": {"status": "Success"}, "product_results": inner}


class FakeSerpApi:
    """An httpx transport that serves canned bodies and records every request."""

    def __init__(self, search=None, details=None, status=200, details_status=200):
        self.search = search if search is not None else search_body()
        self.details = details if details is not None else details_body()
        self.status, self.details_status = status, details_status
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        engine = request.url.params.get("engine")
        if engine == "google_shopping":
            return self._respond(self.search, self.status)
        return self._respond(self.details, self.details_status)

    @staticmethod
    def _respond(body, status):
        if isinstance(body, Exception):
            raise body
        if isinstance(body, str):
            return httpx.Response(status, text=body)
        return httpx.Response(status, json=body)

    @property
    def transport(self):
        return httpx.MockTransport(self)

    def engines(self):
        return [r.url.params.get("engine") for r in self.requests]


def settings(**over) -> SerpApiSettings:
    return SerpApiSettings(api_key=KEY, **over)


def provider(fake: FakeSerpApi, *, cache=None, clock=lambda: NOW, **over) -> SerpApiProvider:
    s = settings(**over)
    return SerpApiProvider(SerpApiClient(s, transport=fake.transport), s, cache=cache, clock=clock)


def cache_for(repo=None, ttl_hours=6, clock=lambda: NOW):
    return SearchCache(repo or InMemoryRepository(SEARCH_CACHE_COLLECTION, id_prefix="sc"),
                       ttl=timedelta(hours=ttl_hours), clock=clock)


# ══ Configuration ═══════════════════════════════════════════════════════════

class TestSettings:
    def test_loaded_from_environment_and_key_never_in_repr(self, tmp_path):
        s = load_serpapi_settings(tmp_path / "none.env", {"SERPAPI_API_KEY": KEY, "SERPAPI_MAX_RESULTS": "5"})
        assert s.api_key == KEY and s.max_results == 5 and s.country == "in" and s.language == "en"
        assert KEY not in repr(s) and KEY not in str(s)

    def test_loaded_from_env_file_and_environment_wins(self, tmp_path):
        f = tmp_path / ".env"
        f.write_text("SERPAPI_API_KEY=from-file-0123456789\nSERPAPI_MAX_DETAIL_CALLS=2\n")
        assert load_serpapi_settings(f, {}).api_key == "from-file-0123456789"
        assert load_serpapi_settings(f, {"SERPAPI_API_KEY": KEY}).api_key == KEY
        assert load_serpapi_settings(f, {}).max_detail_calls == 2

    @pytest.mark.parametrize("value", ["", "   ", "changeme", "<your_serpapi_api_key>", "has space"])
    def test_missing_or_placeholder_key_is_refused(self, value, tmp_path):
        with pytest.raises(SerpApiConfigError):
            load_serpapi_settings(tmp_path / "none.env", {"SERPAPI_API_KEY": value})
        assert not serpapi_configured(tmp_path / "none.env", {"SERPAPI_API_KEY": value})

    @pytest.mark.parametrize("name,value", [("SERPAPI_TIMEOUT_SECONDS", "0"), ("SERPAPI_TIMEOUT_SECONDS", "abc"),
                                            ("SERPAPI_MAX_RESULTS", "0"), ("SERPAPI_MAX_DETAIL_CALLS", "-1"),
                                            ("SERPAPI_CACHE_TTL_HOURS", "0")])
    def test_invalid_values_are_refused(self, name, value, tmp_path):
        with pytest.raises(SerpApiConfigError):
            load_serpapi_settings(tmp_path / "none.env", {"SERPAPI_API_KEY": KEY, name: value})

    def test_configured_reports_true_for_a_real_looking_key(self, tmp_path):
        assert serpapi_configured(tmp_path / "none.env", {"SERPAPI_API_KEY": KEY})


# ══ Client: request shape and error handling ════════════════════════════════

class TestClient:
    def test_search_request_uses_the_documented_endpoint_and_parameters(self):
        fake = FakeSerpApi()
        SerpApiClient(settings(), transport=fake.transport).search("3 seater sofa")
        req = fake.requests[0]
        assert str(req.url).startswith("https://serpapi.com/search.json?")
        p = req.url.params
        assert (p["engine"], p["gl"], p["hl"], p["q"]) == ("google_shopping", "in", "en", "3 seater sofa")
        assert p["api_key"] == KEY

    def test_details_request_uses_the_immersive_engine_and_token(self):
        fake = FakeSerpApi()
        SerpApiClient(settings(), transport=fake.transport).product_details("TOKEN_A")
        p = fake.requests[0].url.params
        assert (p["engine"], p["page_token"]) == ("google_immersive_product", "TOKEN_A")

    @pytest.mark.parametrize("bad", ["", "   ", None, 5])
    def test_empty_search_text_or_token_is_refused_without_a_request(self, bad):
        fake = FakeSerpApi()
        client = SerpApiClient(settings(), transport=fake.transport)
        with pytest.raises(ValueError):
            client.search(bad)
        with pytest.raises(ValueError):
            client.product_details(bad)
        assert fake.requests == []

    @pytest.mark.parametrize("status", [401, 403])
    def test_auth_failures(self, status):
        fake = FakeSerpApi(search={"error": "Invalid API key"}, status=status)
        with pytest.raises(SerpApiAuthError):
            SerpApiClient(settings(), transport=fake.transport).search("sofa")

    def test_http_429_is_a_quota_error(self):
        fake = FakeSerpApi(search={"error": "Your account has run out of searches."}, status=429)
        with pytest.raises(SerpApiQuotaError, match="429"):
            SerpApiClient(settings(), transport=fake.transport).search("sofa")

    def test_quota_error_reported_with_http_200(self):
        fake = FakeSerpApi(search={"error": "Your account has run out of searches."})
        with pytest.raises(SerpApiQuotaError):
            SerpApiClient(settings(), transport=fake.transport).search("sofa")

    def test_invalid_key_reported_with_http_200(self):
        fake = FakeSerpApi(search={"error": "Invalid API key. Your API key should be here: https://serpapi.com"})
        with pytest.raises(SerpApiAuthError):
            SerpApiClient(settings(), transport=fake.transport).search("sofa")

    def test_timeout(self):
        fake = FakeSerpApi(search=httpx.ReadTimeout("slow"))
        with pytest.raises(SerpApiTimeoutError):
            SerpApiClient(settings(), transport=fake.transport).search("sofa")

    def test_network_error(self):
        fake = FakeSerpApi(search=httpx.ConnectError("no route"))
        with pytest.raises(SerpApiError) as info:
            SerpApiClient(settings(), transport=fake.transport).search("sofa")
        assert not isinstance(info.value, (SerpApiAuthError, SerpApiQuotaError))

    @pytest.mark.parametrize("status", [500, 502, 503])
    def test_server_errors(self, status):
        fake = FakeSerpApi(search="<html>bad gateway</html>", status=status)
        with pytest.raises(SerpApiError, match=str(status)):
            SerpApiClient(settings(), transport=fake.transport).search("sofa")

    def test_non_json_success_body(self):
        with pytest.raises(SerpApiResponseError):
            SerpApiClient(settings(), transport=FakeSerpApi(search="not json at all").transport).search("sofa")

    def test_json_that_is_not_an_object(self):
        with pytest.raises(SerpApiResponseError):
            SerpApiClient(settings(), transport=FakeSerpApi(search=[1, 2, 3]).transport).search("sofa")

    def test_google_returning_no_results_is_an_empty_result_not_an_error(self):
        fake = FakeSerpApi(search={"error": "Google hasn't returned any results for this query."})
        assert SerpApiClient(settings(), transport=fake.transport).search("zzzz").get("_empty") is True

    def test_other_error_messages_are_reported_and_redacted(self):
        fake = FakeSerpApi(search={"error": f"Something odd happened for api_key={KEY}"})
        with pytest.raises(SerpApiError) as info:
            SerpApiClient(settings(), transport=fake.transport).search("sofa")
        assert KEY not in str(info.value)

    def test_each_request_is_counted(self):
        fake = FakeSerpApi()
        client = SerpApiClient(settings(), transport=fake.transport)
        client.search("a")
        client.product_details("t")
        assert client.calls == 2

    def test_the_key_never_appears_in_errors_or_logs(self, caplog):
        caplog.set_level(logging.DEBUG)
        for fake in (FakeSerpApi(search=httpx.ConnectError(f"failed for https://serpapi.com?api_key={KEY}")),
                     FakeSerpApi(search={"error": "x"}, status=401), FakeSerpApi(search={"error": "x"}, status=429)):
            with pytest.raises(SerpApiError) as info:
                SerpApiClient(settings(), transport=fake.transport).search("sofa")
            assert KEY not in str(info.value) and KEY not in repr(info.value)
        assert KEY not in caplog.text


# ══ Prices ══════════════════════════════════════════════════════════════════

class TestPrices:
    @pytest.mark.parametrize("text,extracted,amount", [
        ("₹24,999.00", 24999.0, 24999.0), ("₹1,24,999", 124999, 124999.0), ("₹24999", None, 24999.0),
        ("Rs. 18,500", None, 18500.0), ("INR 9,999.50", None, 9999.5), ("₹ 7,499", 7499, 7499.0),
    ])
    def test_indian_prices_are_parsed(self, text, extracted, amount):
        value, currency, problems = parse_price(text, extracted)
        assert (value, currency, problems) == (amount, "INR", [])

    def test_other_currencies_are_recognised_not_converted(self):
        assert parse_price("$499.00", 499.0)[:2] == (499.0, "USD")
        assert parse_price("€120", None)[:2] == (120.0, "EUR")

    def test_currency_is_never_assumed(self):
        amount, currency, problems = parse_price(None, 24999.0)
        assert (amount, currency) == (24999.0, None) and problems == []

    @pytest.mark.parametrize("text,extracted", [(None, None), ("", None), ("Price on request", None),
                                                ("₹0", 0), ("₹-5", -5.0), (None, True), (None, float("nan"))])
    def test_unusable_prices(self, text, extracted):
        value, _, problems = parse_price(text, extracted)
        assert value is None and problems

    def test_text_and_extracted_price_must_agree(self):
        value, _, problems = parse_price("₹24,999.00", 19999.0)
        assert value is None and "disagrees" in problems[0]


# ══ URL classification ══════════════════════════════════════════════════════

class TestUrlClassification:
    @pytest.mark.parametrize("url", ["https://www.pepperfry.com/product/x-1", "https://www.amazon.in/dp/B0ABC",
                                     "https://shop.example.co.in/p/1?utm_source=g", "https://urbanladder.com/products/sofa"])
    def test_retailer_urls(self, url):
        assert classify_url(url) == ("retailer", url)

    @pytest.mark.parametrize("url", ["https://www.google.com/shopping/product/123", "https://google.co.in/search?q=x",
                                     "https://www.google.com/aclk?sa=l&ai=abc", "https://www.googleadservices.com/pagead/aclk?x=1",
                                     "https://shopping.google.com/product/1", "https://goo.gl/abc"])
    def test_google_urls_are_never_retailer_urls(self, url):
        assert classify_url(url) == ("google_shopping", url)

    @pytest.mark.parametrize("url", [None, "", "   ", 5])
    def test_missing(self, url):
        assert classify_url(url) == ("missing", None)

    @pytest.mark.parametrize("url", ["http://insecure.example.com/p", "ftp://x.example.com/p", "javascript:alert(1)",
                                     "https://localhost/p", "https://192.168.1.5/p", "not a url", "https://a b.com/p"])
    def test_invalid(self, url):
        assert classify_url(url) == ("invalid", None)

    def test_a_domain_that_merely_contains_google_is_still_a_retailer(self):
        assert classify_url("https://notgoogle.com/p")[0] == "retailer"
        assert classify_url("https://google.example.com/p")[0] == "retailer"


# ══ Specification parsing ═══════════════════════════════════════════════════

class TestSpecs:
    def feats(self, **pairs):
        return {"about_the_product": {"features": [{"title": k.replace("_", " "), "value": v} for k, v in pairs.items()]}}

    def test_stated_features_are_read(self):
        s = parsing.specs_from(shopping_item(), details_body())
        assert (s["width_cm"], s["depth_cm"], s["material"], s["colour"], s["style"]) == (210.0, 90.0, "Fabric", "Grey", "Minimalist")
        assert s["seating_capacity"] == 3

    @pytest.mark.parametrize("value,expected", [("210 cm", 210.0), ("2.1 m", 210.0), ("2100 mm", 210.0), ("84 in", 213.36),
                                                ("84 inches", 213.36), ("90.5cm", 90.5)])
    def test_units_are_converted_to_cm(self, value, expected):
        assert parsing.specs_from({}, self.feats(Width=value))["width_cm"] == expected

    @pytest.mark.parametrize("value", ["210", "wide", "", "cm 210", "0 cm", "-5 cm"])
    def test_dimensions_without_a_clear_unit_or_value_stay_null(self, value):
        assert parsing.specs_from({}, self.feats(Width=value))["width_cm"] is None

    def test_labelled_dimensions_inside_one_string(self):
        s = parsing.specs_from({}, self.feats(Dimensions="Width: 210 cm, Depth: 90 cm, Height: 80 cm"))
        assert (s["width_cm"], s["depth_cm"]) == (210.0, 90.0)

    def test_unlabelled_combined_dimensions_are_not_guessed(self):
        s = parsing.specs_from({}, self.feats(Dimensions="210 x 90 x 80 cm"))
        assert (s["width_cm"], s["depth_cm"]) == (None, None)

    def test_length_is_not_assumed_to_be_width(self):
        assert parsing.specs_from({}, self.feats(Length="210 cm"))["width_cm"] is None

    def test_no_details_means_everything_unstated_is_none(self):
        s = parsing.specs_from({"title": "A sofa"}, None)
        assert all(v is None for v in s.values())

    @pytest.mark.parametrize("title,expected", [("Comfy 3 Seater Sofa", 3), ("Two-seater loveseat", 2), ("3-Seater", 3),
                                                ("Five seat corner sofa", 5), ("Plain sofa", None),
                                                ("3+1+1 seater sofa set", None), ("2 seater and 3 seater combo", None)])
    def test_seating_comes_only_from_explicit_wording(self, title, expected):
        assert parsing.seating_from_text(title) == expected

    def test_conflicting_seating_between_title_and_feature_stays_null(self):
        s = parsing.specs_from(shopping_item(title="3 Seater Sofa"), self.feats(Seating_capacity="2"))
        assert s["seating_capacity"] is None

    @pytest.mark.parametrize("texts,expected", [(["In stock"], "in_stock"), (["Out of stock"], "out_of_stock"),
                                                (["Free delivery"], "unknown"), ([], "unknown"),
                                                (["In stock", "Out of stock"], "unknown"), (["Currently unavailable"], "out_of_stock")])
    def test_availability_only_when_stated(self, texts, expected):
        assert parsing.availability_from(texts) == expected


# ══ Normalization and the missing-data policy ═══════════════════════════════

NOW_ISO = NOW.isoformat()


def build(item=None, details="default", **kw):
    return parsing.build_listing(item or shopping_item(), details_body() if details == "default" else details,
                                 retrieved_at=NOW, provider_id="serpapi_google_shopping", **kw)


class TestNormalization:
    def test_a_complete_product_becomes_an_optimizer_compatible_record(self):
        built = build()
        r = built.record
        assert built.reasons == [] and r is not None
        assert validate_record(r)[1] == []
        assert (r["name"], r["price"], r["currency"], r["rating"], r["review_count"]) == ("Nordic Lite 3 Seater Sofa", 24999.0, "INR", 4.4, 812)
        assert (r["width_cm"], r["depth_cm"], r["seating_capacity"]) == (210.0, 90.0, 3)
        assert r["source_agent_id"] == "serpapi_google_shopping" and r["verified_at"] == NOW_ISO
        assert r["image_url"].startswith("https://encrypted-tbn0.gstatic.com/")

    def test_the_direct_retailer_offer_is_chosen_over_a_cheaper_google_redirect(self):
        r = build().record
        assert (r["retailer"], r["price"], r["price_source"]) == ("Pepperfry", 24999.0, "retailer_offer")
        assert r["url"] == "https://www.pepperfry.com/product/nordic-lite-3-seater-1234" and r["url_type"] == "retailer"
        assert r["product_page_url"] == "https://www.google.com/shopping/product/19637827?gl=in"

    def test_the_cheapest_direct_offer_wins_and_foreign_currency_offers_are_ignored(self):
        d = details_body()
        d["product_results"]["stores"].append({"name": "Urban Ladder", "link": "https://www.urbanladder.com/products/x", "price": "₹22,500.00",
                            "extracted_price": 22500.0})
        r = build(details=d).record
        assert (r["retailer"], r["price"]) == ("Urban Ladder", 22500.0)

    def test_a_google_shopping_link_is_never_stored_as_the_retailer_url(self):
        d = details_body()
        d["product_results"]["stores"] = [d["product_results"]["stores"][1]]  # only the google.com/aclk offer
        r = build(details=d).record
        assert r["url"] is None and r["url_type"] == "google_shopping"
        assert r["product_page_url"].startswith("https://www.google.com/shopping/")
        assert r["retailer"] == "Pepperfry"  # the search-level merchant, not the Google redirect

    def test_without_details_there_is_no_direct_url_and_specs_stay_null(self):
        built = build(details=None)
        # Only name, price and a usable URL are mandatory: the product is kept, labelled as a Google page.
        assert built.record is not None and built.summary["missing"] == []
        r = built.record
        assert r["url"] is None and r["url_type"] == "google_shopping"
        assert (r["listing_url_type"], r["listing_url"]) == ("google_shopping", r["product_page_url"])
        assert all(r[k] is None for k in ("material", "colour", "style", "width_cm", "depth_cm"))
        assert {"material", "colour", "style", "width_cm", "depth_cm", "height_cm"} <= set(built.summary["unstated"])
        assert built.summary["unstated_dimensions"] == ["width_cm", "depth_cm", "height_cm"]

    def test_missing_optional_fields_stay_null_and_are_listed_as_unstated(self):
        item = shopping_item()
        del item["rating"], item["reviews"]
        built = build(item)
        assert built.record is not None and built.reasons == []
        assert built.record["rating"] is None and built.record["review_count"] is None   # never 0, never an average
        assert {"rating", "review_count"} <= set(built.summary["unstated"])
        assert validate_record(built.record)[1] == []

    def test_missing_dimensions_are_not_invented(self):
        d = details_body()
        d["product_results"]["about_the_product"]["features"] = [f for f in d["product_results"]["about_the_product"]["features"] if f["title"] not in ("Width", "Depth")]
        built = build(details=d)
        # dimensions are optional: the product is kept, with the sizes left as None (never guessed)
        assert built.record is not None and built.reasons == []
        assert (built.record["width_cm"], built.record["depth_cm"]) == (None, None)
        assert validate_record(built.record)[1] == []
        assert built.summary["unstated_dimensions"] == ["width_cm", "depth_cm", "height_cm"]

    def test_a_non_inr_price_is_rejected_not_converted(self):
        built = build(shopping_item(price="$499.00", extracted_price=499.0), details=None)
        assert built.record is None and any("USD" in r for r in built.reasons)

    def test_a_search_price_without_a_currency_is_rejected_not_assumed_inr(self):
        built = build(shopping_item(price=None, extracted_price=24999.0), details=None)
        assert built.record is None and any("not stated" in r and "INR" in r for r in built.reasons)

    def test_unknown_availability_is_kept_explicit(self):
        d = details_body()
        d["product_results"]["stores"][0]["details_and_offers"] = []
        r = build(details=d).record
        assert r["availability"] == "unknown"

    def test_rating_outside_0_to_5_is_treated_as_not_stated(self):
        built = build(shopping_item(rating=9.1))
        assert built.record["rating"] is None and built.summary["rating"] is None   # not clamped, not trusted

    def test_zero_reviews_is_a_real_value_not_missing(self):
        assert build(shopping_item(reviews=0)).record["review_count"] == 0

    def test_reviews_that_are_not_whole_numbers_are_missing(self):
        assert build(shopping_item(reviews="5.3K")).summary["review_count"] is None

    def test_a_missing_title_is_rejected(self):
        built = build(shopping_item(title=None))
        assert built.record is None and "name is missing" in built.reasons

    def test_an_invalid_thumbnail_is_dropped_but_does_not_reject(self):
        assert build(shopping_item(thumbnail="http://insecure.example.com/x.jpg")).record["image_url"] is None

    def test_product_id_is_stable_and_prefixed(self):
        assert build().record["product_id"] == "SERP_19637827"
        item = shopping_item(); del item["product_id"]
        a = parsing.product_id_for(item, "Pepperfry")
        assert a == parsing.product_id_for(item, "Pepperfry") and a != parsing.product_id_for(item, "Other")


# ══ Query generation ════════════════════════════════════════════════════════

class TestQuery:
    def test_query_text_comes_from_the_requirements(self):
        assert shopping_query(ProductQuery(category="sofa", min_seating=3, style="minimalist", colour="grey")) == "3 seater minimalist grey sofa"
        assert shopping_query(ProductQuery(category="sofa")) == "sofa"
        assert shopping_query(ProductQuery(category="sofa", min_seating=2)) == "2 seater sofa"


# ══ Provider ════════════════════════════════════════════════════════════════

class TestProvider:
    def test_successful_discovery_returns_real_fields_and_counts_api_calls(self):
        fake = FakeSerpApi()
        result = provider(fake).discover(QUERY)
        assert [l["name"] for l in result.listings] == ["Nordic Lite 3 Seater Sofa"]
        assert fake.engines() == ["google_shopping", "google_immersive_product"]
        assert result.api_calls == 2 and result.cache_hits == 0 and not result.stale
        assert result.sources == ["https://www.pepperfry.com/product/nordic-lite-3-seater-1234"]
        assert result.search_queries == ["3 seater minimalist sofa"] and result.retrieved_at == NOW

    def test_incomplete_products_are_reported_with_what_is_known_and_why(self):
        usd = shopping_item(product_id="2", title="Import Sofa", price="$499.00", extracted_price=499.0, immersive_product_page_token="TOKEN_B")
        cheap = shopping_item(product_id="3", title="Compact 2 Seater", price="₹9,999", extracted_price=9999.0,
                              immersive_product_page_token=None, rating=None)
        fake = FakeSerpApi(search=search_body(shopping_item(), usd, cheap))
        result = provider(fake).discover(QUERY)
        assert [l["name"] for l in result.listings] == ["Nordic Lite 3 Seater Sofa", "Compact 2 Seater"]
        assert [r.name for r in result.rejected] == ["Import Sofa"]        # only a mandatory problem rejects
        assert any("USD" in x for x in result.rejected[0].reasons)
        cheap = result.listings[1]
        assert cheap["price"] == 9999.0 and cheap["rating"] is None and cheap["listing_url_type"] == "google_shopping"

    def test_only_products_within_budget_get_a_paid_details_lookup(self):
        pricey = shopping_item(product_id="9", title="Luxury Sofa", price="₹90,000", extracted_price=90000.0, immersive_product_page_token="TOKEN_X")
        fake = FakeSerpApi(search=search_body(pricey, shopping_item()))
        provider(fake).discover(QUERY)
        assert [r.url.params.get("page_token") for r in fake.requests if r.url.params.get("page_token")] == ["TOKEN_A"]

    def test_detail_lookups_are_capped(self):
        items = [shopping_item(product_id=str(i), title=f"Sofa {i}", immersive_product_page_token=f"T{i}") for i in range(6)]
        fake = FakeSerpApi(search=search_body(*items))
        provider(fake, max_detail_calls=2).discover(QUERY)
        assert fake.engines().count("google_immersive_product") == 2

    def test_zero_detail_calls_means_search_only(self):
        fake = FakeSerpApi()
        result = provider(fake, max_detail_calls=0).discover(QUERY)
        assert fake.engines() == ["google_shopping"] and len(result.listings) == 1 and result.rejected == []
        assert result.listings[0]["url"] is None and result.listings[0]["url_type"] == "google_shopping"

    def test_result_limit_is_respected(self):
        items = [shopping_item(product_id=str(i), title=f"Sofa {i}", immersive_product_page_token=f"T{i}") for i in range(8)]
        result = provider(FakeSerpApi(search=search_body(*items)), max_results=3, max_detail_calls=0).discover(QUERY)
        assert result.extracted_count == 3 and len(result.listings) == 3 and result.rejected == []

    def test_duplicate_products_appear_once(self):
        fake = FakeSerpApi(search=search_body(shopping_item(), shopping_item(position=2)))
        result = provider(fake).discover(QUERY)
        assert len(result.listings) == 1 and any("duplicate" in r.reasons[0] for r in result.rejected)

    def test_no_results_is_not_an_error(self):
        result = provider(FakeSerpApi(search={"error": "Google hasn't returned any results for this query."})).discover(QUERY)
        assert result.listings == [] and "no results" in result.notes[0]

    def test_empty_shopping_results_list(self):
        result = provider(FakeSerpApi(search={"shopping_results": []})).discover(QUERY)
        assert result.listings == [] and result.notes

    @pytest.mark.parametrize("body", [{"search_metadata": {}}, {"shopping_results": "oops"}, {"shopping_results": [1, 2]},
                                      {"shopping_results": None}])
    def test_responses_of_the_wrong_shape_are_refused(self, body):
        with pytest.raises(ProductDiscoveryError, match="shopping_results"):
            provider(FakeSerpApi(search=body)).discover(QUERY)

    @pytest.mark.parametrize("fake_kwargs,error", [
        (dict(search={"error": "x"}, status=401), "rejected the API key"),
        (dict(search={"error": "x"}, status=429), "429"),
        (dict(search=httpx.ReadTimeout("t")), "did not answer"),
        (dict(search="garbage"), "not valid JSON"),
    ])
    def test_api_failures_become_discovery_errors(self, fake_kwargs, error):
        with pytest.raises(ProductDiscoveryError, match=error) as info:
            provider(FakeSerpApi(**fake_kwargs)).discover(QUERY)
        assert KEY not in str(info.value) and info.value.provider_id == "serpapi_google_shopping"

    def test_a_failed_details_lookup_does_not_lose_the_search(self):
        fake = FakeSerpApi(details={"error": "x"}, details_status=429)
        result = provider(fake).discover(QUERY)
        assert len(result.listings) == 1 and result.rejected == []      # the search result alone is still usable
        assert result.listings[0]["url_type"] == "google_shopping"
        assert any("product details unavailable" in n for n in result.notes)
        assert any("no product-details lookup succeeded" in n for n in result.notes)

    def test_malformed_details_do_not_crash_and_yield_no_invented_data(self):
        for bad in ({"stores": "x", "about_the_product": []}, {"stores": [None, 5, {"name": 3}], "about_the_product": {"features": [1, {"title": 2}]}}):
            result = provider(FakeSerpApi(details=bad)).discover(QUERY)
            r = result.listings[0]
            assert r["url"] is None and r["url_type"] == "google_shopping"
            assert all(r[k] is None for k in ("material", "colour", "style", "width_cm", "depth_cm"))   # nothing invented


# ══ Cache ═══════════════════════════════════════════════════════════════════

class Clock:
    def __init__(self):
        self.now = NOW

    def __call__(self):
        return self.now

    def advance(self, **kw):
        self.now += timedelta(**kw)


class TestCache:
    def test_key_is_stable_ignores_the_api_key_and_depends_on_params(self):
        a = cache_key("search", {"q": "sofa", "gl": "in"})
        assert a == cache_key("search", {"gl": "in", "q": "sofa"}) == cache_key("search", {"q": "sofa", "gl": "in", "api_key": KEY})
        assert a != cache_key("search", {"q": "sofa", "gl": "us"}) and a != cache_key("details", {"q": "sofa", "gl": "in"})

    def test_put_and_get_round_trip_with_fresh_flag(self):
        clock = Clock(); cache = cache_for(clock=clock)
        cache.put("search", {"q": "x"}, {"a": [1, 2]})
        entry = cache.get("search", {"q": "x"})
        assert entry.payload == {"a": [1, 2]} and entry.fresh and not entry.stale and entry.fetched_at == NOW
        clock.advance(hours=7)
        assert cache.get("search", {"q": "x"}).stale

    def test_a_miss_is_none(self):
        assert cache_for().get("search", {"q": "nothing"}) is None

    def test_no_secret_is_stored(self):
        repo = InMemoryRepository(SEARCH_CACHE_COLLECTION, id_prefix="sc")
        cache_for(repo).put("search", {"q": "x", "api_key": KEY}, {"a": 1})
        assert KEY not in repr(repo.find({}))

    def test_corrupt_entries_are_ignored(self):
        repo = InMemoryRepository(SEARCH_CACHE_COLLECTION, id_prefix="sc")
        key = cache_key("search", {"q": "x"})
        for bad in ({"fetched_at": "garbage", "payload_json": "{}"}, {"fetched_at": NOW_ISO, "payload_json": "not json"},
                    {"fetched_at": NOW_ISO, "payload_json": "[1]"}, {"nothing": 1}):
            repo.upsert(key, bad)
            assert cache_for(repo).get("search", {"q": "x"}) is None

    def test_repository_failures_never_break_the_cache(self):
        class Broken:
            def get(self, *_): raise RuntimeError("down")
            def upsert(self, *_): raise RuntimeError("down")
        cache = SearchCache(Broken())
        assert cache.get("search", {"q": "x"}) is None
        cache.put("search", {"q": "x"}, {})  # does not raise

    def test_ttl_must_be_positive(self):
        with pytest.raises(ValueError):
            SearchCache(InMemoryRepository("c"), ttl=timedelta(0))

    def test_a_second_identical_discovery_makes_no_api_calls(self):
        fake = FakeSerpApi(); cache = cache_for()
        p = provider(fake, cache=cache)
        first = p.discover(QUERY)
        calls = len(fake.requests)
        second = p.discover(QUERY)
        assert len(fake.requests) == calls and second.api_calls == 0 and second.cache_hits == 2
        assert [l["product_id"] for l in second.listings] == [l["product_id"] for l in first.listings]

    def test_cached_records_keep_the_original_fetch_time_not_the_current_time(self):
        clock = Clock(); fake = FakeSerpApi(); cache = cache_for(clock=clock)
        p = provider(fake, cache=cache, clock=clock)
        p.discover(QUERY)
        clock.advance(hours=3)
        again = p.discover(QUERY)
        assert again.listings[0]["verified_at"] == NOW_ISO and again.retrieved_at == NOW

    def test_an_expired_entry_triggers_a_new_search(self):
        clock = Clock(); fake = FakeSerpApi(); cache = cache_for(ttl_hours=1, clock=clock)
        p = provider(fake, cache=cache, clock=clock)
        p.discover(QUERY)
        n = len(fake.requests)
        clock.advance(hours=2)
        result = p.discover(QUERY)
        assert len(fake.requests) == n + 2 and result.api_calls == 2 and not result.stale
        assert result.listings[0]["verified_at"] == (NOW + timedelta(hours=2)).isoformat()

    def test_when_serpapi_fails_an_expired_entry_is_used_but_flagged_stale(self):
        clock = Clock(); fake = FakeSerpApi(); cache = cache_for(ttl_hours=1, clock=clock)
        p = provider(fake, cache=cache, clock=clock)
        p.discover(QUERY)
        clock.advance(hours=5)
        fake.status, fake.search = 429, {"error": "out of searches"}
        fake.details_status, fake.details = 429, {"error": "out of searches"}
        result = p.discover(QUERY)
        assert result.stale and result.listings and result.listings[0]["verified_at"] == NOW_ISO
        assert any("STALE" in n for n in result.notes)

    def test_a_failure_with_no_cache_entry_still_raises(self):
        fake = FakeSerpApi(search={"error": "x"}, status=429)
        with pytest.raises(ProductDiscoveryError):
            provider(fake, cache=cache_for()).discover(QUERY)

    def test_different_queries_do_not_share_entries(self):
        fake = FakeSerpApi(); p = provider(fake, cache=cache_for())
        p.discover(QUERY)
        n = len(fake.requests)
        p.discover(ProductQuery(category="sofa", min_seating=2))
        assert len(fake.requests) > n

    def test_the_cache_lives_in_a_repository_collection_named_search_cache(self):
        repo = InMemoryRepository(SEARCH_CACHE_COLLECTION, id_prefix="sc")
        provider(FakeSerpApi(), cache=cache_for(repo)).discover(QUERY)
        docs = repo.find({})
        assert SEARCH_CACHE_COLLECTION == "search_cache" and len(docs) == 2
        assert {d["kind"] for d in docs} == {"search", "details"}


# ══ CandidateSource compatibility and the full pipeline ═════════════════════

def source(fake=None, **kw) -> SerpApiCandidateSource:
    fake = fake or FakeSerpApi()
    return SerpApiCandidateSource(provider(fake, **kw), QUERY, max_age=timedelta(hours=6), clock=lambda: NOW)


class TestCandidateSource:
    def test_it_is_a_live_candidate_source_with_the_planner_interface(self):
        s = source()
        assert isinstance(s, LiveCandidateSource) and callable(s.get_candidates)
        assert s.get_candidates() == []  # nothing until refresh

    def test_get_candidates_never_calls_the_network(self):
        fake = FakeSerpApi(); s = source(fake)
        s.refresh()
        n = len(fake.requests)
        for _ in range(5):
            s.get_candidates()
        assert len(fake.requests) == n

    def test_refresh_fills_the_snapshot_with_valid_records(self):
        s = source(); s.refresh()
        records = s.get_candidates()
        assert len(records) == 1 and validate_record(records[0])[1] == []

    def test_a_failed_refresh_keeps_the_previous_snapshot_and_reports(self):
        fake = FakeSerpApi(); s = source(fake)
        s.refresh()
        fake.status, fake.search = 429, {"error": "x"}
        with pytest.raises(ProductDiscoveryError):
            s.refresh()
        assert len(s.get_candidates()) == 1 and "429" in s.last_error

    def test_from_settings_builds_a_working_source(self):
        fake = FakeSerpApi()
        s = SerpApiCandidateSource.from_settings(settings(), cache_repository=InMemoryRepository("search_cache"),
                                                 transport=fake.transport, query=QUERY)
        s.refresh(); s.refresh()
        assert len(s.get_candidates()) == 1 and len(fake.requests) == 2  # the second refresh came from the cache

    def test_records_pass_through_the_utility_optimizer(self):
        s = source(); s.refresh()
        result = UtilityOptimizer().optimize(s.get_candidates(), REQUIREMENTS)
        assert result.selected is not None and result.selected.name == "Nordic Lite 3 Seater Sofa"
        assert result.selected.price == 24999.0

    def test_unconfirmed_availability_is_unverified_never_a_confirmed_recommendation(self):
        d = details_body(); d["product_results"]["stores"][0]["details_and_offers"] = []
        s = source(FakeSerpApi(details=d)); s.refresh()
        assert len(s.get_candidates()) == 1
        result = UtilityOptimizer().optimize(s.get_candidates(), REQUIREMENTS)
        assert result.selected is None and result.unverified[0].gaps == ["availability"]
        assert "check stock on the retailer" in result.unverified[0].reasons[0]

    def test_price_changes_between_refreshes_become_environment_events(self):
        fake = FakeSerpApi(); s = source(fake); s.refresh()
        d = details_body()
        d["product_results"]["stores"][0].update(price="₹21,999.00", extracted_price=21999.0)
        fake.details = d
        # a different cache-less provider call: refresh again with no cache => new data
        s.refresh()
        events = detect_changes(s.previous_listings, s.current_listings)
        assert any(e.event_type == EventType.PRICE_CHANGE for e in events)

    def test_the_serpapi_package_does_not_import_gemini_search_or_bdi_layers(self):
        import ast, pathlib
        pkg = pathlib.Path(parsing.__file__).parent
        for path in pkg.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                        [node.module or ""] if isinstance(node, ast.ImportFrom) else []
                for name in names:
                    assert not name.startswith(("smartnest.bdi", "smartnest.adaptive", "smartnest.knowledge")), (path.name, name)
                    assert "gemini.client" not in name and "gemini.service" not in name, (path.name, name)


# ══ Hard lifetime cap on requests (protects a small free quota) ═════════════

class TestLifetimeLimit:
    def capped(self, tmp_path, limit, fake=None):
        fake = fake or FakeSerpApi()
        s = SerpApiSettings(api_key=KEY, lifetime_limit=limit, usage_file=tmp_path / "usage.json")
        return SerpApiClient(s, transport=fake.transport), fake, s

    def test_requests_up_to_the_limit_are_sent_and_the_next_is_refused_locally(self, tmp_path):
        client, fake, _ = self.capped(tmp_path, 2)
        client.search("a"); client.search("b")
        with pytest.raises(SerpApiQuotaError, match="local SerpApi budget exhausted"):
            client.search("c")
        assert len(fake.requests) == 2  # the refused request never reached the transport

    def test_the_count_survives_a_new_client_and_a_restart(self, tmp_path):
        client, fake, s = self.capped(tmp_path, 3)
        client.search("a"); client.product_details("t")
        again = SerpApiClient(s, transport=fake.transport)
        assert again.ledger.used == 2 and again.ledger.remaining == 1
        again.search("c")
        with pytest.raises(SerpApiQuotaError):
            SerpApiClient(s, transport=fake.transport).search("d")
        assert len(fake.requests) == 3

    def test_failed_requests_still_count_because_they_may_cost_a_search(self, tmp_path):
        client, _, _ = self.capped(tmp_path, 5, FakeSerpApi(search={"error": "x"}, status=429))
        with pytest.raises(SerpApiQuotaError):
            client.search("a")
        assert client.ledger.used == 1

    def test_a_limit_of_zero_blocks_everything(self, tmp_path):
        client, fake, _ = self.capped(tmp_path, 0)
        with pytest.raises(SerpApiQuotaError):
            client.search("a")
        assert fake.requests == []

    def test_default_settings_touch_no_usage_file(self):
        assert SerpApiClient(settings(), transport=FakeSerpApi().transport).ledger is None

    def test_a_cap_removed_on_purpose_still_counts_but_never_refuses(self, tmp_path):
        fake = FakeSerpApi()
        client = SerpApiClient(settings(usage_file=tmp_path / "u.json", uncapped=True), transport=fake.transport)
        for i in range(5):
            client.search(f"q{i}")
        assert client.ledger.used == 5 and len(fake.requests) == 5

    def test_a_corrupt_usage_file_is_treated_as_zero_used(self, tmp_path):
        (tmp_path / "usage.json").write_text("garbage")
        client, _, _ = self.capped(tmp_path, 2)
        client.search("a")
        assert client.ledger.used == 1

    def test_a_cache_hit_costs_nothing_against_the_cap(self, tmp_path):
        fake = FakeSerpApi()
        s = SerpApiSettings(api_key=KEY, lifetime_limit=2, usage_file=tmp_path / "usage.json")
        client = SerpApiClient(s, transport=fake.transport)
        p = SerpApiProvider(client, s, cache=cache_for(), clock=lambda: NOW)
        p.discover(QUERY); p.discover(QUERY); p.discover(QUERY)
        assert client.ledger.used == 2 and len(fake.requests) == 2

    def test_the_limit_is_read_from_the_environment(self, tmp_path):
        s = load_serpapi_settings(tmp_path / "none.env", {"SERPAPI_API_KEY": KEY, "SERPAPI_LIFETIME_LIMIT": "10"})
        assert s.lifetime_limit == 10
        with pytest.raises(SerpApiConfigError):
            load_serpapi_settings(tmp_path / "none.env", {"SERPAPI_API_KEY": KEY, "SERPAPI_LIFETIME_LIMIT": "-1"})


# ══ Regression: a trimmed copy of a REAL SerpApi response (captured 2026-11-01) ══

class TestRealResponseShape:
    @pytest.fixture(scope="class")
    @classmethod
    def result(cls):
        import json, pathlib
        data = json.loads((pathlib.Path(__file__).parent / "fixtures" / "serpapi_real_sample.json").read_text(encoding="utf-8"))
        details = iter(data["details"])

        def handler(request):
            if request.url.params["engine"] == "google_shopping":
                return httpx.Response(200, json=data["search"])
            return httpx.Response(200, json=next(details))

        s = settings(max_detail_calls=3)
        p = SerpApiProvider(SerpApiClient(s, transport=httpx.MockTransport(handler)), s)
        return p.discover(QUERY)

    def test_google_product_links_with_spaces_are_usable_google_pages_not_retailer_urls(self, result):
        google = [l for l in result.listings if l["url_type"] == "google_shopping"]
        assert google and all(l["url"] is None and l["listing_url_type"] == "google_shopping" for l in google)
        assert all(l["product_page_url"].startswith("https://www.google.com/search?") for l in google)
        assert all(l["listing_url"] == l["product_page_url"] for l in google)    # shown as a Google page, not a retailer
        assert not any(" " in (l["product_page_url"] or "") for l in result.listings)

    def test_details_nested_under_product_results_yield_direct_retailer_urls(self, result):
        by = {l["merchant"]: l for l in result.listings}
        assert by["Home Centre"]["url"].startswith("https://www.homecentre.in/")
        assert by["Amazon.in"]["url"].startswith("https://www.amazon.in/")
        assert by["Wakefit"]["url"].startswith("https://www.wakefit.co/")
        assert all(by[m]["listing_url_type"] == "retailer" and by[m]["listing_url"] == by[m]["url"]
                   for m in ("Home Centre", "Amazon.in", "Wakefit"))
        assert all(l["url_type"] in ("retailer", "google_shopping") for l in result.listings)

    def test_relative_goto_links_are_never_used_as_urls(self, result):
        assert not any((l["url"] or "").startswith("/") for l in result.listings)

    def test_stated_material_is_read_and_unstated_fields_stay_null(self, result):
        helios = next(l for l in result.listings if "Helios" in l["name"])
        assert helios["material"] == "Leather" and helios["seating_capacity"] == 3 and helios["availability"] == "in_stock"
        assert all(helios[k] is None for k in ("colour", "style", "rating", "review_count", "width_cm", "depth_cm"))

    def test_all_real_products_are_accepted_because_only_name_price_and_url_are_mandatory(self, result):
        assert len(result.listings) == 10 and result.rejected == []
        assert all(l["style"] is None and l["colour"] is None for l in result.listings)   # never stated, never invented
