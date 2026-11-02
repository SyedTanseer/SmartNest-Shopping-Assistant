"""Optional product attributes, dynamic utility scoring, three-state availability and live-listing handling.

Rules under test:

* Mandatory: product name, a valid positive price (and, for a live listing, a usable listing URL).  Everything else -
  rating, review count, style, colour, material, seating, dimensions - is optional and stays null when not stated.
* Only scoring components supported by actual data are scored; their weights are scaled up in proportion so the
  effective weights sum to 1.  A missing attribute is never scored as zero.  Every score carries an
  information-completeness indicator and names the attributes that were unavailable.
* Availability is IN_STOCK, OUT_OF_STOCK or UNKNOWN.  Confirmed out of stock is rejected.  Unknown goes to a separately
  identified UNVERIFIED category - never a confirmed recommendation - telling the user to check the retailer's listing.
* A Google Shopping URL is never labelled a direct retailer link.
"""

from __future__ import annotations

import itertools
import math

import pytest

from smartnest.optimization import (
    AVAILABILITY_LABELS,
    ConstraintValidator,
    ScoringConfig,
    ScoringConfigError,
    UserRequirements,
    UtilityModel,
    UtilityOptimizer,
    ranking_key,
    validate_record,
)
from smartnest.optimization.constraint_validator import OUT_OF_STOCK_VALUES

REQ = UserRequirements(budget=30000, min_seating=3, preferred_style="minimalist", preferred_colour="neutral")
REQ_NO_PREF = UserRequirements(budget=30000, min_seating=3)
REQ_ROOM = UserRequirements(budget=30000, max_width_cm=210, max_depth_cm=110, min_seating=3,
                            preferred_style="minimalist", preferred_colour="neutral")


def rec(**over):
    data = dict(product_id="P1", name="Test sofa", price=24000, width_cm=200, depth_cm=100, seating_capacity=3,
                material="fabric", colour="neutral", style="minimalist", availability="in_stock", rating=4.0,
                review_count=50, source_agent_id="src")
    data.update(over)
    return {k: v for k, v in data.items() if v is not ...}


def cand(**over):
    c, errors = validate_record(rec(**over))
    assert c is not None, errors
    return c


def breakdown(req=REQ, **over):
    return UtilityModel().score(cand(**over), req)


# ══ Mandatory vs optional fields ═════════════════════════════════════════════

class TestFieldRequirements:
    def test_only_name_price_and_source_are_mandatory(self):
        minimal = {"product_id": "X", "name": "Plain sofa", "price": 18000, "source_agent_id": "s"}
        c, errors = validate_record(minimal)
        assert errors == [] and c is not None
        for field in ("width_cm", "depth_cm", "height_cm", "seating_capacity", "material", "colour", "style",
                      "rating", "review_count"):
            assert getattr(c, field) is None, field          # stays null - never 0, "", or an average
        assert c.availability == "unknown"                    # not stated is UNKNOWN, never in stock

    @pytest.mark.parametrize("field", ["name", "price", "source_agent_id", "product_id"])
    def test_a_missing_mandatory_field_is_invalid(self, field):
        record = rec(); del record[field]
        c, errors = validate_record(record)
        assert c is None and errors

    @pytest.mark.parametrize("price", [0, -1, math.nan, math.inf, "cheap", None, True])
    def test_the_price_must_be_a_valid_positive_number(self, price):
        assert validate_record(rec(price=price))[0] is None

    @pytest.mark.parametrize("field,bad", [("rating", 5.1), ("rating", -1), ("rating", "good"), ("review_count", 2.5),
                                           ("review_count", -3), ("seating_capacity", 0), ("seating_capacity", "three"),
                                           ("style", ""), ("colour", 7), ("material", "  ")])
    def test_an_optional_value_that_is_present_must_still_be_valid(self, field, bad):
        assert validate_record(rec(**{field: bad}))[0] is None

    def test_explicit_none_equals_absent_for_every_optional_field(self):
        c, errors = validate_record(rec(rating=None, review_count=None, style=None, colour=None, material=None,
                                        seating_capacity=None, availability=None))
        assert errors == [] and c.rating is None and c.availability == "unknown"

    def test_a_rating_of_zero_is_a_real_value_not_missing(self):
        assert cand(rating=0.0).rating == 0.0 and cand(review_count=0).review_count == 0

    def test_the_strict_complete_mode_still_demands_everything_for_cnp(self):
        for field in ("rating", "review_count", "style", "colour", "material", "seating_capacity", "availability"):
            record = rec(); del record[field]
            c, errors = validate_record(record, require_complete=True)
            assert c is None and f"{field} is missing" in errors, field
        assert validate_record(rec(), require_complete=True)[0] is not None

    def test_the_optimizer_exposes_the_strict_mode(self):
        record = rec(); del record["rating"]
        lenient = UtilityOptimizer().evaluate_candidate(record, REQ)
        strict = UtilityOptimizer().evaluate_candidate(record, REQ, require_complete=True)
        assert lenient.eligible and strict.status == "invalid" and "rating is missing" in strict.reasons[0]


# ══ Availability: IN_STOCK / OUT_OF_STOCK / UNKNOWN ═══════════════════════════

class TestAvailability:
    v = ConstraintValidator()

    @pytest.mark.parametrize("value,state", [("in_stock", "in_stock"), ("low_stock", "in_stock"),
                                             ("out_of_stock", "out_of_stock"), ("sold_out", "out_of_stock"),
                                             ("discontinued", "out_of_stock"), ("unavailable", "out_of_stock"),
                                             ("unknown", "unknown"), ("backorder", "unknown"), ("preorder", "unknown"),
                                             ("IN_STOCK", "in_stock"), ("  Out_Of_Stock ", "out_of_stock")])
    def test_state_mapping(self, value, state):
        assert self.v.availability_state(cand(availability=value)) == state

    def test_the_three_states_have_labels_and_unknown_tells_the_user_to_check_the_listing(self):
        assert set(AVAILABILITY_LABELS) == {"in_stock", "out_of_stock", "unknown"}
        assert "check stock on the retailer" in AVAILABILITY_LABELS["unknown"]

    def test_confirmed_out_of_stock_is_rejected_with_a_clear_reason(self):
        res = self.v.evaluate(rec(availability="out_of_stock"), REQ)
        assert res.status == "rejected" and res.reasons == ["confirmed out of stock (availability: out_of_stock)"]
        assert res.availability == "out_of_stock"

    def test_unknown_availability_is_unverified_not_rejected_and_never_in_stock(self):
        res = self.v.evaluate(rec(availability="unknown"), REQ)
        assert res.status == "unverified" and not res.eligible and res.availability == "unknown"
        assert res.gaps == ["availability"]
        assert "check stock on the retailer" in res.reasons[0]

    def test_a_record_that_does_not_state_availability_is_unknown(self):
        record = rec(); del record["availability"]
        res = self.v.evaluate(record, REQ)
        assert res.status == "unverified" and res.availability == "unknown"

    def test_in_stock_with_everything_else_checkable_is_eligible(self):
        res = self.v.evaluate(rec(), REQ)
        assert res.eligible and res.gaps == [] and res.availability == "in_stock"

    def test_out_of_stock_wins_over_unverified_gaps(self):
        res = self.v.evaluate(rec(availability="out_of_stock", seating_capacity=None), REQ)
        assert res.status == "rejected" and res.gaps == []

    def test_available_states_remain_configurable(self):
        strict = ConstraintValidator(ScoringConfig(available_states=["in_stock"]))
        assert strict.availability_state(cand(availability="low_stock")) == "unknown"
        loose = ConstraintValidator(ScoringConfig(available_states=["in_stock", "preorder"]))
        assert loose.availability_state(cand(availability="preorder")) == "in_stock"

    def test_optimizer_keeps_unknown_stock_out_of_the_confirmed_recommendation(self):
        records = [rec(product_id="KNOWN", price=26000), rec(product_id="UNKNOWN", availability="unknown", price=10000, rating=5)]
        r = UtilityOptimizer().optimize(records, REQ)
        assert r.selected.product_id == "KNOWN"                       # the cheaper unknown-stock one is not recommended
        assert [e.product_id for e in r.unverified] == ["UNKNOWN"]
        assert r.unverified[0].availability_label == AVAILABILITY_LABELS["unknown"]
        assert r.unverified[0].utility > r.selected.utility           # higher score, still not confirmed
        assert "check stock on the retailer" in r.explanation

    def test_with_only_unknown_stock_nothing_is_confirmed_but_the_candidates_are_reported(self):
        r = UtilityOptimizer().optimize([rec(availability="unknown")], REQ)
        assert r.selected is None and r.unverified_count == 1 and r.best_unverified.product_id == "P1"
        assert r.unverified_for("availability") == r.unverified and r.unverified_for("fit") == []

    def test_out_of_stock_products_are_never_in_the_unverified_list(self):
        r = UtilityOptimizer().optimize([rec(availability="out_of_stock")], REQ)
        assert r.unverified == [] and r.rejected_count == 1 and r.selected is None

    def test_out_of_stock_synonyms_are_a_closed_set(self):
        assert OUT_OF_STOCK_VALUES == {"out_of_stock", "sold_out", "unavailable", "discontinued"}

    def test_every_gap_is_named_when_several_apply(self):
        res = self.v.evaluate(rec(availability="unknown", seating_capacity=None, depth_cm=None), REQ_ROOM)
        assert res.status == "unverified" and res.gaps == ["fit", "seating", "availability"] and len(res.reasons) == 3

    def test_an_unstated_seating_capacity_matters_only_when_more_than_one_seat_is_required(self):
        one = UserRequirements(budget=30000)  # min_seating defaults to 1
        assert ConstraintValidator().evaluate(rec(seating_capacity=None), one).eligible
        assert ConstraintValidator().evaluate(rec(seating_capacity=None), REQ).gaps == ["seating"]
        assert ConstraintValidator().evaluate(rec(seating_capacity=2), REQ).status == "rejected"   # a stated 2 is known


# ══ Dynamic scoring ═══════════════════════════════════════════════════════════

class TestDynamicScoring:
    def test_complete_data_uses_all_configured_components_and_weights(self):
        b = breakdown(REQ_ROOM)
        assert set(b.components) == {"price", "quality", "spatial", "style", "colour"}
        assert b.weights == pytest.approx(ScoringConfig().weights) and b.completeness == 1.0
        assert b.excluded == {} and b.unavailable == []
        expected = 100 * (0.30 * b.components["price"] + 0.25 * b.components["quality"] + 0.20 * b.components["spatial"]
                          + 0.15 * b.components["style"] + 0.10 * b.components["colour"])
        assert b.total_utility == pytest.approx(expected, abs=1e-4)

    def test_a_missing_rating_excludes_quality_and_redistributes_it(self):
        b = breakdown(rating=None, review_count=None)
        assert "quality" not in b.components and b.excluded["quality"] == "rating not stated"
        assert set(b.unavailable) == {"rating", "review count"}
        assert b.weights == pytest.approx({"price": 0.30 / 0.55, "style": 0.15 / 0.55, "colour": 0.10 / 0.55})
        assert sum(b.weights.values()) == pytest.approx(1.0, abs=1e-12)

    def test_a_missing_rating_is_not_scored_as_zero(self):
        missing = breakdown(rating=None, review_count=None).total_utility
        zero = breakdown(rating=0.0, review_count=0).total_utility
        assert missing > zero and "quality" in breakdown(rating=0.0, review_count=0).components

    def test_a_rating_without_a_review_count_is_scored_on_the_rating_alone(self):
        b = breakdown(rating=4.0, review_count=None)
        assert b.components["quality"] == pytest.approx(0.8)   # 4/5, not diluted by an invented zero review count
        assert b.unavailable == ["review count"] and "quality" not in b.excluded
        assert breakdown(rating=4.0, review_count=100).components["quality"] == pytest.approx(0.8 * 0.8 + 0.2)

    def test_a_review_count_without_a_rating_does_not_create_a_quality_score(self):
        b = breakdown(rating=None, review_count=500)
        assert "quality" not in b.components and b.unavailable == ["rating"]

    @pytest.mark.parametrize("field,component", [("style", "style"), ("colour", "colour")])
    def test_a_missing_style_or_colour_is_excluded_when_the_user_stated_a_preference(self, field, component):
        b = breakdown(**{field: None})
        assert component not in b.components and b.excluded[component] == f"{field} not stated"
        assert field in b.unavailable

    def test_without_a_stated_preference_style_and_colour_are_not_applicable_and_not_scored(self):
        b = breakdown(REQ_NO_PREF)
        assert set(b.components) == {"price", "quality"} and b.completeness == 1.0
        assert b.excluded["style"] == "no style preference given" and b.unavailable == []
        assert breakdown(REQ_NO_PREF, style=None, colour=None).total_utility == b.total_utility   # irrelevant to the score

    def test_price_only_when_nothing_else_is_stated(self):
        b = breakdown(rating=None, review_count=None, style=None, colour=None)
        assert list(b.components) == ["price"] and b.weights == {"price": 1.0}
        assert b.total_utility == pytest.approx(100 * (1 - 24000 / 30000), abs=1e-4)
        assert b.completeness == pytest.approx(0.30 / 0.80)   # price .30 of price+quality+style+colour .80

    def test_completeness_is_the_share_of_applicable_weight_that_was_scored(self):
        assert breakdown(REQ_ROOM).completeness == 1.0
        assert breakdown(REQ_ROOM, style=None).completeness == pytest.approx(0.85 / 1.0)
        assert breakdown(REQ_ROOM, rating=None, review_count=None).completeness == pytest.approx(0.75)
        assert breakdown(REQ_ROOM, depth_cm=None).completeness == pytest.approx(0.80)   # spatial 0.20 lost
        assert breakdown(REQ_NO_PREF, rating=None, review_count=None).completeness == pytest.approx(0.30 / 0.55)

    def test_spatial_not_applicable_without_limits_does_not_count_as_missing_information(self):
        assert breakdown(REQ).completeness == 1.0 and "spatial" not in breakdown(REQ).components

    def test_the_existing_spatial_scoring_is_preserved_when_it_applies(self):
        b = breakdown(REQ_ROOM, rating=None, review_count=None)
        assert b.spatial_included and b.components["spatial"] == pytest.approx(0.5 * 200 / 210 + 0.5 * 100 / 110, abs=1e-6)
        assert b.weights["spatial"] == pytest.approx(0.20 / 0.75)

    def test_unknown_size_with_limits_marks_the_size_unavailable(self):
        b = breakdown(REQ_ROOM, width_cm=None)
        assert "spatial" not in b.components and "width" in b.unavailable and b.excluded["spatial"].startswith("size not stated")

    @pytest.mark.parametrize("have", [set(c) for r in range(1, 5) for c in itertools.combinations(("quality", "style", "colour", "spatial"), r)])
    def test_effective_weights_always_sum_to_one_for_every_combination_of_available_data(self, have):
        over = {}
        if "quality" not in have:
            over.update(rating=None, review_count=None)
        if "style" not in have:
            over["style"] = None
        if "colour" not in have:
            over["colour"] = None
        if "spatial" not in have:
            over["width_cm"] = None
        b = breakdown(REQ_ROOM, **over)
        assert set(b.components) == {"price"} | have
        assert sum(b.weights.values()) == pytest.approx(1.0, abs=1e-12)
        assert all(w > 0 for w in b.weights.values())
        assert 0 <= b.total_utility <= 100
        assert sum(b.contributions.values()) == pytest.approx(b.total_utility, abs=1e-4)

    def test_ratios_between_remaining_weights_are_preserved(self):
        b = breakdown(REQ_ROOM, style=None, rating=None, review_count=None)
        base = ScoringConfig().weights
        assert b.weights["price"] / b.weights["colour"] == pytest.approx(base["price"] / base["colour"])
        assert b.weights["price"] / b.weights["spatial"] == pytest.approx(base["price"] / base["spatial"])

    def test_no_artificial_zero_appears_anywhere_for_missing_information(self):
        b = breakdown(REQ_ROOM, rating=None, review_count=None, style=None, colour=None, width_cm=None)
        assert list(b.components) == ["price"]
        assert all(k not in b.contributions for k in ("quality", "style", "colour", "spatial"))

    def test_the_completeness_note_names_what_was_unavailable(self):
        note = breakdown(rating=None, review_count=None, style=None).completeness_note
        assert "Scored on" in note and "Not available, so not scored: rating, review count, style." in note
        assert "Not available" not in breakdown().completeness_note

    def test_weights_for_validates_its_input(self):
        cfg = ScoringConfig()
        assert cfg.weights_for(["price", "quality"]) == pytest.approx({"price": 0.30 / 0.55, "quality": 0.25 / 0.55})
        assert cfg.weights_for(list(cfg.weights)) == cfg.weights
        with pytest.raises(ScoringConfigError, match="unknown"):
            cfg.weights_for(["price", "taste"])
        with pytest.raises(ScoringConfigError, match="cannot be redistributed"):
            cfg.weights_for([])

    def test_a_configuration_with_zero_weight_everywhere_remaining_is_refused(self):
        cfg = ScoringConfig(weights={"price": 0, "quality": 0, "spatial": 1.0, "style": 0, "colour": 0})
        with pytest.raises(ScoringConfigError):
            cfg.weights_for(["price", "quality"])


# ══ Ranking with incomplete data ═════════════════════════════════════════════

class TestRanking:
    def test_a_product_missing_data_is_ranked_on_what_is_stated_and_flagged(self):
        records = [rec(product_id="FULL", price=24000), rec(product_id="SPARSE", price=24000, rating=None, review_count=None, style=None)]
        r = UtilityOptimizer().optimize(records, REQ)
        assert {e.product_id for e in r.ranking} == {"FULL", "SPARSE"}
        sparse = next(e for e in r.ranking if e.product_id == "SPARSE")
        assert sparse.completeness < 1 and sparse.unavailable == ["rating", "review count", "style"]

    def test_exact_utility_ties_go_to_the_more_complete_product(self):
        # Same price, neither has a rating (so the quality key cannot decide), ids chosen so id order favours the sparse one.
        records = [rec(product_id="A_SPARSE", price=24000, rating=None, review_count=None, style=None, colour=None),
                   rec(product_id="B_FULL", price=24000, rating=None, review_count=None)]
        r = UtilityOptimizer().optimize(records, REQ)
        sparse = next(e for e in r.ranking if e.product_id == "A_SPARSE")
        full = next(e for e in r.ranking if e.product_id == "B_FULL")
        assert full.completeness > sparse.completeness
        # Force exactly equal utilities: only completeness (then, if equal, quality, price, id) may decide.
        tied_sparse = sparse.model_copy(update={"breakdown": sparse.breakdown.model_copy(update={"total_utility": 50.0})})
        tied_full = full.model_copy(update={"breakdown": full.breakdown.model_copy(update={"total_utility": 50.0})})
        assert [e.product_id for e in sorted([tied_sparse, tied_full], key=ranking_key)] == ["B_FULL", "A_SPARSE"]

    def test_utility_still_comes_first(self):
        cheap_sparse = rec(product_id="CHEAP", price=10000, rating=None, review_count=None, style=None, colour=None)
        full = rec(product_id="FULL", price=26000)
        r = UtilityOptimizer().optimize([full, cheap_sparse], REQ)
        assert r.selected.product_id == "CHEAP"          # documented: scored on price alone
        assert r.selected.completeness < r.ranking[1].completeness and r.selected.completeness is not None

    def test_ranking_key_tolerates_a_missing_quality_component(self):
        r = UtilityOptimizer().optimize([rec(rating=None, review_count=None)], REQ)
        assert ranking_key(r.ranking[0])      # no KeyError

    def test_ordering_is_deterministic_for_mixed_data(self):
        records = [rec(product_id=f"P{i}", price=20000 + 1000 * i, rating=None if i % 2 else 4.0, style=None if i % 3 == 0 else "minimalist")
                   for i in range(8)]
        a = UtilityOptimizer().optimize(records, REQ)
        b = UtilityOptimizer().optimize(list(reversed(records)), REQ)
        assert [e.product_id for e in a.ranking] == [e.product_id for e in b.ranking]
        assert a.selected.product_id == b.selected.product_id and a.verified_optimal

    def test_confirmed_and_unverified_are_ranked_in_separate_lists(self):
        records = [rec(product_id="OK1", price=25000), rec(product_id="OK2", price=26000),
                   rec(product_id="U1", availability="unknown", price=12000), rec(product_id="U2", seating_capacity=None, price=15000)]
        r = UtilityOptimizer().optimize(records, REQ)
        assert [e.product_id for e in r.ranking] == ["OK1", "OK2"]
        assert [e.product_id for e in r.unverified] == ["U1", "U2"] and [e.rank for e in r.unverified] == [1, 2]
        assert r.eligible_count + r.unverified_count + r.rejected_count == r.total_inspected == 4

    def test_the_explanation_mentions_incompleteness_and_the_unverified_candidates(self):
        r = UtilityOptimizer().optimize([rec(rating=None, review_count=None), rec(product_id="U", availability="unknown")], REQ)
        assert "Not available, so not scored: rating, review count." in r.explanation
        assert "1 other candidate(s)" in r.explanation and "stock" in r.explanation and "check stock" in r.explanation

    def test_complete_records_rank_exactly_as_before(self):
        records = [rec(product_id=f"P{i}", price=20000 + 800 * i, rating=3 + i / 4) for i in range(5)]
        r = UtilityOptimizer().optimize(records, REQ_ROOM)
        assert all(e.completeness == 1.0 and e.gaps == [] for e in r.ranking) and r.unverified == []


# ══ Live listings: URLs, incomplete records, the real response shape ═════════

class TestListingUrls:
    @staticmethod
    def build(item_over=None, details="default"):
        from tests.test_serpapi_discovery import build, shopping_item
        return build(shopping_item(**(item_over or {})), details=details)

    def test_a_direct_retailer_listing_is_labelled_retailer(self):
        r = self.build().record
        assert (r["url_type"], r["listing_url_type"]) == ("retailer", "retailer")
        assert r["listing_url"] == r["url"] == "https://www.pepperfry.com/product/nordic-lite-3-seater-1234"

    def test_a_google_shopping_page_is_never_labelled_a_retailer_link(self):
        r = self.build(details=None).record
        assert r["url"] is None and r["url_type"] == "google_shopping"
        assert r["listing_url_type"] == "google_shopping" and r["listing_url"] == r["product_page_url"]
        assert r["listing_url"].startswith("https://www.google.com/")

    def test_a_listing_with_no_link_at_all_is_rejected_for_lack_of_a_usable_url(self):
        built = self.build({"product_link": None}, details=None)
        assert built.record is None and built.summary["missing"] == ["listing_url"]
        assert any("usable listing URL" in x for x in built.reasons) and built.summary["listing_url_type"] == "missing"

    @pytest.mark.parametrize("link", ["http://insecure.example.com/p", "not a url", "javascript:alert(1)", "/goto?url=abc",
                                      "https://localhost/p", "ftp://x.example.com/f"])
    def test_an_invalid_link_is_rejected(self, link):
        built = self.build({"product_link": link}, details=None)
        assert built.record is None and built.summary["listing_url_type"] == "invalid"

    def test_a_product_page_on_a_non_google_domain_counts_as_a_retailer_link(self):
        r = self.build({"product_link": "https://www.urbanladder.com/products/sofa-1"}, details=None).record
        assert r["listing_url_type"] == "retailer" and r["listing_url"].startswith("https://www.urbanladder.com/")

    def test_the_name_and_price_are_mandatory_for_a_live_listing(self):
        assert self.build({"title": None}).record is None
        assert self.build({"price": None, "extracted_price": None}, details=None).record is None
        assert self.build({"price": "$499", "extracted_price": 499.0}, details=None).record is None   # not INR

    def test_everything_else_may_be_missing(self):
        item = {"rating": None, "reviews": None, "snippet": None, "extensions": None, "thumbnail": None, "source": None}
        r = self.build(item, details=None).record
        assert r is not None and r["rating"] is None and r["review_count"] is None and r["image_url"] is None
        assert validate_record(r)[1] == []


class TestRealResponseThroughTheOptimizer:
    @pytest.fixture(scope="class")
    @classmethod
    def run(cls):
        import json, pathlib
        import httpx
        from smartnest.integrations.products import ProductQuery
        from smartnest.integrations.products.serpapi import SerpApiClient, SerpApiProvider, SerpApiSettings
        data = json.loads((pathlib.Path(__file__).parent / "fixtures" / "serpapi_real_sample.json").read_text(encoding="utf-8"))
        details = iter(data["details"])

        def handler(request):
            if request.url.params["engine"] == "google_shopping":
                return httpx.Response(200, json=data["search"])
            return httpx.Response(200, json=next(details))

        s = SerpApiSettings(api_key="test-key-0123456789abcdef", max_detail_calls=3)
        result = SerpApiProvider(SerpApiClient(s, transport=httpx.MockTransport(handler)), s).discover(
            ProductQuery(category="sofa", max_price=30000, min_seating=3, style="minimalist"))
        return result, UtilityOptimizer().optimize(result.listings, REQ)

    def test_real_listings_reach_the_recommendation_stage(self, run):
        result, r = run
        assert len(result.listings) == 10 and result.rejected == []
        assert r.selected is not None and r.eligible_count == 2
        assert r.selected.availability == "in_stock" and r.selected.fit_label == "No room limits specified"

    def test_the_recommendation_is_a_direct_retailer_listing_with_a_real_price(self, run):
        result, r = run
        rec_ = next(l for l in result.listings if l["product_id"] == r.selected.product_id)
        assert rec_["listing_url_type"] == "retailer" and rec_["listing_url"].startswith("https://www.")
        assert rec_["price"] == r.selected.price and rec_["merchant"]

    def test_unknown_stock_and_unstated_seating_candidates_are_held_back(self, run):
        _, r = run
        assert [e.gaps for e in r.unverified] == [["seating"]]                  # Wakefit: in stock, seating not stated
        assert all(e.availability == "unknown" for e in r.rejected if "price" in e.reasons[0])  # over budget anyway

    def test_what_real_data_does_not_state_is_reported_not_invented(self, run):
        _, r = run
        for e in r.ranking:
            assert {"rating", "style"} <= set(e.unavailable) and e.completeness < 1
            assert set(e.breakdown.components) == {"price"}                     # only price had data

    def test_google_only_results_are_labelled_and_their_stock_is_unknown(self, run):
        result, _ = run
        google = [l for l in result.listings if l["listing_url_type"] == "google_shopping"]
        assert google and all(l["url"] is None and l["availability"] == "unknown" for l in google)


# ══ Recommendation wording and the session ═══════════════════════════════════

class TestRecommendationOutput:
    @staticmethod
    def text(**over):
        from smartnest.integrations.gemini import GeminiService
        from smartnest.integrations.gemini.schemas import RecommendationFacts
        base = dict(product_id="P1", product_name="Test sofa", price=24000, utility=60.0)
        return GeminiService.template_explanation(RecommendationFacts(**{**base, **over})).text

    def test_the_template_states_incompleteness_and_names_the_unavailable_attributes(self):
        t = self.text(completeness_percent=42.9, unavailable_attributes=["rating", "style"])
        assert "42.9% of the applicable criteria" in t and "rating, style" in t

    def test_a_complete_score_makes_no_such_claim(self):
        assert "applicable criteria" not in self.text(completeness_percent=100.0)
        assert "applicable criteria" not in self.text()

    def test_a_google_link_is_called_a_google_shopping_page(self):
        assert "Google Shopping page, not a direct retailer listing" in self.text(url_type="google_shopping")
        assert "Google Shopping" not in self.text(url_type="retailer")

    def test_the_session_exposes_unverified_options_separately(self):
        from smartnest.integrations.products import DiscoveryResult, ProductQuery
        from tests.bdi_support import SELECT
        from tests.test_adaptive_session import World
        w = World()
        w.session.request("I'd like a minimalist three-seater sofa under ₹30,000 for my living room.")
        records = [rec(product_id="U1", name="Unknown stock sofa", availability="unknown", price=15000),
                   rec(product_id="OK", name="Known sofa", price=26000)]
        w.source._listings = [dict(r, merchant="Shop", listing_url="https://shop.example.com/p/1",
                                   listing_url_type="retailer", url="https://shop.example.com/p/1") for r in records]
        outcome = w.agent.last_outcomes[SELECT]
        outcome.optimization = UtilityOptimizer().optimize(records, REQ)
        options = w.session._unverified_options()
        assert [o["product_id"] for o in options] == ["U1"] and options[0]["gaps"] == ["availability"]
        assert options[0]["availability_label"] == AVAILABILITY_LABELS["unknown"]
        assert options[0]["merchant"] == "Shop" and options[0]["url_type"] == "retailer"

    def test_no_unverified_options_when_everything_is_confirmed(self):
        from tests.test_adaptive_session import World
        w = World()
        turn = w.session.request("I'd like a minimalist three-seater sofa under ₹30,000 for my living room.")
        assert turn.unverified_options == [] and turn.recommendation.availability == "in_stock"
        assert turn.recommendation.completeness == 1.0


class TestKnowledgePersistenceOfRealListings:
    """Regression: the first live run found that listings with a URL type / Google page could not be saved."""

    def test_a_real_shaped_listing_is_persisted_with_its_url_labels(self):
        from smartnest.adaptive.session import SessionTurn
        from smartnest.integrations.products import DiscoveryResult, ProductQuery
        from smartnest.knowledge import KnowledgeKind
        from tests.test_adaptive_session import World
        from tests.test_serpapi_discovery import build, shopping_item
        w = World()
        direct = build(shopping_item()).record
        google = build(shopping_item(product_id="G1", title="Google only sofa"), details=None).record
        result = DiscoveryResult(provider_id="serpapi_google_shopping", query=ProductQuery(), listings=[direct, google])
        turn = SessionTurn(kind="request")
        w.session._remember_products(result, turn)
        assert not any("Knowledge was not saved" in n for n in turn.notes), turn.notes
        assert w.knowledge.count(KnowledgeKind.PRODUCT_SPEC) == 2
        saved = {r.attributes["product_id"]: r.attributes for r in w.knowledge.find(KnowledgeKind.PRODUCT_SPEC)}
        assert saved[direct["product_id"]]["url_type"] == "retailer"
        assert saved[direct["product_id"]]["url"].startswith("https://www.pepperfry.com/")
        g = saved["SERP_G1"]
        assert g["url_type"] == "google_shopping" and g.get("url") is None and g["product_page_url"].startswith("https://www.google.com/")
        assert g["rating"] == 4.4                                                          # stated -> stored
        assert all(g.get(k) is None for k in ("style", "colour", "width_cm"))             # unstated stays null
