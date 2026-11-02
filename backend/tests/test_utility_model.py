"""Tests for ScoringConfig and UtilityModel (Assignment 3.2).

Verifies each normalised component against the assignment formulas,
the weighted total, configuration validation and determinism.
"""

from __future__ import annotations

import math

import pytest

from smartnest.optimization import (
    COMPONENTS,
    DEFAULT_WEIGHTS,
    ProductCandidate,
    ScoringConfig,
    ScoringConfigError,
    UserRequirements,
    UtilityModel,
)

REQ = UserRequirements(budget=30000, max_width_cm=210, max_depth_cm=110, min_seating=3,
                       preferred_style="minimalist", preferred_colour="neutral")


def _product(**overrides) -> ProductCandidate:
    data = dict(product_id="P1", name="Test", price=27499, width_cm=195, depth_cm=92,
                seating_capacity=3, material="leather", colour="neutral", style="minimalist",
                availability="in_stock", rating=4.5, review_count=88, source_agent_id="a")
    data.update(overrides)
    return ProductCandidate(**data)


# ── Configuration ────────────────────────────────────────────────────────────

class TestScoringConfig:
    def test_default_weights_match_assignment(self):
        assert ScoringConfig().weights == {
            "price": 0.30, "quality": 0.25, "spatial": 0.20, "style": 0.15, "colour": 0.10,
        }
        assert set(DEFAULT_WEIGHTS) == set(COMPONENTS)

    def test_weights_not_summing_to_one_rejected(self):
        with pytest.raises(ScoringConfigError, match="sum to 1"):
            ScoringConfig(weights={**DEFAULT_WEIGHTS, "price": 0.40})

    def test_negative_weight_rejected(self):
        with pytest.raises(ScoringConfigError, match="non-negative"):
            ScoringConfig(weights={**DEFAULT_WEIGHTS, "price": -0.10, "quality": 0.65})

    @pytest.mark.parametrize("bad", [math.nan, math.inf, "0.3", True])
    def test_non_finite_or_non_numeric_weight_rejected(self, bad):
        with pytest.raises(ScoringConfigError):
            ScoringConfig(weights={**DEFAULT_WEIGHTS, "price": bad})

    def test_missing_and_unknown_weight_keys_rejected(self):
        with pytest.raises(ScoringConfigError, match="missing"):
            ScoringConfig(weights={"price": 1.0})
        with pytest.raises(ScoringConfigError, match="unknown"):
            ScoringConfig(weights={**DEFAULT_WEIGHTS, "comfort": 0.0})

    def test_sub_weights_must_sum_to_one(self):
        with pytest.raises(ScoringConfigError, match="quality"):
            ScoringConfig(quality_rating_weight=0.9, quality_review_weight=0.2)
        with pytest.raises(ScoringConfigError, match="spatial"):
            ScoringConfig(spatial_width_weight=0.7, spatial_depth_weight=0.7)

    def test_match_scores_must_be_ordered(self):
        with pytest.raises(ScoringConfigError):
            ScoringConfig(compatible_score=1.5)
        with pytest.raises(ScoringConfigError):
            ScoringConfig(exact_match_score=0.4, compatible_score=0.5)

    def test_invalid_review_saturation_rejected(self):
        with pytest.raises(ScoringConfigError):
            ScoringConfig(review_saturation=0)

    def test_with_weights_revalidates(self):
        cfg = ScoringConfig().with_weights(price=0.40, quality=0.15)
        assert cfg.weights["price"] == 0.40
        with pytest.raises(ScoringConfigError):
            ScoringConfig().with_weights(price=0.99)

    def test_normalised_helper(self):
        w = ScoringConfig.normalised({"price": 3, "quality": 2.5, "spatial": 2, "style": 1.5, "colour": 1})
        assert ScoringConfig(weights=w).weights == pytest.approx(DEFAULT_WEIGHTS)
        with pytest.raises(ScoringConfigError):
            ScoringConfig.normalised({"price": 0})

    def test_compatibility_tables_are_configurable_and_normalised(self):
        cfg = ScoringConfig(style_compatibility={"Boho": ["Eclectic", "eclectic"]})
        assert cfg.style_compatibility == {"boho": ["eclectic"]}
        with pytest.raises(ScoringConfigError):
            ScoringConfig(colour_families={"neutral": [""]})

    def test_config_is_serialisable(self):
        cfg = ScoringConfig()
        assert ScoringConfig(**cfg.model_dump()) == cfg
        assert '"price":0.3' in cfg.model_dump_json()


# ── Components ───────────────────────────────────────────────────────────────

class TestPriceScore:
    def test_formula(self):
        assert UtilityModel().price_score(27499, 30000) == pytest.approx(1 - 27499 / 30000)

    def test_price_equal_to_budget_scores_zero(self):
        assert UtilityModel().price_score(30000, 30000) == 0.0

    def test_clamped_to_unit_interval(self):
        m = UtilityModel()
        assert m.price_score(40000, 30000) == 0.0
        assert m.price_score(0, 30000) == 1.0

    @pytest.mark.parametrize("budget", [0, -1, math.nan, math.inf])
    def test_invalid_budget_raises_not_divides_by_zero(self, budget):
        with pytest.raises(ValueError):
            UtilityModel().price_score(100, budget)


class TestQualityScore:
    def test_formula(self):
        assert UtilityModel().quality_score(4.5, 88) == pytest.approx(0.8 * 4.5 / 5 + 0.2 * 0.88)

    def test_review_term_saturates_at_100(self):
        m = UtilityModel()
        assert m.quality_score(5, 100) == pytest.approx(1.0)
        assert m.quality_score(5, 10_000) == pytest.approx(1.0)

    def test_extremes(self):
        m = UtilityModel()
        assert m.quality_score(0, 0) == 0.0
        assert m.quality_score(5, 0) == pytest.approx(0.8)

    @pytest.mark.parametrize("rating", [-0.1, 5.01, math.nan, math.inf])
    def test_rating_outside_0_5_raises(self, rating):
        with pytest.raises(ValueError):
            UtilityModel().quality_score(rating, 10)


class TestSpatialScore:
    def test_formula(self):
        assert UtilityModel().spatial_score(195, 92, 210, 110) == pytest.approx(
            0.5 * 195 / 210 + 0.5 * 92 / 110)

    def test_exact_fit_scores_one(self):
        assert UtilityModel().spatial_score(210, 110, 210, 110) == pytest.approx(1.0)

    def test_oversized_product_is_not_scored(self):
        with pytest.raises(ValueError, match="within the room limits"):
            UtilityModel().spatial_score(211, 100, 210, 110)

    @pytest.mark.parametrize("args", [(0, 100, 210, 110), (-5, 100, 210, 110),
                                      (100, 100, 0, 110), (100, math.nan, 210, 110)])
    def test_invalid_dimensions_raise(self, args):
        with pytest.raises(ValueError):
            UtilityModel().spatial_score(*args)


class TestStyleAndColourScores:
    def test_style_exact_compatible_incompatible(self):
        m = UtilityModel()
        assert m.style_score("minimalist", "minimalist") == 1.0
        assert m.style_score("Scandinavian", "minimalist") == 0.5
        assert m.style_score("traditional", "minimalist") == 0.0
        assert m.style_score("boho", "minimalist") == 0.0  # unknown → incompatible

    def test_colour_exact_family_incompatible(self):
        m = UtilityModel()
        assert m.colour_score("neutral", "neutral") == 1.0
        assert m.colour_score("beige", "neutral") == 0.5
        assert m.colour_score("burgundy", "neutral") == 0.0

    def test_no_preference_scores_exact(self):
        m = UtilityModel()
        assert m.style_score("industrial", None) == 1.0
        assert m.colour_score("red", None) == 1.0

    def test_empty_value_raises(self):
        with pytest.raises(ValueError):
            UtilityModel().style_score("  ", "minimalist")

    def test_custom_tables_change_scores(self):
        cfg = ScoringConfig(style_compatibility={"minimalist": ["industrial"]})
        m = UtilityModel(cfg)
        assert m.style_score("industrial", "minimalist") == 0.5
        assert m.style_score("scandinavian", "minimalist") == 0.0


# ── Total utility ────────────────────────────────────────────────────────────

class TestTotalUtility:
    def test_total_matches_formula(self):
        b = UtilityModel().score(_product(), REQ)
        P = 1 - 27499 / 30000
        Q = 0.8 * 4.5 / 5 + 0.2 * 0.88
        F = 0.5 * 195 / 210 + 0.5 * 92 / 110
        expected = 100 * (0.30 * P + 0.25 * Q + 0.20 * F + 0.15 * 1 + 0.10 * 1)
        assert b.total_utility == pytest.approx(expected, abs=1e-6)
        assert b.total_utility == pytest.approx(67.550351, abs=1e-6)

    def test_contributions_sum_to_total(self):
        b = UtilityModel().score(_product(), REQ)
        assert sum(b.contributions.values()) == pytest.approx(b.total_utility, abs=1e-5)
        assert b.normalised_utility == pytest.approx(b.total_utility / 100, abs=1e-6)

    def test_all_components_in_unit_interval(self):
        b = UtilityModel().score(_product(style="traditional", colour="red"), REQ)
        assert all(0.0 <= v <= 1.0 for v in b.components.values())
        assert 0.0 <= b.total_utility <= 100.0

    def test_perfect_product_scores_bounded(self):
        best = _product(price=1, width_cm=210, depth_cm=110, rating=5, review_count=1000)
        assert UtilityModel().score(best, REQ).total_utility <= 100.0

    def test_weights_change_the_result(self):
        price_heavy = ScoringConfig(weights={"price": 0.9, "quality": 0.025, "spatial": 0.025,
                                             "style": 0.025, "colour": 0.025})
        quality_heavy = ScoringConfig(weights={"price": 0.025, "quality": 0.9, "spatial": 0.025,
                                               "style": 0.025, "colour": 0.025})
        cheap = _product(price=10000, rating=3.0)
        dear = _product(price=29000, rating=5.0, review_count=500)
        by_price, by_quality = UtilityModel(price_heavy), UtilityModel(quality_heavy)
        # The same two sofas swap places when the weights change.
        assert by_price.score(cheap, REQ).total_utility > by_price.score(dear, REQ).total_utility
        assert by_quality.score(dear, REQ).total_utility > by_quality.score(cheap, REQ).total_utility

    def test_repeated_evaluation_is_identical(self):
        m = UtilityModel()
        results = {m.score(_product(), REQ).model_dump_json() for _ in range(25)}
        assert len(results) == 1

    def test_breakdown_is_serialisable(self):
        b = UtilityModel().score(_product(), REQ)
        assert type(b).model_validate_json(b.model_dump_json()) == b
