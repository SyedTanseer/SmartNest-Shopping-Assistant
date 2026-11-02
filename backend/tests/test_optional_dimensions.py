"""Optional product dimensions, optional room limits, conditional spatial validation and dynamic weights.

The rules under test:

* Room limits (width / depth / height) are optional user inputs; a product's width / depth / height may be null.
* A limit is enforced only if the user gave it.  A stated size over a limit  -> "Dimensionally incompatible" (rejected).
  A needed size that is missing                                             -> "Fit not verified" (unverified, kept apart).
  Everything needed present and within the limits                           -> "Fit verified".
  No limits at all                                                          -> "No room limits specified".
* SpaceFit (weight 0.20) counts only for a verified fit; otherwise it is excluded and the other weights are scaled up
  proportionally so the effective weights still sum to 1.  No spatial score is ever invented.
"""

from __future__ import annotations

import math
import random

import pytest

from smartnest.bdi import BDIAgent, BDIPlanner
from smartnest.optimization import (
    DEFAULT_WEIGHTS,
    ConstraintValidator,
    InvalidRequirementsError,
    ScoringConfig,
    ScoringConfigError,
    UserRequirements,
    UtilityModel,
    UtilityOptimizer,
    assess_space,
    validate_record,
)
from tests.bdi_support import SELECT, make_agent, make_planner

LIMITS = UserRequirements(budget=30000, max_width_cm=210, max_depth_cm=110, min_seating=3,
                          preferred_style="minimalist", preferred_colour="neutral")
NO_LIMITS = UserRequirements(budget=30000, min_seating=3, preferred_style="minimalist", preferred_colour="neutral")
VERIFIED, UNVERIFIED, INCOMPATIBLE, NOT_REQUIRED = (
    "Fit verified", "Fit not verified", "Dimensionally incompatible", "No room limits specified")


def rec(**over):
    data = dict(product_id="P1", name="Test sofa", price=24000, width_cm=200, depth_cm=100, seating_capacity=3,
                material="fabric", colour="neutral", style="minimalist", availability="in_stock", rating=4.0,
                review_count=50, source_agent_id="src")
    data.update(over)
    return {k: v for k, v in data.items() if v is not ...}


def candidate(**over):
    return validate_record(rec(**over))[0]


def evaluate(record, req=LIMITS):
    return UtilityOptimizer().evaluate_candidate(record, req)


# ══ Requirements: room limits are optional ═══════════════════════════════════

class TestOptionalRoomLimits:
    def test_the_room_limits_default_to_none(self):
        r = UserRequirements(budget=1000)
        assert (r.max_width_cm, r.max_depth_cm, r.max_height_cm) == (None, None, None) and not r.has_spatial_limits

    @pytest.mark.parametrize("given", [{"max_width_cm": 200}, {"max_depth_cm": 90}, {"max_height_cm": 100},
                                       {"max_width_cm": 200, "max_depth_cm": 90, "max_height_cm": 100}])
    def test_any_given_limit_counts(self, given):
        assert UserRequirements(budget=1000, **given).has_spatial_limits

    @pytest.mark.parametrize("field", ["max_width_cm", "max_depth_cm", "max_height_cm"])
    @pytest.mark.parametrize("bad", [0, -5, float("nan"), float("inf"), "wide", True])
    def test_a_limit_that_is_given_must_be_valid(self, field, bad):
        with pytest.raises(InvalidRequirementsError, match=field):
            UserRequirements(budget=1000, **{field: bad})

    def test_explicit_none_is_the_same_as_absent(self):
        r = UserRequirements(budget=1000, max_width_cm=None, max_depth_cm=None, max_height_cm=None)
        assert not r.has_spatial_limits

    def test_the_budget_is_still_required(self):
        with pytest.raises(InvalidRequirementsError, match="budget"):
            UserRequirements(max_width_cm=200)

    def test_from_cnp_still_requires_the_room_limits_so_negotiations_are_unchanged(self):
        with pytest.raises(InvalidRequirementsError, match="max_depth_cm is missing"):
            UserRequirements.from_cnp({"budget": 30000, "max_width": 210})
        with pytest.raises(InvalidRequirementsError, match="max_width_cm is missing"):
            UserRequirements.from_cnp({"budget": 30000, "max_depth": 110})

    def test_from_cnp_can_opt_out_of_requiring_them(self):
        r = UserRequirements.from_cnp({"budget": 30000, "seating_capacity": 3}, require_room_limits=False)
        assert not r.has_spatial_limits and r.min_seating == 3

    def test_from_cnp_reads_a_height_limit(self):
        r = UserRequirements.from_cnp({"budget": 30000, "max_width": 210, "max_depth": 110, "max_height": 90})
        assert r.max_height_cm == 90


# ══ Product records: dimensions may be null, never invented ══════════════════

class TestOptionalProductDimensions:
    @pytest.mark.parametrize("missing", [("width_cm",), ("depth_cm",), ("width_cm", "depth_cm"),
                                         ("width_cm", "depth_cm", "height_cm")])
    def test_absent_dimensions_are_valid_and_stay_none(self, missing):
        record = rec(**{m: ... for m in missing})
        candidate_, errors = validate_record(record)
        assert errors == []
        assert all(getattr(candidate_, m) is None for m in missing)  # never defaulted to 0 or an average

    def test_explicit_none_is_valid(self):
        c, errors = validate_record(rec(width_cm=None, depth_cm=None, height_cm=None))
        assert errors == [] and (c.width_cm, c.depth_cm, c.height_cm) == (None, None, None)

    def test_height_is_read_when_present(self):
        assert candidate(height_cm=85).height_cm == 85.0

    @pytest.mark.parametrize("field", ["width_cm", "depth_cm", "height_cm"])
    @pytest.mark.parametrize("bad,fragment", [(0, "greater than 0"), (-3, "greater than 0"),
                                              (float("nan"), "finite"), ("wide", "must be a number"),
                                              (True, "must be a number")])
    def test_a_present_but_malformed_dimension_is_still_invalid(self, field, bad, fragment):
        c, errors = validate_record(rec(**{field: bad}))
        assert c is None and any(field in e and fragment in e for e in errors)

    def test_other_required_fields_are_still_required(self):
        for field in ("price", "name", "source_agent_id"):          # the only mandatory fields now
            record = rec(); del record[field]
            assert validate_record(record)[0] is None, field
        for field in ("rating", "review_count", "seating_capacity", "material", "colour", "style", "availability"):
            record = rec(); del record[field]
            assert validate_record(record)[0] is not None, field   # optional (see test_optional_attributes.py)


# ══ Spatial assessment ═══════════════════════════════════════════════════════

class TestSpatialAssessment:
    def test_no_room_limits_means_fit_is_not_required_whatever_the_product_states(self):
        for dims in ({}, {"width_cm": ..., "depth_cm": ...}, {"width_cm": 900}):
            assert assess_space(candidate(**dims), NO_LIMITS).status == "not_required"

    def test_all_limits_checked_and_satisfied_is_verified(self):
        a = assess_space(candidate(), LIMITS)
        assert (a.status, a.label, a.verified_axes, a.unknown_axes, a.violations) == ("verified", VERIFIED, ["width", "depth"], [], [])

    def test_boundaries_are_inclusive(self):
        assert assess_space(candidate(width_cm=210, depth_cm=110), LIMITS).status == "verified"

    @pytest.mark.parametrize("dims,axis", [({"width_cm": 211}, "width"), ({"depth_cm": 111}, "depth")])
    def test_a_stated_size_over_a_limit_is_incompatible_with_a_clear_reason(self, dims, axis):
        a = assess_space(candidate(**dims), LIMITS)
        assert a.status == "incompatible" and a.label == INCOMPATIBLE
        assert len(a.violations) == 1 and a.violations[0].startswith(f"{axis} ") and "exceeds maximum" in a.violations[0]

    def test_a_missing_size_is_unknown_not_incompatible_and_not_verified(self):
        a = assess_space(candidate(width_cm=...), LIMITS)
        assert (a.status, a.label) == ("unknown", UNVERIFIED)
        assert a.unknown_axes == ["width"] and a.verified_axes == ["depth"] and a.violations == []
        assert "width is not stated" in a.notes[0]

    def test_all_sizes_missing_is_unknown(self):
        a = assess_space(candidate(width_cm=..., depth_cm=...), LIMITS)
        assert a.status == "unknown" and a.unknown_axes == ["width", "depth"] and a.verified_axes == []

    def test_one_exceeded_limit_makes_it_incompatible_even_if_another_size_is_unknown(self):
        a = assess_space(candidate(width_cm=250, depth_cm=...), LIMITS)
        assert a.status == "incompatible" and a.unknown_axes == ["depth"]

    def test_only_the_limits_the_user_gave_are_checked(self):
        only_width = UserRequirements(budget=30000, max_width_cm=210)
        assert assess_space(candidate(depth_cm=...), only_width).status == "verified"      # depth not asked about
        assert assess_space(candidate(width_cm=...), only_width).status == "unknown"       # width asked, not stated
        only_depth = UserRequirements(budget=30000, max_depth_cm=110)
        assert assess_space(candidate(width_cm=999), only_depth).status == "verified"      # width not asked about

    def test_a_height_limit_needs_a_stated_height(self):
        req = UserRequirements(budget=30000, max_height_cm=90)
        assert assess_space(candidate(), req).status == "unknown"
        assert assess_space(candidate(height_cm=85), req).status == "verified"
        assert assess_space(candidate(height_cm=95), req).status == "incompatible"

    def test_the_four_labels_are_distinct(self):
        assert len({VERIFIED, UNVERIFIED, INCOMPATIBLE, NOT_REQUIRED}) == 4


# ══ Validator statuses ═══════════════════════════════════════════════════════

class TestValidatorStatuses:
    def test_statuses(self):
        v = ConstraintValidator()
        assert v.evaluate(rec(), LIMITS).status == "eligible"
        assert v.evaluate(rec(depth_cm=...), LIMITS).status == "unverified"
        assert v.evaluate(rec(width_cm=300), LIMITS).status == "rejected"
        assert v.evaluate(rec(width_cm=...), NO_LIMITS).status == "eligible"

    def test_an_unverified_product_is_never_eligible_and_explains_why(self):
        res = ConstraintValidator().evaluate(rec(depth_cm=...), LIMITS)
        assert not res.eligible and res.status == "unverified"
        assert res.reasons == ["depth is not stated, so the 110 cm limit cannot be checked"]

    def test_incompatibility_is_a_hard_rejection_with_the_old_wording(self):
        res = ConstraintValidator().evaluate(rec(width_cm=215), LIMITS)
        assert res.status == "rejected" and res.reasons == ["width 215 cm exceeds maximum width 210 cm by 5 cm"]

    def test_other_constraints_still_apply_to_an_unverified_product(self):
        res = ConstraintValidator().evaluate(rec(depth_cm=..., price=40000), LIMITS)
        assert res.status == "rejected" and "exceeds budget" in res.reasons[0]

    def test_check_constraints_ignores_missing_sizes(self):
        v = ConstraintValidator()
        assert v.check_constraints(candidate(width_cm=..., depth_cm=...), LIMITS) == []


# ══ Dynamic weights ══════════════════════════════════════════════════════════

class TestEffectiveWeights:
    def test_with_spatial_the_configured_weights_are_unchanged(self):
        assert ScoringConfig().effective_weights(True) == DEFAULT_WEIGHTS

    def test_without_spatial_the_rest_are_scaled_up_proportionally(self):
        w = ScoringConfig().effective_weights(False)
        assert set(w) == {"price", "quality", "style", "colour"}
        assert w == pytest.approx({"price": 0.375, "quality": 0.3125, "style": 0.1875, "colour": 0.125})
        assert sum(w.values()) == pytest.approx(1.0, abs=1e-12)

    def test_the_ratios_between_the_remaining_weights_are_preserved(self):
        w, base = ScoringConfig().effective_weights(False), DEFAULT_WEIGHTS
        for a in w:
            for b in w:
                assert w[a] / w[b] == pytest.approx(base[a] / base[b])

    @pytest.mark.parametrize("seed", range(25))
    def test_any_valid_configuration_sums_to_one_either_way(self, seed):
        rnd = random.Random(seed)
        raw = {k: rnd.uniform(0.01, 1) for k in DEFAULT_WEIGHTS}
        cfg = ScoringConfig(weights=ScoringConfig.normalised(raw))
        for spatial in (True, False):
            w = cfg.effective_weights(spatial)
            assert sum(w.values()) == pytest.approx(1.0, abs=1e-9) and all(v >= 0 for v in w.values())
            assert ("spatial" in w) is spatial

    def test_it_cannot_exclude_spatial_when_nothing_else_has_weight(self):
        cfg = ScoringConfig(weights={"price": 0, "quality": 0, "spatial": 1.0, "style": 0, "colour": 0})
        assert cfg.effective_weights(True)["spatial"] == 1.0
        with pytest.raises(ScoringConfigError, match="cannot be redistributed"):
            cfg.effective_weights(False)

    def test_the_configured_weights_are_not_mutated(self):
        cfg = ScoringConfig()
        cfg.effective_weights(False)
        assert cfg.weights == DEFAULT_WEIGHTS


# ══ Utility ══════════════════════════════════════════════════════════════════

def manual_utility(c, req, weights):
    m = UtilityModel()
    parts = {"price": m.price_score(c.price, req.budget), "quality": m.quality_score(c.rating, c.review_count),
             "style": m.style_score(c.style, req.preferred_style), "colour": m.colour_score(c.colour, req.preferred_colour)}
    return 100 * sum(weights[k] * parts[k] for k in parts)


class TestUtility:
    def test_verified_fit_keeps_the_existing_formula_and_the_20_percent_weight(self):
        c = candidate()
        b = UtilityModel().score(c, LIMITS)
        assert b.spatial_included and b.weights == pytest.approx(DEFAULT_WEIGHTS)
        f = 0.5 * 200 / 210 + 0.5 * 100 / 110
        assert b.components["spatial"] == pytest.approx(f, abs=1e-6)
        expected = manual_utility(c, LIMITS, DEFAULT_WEIGHTS) + 100 * 0.20 * f
        assert b.total_utility == pytest.approx(expected, abs=1e-4)

    @pytest.mark.parametrize("req_name", ["no limits", "unknown fit"])
    def test_without_spatial_evaluation_spacefit_is_excluded_and_weights_redistributed(self, req_name):
        req, c = (NO_LIMITS, candidate()) if req_name == "no limits" else (LIMITS, candidate(depth_cm=...))
        b = UtilityModel().score(c, req)
        assert not b.spatial_included
        assert "spatial" not in b.components and "spatial" not in b.weights and "spatial" not in b.contributions
        assert b.weights == pytest.approx({"price": 0.375, "quality": 0.3125, "style": 0.1875, "colour": 0.125})
        assert sum(b.weights.values()) == pytest.approx(1.0, abs=1e-12)
        assert b.total_utility == pytest.approx(manual_utility(c, req, b.weights), abs=1e-4)
        assert sum(b.contributions.values()) == pytest.approx(b.total_utility, abs=1e-4)

    def test_no_artificial_spatial_score_is_assigned_to_missing_dimensions(self):
        c = candidate(width_cm=..., depth_cm=...)
        b = UtilityModel().score(c, LIMITS)
        with_zero = manual_utility(c, LIMITS, DEFAULT_WEIGHTS)  # what an artificial F = 0 would give
        assert b.total_utility != pytest.approx(with_zero, abs=0.5)
        assert b.total_utility > with_zero  # not penalised for the missing size

    def test_a_product_with_a_partial_size_gets_no_spatial_score_either(self):
        b = UtilityModel().score(candidate(depth_cm=...), LIMITS)
        assert not b.spatial_included and "spatial" not in b.components

    def test_with_one_limit_only_that_axis_is_scored(self):
        req = UserRequirements(budget=30000, max_width_cm=250, min_seating=3)
        b = UtilityModel().score(candidate(depth_cm=...), req)  # depth is irrelevant here
        assert b.spatial_included and b.components["spatial"] == pytest.approx(200 / 250, abs=1e-6)
        req_d = UserRequirements(budget=30000, max_depth_cm=125, min_seating=3)
        assert UtilityModel().score(candidate(width_cm=...), req_d).components["spatial"] == pytest.approx(100 / 125, abs=1e-6)

    def test_a_height_only_limit_verifies_fit_but_has_no_spacefit_score(self):
        req = UserRequirements(budget=30000, max_height_cm=90, min_seating=3)
        b = UtilityModel().score(candidate(height_cm=80), req)
        assert not b.spatial_included and assess_space(candidate(height_cm=80), req).status == "verified"

    def test_scoring_an_incompatible_product_is_refused(self):
        with pytest.raises(ValueError, match="within the room limits"):
            UtilityModel().score(candidate(width_cm=300), LIMITS)

    def test_the_existing_spatial_score_function_is_unchanged(self):
        m = UtilityModel()
        assert m.spatial_score(200, 100, 210, 110) == pytest.approx(0.5 * 200 / 210 + 0.5 * 100 / 110)
        with pytest.raises(ValueError):
            m.spatial_score(300, 100, 210, 110)

    def test_effective_weights_sum_to_one_for_every_breakdown_in_a_mixed_catalogue(self):
        records = [rec(product_id=f"P{i}", **d) for i, d in enumerate(
            [{}, {"width_cm": ...}, {"depth_cm": ...}, {"width_cm": ..., "depth_cm": ...}, {"width_cm": 150, "depth_cm": 80}])]
        for req in (LIMITS, NO_LIMITS):
            for e in UtilityOptimizer().optimize(records, req).evaluations:
                assert math.isclose(sum(e.breakdown.weights.values()), 1.0, abs_tol=1e-9)
                assert 0 <= e.utility <= 100

    def test_unknown_and_not_required_products_with_the_same_other_attributes_score_identically(self):
        a = UtilityModel().score(candidate(depth_cm=...), LIMITS)
        b = UtilityModel().score(candidate(), NO_LIMITS)
        assert a.total_utility == pytest.approx(b.total_utility, abs=1e-9)


# ══ Optimizer: separation and output ═════════════════════════════════════════

class TestOptimizerSeparation:
    def catalogue(self):
        return [
            rec(product_id="VERIFIED_OK", price=26000, rating=3.5),
            rec(product_id="TOO_WIDE", width_cm=230),
            rec(product_id="NO_DEPTH_BEST", depth_cm=..., price=15000, rating=5.0, review_count=500),   # would win on utility
            rec(product_id="NO_SIZES", width_cm=..., depth_cm=..., price=20000),
            rec(product_id="OVER_BUDGET", price=99999),
            rec(product_id="BROKEN", rating=9),
        ]

    def test_unverified_candidates_are_listed_separately_and_never_selected(self):
        result = UtilityOptimizer().optimize(self.catalogue(), LIMITS)
        assert [e.product_id for e in result.ranking] == ["VERIFIED_OK"]
        assert result.selected.product_id == "VERIFIED_OK"
        assert {e.product_id for e in result.unverified} == {"NO_DEPTH_BEST", "NO_SIZES"} and result.unverified_count == 2
        best_unverified = max(result.unverified, key=lambda e: e.utility)
        assert best_unverified.utility > result.selected.utility  # higher utility, still not recommended as a fit

    def test_unverified_candidates_are_not_rejected_and_the_counts_add_up(self):
        r = UtilityOptimizer().optimize(self.catalogue(), LIMITS)
        assert r.eligible_count == 1 and r.unverified_count == 2 and r.rejected_count == 3 and r.invalid_count == 1
        assert r.eligible_count + r.unverified_count + r.rejected_count == r.total_inspected == 6
        assert {e.product_id for e in r.rejected} == {"TOO_WIDE", "OVER_BUDGET", "BROKEN"}

    def test_the_three_fit_categories_are_reported_distinctly(self):
        r = UtilityOptimizer().optimize(self.catalogue(), LIMITS)
        groups = {label: sorted(e.product_id for e in items) for label, items in r.by_fit().items()}
        assert groups[VERIFIED] == ["OVER_BUDGET", "VERIFIED_OK"]   # sizes check out (OVER_BUDGET fails on price instead)
        assert groups[UNVERIFIED] == ["NO_DEPTH_BEST", "NO_SIZES"]
        assert groups[INCOMPATIBLE] == ["TOO_WIDE"]
        by = {e.product_id: e.fit_label for e in r.evaluations}
        assert by["VERIFIED_OK"] == VERIFIED and by["NO_DEPTH_BEST"] == UNVERIFIED and by["TOO_WIDE"] == INCOMPATIBLE
        assert [e.product_id for e in r.incompatible] == ["TOO_WIDE"]

    def test_the_incompatible_product_says_why(self):
        r = UtilityOptimizer().optimize(self.catalogue(), LIMITS)
        wide = next(e for e in r.evaluations if e.product_id == "TOO_WIDE")
        assert wide.status == "rejected" and "width 230 cm exceeds maximum width 210 cm" in wide.reasons[0]

    def test_unverified_products_have_their_own_ranking_and_a_reason(self):
        r = UtilityOptimizer().optimize(self.catalogue(), LIMITS)
        assert [(e.product_id, e.rank) for e in r.unverified] == [("NO_DEPTH_BEST", 1), ("NO_SIZES", 2)]
        assert r.unverified[0].reasons == ["depth is not stated, so the 110 cm limit cannot be checked"]
        assert all(e.status == "unverified" and not e.eligible for e in r.unverified)

    def test_without_any_limits_everything_is_ranked_together_and_nothing_is_unverified(self):
        r = UtilityOptimizer().optimize(self.catalogue(), NO_LIMITS)
        assert r.unverified == [] and r.unverified_count == 0
        assert {e.product_id for e in r.ranking} == {"VERIFIED_OK", "TOO_WIDE", "NO_DEPTH_BEST", "NO_SIZES"}
        assert r.selected.product_id == "NO_DEPTH_BEST"                     # highest utility, no room question asked
        assert {e.fit_label for e in r.ranking} == {NOT_REQUIRED}
        assert all(not e.breakdown.spatial_included for e in r.ranking)     # SpaceFit excluded for everyone

    def test_oversized_products_are_not_rejected_when_the_user_gave_no_limits(self):
        r = UtilityOptimizer().optimize([rec(width_cm=400, depth_cm=300)], NO_LIMITS)
        assert r.selected is not None and r.selected.fit_label == NOT_REQUIRED

    def test_with_only_unverified_candidates_nothing_is_recommended_and_the_report_says_so(self):
        r = UtilityOptimizer().optimize([rec(depth_cm=...), rec(product_id="P2", width_cm=...)], LIMITS)
        assert r.selected is None and not r.has_recommendation and r.eligible_count == 0
        assert r.unverified_count == 2 and r.verified_optimal is False
        assert "No feasible candidate" in r.explanation and "not verified" in r.explanation and "2 other" in r.explanation

    def test_the_explanation_of_a_recommendation_mentions_the_unverified_ones(self):
        r = UtilityOptimizer().optimize(self.catalogue(), LIMITS)
        assert "2 other candidate(s)" in r.explanation and "listed separately" in r.explanation

    def test_no_note_when_there_are_no_unverified_candidates(self):
        r = UtilityOptimizer().optimize([rec()], LIMITS)
        assert "not verified" not in r.explanation

    def test_results_are_deterministic(self):
        a = UtilityOptimizer().optimize(self.catalogue(), LIMITS)
        b = UtilityOptimizer().optimize(list(reversed(self.catalogue())), LIMITS)
        assert [e.product_id for e in a.unverified] == [e.product_id for e in b.unverified]
        assert a.selected.product_id == b.selected.product_id

    def test_the_selected_product_is_still_the_best_verified_one(self):
        records = [rec(product_id=f"V{i}", price=20000 + 1000 * i) for i in range(4)] + [rec(product_id="U", depth_cm=...)]
        r = UtilityOptimizer().optimize(records, LIMITS)
        assert r.verified_optimal and r.selected.product_id == "V0"

    def test_results_serialise(self):
        r = UtilityOptimizer().optimize(self.catalogue(), LIMITS)
        assert r.model_dump_json() and "unverified" in r.model_dump()

    def test_old_style_complete_records_give_identical_results_to_before(self):
        records = [rec(product_id=f"P{i}", price=20000 + 500 * i, width_cm=190 + i, rating=3 + i / 5) for i in range(5)]
        r = UtilityOptimizer().optimize(records, LIMITS)
        assert r.unverified == [] and r.eligible_count == 5
        assert all(e.breakdown.spatial_included and e.breakdown.weights == pytest.approx(DEFAULT_WEIGHTS) for e in r.ranking)
        assert all(e.fit_label == VERIFIED for e in r.ranking)


# ══ BDI integration (planner and agent) ═════════════════════════════════════

class ListSource:
    def __init__(self, records):
        self.records = records

    def get_candidates(self):
        return [dict(r) for r in self.records]


NO_ROOM = {"total_budget": 200000, "sofa_budget": 30000, "min_seating": 3,
           "preferred_style": "minimalist", "preferred_colour": "neutral"}


class TestBdiWithOptionalDimensions:
    def agent(self, records, requirements):
        return make_agent(requirements=requirements, planner=make_planner(source=ListSource(records)))

    def test_the_agent_plans_without_any_room_limits(self):
        agent = self.agent([rec(width_cm=..., depth_cm=...)], NO_ROOM)
        intention = agent.current_intention(SELECT)
        assert intention is not None and intention.params["product_id"] == "P1"
        outcome = agent.last_outcomes[SELECT]
        assert outcome.optimization.selected.fit_label == NOT_REQUIRED
        assert agent.beliefs.value("user", "max_width_cm") is None  # nothing was invented

    def test_no_fit_preconditions_are_created_when_no_limit_was_given(self):
        agent = self.agent([rec()], NO_ROOM)
        descriptions = [c.description for c in agent.current_intention(SELECT).preconditions]
        assert not any("fits the room" in d for d in descriptions) and any("enough seats" in d for d in descriptions)

    def test_fit_preconditions_exist_only_for_the_limits_given(self):
        agent = self.agent([rec()], {**NO_ROOM, "max_width_cm": 210})
        descriptions = [c.description for c in agent.current_intention(SELECT).preconditions]
        assert "fits the room width" in descriptions and "fits the room depth" not in descriptions

    def test_with_limits_an_unverified_product_is_not_adopted(self):
        agent = self.agent([rec(depth_cm=...)], {**NO_ROOM, "max_width_cm": 210, "max_depth_cm": 110})
        assert agent.current_intention(SELECT) is None
        outcome = agent.last_outcomes[SELECT]
        assert outcome.status == "blocked" and "not verified" in outcome.explanation
        assert outcome.optimization.unverified_count == 1

    def test_the_verified_product_is_adopted_over_a_better_unverified_one(self):
        records = [rec(product_id="OK", price=27000), rec(product_id="UNV", depth_cm=..., price=10000, rating=5)]
        agent = self.agent(records, {**NO_ROOM, "max_width_cm": 210, "max_depth_cm": 110})
        assert agent.current_intention(SELECT).params["product_id"] == "OK"

    def test_a_room_limit_added_later_triggers_replanning(self):
        agent = self.agent([rec(product_id="BIG", width_cm=230), rec(product_id="SMALL", width_cm=190, price=27000)], NO_ROOM)
        first = agent.current_intention(SELECT).params["product_id"]
        agent.set_requirement("max_width_cm", 200)
        assert agent.current_intention(SELECT).params["product_id"] == "SMALL"
        assert first in {"BIG", "SMALL"}

    def test_an_invalidated_room_limit_is_unknown_not_dropped(self):
        agent = self.agent([rec()], {**NO_ROOM, "max_width_cm": 210})
        agent.invalidate_belief("user", "max_width_cm", "remeasuring")
        assert agent.last_outcomes[SELECT].status == "waiting"
        assert "user.max_width_cm" in agent.last_outcomes[SELECT].missing

    def test_the_seating_requirement_is_still_required(self):
        agent = self.agent([rec()], {k: v for k, v in NO_ROOM.items() if k != "min_seating"})
        assert agent.last_outcomes[SELECT].status == "waiting" and "user.min_seating" in agent.last_outcomes[SELECT].missing


# ══ Live-style data: SerpApi products without dimensions ════════════════════

class TestSerpApiStyleRecords:
    def serp_record(self):
        from tests.test_serpapi_discovery import build, details_body, shopping_item
        d = details_body()
        d["product_results"]["about_the_product"]["features"] = [
            f for f in d["product_results"]["about_the_product"]["features"] if f["title"] not in ("Width", "Depth")]
        built = build(shopping_item(), details=d)
        assert built.record is not None and built.record["width_cm"] is None
        return built.record

    def test_a_listing_without_dimensions_is_recommendable_when_the_user_gave_no_room_limits(self):
        r = UtilityOptimizer().optimize([self.serp_record()], NO_LIMITS)
        assert r.selected is not None and r.selected.fit_label == NOT_REQUIRED
        assert r.selected.breakdown.weights == pytest.approx({"price": 0.375, "quality": 0.3125, "style": 0.1875, "colour": 0.125})

    def test_the_same_listing_is_only_fit_not_verified_when_limits_are_given(self):
        r = UtilityOptimizer().optimize([self.serp_record()], LIMITS)
        assert r.selected is None and [e.fit_label for e in r.unverified] == [UNVERIFIED]


# ══ Recommendation output ═══════════════════════════════════════════════════

class TestRecommendationWording:
    def facts(self, fit):
        from smartnest.integrations.gemini import GeminiService
        from smartnest.integrations.gemini.schemas import RecommendationFacts
        facts = RecommendationFacts(product_id="P1", product_name="Test sofa", price=24000, utility=60.0, fit_status=fit)
        return GeminiService.template_explanation(facts).text

    def test_the_template_states_each_fit_status_without_overclaiming(self):
        assert "checked against your room limits and fits" in self.facts(VERIFIED)
        assert "whether it fits your room is not verified" in self.facts(UNVERIFIED)
        assert "and fits" not in self.facts(UNVERIFIED) and "checked against" not in self.facts(UNVERIFIED)
        assert "fit was not assessed" in self.facts(NOT_REQUIRED)
        assert "fit" not in self.facts(None).lower()
