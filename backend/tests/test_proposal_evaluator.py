"""Tests for the ProposalEvaluator — utility scoring, constraint rejection, ranking.

Verifies:
    - Hard constraint rejection (price, width, availability).
    - Utility score calculation with default weights.
    - Score components are normalized to [0, 1].
    - Ranking by utility score descending.
    - Infeasible proposals excluded from ranking.
    - Custom weights.
    - Data-dependent results (not hardcoded).
"""

from __future__ import annotations

import pytest

from smartnest.cnp.proposal_evaluator import ProposalEvaluation, ProposalEvaluator


REQUIREMENTS = {
    "max_budget": 30000,
    "room_width": 210,
    "preferred_style": "minimalist",
    "seating_capacity": 3,
}


def _proposal(
    agent_id="agent_a",
    product_id="SOFA_001",
    price=24999,
    width_cm=200,
    style="minimalist",
    availability="in_stock",
    **extra,
):
    return {
        "agent_id": agent_id,
        "product_id": product_id,
        "product": {
            "product_id": product_id,
            "name": "Test Sofa",
            "price": price,
            "width_cm": width_cm,
            "style": style,
            "availability": availability,
            **extra,
        },
        "proposed_price": price,
    }


class TestHardConstraints:
    def test_price_over_budget_rejected(self):
        prop = _proposal(price=31000)
        ev = ProposalEvaluator().evaluate(prop, REQUIREMENTS)
        assert not ev.feasible
        assert "exceeds max budget" in ev.rejection_reason
        assert ev.utility_score == 0.0

    def test_width_over_room_rejected(self):
        prop = _proposal(width_cm=230)
        ev = ProposalEvaluator().evaluate(prop, REQUIREMENTS)
        assert not ev.feasible
        assert "exceeds room width" in ev.rejection_reason

    def test_out_of_stock_rejected(self):
        prop = _proposal(availability="out_of_stock")
        ev = ProposalEvaluator().evaluate(prop, REQUIREMENTS)
        assert not ev.feasible
        assert "not in stock" in ev.rejection_reason

    def test_feasible_proposal_passes(self):
        prop = _proposal(price=24999, width_cm=200)
        ev = ProposalEvaluator().evaluate(prop, REQUIREMENTS)
        assert ev.feasible
        assert ev.utility_score > 0.0
        assert ev.rejection_reason == ""


class TestUtilityScore:
    def test_score_in_range(self):
        prop = _proposal(price=25000, width_cm=200)
        ev = ProposalEvaluator().evaluate(prop, REQUIREMENTS)
        assert 0.0 <= ev.utility_score <= 1.0

    def test_cheaper_product_scores_higher_on_price(self):
        cheap = _proposal(product_id="CHEAP", price=20000, width_cm=200)
        expensive = _proposal(product_id="EXP", price=29000, width_cm=200)
        evaluator = ProposalEvaluator()
        ev_cheap = evaluator.evaluate(cheap, REQUIREMENTS)
        ev_exp = evaluator.evaluate(expensive, REQUIREMENTS)
        assert ev_cheap.scores["price"] > ev_exp.scores["price"]

    def test_wider_product_scores_higher_on_spatial(self):
        narrow = _proposal(product_id="NARROW", price=25000, width_cm=190)
        wide = _proposal(product_id="WIDE", price=25000, width_cm=205)
        evaluator = ProposalEvaluator()
        ev_narrow = evaluator.evaluate(narrow, REQUIREMENTS)
        ev_wide = evaluator.evaluate(wide, REQUIREMENTS)
        assert ev_wide.scores["spatial"] > ev_narrow.scores["spatial"]

    def test_style_match_scores_higher(self):
        matching = _proposal(product_id="MATCH", price=25000, width_cm=200, style="minimalist")
        mismatch = _proposal(product_id="MIS", price=25000, width_cm=200, style="modern")
        evaluator = ProposalEvaluator()
        ev_match = evaluator.evaluate(matching, REQUIREMENTS)
        ev_mis = evaluator.evaluate(mismatch, REQUIREMENTS)
        assert ev_match.scores["style"] == 1.0
        assert ev_mis.scores["style"] == 0.0
        assert ev_match.utility_score > ev_mis.utility_score

    def test_all_scores_in_zero_one(self):
        prop = _proposal(price=25000, width_cm=200)
        ev = ProposalEvaluator().evaluate(prop, REQUIREMENTS)
        for key, val in ev.scores.items():
            assert 0.0 <= val <= 1.0, f"{key}={val} not in [0,1]"


class TestRanking:
    def test_rank_by_utility_descending(self):
        proposals = [
            _proposal(product_id="LOW", price=28000, width_cm=190),
            _proposal(product_id="HIGH", price=20000, width_cm=205),
            _proposal(product_id="MID", price=25000, width_cm=195),
        ]
        evaluator = ProposalEvaluator()
        evals = [evaluator.evaluate(p, REQUIREMENTS) for p in proposals]
        ranked = evaluator.rank(evals)
        assert len(ranked) == 3
        assert ranked[0].utility_score >= ranked[1].utility_score >= ranked[2].utility_score
        assert ranked[0].product_id == "HIGH"

    def test_infeasible_excluded_from_ranking(self):
        proposals = [
            _proposal(product_id="GOOD", price=25000, width_cm=200),
            _proposal(product_id="OVERTHICK", price=25000, width_cm=250),
        ]
        evaluator = ProposalEvaluator()
        evals = [evaluator.evaluate(p, REQUIREMENTS) for p in proposals]
        ranked = evaluator.rank(evals)
        assert len(ranked) == 1
        assert ranked[0].product_id == "GOOD"

    def test_empty_ranking_when_all_infeasible(self):
        proposals = [
            _proposal(product_id="BAD1", price=50000, width_cm=200),
            _proposal(product_id="BAD2", price=25000, width_cm=250),
        ]
        evaluator = ProposalEvaluator()
        evals = [evaluator.evaluate(p, REQUIREMENTS) for p in proposals]
        ranked = evaluator.rank(evals)
        assert len(ranked) == 0


class TestCustomWeights:
    def test_custom_weights_affect_ranking(self):
        """The same product should score differently under different weights."""
        cheap_narrow = _proposal(
            product_id="CN", price=20000, width_cm=180, style="minimalist"
        )

        # Weight price heavily
        price_heavy = ProposalEvaluator(
            weights={"price": 0.9, "spatial": 0.03, "style": 0.03, "availability": 0.04}
        )
        # Weight spatial heavily
        spatial_heavy = ProposalEvaluator(
            weights={"price": 0.03, "spatial": 0.9, "style": 0.03, "availability": 0.04}
        )

        ev_price = price_heavy.evaluate(cheap_narrow, REQUIREMENTS)
        ev_spatial = spatial_heavy.evaluate(cheap_narrow, REQUIREMENTS)

        # Under price-heavy weights, the price component dominates the score
        # Under spatial-heavy weights, the (low) spatial component dominates
        # Since width=180/210=0.857 is high and price=20000/30000=0.667 is also decent,
        # the key is that the price score contributes more under price_heavy
        # and spatial contributes more under spatial_heavy.
        # Verify the component scores are the same but total differs:
        assert ev_price.scores["price"] == ev_spatial.scores["price"]
        assert ev_price.scores["spatial"] == ev_spatial.scores["spatial"]
        # The total scores differ because weights differ
        assert ev_price.utility_score != ev_spatial.utility_score

    def test_price_heavy_prefers_cheap(self):
        """Under price-heavy weights, a cheaper product ranks above an expensive one
        even if the expensive one has better spatial fit."""
        cheap_narrow = _proposal(product_id="CN", price=20000, width_cm=180)
        expensive_wide = _proposal(product_id="EW", price=29000, width_cm=208)

        price_heavy = ProposalEvaluator(
            weights={"price": 0.8, "spatial": 0.1, "style": 0.05, "availability": 0.05}
        )

        ev_cheap = price_heavy.evaluate(cheap_narrow, REQUIREMENTS)
        ev_expensive = price_heavy.evaluate(expensive_wide, REQUIREMENTS)

        assert ev_cheap.utility_score > ev_expensive.utility_score


class TestDataDependence:
    def test_different_requirements_different_result(self):
        """Same product, different requirements → different scores."""
        prop = _proposal(price=25000, width_cm=200, style="minimalist")

        generous = {"max_budget": 100000, "room_width": 300, "preferred_style": "minimalist"}
        tight = {"max_budget": 26000, "room_width": 210, "preferred_style": "minimalist"}

        evaluator = ProposalEvaluator()
        ev_generous = evaluator.evaluate(prop, generous)
        ev_tight = evaluator.evaluate(prop, tight)

        # Under generous budget, price score is higher (price/budget ratio lower)
        assert ev_generous.scores["price"] > ev_tight.scores["price"]
        # Under generous room, spatial score is lower (width/room ratio lower)
        assert ev_generous.scores["spatial"] < ev_tight.scores["spatial"]


# ── Fixes: style, seating, availability, weights ────────────────────────────

class TestStyleScoring:
    def test_missing_style_gets_no_partial_credit(self):
        ev = ProposalEvaluator().evaluate(_proposal(style=""), REQUIREMENTS)
        assert ev.feasible
        assert ev.scores["style"] == 0.0

    def test_partial_style_match(self):
        ev = ProposalEvaluator().evaluate(_proposal(style="scandi minimalist"), REQUIREMENTS)
        assert ev.scores["style"] == 0.5

    def test_no_style_preference_accepts_any_style(self):
        reqs = {**REQUIREMENTS, "preferred_style": ""}
        ev = ProposalEvaluator().evaluate(_proposal(style="industrial"), reqs)
        assert ev.scores["style"] == 1.0


class TestSeatingConstraint:
    def test_too_few_seats_rejected(self):
        ev = ProposalEvaluator().evaluate(_proposal(seating_capacity=2), REQUIREMENTS)
        assert not ev.feasible
        assert "seating" in ev.rejection_reason.lower()

    def test_enough_seats_accepted(self):
        ev = ProposalEvaluator().evaluate(_proposal(seating_capacity=3), REQUIREMENTS)
        assert ev.feasible


class TestAvailabilityScoring:
    def test_low_stock_feasible_but_scores_lower(self):
        evaluator = ProposalEvaluator()
        in_stock = evaluator.evaluate(_proposal(availability="in_stock"), REQUIREMENTS)
        low = evaluator.evaluate(_proposal(availability="low_stock"), REQUIREMENTS)
        assert low.feasible
        assert low.scores["availability"] < in_stock.scores["availability"]
        assert low.utility_score < in_stock.utility_score

    def test_unavailable_rejected(self):
        ev = ProposalEvaluator().evaluate(_proposal(availability="discontinued"), REQUIREMENTS)
        assert not ev.feasible

    def test_missing_width_rejected(self):
        ev = ProposalEvaluator().evaluate(_proposal(width_cm=None), REQUIREMENTS)
        assert not ev.feasible
        assert "width" in ev.rejection_reason.lower()


class TestWeightValidation:
    def test_weights_are_normalised(self):
        evaluator = ProposalEvaluator(weights={"price": 2, "spatial": 2})
        assert evaluator.weights == {
            "price": 0.5, "spatial": 0.5, "style": 0.0, "availability": 0.0,
        }

    def test_utility_stays_in_unit_interval_with_unnormalised_weights(self):
        evaluator = ProposalEvaluator(weights={"price": 5, "spatial": 5, "style": 5, "availability": 5})
        ev = evaluator.evaluate(_proposal(price=1000, width_cm=210), REQUIREMENTS)
        assert 0.0 <= ev.utility_score <= 1.0

    def test_unknown_weight_key_rejected(self):
        with pytest.raises(ValueError, match="Unknown"):
            ProposalEvaluator(weights={"colour": 1.0})

    def test_negative_weight_rejected(self):
        with pytest.raises(ValueError):
            ProposalEvaluator(weights={"price": -1.0, "spatial": 2.0})

    def test_zero_weights_rejected(self):
        with pytest.raises(ValueError):
            ProposalEvaluator(weights={"price": 0.0})

    def test_empty_weights_rejected_not_silently_defaulted(self):
        with pytest.raises(ValueError):
            ProposalEvaluator(weights={})


class TestRankingTieBreak:
    def test_equal_utility_prefers_lower_price(self):
        a = ProposalEvaluation("a", "A", True, 0.7, proposed_price=25000, round=1)
        b = ProposalEvaluation("b", "B", True, 0.7, proposed_price=24000, round=2)
        assert ProposalEvaluator().rank([a, b])[0].product_id == "B"

    def test_evaluation_records_price_and_round(self):
        prop = {**_proposal(price=20000), "negotiation_round": 3}
        ev = ProposalEvaluator().evaluate(prop, REQUIREMENTS)
        assert ev.proposed_price == 20000
        assert ev.round == 3
