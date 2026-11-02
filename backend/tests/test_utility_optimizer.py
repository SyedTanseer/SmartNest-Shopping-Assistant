"""Tests for the UtilityOptimizer (Assignment 3.2).

Covers exhaustive search, filtering, ranking, deterministic tie-breaking,
selection of the provably highest-utility feasible candidate, empty and
infeasible sets, malformed input, multiple catalogues and the demo.
"""

from __future__ import annotations

import asyncio
import itertools
import random

import pytest

from smartnest.cnp.product_catalog import catalog_agent_a, catalog_agent_b
from smartnest.optimization import (
    InvalidRequirementsError,
    OptimizationResult,
    ScoringConfig,
    UserRequirements,
    UtilityModel,
    UtilityOptimizer,
)

REQ = UserRequirements(budget=30000, max_width_cm=210, max_depth_cm=110, min_seating=3,
                       preferred_style="minimalist", preferred_colour="neutral")


def _p(pid, **overrides):
    data = dict(product_id=pid, name=f"Sofa {pid}", price=25000, width_cm=200, depth_cm=100,
                seating_capacity=3, material="fabric", colour="neutral", style="minimalist",
                availability="in_stock", rating=4.0, review_count=50, source_agent_id="agent_x")
    data.update(overrides)
    return data


def _scenario_candidates():
    out = []
    for cat in (catalog_agent_a(extended=True), catalog_agent_b(extended=True)):
        out += cat.get_all_products()
    return out


def _formula(p, req=REQ):
    """Independent re-implementation of the assignment formula (for checking)."""
    P = max(0.0, min(1.0, 1 - p["price"] / req.budget))
    Q = 0.8 * p["rating"] / 5 + 0.2 * min(p["review_count"] / 100, 1)
    F = 0.5 * p["width_cm"] / req.max_width_cm + 0.5 * p["depth_cm"] / req.max_depth_cm
    style = {"minimalist": 1.0, "scandinavian": 0.5, "contemporary": 0.5}.get(p["style"], 0.0)
    colour = {"neutral": 1.0, "beige": 0.5, "grey": 0.5, "cream": 0.5, "taupe": 0.5,
              "white": 0.5}.get(p.get("color", p.get("colour")), 0.0)
    return 100 * (0.30 * P + 0.25 * Q + 0.20 * F + 0.15 * style + 0.10 * colour)


def _feasible(p, req=REQ):
    return (p["price"] <= req.budget and p["width_cm"] <= req.max_width_cm
            and p["depth_cm"] <= req.max_depth_cm and p["seating_capacity"] >= req.min_seating
            and p["availability"] in ("in_stock", "low_stock"))


# ── The assignment scenario ─────────────────────────────────────────────────

class TestSofaScenario:
    def test_at_least_twelve_candidates_from_both_agents(self):
        cands = _scenario_candidates()
        assert len(cands) >= 12
        assert {c["source_agent_id"] for c in cands} == {"research_agent_a", "research_agent_b"}
        required = {"product_id", "name", "price", "width_cm", "depth_cm", "seating_capacity",
                    "material", "color", "style", "availability", "rating", "review_count",
                    "source_agent_id"}
        assert all(required <= set(c) for c in cands)

    def test_counts(self):
        r = UtilityOptimizer().optimize(_scenario_candidates(), REQ)
        assert r.total_inspected == 14
        assert r.eligible_count + r.rejected_count == r.total_inspected
        assert r.eligible_count == sum(_feasible(c) for c in _scenario_candidates())

    def test_selected_is_highest_utility_feasible(self):
        """Brute-force check with an independent formula implementation."""
        cands = _scenario_candidates()
        r = UtilityOptimizer().optimize(cands, REQ)
        expected = max((c for c in cands if _feasible(c)), key=_formula)
        assert r.selected.product_id == expected["product_id"]
        assert r.selected.utility == pytest.approx(_formula(expected), abs=1e-5)
        assert r.verified_optimal

    def test_every_eligible_utility_matches_formula(self):
        cands = {c["product_id"]: c for c in _scenario_candidates()}
        r = UtilityOptimizer().optimize(cands.values(), REQ)
        for e in r.ranking:
            assert e.utility == pytest.approx(_formula(cands[e.product_id]), abs=1e-5)

    def test_rejected_candidates_have_reasons(self):
        r = UtilityOptimizer().optimize(_scenario_candidates(), REQ)
        reasons = {e.product_id: e.reasons for e in r.rejected}
        assert all(reasons.values())
        assert "1 cm" in reasons["SOFA_B7"][0]
        assert "by ₹1" in reasons["SOFA_B5"][0]
        assert "depth" in reasons["SOFA_A4"][0]
        assert "seating" in reasons["SOFA_A5"][0]
        assert "availability" in reasons["SOFA_B6"][0]

    def test_high_utility_cannot_override_hard_constraint(self):
        """B5 would beat everything on quality, but it is ₹1 over budget."""
        r = UtilityOptimizer().optimize(_scenario_candidates(), REQ)
        assert "SOFA_B5" not in [e.product_id for e in r.ranking]
        b5 = next(c for c in _scenario_candidates() if c["product_id"] == "SOFA_B5")
        relaxed = REQ.model_copy(update={"budget": 30001})
        assert _formula(b5, relaxed) > r.selected.utility

    def test_boundary_product_is_eligible(self):
        """A6 sits exactly on every limit."""
        r = UtilityOptimizer().optimize(_scenario_candidates(), REQ)
        a6 = next(e for e in r.ranking if e.product_id == "SOFA_A6")
        assert a6.breakdown.components["price"] == 0.0
        assert a6.breakdown.components["spatial"] == 1.0

    def test_ranking_is_sorted_and_ranked(self):
        r = UtilityOptimizer().optimize(_scenario_candidates(), REQ)
        utilities = [e.utility for e in r.ranking]
        assert utilities == sorted(utilities, reverse=True)
        assert [e.rank for e in r.ranking] == list(range(1, r.eligible_count + 1))
        assert all(e.rank is None for e in r.rejected)

    def test_explanation_names_winner_and_runner_up(self):
        r = UtilityOptimizer().optimize(_scenario_candidates(), REQ)
        assert r.selected.product_id in r.explanation
        assert r.ranking[1].product_id in r.explanation
        assert "highest utility" in r.explanation


# ── Verifiable example ──────────────────────────────────────────────────────

class TestHandCalculatedExample:
    def test_known_winner(self):
        req = UserRequirements(budget=10000, max_width_cm=200, max_depth_cm=100, min_seating=3,
                               preferred_style="minimalist", preferred_colour="neutral")
        x = _p("X", price=5000, width_cm=200, depth_cm=100, rating=5, review_count=100)
        y = _p("Y", price=2000, width_cm=100, depth_cm=50, rating=4, review_count=100,
               style="scandinavian", colour="red")
        z = _p("Z", price=1000, width_cm=201, rating=5, review_count=500)
        r = UtilityOptimizer().optimize([x, y, z], req)
        assert r.selected.product_id == "X"
        assert r.selected.utility == pytest.approx(85.0)
        assert r.ranking[1].utility == pytest.approx(62.5)
        assert [e.product_id for e in r.rejected] == ["Z"]


# ── Empty / infeasible / invalid sets ───────────────────────────────────────

class TestNoRecommendation:
    def test_empty_candidate_list(self):
        r = UtilityOptimizer().optimize([], REQ)
        assert r.selected is None
        assert not r.has_recommendation
        assert not r.verified_optimal
        assert r.total_inspected == 0
        assert "no recommendation" in r.explanation.lower()

    def test_all_candidates_infeasible(self):
        r = UtilityOptimizer().optimize(
            [_p("A", price=40000), _p("B", width_cm=300), _p("C", availability="out_of_stock")], REQ)
        assert r.selected is None
        assert r.eligible_count == 0
        assert r.rejected_count == 3
        assert r.ranking == []

    def test_all_candidates_invalid(self):
        r = UtilityOptimizer().optimize([_p("A", price=float("nan")), None, {"product_id": "Q"}], REQ)
        assert r.selected is None
        assert r.invalid_count == 3

    def test_invalid_records_do_not_stop_valid_ones(self):
        r = UtilityOptimizer().optimize([_p("BAD", rating=9), _p("GOOD")], REQ)
        assert r.selected.product_id == "GOOD"
        assert r.invalid_count == 1

    def test_invalid_requirements_raise(self):
        with pytest.raises(InvalidRequirementsError):
            UtilityOptimizer().optimize([_p("A")], {"budget": 0, "max_width_cm": 1, "max_depth_cm": 1})

    def test_requirements_dict_accepted(self):
        r = UtilityOptimizer().optimize([_p("A")], REQ.model_dump())
        assert r.selected.product_id == "A"

    def test_duplicate_records_flagged(self):
        r = UtilityOptimizer().optimize([_p("A"), _p("A")], REQ)
        assert r.eligible_count == 1
        assert "duplicate" in r.evaluations[1].reasons[0]

    def test_same_id_from_different_agents_is_not_duplicate(self):
        r = UtilityOptimizer().optimize([_p("A", source_agent_id="x"), _p("A", source_agent_id="y")], REQ)
        assert r.eligible_count == 2


# ── Tie-breaking and determinism ────────────────────────────────────────────

class TestTieBreaking:
    def test_identical_utility_broken_by_quality(self):
        # Same U = 100·(0.5·Q + 0.5·F) = 75, reached two different ways.
        model = UtilityModel(ScoringConfig(weights={"price": 0.0, "quality": 0.5, "spatial": 0.5,
                                                    "style": 0.0, "colour": 0.0}))
        opt = UtilityOptimizer(model)
        c = _p("C", rating=2.5, review_count=50, width_cm=210, depth_cm=110)    # Q=0.5 F=1.0 → 75
        d = _p("D", rating=5.0, review_count=100, width_cm=105, depth_cm=55)    # Q=1.0 F=0.5 → 75
        r = opt.optimize([c, d], REQ)
        assert r.ranking[0].utility == pytest.approx(r.ranking[1].utility)
        assert r.selected.product_id == "D"  # higher quality wins the tie
        assert "tie-break" in r.explanation

    def test_identical_utility_and_quality_broken_by_price_then_id(self):
        model = UtilityModel(ScoringConfig(weights={"price": 0.0, "quality": 1.0, "spatial": 0.0,
                                                    "style": 0.0, "colour": 0.0}))
        opt = UtilityOptimizer(model)
        r = opt.optimize([_p("B", price=20000), _p("C", price=19000), _p("A", price=20000)], REQ)
        assert [e.product_id for e in r.ranking] == ["C", "A", "B"]

    def test_order_of_input_does_not_change_result(self):
        cands = _scenario_candidates()
        baseline = [e.product_id for e in UtilityOptimizer().optimize(cands, REQ).ranking]
        rng = random.Random(42)
        for _ in range(10):
            shuffled = cands[:]
            rng.shuffle(shuffled)
            assert [e.product_id for e in UtilityOptimizer().optimize(shuffled, REQ).ranking] == baseline

    def test_repeated_evaluation_gives_identical_output(self):
        cands = _scenario_candidates()
        outputs = {UtilityOptimizer().optimize(cands, REQ).model_dump_json() for _ in range(10)}
        assert len(outputs) == 1

    def test_permutations_of_tied_candidates_are_stable(self):
        tied = [_p("T1"), _p("T2"), _p("T3")]
        orders = {tuple(e.product_id for e in UtilityOptimizer().optimize(list(p), REQ).ranking)
                  for p in itertools.permutations(tied)}
        assert orders == {("T1", "T2", "T3")}


# ── Multiple catalogues, async, serialisation ───────────────────────────────

class TestIntegrationPoints:
    def test_each_agent_catalogue_has_own_best(self):
        a = UtilityOptimizer().optimize(catalog_agent_a(extended=True).get_all_products(), REQ)
        b = UtilityOptimizer().optimize(catalog_agent_b(extended=True).get_all_products(), REQ)
        combined = UtilityOptimizer().optimize(_scenario_candidates(), REQ)
        assert combined.selected.utility == max(a.selected.utility, b.selected.utility)

    def test_default_catalogues_unchanged_for_3_1(self):
        assert len(catalog_agent_a().get_all_products()) == 3
        assert len(catalog_agent_a(extended=True).get_all_products()) == 7

    def test_optimize_async_matches_sync(self):
        cands = _scenario_candidates()
        sync = UtilityOptimizer().optimize(cands, REQ)
        result = asyncio.run(UtilityOptimizer().optimize_async(cands, REQ))
        assert result.model_dump() == sync.model_dump()

    def test_result_round_trips_through_json(self):
        r = UtilityOptimizer().optimize(_scenario_candidates(), REQ)
        restored = OptimizationResult.model_validate_json(r.model_dump_json())
        assert restored.selected.product_id == r.selected.product_id
        assert restored.ranking[0].breakdown == r.ranking[0].breakdown


# ── Demo ─────────────────────────────────────────────────────────────────────

class TestDemo:
    def test_demo_selects_brute_force_optimum(self, capsys):
        import utility_optimization_run as demo

        result = demo.run()
        out = capsys.readouterr().out
        cands = demo.load_candidates()
        expected = max((c for c in cands if _feasible(c)), key=_formula)
        assert result.total_inspected >= 12
        assert result.selected.product_id == expected["product_id"]
        assert "MATCHES hand calculation" in out
        assert "consistent with the negotiated outcome" in out
