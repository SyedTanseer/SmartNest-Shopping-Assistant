"""Integration of the 3.2 utility model with the 3.1 Contract Net Protocol.

Verifies:
    - ProposalEvaluator stays in legacy mode by default (3.1 unchanged) and
      delegates to the optimizer in utility mode.
    - The coordinator recalculates utility from the offered price and
      ignores (but audits) any utility a bidder reports about itself.
    - Bidding agents find their own highest-utility feasible products.
    - A full TCP negotiation in utility mode selects the same product as
      the exhaustive optimizer over both catalogues.
"""

from __future__ import annotations

import pytest

from smartnest.cnp.bidding_agent import BiddingAgent
from smartnest.cnp.coordinator_agent import CoordinatorAgent
from smartnest.cnp.negotiation_state import NegotiationState
from smartnest.cnp.product_catalog import catalog_agent_a, catalog_agent_b
from smartnest.cnp.proposal_evaluator import ProposalEvaluation, ProposalEvaluator
from smartnest.core.models import MessageEnvelope, MessageType
from smartnest.optimization import (
    InvalidRequirementsError,
    UserRequirements,
    UtilityModel,
    UtilityOptimizer,
)

CNP_REQ = {
    "task_description": "sofa",
    "max_budget": 30000,
    "room_width": 210,
    "room_depth": 110,
    "preferred_style": "minimalist",
    "preferred_colour": "neutral",
    "seating_capacity": 3,
}
REQ = UserRequirements.from_cnp(CNP_REQ)


def _product(pid: str, agent: str = "research_agent_a", extended: bool = True) -> dict:
    cat = catalog_agent_a(extended) if agent.endswith("_a") else catalog_agent_b(extended)
    return cat.get_product(pid)


def _proposal(pid: str, agent: str = "research_agent_a", price=None, **extra) -> dict:
    product = _product(pid, agent)
    return {
        "agent_id": agent,
        "product_id": pid,
        "product": product,
        "proposed_price": product["price"] if price is None else price,
        "negotiation_round": 1,
        **extra,
    }


def _optimizer_utility(pid: str, agent: str = "research_agent_a", price=None) -> float:
    record = {**_product(pid, agent), "source_agent_id": agent}
    if price is not None:
        record["price"] = price
    return UtilityOptimizer().evaluate_candidate(record, REQ).breakdown.normalised_utility


# ── ProposalEvaluator modes ─────────────────────────────────────────────────

class TestProposalEvaluatorModes:
    def test_default_is_legacy_and_unchanged(self):
        ev = ProposalEvaluator()
        assert ev.mode == "legacy"
        result = ev.evaluate(_proposal("SOFA_A3"), {**CNP_REQ})
        assert set(result.scores) == {"price", "spatial", "style", "availability"}
        assert result.utility_score == pytest.approx(0.6967, abs=1e-4)  # 3.1 value

    def test_utility_mode_delegates_to_optimizer(self):
        ev = ProposalEvaluator.utility_based()
        assert ev.mode == "utility"
        result = ev.evaluate(_proposal("SOFA_A2"), CNP_REQ)
        assert result.feasible
        assert result.utility_score == pytest.approx(_optimizer_utility("SOFA_A2"))
        assert set(result.scores) == {"price", "quality", "spatial", "style", "colour"}
        assert result.utility_breakdown["total_utility"] == pytest.approx(67.550351, abs=1e-6)
        assert result.quality_score == pytest.approx(0.896)

    def test_constructor_with_model(self):
        assert ProposalEvaluator(utility_model=UtilityModel()).mode == "utility"

    def test_weights_and_model_together_rejected(self):
        with pytest.raises(ValueError):
            ProposalEvaluator(weights={"price": 1.0}, utility_model=UtilityModel())

    def test_hard_constraints_enforced_in_utility_mode(self):
        ev = ProposalEvaluator.utility_based()
        too_deep = ev.evaluate(_proposal("SOFA_A4"), CNP_REQ)
        assert not too_deep.feasible
        assert "depth" in too_deep.rejection_reason
        assert too_deep.utility_score == 0.0

    def test_depth_is_not_checked_in_legacy_mode(self):
        """Documented difference: 3.1 scoring has no depth constraint."""
        assert ProposalEvaluator().evaluate(_proposal("SOFA_A4"), CNP_REQ).feasible
        assert not ProposalEvaluator.utility_based().evaluate(_proposal("SOFA_A4"), CNP_REQ).feasible

    def test_missing_depth_requirement_raises_in_utility_mode(self):
        req = {k: v for k, v in CNP_REQ.items() if k != "room_depth"}
        with pytest.raises(InvalidRequirementsError):
            ProposalEvaluator.utility_based().evaluate(_proposal("SOFA_A2"), req)

    def test_malformed_proposal_is_infeasible_not_crash(self):
        proposal = _proposal("SOFA_A2")
        del proposal["product"]["rating"]
        result = ProposalEvaluator.utility_based().evaluate(proposal, CNP_REQ)
        assert not result.feasible
        assert "rating is missing" in result.rejection_reason

    def test_non_numeric_offer_is_infeasible(self):
        result = ProposalEvaluator.utility_based().evaluate(
            _proposal("SOFA_A2", price="cheap"), CNP_REQ)
        assert not result.feasible
        assert result.proposed_price == 0.0


# ── Coordinator-side recalculation ──────────────────────────────────────────

class TestRecalculation:
    def test_discounted_offer_is_rescored_at_offered_price(self):
        ev = ProposalEvaluator.utility_based()
        full = ev.evaluate(_proposal("SOFA_B3", "research_agent_b"), CNP_REQ)
        cut = ev.evaluate(_proposal("SOFA_B3", "research_agent_b", price=26324), CNP_REQ)
        assert cut.utility_score > full.utility_score
        assert cut.utility_score == pytest.approx(
            _optimizer_utility("SOFA_B3", "research_agent_b", price=26324))

    def test_offer_above_budget_rejected_even_if_list_price_fits(self):
        result = ProposalEvaluator.utility_based().evaluate(
            _proposal("SOFA_A2", price=30500), CNP_REQ)
        assert not result.feasible
        assert "budget" in result.rejection_reason

    def test_reported_utility_is_ignored_but_recorded(self):
        ev = ProposalEvaluator.utility_based()
        honest = ev.evaluate(_proposal("SOFA_A7"), CNP_REQ)
        liar = ev.evaluate(_proposal("SOFA_A7", self_reported_utility=0.99), CNP_REQ)
        assert liar.utility_score == honest.utility_score
        assert liar.reported_utility == 0.99
        assert liar.utility_discrepancy == pytest.approx(0.99 - honest.utility_score, abs=1e-6)

    @pytest.mark.parametrize("bogus", [float("nan"), float("inf"), "high", True])
    def test_garbage_reported_utility_is_discarded(self, bogus):
        result = ProposalEvaluator.utility_based().evaluate(
            _proposal("SOFA_A2", self_reported_utility=bogus), CNP_REQ)
        assert result.reported_utility is None

    def test_inflated_report_cannot_change_ranking(self):
        ev = ProposalEvaluator.utility_based()
        evals = [
            ev.evaluate(_proposal("SOFA_A2"), CNP_REQ),
            ev.evaluate(_proposal("SOFA_A7", self_reported_utility=1.0), CNP_REQ),
        ]
        assert ev.rank(evals)[0].product_id == "SOFA_A2"

    def test_rank_tie_breaks_on_quality_then_price(self):
        a = ProposalEvaluation("a", "A", True, 0.6, proposed_price=20000, quality_score=0.8)
        b = ProposalEvaluation("b", "B", True, 0.6, proposed_price=19000, quality_score=0.9)
        c = ProposalEvaluation("c", "C", True, 0.6, proposed_price=18000, quality_score=0.8)
        assert [e.product_id for e in ProposalEvaluator().rank([a, b, c])] == ["B", "C", "A"]


# ── Bidding agents evaluating their own catalogues ──────────────────────────

def _cfp(requirements: dict, **payload_extra) -> MessageEnvelope:
    return MessageEnvelope(
        sender="coordinator", receiver="x", message_type=MessageType.CFP, task_id="t1",
        payload={"negotiation_id": "neg_u", "round": 1, "requirements": requirements,
                 **payload_extra},
    )


class TestBidderSelfEvaluation:
    def test_agent_finds_its_own_best(self):
        agent = BiddingAgent("research_agent_a", catalog_agent_a(True), utility_optimizer=UtilityOptimizer())
        own = agent.evaluate_own_catalog(REQ)
        records = [{**p, "source_agent_id": agent.agent_id}
                   for p in catalog_agent_a(True).get_all_products()]
        assert own.selected.product_id == UtilityOptimizer().optimize(records, REQ).selected.product_id
        assert own.total_inspected == 7

    def test_agent_without_optimizer_cannot_self_evaluate(self):
        with pytest.raises(RuntimeError):
            BiddingAgent("x", catalog_agent_a()).evaluate_own_catalog(REQ)

    def test_cfp_proposal_is_highest_utility_with_self_assessment(self):
        agent = BiddingAgent("research_agent_b", catalog_agent_b(True), utility_optimizer=UtilityOptimizer())
        reply = agent._handle_cfp(_cfp(CNP_REQ))
        p = reply.payload
        assert reply.message_type == "proposal"
        assert p["product_id"] == agent.evaluate_own_catalog(REQ).selected.product_id
        assert p["self_reported_utility"] == pytest.approx(
            _optimizer_utility(p["product_id"], "research_agent_b"))
        assert set(p["utility_breakdown"]["components"]) == {"price", "quality", "spatial", "style", "colour"}

    def test_self_assessment_uses_discounted_price(self):
        agent = BiddingAgent("research_agent_b", catalog_agent_b(True), utility_optimizer=UtilityOptimizer())
        agent._handle_cfp(_cfp(CNP_REQ))
        reply = agent._handle_cfp(_cfp(CNP_REQ, round=2, offer_feedback={
            "request": "improve_offer", "you_are_leading": False}))
        p = reply.payload
        assert p["proposed_price"] < p["list_price"]
        assert p["self_reported_utility"] == pytest.approx(
            _optimizer_utility(p["product_id"], "research_agent_b", price=p["proposed_price"]))

    def test_falls_back_to_heuristic_when_width_unannounced(self):
        agent = BiddingAgent("research_agent_a", catalog_agent_a(True), utility_optimizer=UtilityOptimizer())
        req = {k: v for k, v in CNP_REQ.items() if k != "room_width"}
        p = agent._handle_cfp(_cfp(req)).payload
        assert "self_reported_utility" not in p
        assert p["product_id"] == "SOFA_A1"  # 3.1 best-value choice

    def test_no_feasible_product_returns_failure(self):
        agent = BiddingAgent("research_agent_a", catalog_agent_a(True), utility_optimizer=UtilityOptimizer())
        reply = agent._handle_cfp(_cfp({**CNP_REQ, "max_budget": 1000}))
        assert reply.message_type == "failure"
        assert reply.payload["reason"] == "no_eligible_products"


# ── Full negotiation over TCP in utility mode ───────────────────────────────

class InflatingBiddingAgent(BiddingAgent):
    """Self-interested bidder that lies about its utility."""

    def _attach_self_assessment(self, proposal, product, req):
        super()._attach_self_assessment(proposal, product, req)
        proposal["self_reported_utility"] = 1.0


@pytest.fixture()
def utility_agents():
    a = BiddingAgent("research_agent_a", catalog_agent_a(True), max_discount_pct=0.05,
                     utility_optimizer=UtilityOptimizer())
    b = BiddingAgent("research_agent_b", catalog_agent_b(True), max_discount_pct=0.04,
                     utility_optimizer=UtilityOptimizer())
    a.start()
    b.start()
    yield a, b
    a.stop(timeout=3.0)
    b.stop(timeout=3.0)


def _utility_coordinator(agents, **kwargs) -> CoordinatorAgent:
    coord = CoordinatorAgent(evaluator=ProposalEvaluator.utility_based(), **kwargs)
    coord.set_requirements("sofa", 30000, 210, "minimalist", 3,
                           room_depth=110, preferred_colour="neutral")
    for agent in agents:
        coord.add_bidder(agent.agent_id, "127.0.0.1", agent.actual_port)
    return coord


def _exhaustive_best():
    records = []
    for cat in (catalog_agent_a(True), catalog_agent_b(True)):
        records += [{**p, "source_agent_id": cat.agent_id} for p in cat.get_all_products()]
    return UtilityOptimizer().optimize(records, REQ).selected


class TestUtilityNegotiation:
    @pytest.mark.parametrize("progressive", [True, False])
    def test_negotiation_matches_exhaustive_optimizer(self, utility_agents, progressive):
        coord = _utility_coordinator(utility_agents, reveal_constraints_progressively=progressive)
        result = coord.run_negotiation()
        best = _exhaustive_best()
        assert result.success
        assert result.winner.product_id == best.product_id
        assert result.winner.agent_id == best.source_agent_id
        assert coord.state == NegotiationState.TERMINATED
        coord.close_all()

    def test_every_score_is_coordinator_recalculated(self, utility_agents):
        coord = _utility_coordinator(utility_agents)
        coord.run_negotiation()
        for evaluations in coord.round_evaluations.values():
            for ev in evaluations:
                if ev.feasible:
                    assert ev.utility_score == pytest.approx(
                        _optimizer_utility(ev.product_id, ev.agent_id, price=ev.proposed_price))
        assert not [e for e in coord.message_log if e["event_type"] == "utility_discrepancy"]
        coord.close_all()

    def test_winner_and_award_use_offered_price(self, utility_agents):
        a, b = utility_agents
        coord = _utility_coordinator(utility_agents)
        result = coord.run_negotiation()
        winner = a if result.winner.agent_id == a.agent_id else b
        assert winner.award_details["final_price"] == result.winner.proposed_price
        coord.close_all()

    def test_inflated_utility_is_detected_and_ignored(self, utility_agents):
        honest_a, _ = utility_agents
        liar = InflatingBiddingAgent("research_agent_b", catalog_agent_b(True),
                                     max_discount_pct=0.04, utility_optimizer=UtilityOptimizer())
        liar.start()
        try:
            coord = _utility_coordinator([honest_a, liar])
            result = coord.run_negotiation()
            flagged = [e for e in coord.message_log if e["event_type"] == "utility_discrepancy"]
            assert flagged and all(e["agent"] == "research_agent_b" for e in flagged)
            assert result.winner.product_id == _exhaustive_best().product_id
            assert result.winner.agent_id == "research_agent_a"
            coord.close_all()
        finally:
            liar.stop(timeout=3.0)

    def test_utility_breakdown_in_timeline(self, utility_agents):
        coord = _utility_coordinator(utility_agents, reveal_constraints_progressively=False)
        coord.run_negotiation()
        feasible = [e for e in coord.message_log if e["event_type"] == "proposal_feasible"]
        assert feasible and all("breakdown" in e["details"] for e in feasible)
        coord.close_all()
