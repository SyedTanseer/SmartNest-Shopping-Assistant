"""Integration tests for the full CNP negotiation over TCP.

Spins up independent bidding agent servers on OS-assigned ports, then
runs the coordinator over real sockets.

Verifies:
    - Independent agent initialization.
    - Round 1 (width not yet announced): too-wide sofas are proposed and
      rejected by the coordinator's hard constraints.
    - Constraint feedback: the violated width is revealed and bidders
      switch to different, narrower products.
    - Offer feedback: a trailing bidder concedes on price, a leading
      bidder holds, and the concession changes the ranking.
    - The coordinator's own stopping rule (min rounds, convergence,
      max rounds) and award of the best offer from any round.
    - AWARD → INFORM_DONE, REJECT_PROPOSAL to losers, INFORM_RESULT to all.
    - Award refusal → runner-up is awarded; all refuse → failure.
    - A bidder dropping out mid-negotiation.
    - No-proposal scenario → FAILED → TERMINATED.
    - Timeline labels, timestamps and task/negotiation ID correlation.
    - Bidding agent handler validation (negotiation, task, price).
"""

from __future__ import annotations

import time
from datetime import datetime

import pytest

from smartnest.cnp.bidding_agent import BiddingAgent
from smartnest.cnp.coordinator_agent import CoordinatorAgent
from smartnest.cnp.negotiation_state import NegotiationState
from smartnest.cnp.product_catalog import ProductCatalog, catalog_agent_a, catalog_agent_b
from smartnest.core.models import MessageEnvelope, MessageType


# ── Fixtures & helpers ────────────────────────────────────────────────────────

def _start(agent: BiddingAgent) -> BiddingAgent:
    agent.start()  # binds synchronously
    return agent


@pytest.fixture()
def agents():
    """Start two bidding agents with independent catalogues and strategies."""
    agent_a = _start(BiddingAgent("research_agent_a", catalog_agent_a(), max_discount_pct=0.05))
    agent_b = _start(BiddingAgent("research_agent_b", catalog_agent_b(), max_discount_pct=0.04))
    yield agent_a, agent_b
    agent_a.stop(timeout=3.0)
    agent_b.stop(timeout=3.0)


def _make_coordinator(max_rounds=3, min_rounds=2, progressive=True) -> CoordinatorAgent:
    coord = CoordinatorAgent(
        agent_id="coordinator",
        max_rounds=max_rounds,
        min_rounds=min_rounds,
        reveal_constraints_progressively=progressive,
    )
    coord.set_requirements(
        task_description="Find a three-seater minimalist sofa",
        max_budget=30000,
        room_width=210,
        preferred_style="minimalist",
        seating_capacity=3,
    )
    return coord


def _connected(agents_, **kwargs) -> CoordinatorAgent:
    coord = _make_coordinator(**kwargs)
    for agent in agents_:
        coord.add_bidder(agent.agent_id, "127.0.0.1", agent.actual_port)
    coord.start_negotiation()
    coord.connect_to_bidders()
    return coord


def _wait_status(agent: BiddingAgent, negotiation_id: str, expected: str, timeout=2.0) -> str:
    """Wait for one-way notifications (REJECT/INFORM_RESULT/CANCEL) to land."""
    deadline = time.monotonic() + timeout
    status = agent.negotiation_status(negotiation_id)
    while status != expected and time.monotonic() < deadline:
        time.sleep(0.02)
        status = agent.negotiation_status(negotiation_id)
    return status


class RefusingBiddingAgent(BiddingAgent):
    """A bidder that proposes normally but refuses every award."""

    def _handle_award(self, message):
        return MessageEnvelope(
            sender=self.agent_id,
            receiver=message.sender,
            message_type=MessageType.FAILURE,
            task_id=message.task_id,
            payload={"reason": "award_does_not_match_proposal", "detail": "test refusal"},
        )


# ── Agent initialization ─────────────────────────────────────────────────

class TestAgentInit:
    def test_bidding_agent_starts(self, agents):
        agent_a, agent_b = agents
        assert agent_a.is_running
        assert agent_b.is_running

    def test_bidding_agents_have_different_real_ports(self, agents):
        agent_a, agent_b = agents
        assert agent_a.actual_port != 0
        assert agent_a.actual_port != agent_b.actual_port

    def test_bidding_agents_have_independent_catalogues(self, agents):
        agent_a, agent_b = agents
        a_ids = {p["product_id"] for p in agent_a.catalog.get_all_products()}
        b_ids = {p["product_id"] for p in agent_b.catalog.get_all_products()}
        assert a_ids.isdisjoint(b_ids)

    def test_coordinator_initial_state(self):
        assert _make_coordinator().state is None

    def test_coordinator_after_start(self):
        coord = _make_coordinator()
        neg_id = coord.start_negotiation()
        assert coord.state == NegotiationState.CREATED
        assert coord.negotiation_id == neg_id

    def test_invalid_round_limits_rejected(self):
        with pytest.raises(ValueError):
            CoordinatorAgent(max_rounds=0)
        with pytest.raises(ValueError):
            CoordinatorAgent(max_rounds=2, min_rounds=3)


# ── Round 1 and constraint feedback ───────────────────────────────────────

class TestConstraintFeedback:
    def test_round1_rejects_too_wide_products(self, agents):
        """Width isn't announced in round 1, so both agents offer their
        best-value sofa (A1 230 cm, B1 225 cm) and both are rejected."""
        coord = _connected(agents)
        evals = coord.run_cfp_round()

        assert {e.product_id for e in evals} == {"SOFA_A1", "SOFA_B1"}
        assert all(not e.feasible for e in evals)
        assert all("width" in e.rejection_reason.lower() for e in evals)
        coord.close_all()

    def test_round1_cfp_does_not_announce_width(self, agents):
        coord = _connected(agents)
        coord.run_cfp_round()
        proposals = coord.round_proposals[1]
        # The bidders couldn't verify fit, and said so honestly.
        assert all(p["compatibility"]["fits_room"] is None for p in proposals)
        assert all(p["feasibility_status"] == "unverified" for p in proposals)
        coord.close_all()

    def test_violated_width_is_derived_as_new_constraint(self, agents):
        coord = _connected(agents)
        evals = coord.run_cfp_round()
        assert coord._derive_constraints(evals) == {"max_width": 210}
        coord.close_all()

    def test_round2_bidders_switch_to_different_narrower_products(self, agents):
        coord = _connected(agents)
        coord.run_cfp_round()
        evals_r2 = coord.run_cfp_round(updated_constraints={"max_width": 210})

        r1_ids = {p["product_id"] for p in coord.round_proposals[1]}
        r2_ids = {p["product_id"] for p in coord.round_proposals[2]}
        assert r1_ids.isdisjoint(r2_ids), "Both agents should have revised their product"
        assert all(p["product"]["width_cm"] <= 210 for p in coord.round_proposals[2])
        assert all(e.feasible for e in evals_r2)
        assert coord.revealed_constraints == {"max_width": 210}
        coord.close_all()

    def test_upfront_mode_makes_round1_feasible(self, agents):
        coord = _connected(agents, progressive=False)
        evals = coord.run_cfp_round()
        assert all(e.feasible for e in evals)
        coord.close_all()


# ── Offer feedback and concessions ────────────────────────────────────────

class TestConcessions:
    def test_trailing_bidder_concedes_leading_bidder_holds(self, agents):
        coord = _connected(agents, progressive=False)
        evals_r1 = coord.run_cfp_round()
        best = coord.get_best_proposal(evals_r1)
        coord._best = best
        coord.run_cfp_round(offer_feedback=coord._offer_feedback())

        r1 = {p["agent_id"]: p for p in coord.round_proposals[1]}
        r2 = {p["agent_id"]: p for p in coord.round_proposals[2]}
        leader = best.agent_id
        trailer = next(a for a in r1 if a != leader)

        assert r2[leader]["proposed_price"] == r1[leader]["proposed_price"]
        assert r2[trailer]["proposed_price"] < r1[trailer]["proposed_price"]
        assert r2[trailer]["discount_pct"] > 0
        coord.close_all()

    def test_concession_never_exceeds_max_discount(self, agents):
        agent_a, _ = agents
        coord = _connected([agent_a], progressive=False, max_rounds=10)
        coord.run_cfp_round()
        trailing = {"research_agent_a": {"request": "improve_offer", "you_are_leading": False}}
        for _ in range(6):
            coord.run_cfp_round(offer_feedback=trailing)
        last = coord.round_proposals[coord.current_round][0]
        assert last["discount_pct"] == pytest.approx(agent_a.max_discount_pct)
        assert last["proposed_price"] == round(last["list_price"] * (1 - agent_a.max_discount_pct))
        coord.close_all()


# ── Full negotiation driven by the coordinator ────────────────────────────

class TestRunNegotiation:
    def test_default_scenario_end_to_end(self, agents):
        """R1 width rejections → R2 feasible → R3 concession flips the winner."""
        agent_a, agent_b = agents
        coord = _connected(agents)
        result = coord.run_negotiation()

        assert result.success
        assert result.rounds == 3
        assert result.winner.agent_id == "research_agent_a"
        assert result.winner.product_id == "SOFA_A3"
        assert result.winner.round == 3
        assert result.winner.proposed_price == round(29499 * 0.975)

        # Round 2's leader was B; A only wins because it conceded in round 3.
        r2_best = coord.get_best_proposal(coord.round_evaluations[2])
        assert r2_best.agent_id == "research_agent_b"

        assert coord.state == NegotiationState.TERMINATED
        assert agent_a.awarded
        assert agent_a.award_details["final_price"] == result.winner.proposed_price
        assert not agent_b.awarded
        assert _wait_status(agent_b, result.negotiation_id, "rejected") == "rejected"
        coord.close_all()

    def test_minimum_rounds_enforced_even_if_round1_feasible(self, agents):
        coord = _connected(agents, progressive=False, max_rounds=3, min_rounds=2)
        result = coord.run_negotiation()
        assert result.success
        assert result.rounds >= 2
        coord.close_all()

    def test_stops_when_offers_converge(self, agents):
        coord = _connected(agents, max_rounds=10)
        result = coord.run_negotiation()
        assert result.success
        assert result.rounds < 10
        decisions = [e["message"] for e in coord.message_log if e["event_type"] == "decision"]
        assert any("converged" in d for d in decisions)
        coord.close_all()

    def test_awarded_offer_is_best_across_all_rounds(self, agents):
        coord = _connected(agents, max_rounds=6)
        result = coord.run_negotiation()
        all_feasible = [e for evs in coord.round_evaluations.values() for e in evs if e.feasible]
        assert result.winner.utility_score == max(e.utility_score for e in all_feasible)
        coord.close_all()

    def test_state_history_shows_full_lifecycle(self, agents):
        coord = _connected(agents)
        coord.run_negotiation()
        states = [h["state"] for h in coord.state_history]
        assert states[0] == "created"
        assert states.count("cfp_sent") == 3
        assert states[-4:] == ["awarded", "executing", "completed", "terminated"]
        coord.close_all()

    def test_run_negotiation_connects_by_itself(self, agents):
        coord = _make_coordinator()
        for agent in agents:
            coord.add_bidder(agent.agent_id, "127.0.0.1", agent.actual_port)
        result = coord.run_negotiation()
        assert result.success
        coord.close_all()


# ── Award refusal and fallback ──────────────────────────────────────────────

class TestAwardFallback:
    def test_refused_award_falls_back_to_runner_up(self, agents):
        _, agent_b = agents
        refusing_a = _start(RefusingBiddingAgent("research_agent_a", catalog_agent_a()))
        try:
            coord = _connected([refusing_a, agent_b])
            result = coord.run_negotiation()

            assert result.success
            assert result.winner.agent_id == "research_agent_b"
            assert agent_b.awarded
            refused = [e for e in coord.message_log if e["event_type"] == "award_refused"]
            assert refused and refused[0]["agent"] == "research_agent_a"
            states = [h["state"] for h in coord.state_history]
            assert "awarded" in states and states[-1] == "terminated"
            coord.close_all()
        finally:
            refusing_a.stop(timeout=3.0)

    def test_all_awards_refused_reports_failure(self):
        refusing = _start(RefusingBiddingAgent("research_agent_a", catalog_agent_a()))
        try:
            coord = _connected([refusing])
            result = coord.run_negotiation()
            assert not result.success
            assert "refused" in result.reason
            assert coord.state == NegotiationState.TERMINATED
            assert "failed" in [h["state"] for h in coord.state_history]
            coord.close_all()
        finally:
            refusing.stop(timeout=3.0)


# ── Bidder failures ─────────────────────────────────────────────────────────

class TestBidderFailure:
    def test_bidder_dropping_out_mid_negotiation(self, agents):
        agent_a, agent_b = agents
        coord = _connected(agents)
        agent_b.stop(timeout=3.0)  # closes B's side of the connection
        result = coord.run_negotiation()

        assert "research_agent_b" in coord.failed_bidders
        assert result.success
        assert result.winner.agent_id == "research_agent_a"
        coord.close_all()

    def test_unreachable_bidder_is_skipped_at_connect(self, agents):
        agent_a, _ = agents
        coord = _make_coordinator()
        coord.add_bidder("research_agent_a", "127.0.0.1", agent_a.actual_port)
        coord.add_bidder("ghost", "127.0.0.1", 1)
        coord.start_negotiation()
        coord.connect_to_bidders()
        assert "ghost" in coord.failed_bidders
        assert coord.run_negotiation().success
        coord.close_all()


# ── No-proposal scenario ────────────────────────────────────────────────────

class TestNoProposalScenario:
    def test_no_feasible_products_reports_failure(self):
        impossible_catalog = ProductCatalog("impossible", [
            {"product_id": "EXP1", "name": "Expensive 1", "price": 50000,
             "width_cm": 200, "style": "minimalist", "seating_capacity": 3,
             "availability": "in_stock"},
            {"product_id": "EXP2", "name": "Expensive 2", "price": 60000,
             "width_cm": 180, "style": "minimalist", "seating_capacity": 3,
             "availability": "in_stock"},
        ])
        agent = _start(BiddingAgent("expensive_agent", impossible_catalog))
        try:
            coord = _make_coordinator(max_rounds=5)
            coord.set_requirements(
                task_description="Impossible sofa",
                max_budget=10000,
                room_width=210,
                preferred_style="minimalist",
                seating_capacity=3,
            )
            coord.add_bidder("expensive_agent", "127.0.0.1", agent.actual_port)
            result = coord.run_negotiation()

            assert not result.success
            # Min rounds run, then it stops: nothing new to reveal.
            assert result.rounds == 2
            assert coord.state == NegotiationState.TERMINATED
            assert _wait_status(agent, result.negotiation_id, "cancelled") == "cancelled"
            assert not agent.awarded
            coord.close_all()
        finally:
            agent.stop(timeout=3.0)


# ── Timeline and correlation ────────────────────────────────────────────────

class TestTimelineAndCorrelation:
    def test_timeline_labels_match_the_evaluated_agent(self, agents):
        coord = _connected(agents)
        coord.run_negotiation()
        entries = [e for e in coord.message_log
                   if e["event_type"] in ("proposal_feasible", "proposal_rejected")]
        assert entries
        for e in entries:
            assert e["message"].strip().startswith(e["agent"])
        coord.close_all()

    def test_timeline_has_real_ordered_timestamps(self, agents):
        coord = _connected(agents)
        coord.run_negotiation()
        log = coord.message_log
        assert [e["seq"] for e in log] == list(range(1, len(log) + 1))
        stamps = [datetime.fromisoformat(e["timestamp"]) for e in log]
        assert stamps == sorted(stamps)
        coord.close_all()

    def test_task_id_correlation(self, agents):
        agent_a, agent_b = agents
        coord = _connected(agents)
        result = coord.run_negotiation()
        winner = agent_a if result.winner.agent_id == agent_a.agent_id else agent_b
        assert winner.award_details["task_id"] == coord.task_id
        assert winner.award_details["negotiation_id"] == coord.negotiation_id
        coord.close_all()

    def test_negotiation_id_in_every_proposal(self, agents):
        coord = _connected(agents)
        coord.run_cfp_round()
        assert all(p["negotiation_id"] == coord.negotiation_id
                   for p in coord.round_proposals[1])
        coord.close_all()


# ── Bidding agent handlers ─────────────────────────────────────────────────

def _cfp(neg_id="neg_test", task_id="task_test", round_num=1, room_width=210, **extra):
    requirements = {"max_budget": 30000, "preferred_style": "minimalist", "seating_capacity": 3}
    if room_width is not None:
        requirements["room_width"] = room_width
    return MessageEnvelope(
        sender="coordinator",
        receiver="research_agent_a",
        message_type=MessageType.CFP,
        task_id=task_id,
        payload={"negotiation_id": neg_id, "round": round_num,
                 "requirements": requirements, **extra},
    )


def _award(neg_id, product_id, task_id="task_test", price=None):
    payload = {"negotiation_id": neg_id, "product_id": product_id, "round": 1}
    if price is not None:
        payload["proposed_price"] = price
    return MessageEnvelope(
        sender="coordinator",
        receiver="research_agent_a",
        message_type=MessageType.AWARD,
        task_id=task_id,
        payload=payload,
    )


class TestBiddingAgentHandlers:
    def test_cfp_handler_returns_honest_proposal(self, agents):
        agent_a, _ = agents
        response = agent_a._handle_cfp(_cfp())
        assert response.message_type == "proposal"
        p = response.payload
        assert p["agent_id"] == "research_agent_a"
        assert p["negotiation_round"] == 1
        assert p["compatibility"] == {"within_budget": True, "fits_room": True, "style_match": True}
        assert p["feasibility_status"] == "feasible"
        assert "utility_score" not in p  # the coordinator computes utility

    def test_cfp_handler_returns_failure_when_no_products(self):
        catalog = ProductCatalog("empty", [
            {"product_id": "P1", "name": "Too Wide", "price": 20000,
             "width_cm": 300, "style": "minimalist", "seating_capacity": 3,
             "availability": "in_stock"},
        ])
        agent = BiddingAgent("empty_agent", catalog)
        response = agent._handle_cfp(_cfp())
        assert response.message_type == "failure"
        assert response.payload["reason"] == "no_eligible_products"

    def test_award_handler_verifies_and_executes(self, agents):
        agent_a, _ = agents
        proposal = agent_a._handle_cfp(_cfp(neg_id="neg_award")).payload
        done = agent_a._handle_award(
            _award("neg_award", proposal["product_id"], price=proposal["proposed_price"])
        )
        assert done.message_type == "inform_done"
        assert done.payload["product_id"] == proposal["product_id"]
        assert done.payload["final_price"] == proposal["proposed_price"]
        assert agent_a.awarded

    def test_award_for_unknown_product_refused(self, agents):
        agent_a, _ = agents
        agent_a._handle_cfp(_cfp(neg_id="neg_x"))
        response = agent_a._handle_award(_award("neg_x", "NONEXISTENT_PRODUCT"))
        assert response.message_type == "failure"
        assert response.payload["reason"] == "award_does_not_match_proposal"
        assert not agent_a.awarded

    def test_award_for_unknown_negotiation_refused(self, agents):
        agent_a, _ = agents
        proposal = agent_a._handle_cfp(_cfp(neg_id="neg_real")).payload
        response = agent_a._handle_award(_award("neg_other", proposal["product_id"]))
        assert response.payload["reason"] == "award_does_not_match_proposal"
        assert "unknown negotiation" in response.payload["detail"]

    def test_award_with_wrong_task_id_refused(self, agents):
        agent_a, _ = agents
        proposal = agent_a._handle_cfp(_cfp(neg_id="neg_t", task_id="task_1")).payload
        response = agent_a._handle_award(_award("neg_t", proposal["product_id"], task_id="task_2"))
        assert response.message_type == "failure"
        assert "task_id" in response.payload["detail"]

    def test_award_at_price_never_offered_refused(self, agents):
        agent_a, _ = agents
        proposal = agent_a._handle_cfp(_cfp(neg_id="neg_p")).payload
        response = agent_a._handle_award(
            _award("neg_p", proposal["product_id"], price=proposal["proposed_price"] - 1000)
        )
        assert response.message_type == "failure"

    def test_award_of_earlier_round_offer_accepted(self, agents):
        """Offers are binding: an earlier round's offer can still be awarded."""
        agent_a, _ = agents
        r1 = agent_a._handle_cfp(_cfp(neg_id="neg_e", room_width=None)).payload
        agent_a._handle_cfp(_cfp(neg_id="neg_e", round_num=2,
                                 updated_constraints={"max_width": 210}))
        done = agent_a._handle_award(_award("neg_e", r1["product_id"], price=r1["proposed_price"]))
        assert done.message_type == "inform_done"

    def test_negotiations_are_tracked_separately(self, agents):
        agent_a, _ = agents
        agent_a._handle_cfp(_cfp(neg_id="neg_1", room_width=None))
        agent_a._handle_cfp(_cfp(neg_id="neg_2"))
        assert agent_a.offers_for("neg_1")[0]["product_id"] == "SOFA_A1"
        assert agent_a.offers_for("neg_2")[0]["product_id"] == "SOFA_A3"
        assert agent_a.offers_for("neg_1") != agent_a.offers_for("neg_2")
