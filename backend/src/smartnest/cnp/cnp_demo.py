"""CNP demonstration runner for Assignment 3.1.

Starts two bidding agent servers, then lets the coordinator run a full
multi-round Contract Net Protocol negotiation over TCP.  The coordinator
owns the negotiation policy (``CoordinatorAgent.run_negotiation``); this
module only sets up the scenario and prints what happened.

Expected behaviour with the default scenario (room width revealed
progressively, max 3 rounds):
    Round 1 — the CFP announces item, budget, style and seating but not
              the room width.  Both agents offer their best-value sofa
              (230 cm and 225 cm); both are rejected as too wide.
    Round 2 — the CFP reveals max width 210 cm.  Both agents switch to
              narrower sofas; both are feasible.
    Round 3 — each agent is told whether it leads.  The trailing agent
              concedes on price, which changes the ranking.
    Award   — the best offer across all rounds wins; the winner confirms
              with INFORM_DONE, the loser gets REJECT_PROPOSAL, and both
              get INFORM_RESULT.

Usage:
    python src/cnp_run.py
"""

from __future__ import annotations

import logging
import sys
import time

from smartnest.cnp.bidding_agent import BiddingAgent
from smartnest.cnp.coordinator_agent import CoordinatorAgent, NegotiationResult
from smartnest.cnp.product_catalog import catalog_agent_a, catalog_agent_b

MAX_BUDGET = 30000
ROOM_WIDTH = 210
MAX_ROUNDS = 3


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)-8s] [%(threadName)-16s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


def _print_separator(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def _print_rounds(coordinator: CoordinatorAgent) -> None:
    _print_separator("Round-by-Round Summary")
    proposals = coordinator.round_proposals
    for round_num, evaluations in coordinator.round_evaluations.items():
        print(f"\n  ROUND {round_num}")
        by_key = {(p["agent_id"], p["product_id"]): p for p in proposals.get(round_num, [])}
        if not evaluations:
            print("     (no proposals received)")
        for ev in evaluations:
            p = by_key.get((ev.agent_id, ev.product_id), {})
            width = p.get("product", {}).get("width_cm", "?")
            discount = p.get("discount_pct") or 0
            price = f"₹{ev.proposed_price:,}" + (f" (-{discount * 100:.1f}%)" if discount else "")
            verdict = (
                f"utility {ev.utility_score:.4f}" if ev.feasible
                else f"REJECTED: {ev.rejection_reason}"
            )
            print(f"     {ev.agent_id:18s} {ev.product_id:8s} {width:>4} cm  {price:22s} {verdict}")


def _print_timeline(coordinator: CoordinatorAgent) -> None:
    _print_separator("Negotiation Timeline")
    for entry in coordinator.message_log:
        print(f"  {entry['seq']:>3}. {entry['message']}")
        for key in ("new_constraints", "offer_feedback"):
            if key in entry["details"]:
                print(f"         {key}: {entry['details'][key]}")


def _wait_for_bidders(agents: list[BiddingAgent], negotiation_id: str, timeout: float = 2.0) -> None:
    """Let bidders finish processing the final one-way notifications.

    REJECT_PROPOSAL / INFORM_RESULT / CANCEL need no reply, so the
    coordinator doesn't wait for them; give the bidder threads a moment
    so the summary shows their final status.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if all(a.negotiation_status(negotiation_id) not in ("open", None) for a in agents):
            return
        time.sleep(0.02)


def _print_state_history(coordinator: CoordinatorAgent) -> None:
    _print_separator("Negotiation State History")
    for entry in coordinator.state_history:
        print(f"  {entry['state']:25s}  @ {entry['timestamp']}")


def _print_summary(
    coordinator: CoordinatorAgent,
    result: NegotiationResult,
    agents: list[BiddingAgent],
) -> None:
    _print_separator("Summary")
    print(f"  Negotiation ID:   {result.negotiation_id}")
    print(f"  Task ID:          {result.task_id}")
    print(f"  Rounds executed:  {result.rounds}")
    print(f"  Final state:      {coordinator.state.value if coordinator.state else 'N/A'}")
    print(f"  Outcome:          {'AGREEMENT' if result.success else 'NO AGREEMENT'}")
    if result.winner:
        w = result.winner
        print(f"  Winner:           {w.agent_id} / {w.product_id} at ₹{w.proposed_price:,} "
              f"(round {w.round}, utility {w.utility_score:.4f})")
    else:
        print(f"  Reason:           {result.reason}")
    for agent in agents:
        print(f"  {agent.agent_id} awarded: {agent.awarded} "
              f"(status: {agent.negotiation_status(result.negotiation_id)})")


def run() -> NegotiationResult:
    """Execute the CNP demonstration and return the result."""
    _configure_logging()

    _print_separator("SmartNest — Assignment 3.1: Multi-Agent Coordination (CNP)")
    print("  Coordinator seeking a three-seater minimalist sofa")
    print(f"  Max budget: ₹{MAX_BUDGET:,}  |  Room width: {ROOM_WIDTH} cm (private, "
          f"revealed when violated)  |  Style: minimalist")
    print("  Bidders: Agent A (3 sofas, concedes up to 5%) + "
          "Agent B (3 sofas, concedes up to 4%)")
    print(f"  Max rounds: {MAX_ROUNDS}  |  Min rounds before award: 2")

    # ── Start bidding agents ────────────────────────────────────────────
    _print_separator("Starting Bidding Agents")
    agent_a = BiddingAgent(
        agent_id="research_agent_a",
        catalog=catalog_agent_a(),
        port=0,  # OS-assigned port
        max_discount_pct=0.05,
    )
    agent_b = BiddingAgent(
        agent_id="research_agent_b",
        catalog=catalog_agent_b(),
        port=0,
        max_discount_pct=0.04,
    )
    agents = [agent_a, agent_b]
    for agent in agents:
        agent.start()  # binds synchronously; ready when this returns
        print(f"  {agent.agent_id} listening on {agent.host}:{agent.actual_port}")

    coordinator = CoordinatorAgent(agent_id="coordinator", max_rounds=MAX_ROUNDS, min_rounds=2)
    coordinator.set_requirements(
        task_description=(
            f"Find a three-seater minimalist sofa ≤ ₹{MAX_BUDGET:,} "
            f"fitting a {ROOM_WIDTH} cm wall"
        ),
        max_budget=MAX_BUDGET,
        room_width=ROOM_WIDTH,
        preferred_style="minimalist",
        seating_capacity=3,
    )
    for agent in agents:
        coordinator.add_bidder(agent.agent_id, agent.host, agent.actual_port)

    try:
        _print_separator("Running Negotiation")
        result = coordinator.run_negotiation()
        print(f"  Negotiation {result.negotiation_id} finished after {result.rounds} round(s).")
        _wait_for_bidders(agents, result.negotiation_id)

        _print_rounds(coordinator)
        _print_timeline(coordinator)
        _print_state_history(coordinator)
        _print_summary(coordinator, result, agents)
    finally:
        coordinator.close_all()
        for agent in agents:
            agent.stop(timeout=3.0)

    _print_separator("CNP Demonstration Complete")
    print()
    return result


if __name__ == "__main__":
    run()
