#!/usr/bin/env python3
"""SmartNest — Assignment 3.2: Self-Interested Agent Utility Optimization.

Run from the ``backend`` directory:

    python src/utility_optimization_run.py

What it shows (everything is calculated from the data — the winner and
the ranking are not hardcoded anywhere):

    1. The user's requirements and the utility function with its weights.
    2. 14 simulated sofas from the catalogues of Product Research Agent A
       and Agent B, each validated and checked against the hard constraints.
    3. Rejected candidates with their explicit reasons.
    4. Normalised component scores (P, Q, F, S, C) and U for every
       feasible candidate, then the ranking and the selected product.
    5. A worked, substituted calculation of the winner's utility.
    6. Each agent's own highest-utility option (self-interested agents
       evaluating their own catalogues).
    7. An independently verifiable example whose answer is known by hand.
    8. Boundary checks (exactly at / just over each limit).
    9. The same decision reached through the Contract Net Protocol over
       TCP, with the coordinator recalculating every bidder's score.
"""

from __future__ import annotations

import logging
import sys

from smartnest.cnp.bidding_agent import BiddingAgent
from smartnest.cnp.coordinator_agent import CoordinatorAgent
from smartnest.cnp.product_catalog import catalog_agent_a, catalog_agent_b
from smartnest.cnp.proposal_evaluator import ProposalEvaluator
from smartnest.optimization import (
    COMPONENTS,
    OptimizationResult,
    UserRequirements,
    UtilityModel,
    UtilityOptimizer,
)

# ── Scenario: the user's request (structured requirements) ──────────────────
REQUIREMENTS = UserRequirements(
    budget=30000,
    max_width_cm=210,
    max_depth_cm=110,
    min_seating=3,
    preferred_style="minimalist",
    preferred_colour="neutral",
)

SYMBOL = {"price": "P", "quality": "Q", "spatial": "F", "style": "S", "colour": "C"}


def _sep(title: str) -> None:
    print(f"\n{'=' * 78}\n  {title}\n{'=' * 78}")


def _rupees(value: float) -> str:
    return f"₹{value:,.0f}"


def load_candidates() -> list[dict]:
    """All simulated sofas from both agents' (extended) catalogues."""
    products: list[dict] = []
    for catalog in (catalog_agent_a(extended=True), catalog_agent_b(extended=True)):
        for p in catalog.get_all_products():
            products.append({**p, "source_agent_id": catalog.agent_id})
    return products


# ── Sections ─────────────────────────────────────────────────────────────────

def show_requirements(model: UtilityModel) -> None:
    _sep("SmartNest — Assignment 3.2: Utility Optimization for Sofa Selection")
    r = REQUIREMENTS
    print(f"  Budget ≤ {_rupees(r.budget)} | width ≤ {r.max_width_cm:g} cm | "
          f"depth ≤ {r.max_depth_cm:g} cm | seats ≥ {r.min_seating}")
    print(f"  Preferred style: {r.preferred_style} | preferred colour: {r.preferred_colour} | "
          f"availability must be confirmed")
    w = model.weights
    print("\n  Utility function:")
    print("    U(p) = 100 × (" + " + ".join(f"{w[k]:.2f}{SYMBOL[k]}" for k in COMPONENTS) + ")")
    print("    P = clamp(1 − price / budget, 0, 1)")
    print("    Q = 0.80 × rating/5 + 0.20 × min(reviews/100, 1)")
    print("    F = 0.50 × width/max_width + 0.50 × depth/max_depth   (feasible products only)")
    print("    S = 1.0 exact style | 0.5 compatible style | 0.0 otherwise   (configurable table)")
    print("    C = 1.0 exact colour | 0.5 same colour family | 0.0 otherwise (configurable table)")


def show_all_candidates(result: OptimizationResult, raw: list[dict]) -> None:
    _sep(f"Candidates Inspected: {result.total_inspected}")
    by_id = {(p["source_agent_id"], p["product_id"]): p for p in raw}
    print(f"  {'ID':8} {'Agent':17} {'Price':>9} {'W×D cm':>9} {'Seats':>5} "
          f"{'Style':13} {'Colour':9} {'Rating':>6} {'Reviews':>7}  Status")
    for e in result.evaluations:
        p = by_id[(e.source_agent_id, e.product_id)]
        print(f"  {e.product_id:8} {e.source_agent_id:17} {_rupees(p['price']):>9} "
              f"{p['width_cm']:>4}×{p['depth_cm']:<4} {p['seating_capacity']:>5} "
              f"{p['style']:13} {p['color']:9} {p['rating']:>6} {p['review_count']:>7}  "
              f"{e.status.upper()}")
    print(f"\n  Eligible: {result.eligible_count}   Rejected: {result.rejected_count} "
          f"(of which invalid records: {result.invalid_count})")


def show_rejections(result: OptimizationResult) -> None:
    _sep("Rejected Candidates (hard constraints are never overridden by utility)")
    if not result.rejected:
        print("  (none)")
    for e in result.rejected:
        print(f"  ✗ {e.product_id:8} {e.name}")
        for reason in e.reasons:
            print(f"        - {reason}")


def show_scores_and_ranking(result: OptimizationResult) -> None:
    _sep("Normalised Component Scores and Ranking (feasible candidates)")
    print(f"  {'Rank':>4}  {'ID':8} {'Price':>9}   " +
          "  ".join(f"{SYMBOL[k]:>6}" for k in COMPONENTS) + f"   {'U':>7}")
    for e in result.ranking:
        c = e.breakdown.components
        print(f"  {e.rank:>4}  {e.product_id:8} {_rupees(e.price):>9}   " +
              "  ".join(f"{c[k]:>6.4f}" for k in COMPONENTS) + f"   {e.utility:>7.2f}")
    print("\n  Ranking order: higher U → higher Q → lower price → product ID.")


def show_selection(result: OptimizationResult, model: UtilityModel) -> None:
    _sep("Selected Product")
    if result.selected is None:
        print(f"  No recommendation: {result.explanation}")
        return
    s = result.selected
    c, w = s.breakdown.components, model.weights
    print(f"  ★ {s.product_id} — {s.name} ({s.source_agent_id}) at {_rupees(s.price)}")
    print(f"    U = 100 × (" + " + ".join(f"{w[k]:.2f}×{c[k]:.4f}" for k in COMPONENTS) + ")")
    print(f"      = " + " + ".join(f"{s.breakdown.contributions[k]:.2f}" for k in COMPONENTS)
          + f" = {s.utility:.2f}")
    print(f"\n  Why it won: {result.explanation}")
    best_other = max((e.utility for e in result.ranking[1:]), default=None)
    print(f"\n  Exhaustive check: selected U = {s.utility:.4f}"
          + (f" ≥ every other feasible U (next best {best_other:.4f})" if best_other is not None else "")
          + f" → verified_optimal = {result.verified_optimal}")


def show_agent_self_evaluation(optimizer: UtilityOptimizer) -> None:
    _sep("Self-Interested Agents: Each Agent's Own Highest-Utility Option")
    for catalog in (catalog_agent_a(extended=True), catalog_agent_b(extended=True)):
        agent = BiddingAgent(catalog.agent_id, catalog, utility_optimizer=optimizer)
        own = agent.evaluate_own_catalog(REQUIREMENTS)
        best = own.selected
        print(f"  {agent.agent_id}: {own.total_inspected} products, {own.eligible_count} feasible → "
              + (f"best {best.product_id} (U = {best.utility:.2f})" if best else "nothing feasible"))


def show_verifiable_example(optimizer: UtilityOptimizer) -> bool:
    """A tiny example whose answer can be checked with pen and paper."""
    _sep("Independently Verifiable Example")
    req = UserRequirements(budget=10000, max_width_cm=200, max_depth_cm=100, min_seating=3,
                           preferred_style="minimalist", preferred_colour="neutral")
    common = dict(seating_capacity=3, material="fabric", availability="in_stock",
                  source_agent_id="worked_example")
    x = dict(product_id="X", name="Exact fit", price=5000, width_cm=200, depth_cm=100, rating=5,
             review_count=100, style="minimalist", colour="neutral", **common)
    y = dict(product_id="Y", name="Cheap and small", price=2000, width_cm=100, depth_cm=50,
             rating=4, review_count=100, style="scandinavian", colour="red", **common)
    z = dict(product_id="Z", name="Bargain but 1 cm too wide", price=1000, width_cm=201,
             depth_cm=100, rating=5, review_count=500, style="minimalist", colour="neutral", **common)
    print("  Budget ₹10,000, room 200 × 100 cm, minimalist, neutral.")
    print("  X: P=1−5000/10000=0.5, Q=0.8·5/5+0.2·1=1, F=0.5·1+0.5·1=1, S=1, C=1")
    print("     U = 100(0.30·0.5 + 0.25·1 + 0.20·1 + 0.15·1 + 0.10·1) = 85.00")
    print("  Y: P=0.8, Q=0.8·4/5+0.2=0.84, F=0.5·0.5+0.5·0.5=0.5, S=0.5 (compatible), C=0")
    print("     U = 100(0.24 + 0.21 + 0.10 + 0.075 + 0) = 62.50")
    print("  Z: would score highest on price, but is 1 cm too wide → must be rejected.")
    result = optimizer.optimize([x, y, z], req)
    got = {e.product_id: e.utility for e in result.ranking}
    ok = (result.selected is not None and result.selected.product_id == "X"
          and abs(got["X"] - 85.0) < 1e-6 and abs(got["Y"] - 62.5) < 1e-6
          and "Z" not in got)
    print(f"\n  Computed: X = {got.get('X')}, Y = {got.get('Y')}, Z rejected = {'Z' not in got}"
          f" → selected {result.selected.product_id if result.selected else None}"
          f"  [{'MATCHES hand calculation' if ok else 'MISMATCH'}]")
    return ok


def show_boundaries(optimizer: UtilityOptimizer) -> None:
    _sep("Constraint Boundary Checks (limits are inclusive)")
    base = dict(product_id="T", name="Boundary probe", price=25000, width_cm=200, depth_cm=100,
                seating_capacity=3, material="fabric", colour="neutral", style="minimalist",
                availability="in_stock", rating=4.0, review_count=50, source_agent_id="probe")
    probes = [
        ("price = budget (₹30,000)", {"price": 30000}),
        ("price = budget + ₹1", {"price": 30001}),
        ("width = 210 cm", {"width_cm": 210}),
        ("width = 211 cm", {"width_cm": 211}),
        ("depth = 110 cm", {"depth_cm": 110}),
        ("depth = 111 cm", {"depth_cm": 111}),
        ("seats = 3 (minimum)", {"seating_capacity": 3}),
        ("seats = 2", {"seating_capacity": 2}),
        ("out of stock", {"availability": "out_of_stock"}),
        ("depth not stated", {"depth_cm": None}),
        ("negative price", {"price": -100}),
        ("rating 5.5", {"rating": 5.5}),
        ("NaN width", {"width_cm": float("nan")}),
        ("empty style", {"style": ""}),
    ]
    for label, change in probes:
        record = {**base, **change}
        if change.get("depth_cm", 0) is None:
            record.pop("depth_cm")
        e = optimizer.evaluate_candidate(record, REQUIREMENTS)
        detail = f"U = {e.utility:.2f}" if e.eligible else e.reasons[0]
        print(f"  {label:26} → {e.status.upper():8} {detail}")
    empty = optimizer.optimize([], REQUIREMENTS)
    print(f"  {'empty candidate list':26} → selected = {empty.selected} ({empty.explanation})")


def show_cnp_integration(optimizer_result: OptimizationResult) -> bool:
    _sep("CNP Integration: Same Decision Through Negotiation over TCP")
    agents = [
        BiddingAgent("research_agent_a", catalog_agent_a(extended=True),
                     max_discount_pct=0.05, utility_optimizer=UtilityOptimizer()),
        BiddingAgent("research_agent_b", catalog_agent_b(extended=True),
                     max_discount_pct=0.04, utility_optimizer=UtilityOptimizer()),
    ]
    for a in agents:
        a.start()
    coordinator = CoordinatorAgent(evaluator=ProposalEvaluator.utility_based(), max_rounds=3)
    r = REQUIREMENTS
    coordinator.set_requirements(
        task_description="Three-seater minimalist sofa for the living room",
        max_budget=r.budget, room_width=r.max_width_cm, room_depth=r.max_depth_cm,
        preferred_style=r.preferred_style, preferred_colour=r.preferred_colour,
        seating_capacity=r.min_seating,
    )
    for a in agents:
        coordinator.add_bidder(a.agent_id, a.host, a.actual_port)
    try:
        result = coordinator.run_negotiation()
        for round_num, evaluations in coordinator.round_evaluations.items():
            print(f"\n  ROUND {round_num}")
            for ev in evaluations:
                if ev.feasible:
                    reported = (f"bidder reported U = {100 * ev.reported_utility:.2f}, "
                                if ev.reported_utility is not None else "")
                    print(f"     {ev.agent_id:17} {ev.product_id:8} {_rupees(ev.proposed_price):>9}  "
                          f"{reported}coordinator recalculated U = {100 * ev.utility_score:.2f}")
                else:
                    print(f"     {ev.agent_id:17} {ev.product_id:8} {_rupees(ev.proposed_price):>9}  "
                          f"REJECTED: {ev.rejection_reason}")
        w = result.winner
        print(f"\n  Negotiation winner: {w.agent_id} / {w.product_id} at {_rupees(w.proposed_price)} "
              f"(U = {100 * w.utility_score:.2f}) after {result.rounds} round(s)")
        same = w.product_id == optimizer_result.selected.product_id
        print(f"  Exhaustive optimizer selected {optimizer_result.selected.product_id} → "
              f"{'consistent' if same else 'DIFFERENT'} with the negotiated outcome")
        return same
    finally:
        coordinator.close_all()
        for a in agents:
            a.stop(timeout=3.0)


def run() -> OptimizationResult:
    """Run the full Assignment 3.2 demonstration and return the main result."""
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    model = UtilityModel()
    optimizer = UtilityOptimizer(model)

    show_requirements(model)
    candidates = load_candidates()
    result = optimizer.optimize(candidates, REQUIREMENTS)

    show_all_candidates(result, candidates)
    show_rejections(result)
    show_scores_and_ranking(result)
    show_selection(result, model)
    show_agent_self_evaluation(optimizer)
    show_verifiable_example(optimizer)
    show_boundaries(optimizer)
    if result.selected is not None:
        show_cnp_integration(result)

    _sep("Assignment 3.2 Demonstration Complete")
    print()
    return result


if __name__ == "__main__":
    run()
