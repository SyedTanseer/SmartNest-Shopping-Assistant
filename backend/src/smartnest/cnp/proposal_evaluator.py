"""Proposal evaluation and utility scoring for CNP.

The ``ProposalEvaluator`` is intentionally isolated from the negotiation
protocol so that Assignment 3.2 can replace it with a more sophisticated
mathematical utility optimization engine without touching the
coordinator or bidding agents.  Any replacement only needs:

    evaluate(proposal: dict, requirements: dict) -> ProposalEvaluation
    rank(evaluations: list[ProposalEvaluation]) -> list[ProposalEvaluation]

Hard constraints (any violation → infeasible, utility 0):
    - Product must be purchasable (not out of stock / unavailable).
    - Proposed price ≤ ``max_budget``.
    - Product width must be known and ≤ ``room_width``.
    - Seating capacity ≥ required ``seating_capacity`` (when both are given).

Soft scoring (each normalised to [0, 1], then weighted):
    - price:        1 − price / budget (cheaper is better).
    - spatial:      width / room_width (better use of the space, never over).
    - style:        1.0 exact match, 0.5 partial match (one style name
                    contains the other, e.g. "scandi minimalist" vs
                    "minimalist"), 0.0 otherwise or if the product has no
                    style.  If no style is requested, every product gets 1.0.
    - availability: in_stock 1.0, low_stock 0.7, backorder/preorder 0.4 —
                    so delivery risk actually affects the ranking.

Weights are validated (known keys, non-negative, positive sum) and
normalised to sum to 1, so utility stays in [0, 1].

Two scoring modes (Assignment 3.2):
    - **legacy** (default) — the Assignment 3.1 weighted score described
      above.  Kept unchanged so existing CNP callers behave exactly as
      before.
    - **utility** — ``ProposalEvaluator(utility_model=UtilityModel(...))``
      or ``ProposalEvaluator.utility_based()``.  ``evaluate`` delegates
      to :mod:`smartnest.optimization`: the proposal is validated, checked
      against the hard constraints (budget, width, depth, seating,
      availability) and scored with
      U = 100 × (0.30P + 0.25Q + 0.20F + 0.15S + 0.10C).
      ``utility_score`` is U / 100 so both modes share the [0, 1] scale;
      the full breakdown (0–100 points) is in ``utility_breakdown``.

In both modes the evaluator scores the *offered* price
(``proposed_price``), so a negotiated discount is always re-scored, and
any utility a bidder reports about itself is ignored for ranking (it is
only recorded in ``reported_utility`` for auditing).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

from smartnest.optimization import (
    ScoringConfig,
    UserRequirements,
    UtilityModel,
    UtilityOptimizer,
)

logger = logging.getLogger(__name__)

# Availability values that make a product impossible to buy.
UNAVAILABLE_STATES = frozenset({"out_of_stock", "unavailable", "discontinued", ""})

# Soft availability scores for purchasable states.  Unknown purchasable
# states get the backorder score (treated as a delivery risk).
AVAILABILITY_SCORES: dict[str, float] = {
    "in_stock": 1.0,
    "low_stock": 0.7,
    "backorder": 0.4,
    "preorder": 0.4,
}
_DEFAULT_AVAILABILITY_SCORE = 0.4


@dataclass
class ProposalEvaluation:
    """The result of evaluating a single proposal.

    Attributes:
        agent_id: ID of the proposing agent.
        product_id: ID of the proposed product.
        feasible: Whether the proposal satisfies all hard constraints.
        utility_score: Normalised score in [0, 1] (0 if infeasible).
        scores: Breakdown of individual component scores.
        rejection_reason: Why the proposal was rejected (if infeasible).
        proposed_price: The price that was evaluated (the bid price).
        round: Negotiation round the proposal came from (0 if unknown).
        quality_score: Q component (utility mode; 0.0 in legacy mode),
            used as the first tie-breaker after utility.
        utility_breakdown: Full utility-model breakdown (utility mode).
        reported_utility: Utility the bidder claimed for itself, if any.
            Recorded for auditing only — never used for ranking.
    """

    agent_id: str
    product_id: str
    feasible: bool
    utility_score: float
    scores: dict[str, float] = field(default_factory=dict)
    rejection_reason: str = ""
    proposed_price: float = 0.0
    round: int = 0
    quality_score: float = 0.0
    utility_breakdown: dict[str, Any] = field(default_factory=dict)
    reported_utility: float | None = None

    @property
    def utility_discrepancy(self) -> float | None:
        """reported − recalculated utility (``None`` if nothing comparable)."""
        if self.reported_utility is None or not self.feasible:
            return None
        return round(self.reported_utility - self.utility_score, 6)


class ProposalEvaluator:
    """Evaluates CNP proposals against requirements using weighted utility.

    Args:
        weights: Optional dict of component weights (keys from
            :attr:`COMPONENTS`).  They are normalised to sum to 1.
            Defaults prioritise spatial fit, then price, style, availability.

    Raises:
        ValueError: If weights contain unknown keys, negative values, or
            sum to zero.
    """

    COMPONENTS: tuple[str, ...] = ("price", "spatial", "style", "availability")

    DEFAULT_WEIGHTS: dict[str, float] = {
        "price": 0.30,
        "spatial": 0.35,
        "style": 0.20,
        "availability": 0.15,
    }

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        utility_model: UtilityModel | None = None,
    ) -> None:
        if weights is not None and utility_model is not None:
            raise ValueError(
                "Pass either legacy weights or a utility_model, not both "
                "(utility weights live in the model ScoringConfig)"
            )
        self._weights = self._normalise_weights(
            dict(self.DEFAULT_WEIGHTS) if weights is None else dict(weights)
        )
        self._optimizer = UtilityOptimizer(utility_model) if utility_model is not None else None

    @classmethod
    def utility_based(cls, config: ScoringConfig | None = None) -> "ProposalEvaluator":
        """An evaluator that delegates to the Assignment 3.2 utility model."""
        return cls(utility_model=UtilityModel(config))

    @property
    def mode(self) -> str:
        """``"utility"`` when delegating to the utility model, else ``"legacy"``."""
        return "utility" if self._optimizer is not None else "legacy"

    @property
    def optimizer(self) -> UtilityOptimizer | None:
        return self._optimizer

    @classmethod
    def _normalise_weights(cls, weights: dict[str, float]) -> dict[str, float]:
        unknown = set(weights) - set(cls.COMPONENTS)
        if unknown:
            raise ValueError(f"Unknown weight component(s): {sorted(unknown)}")
        if any(w < 0 for w in weights.values()):
            raise ValueError("Weights must be non-negative")
        total = sum(weights.values())
        if total <= 0:
            raise ValueError("Weights must sum to a positive value")
        return {k: weights.get(k, 0.0) / total for k in cls.COMPONENTS}

    @property
    def weights(self) -> dict[str, float]:
        return dict(self._weights)

    def evaluate(
        self,
        proposal: dict[str, Any],
        requirements: dict[str, Any],
    ) -> ProposalEvaluation:
        """Evaluate a single proposal against requirements.

        Args:
            proposal: Dict with ``agent_id``, ``product_id``, ``product``
                (attributes) and ``proposed_price``.
            requirements: Dict with ``max_budget``, ``room_width``, and
                optionally ``preferred_style`` and ``seating_capacity``.

        Returns:
            A ``ProposalEvaluation`` with feasibility and utility score.

        Raises:
            InvalidRequirementsError: In utility mode, if ``requirements``
                lack a valid budget, width, depth or seating value.
        """
        if self._optimizer is not None:
            return self._evaluate_with_utility_model(proposal, requirements)

        agent_id = proposal.get("agent_id", "unknown")
        product_id = proposal.get("product_id", "unknown")
        product = proposal.get("product", {}) or {}
        price = proposal.get("proposed_price", product.get("price", 0))
        width = product.get("width_cm") or 0
        style = str(product.get("style", "") or "").strip().lower()
        availability = str(product.get("availability", "") or "").strip().lower()
        seats = product.get("seating_capacity")
        round_num = int(proposal.get("negotiation_round", 0) or 0)

        max_budget = requirements.get("max_budget", 0) or 0
        room_width = requirements.get("room_width", 0) or 0
        preferred_style = str(requirements.get("preferred_style", "") or "").strip().lower()
        required_seats = requirements.get("seating_capacity")

        # ── Hard constraints ─────────────────────────────────────────────
        rejection_reason = ""

        if availability in UNAVAILABLE_STATES:
            rejection_reason = (
                f"Product {product_id} is not in stock ({availability or 'unknown'})"
            )
        elif price > max_budget:
            rejection_reason = f"Price ₹{price:,} exceeds max budget ₹{max_budget:,}"
        elif width <= 0:
            rejection_reason = f"Product {product_id} has no width, so fit cannot be checked"
        elif width > room_width:
            rejection_reason = f"Width {width} cm exceeds room width {room_width} cm"
        elif required_seats and seats is not None and seats < required_seats:
            rejection_reason = (
                f"Seating capacity {seats} is below the required {required_seats}"
            )

        if rejection_reason:
            return ProposalEvaluation(
                agent_id=agent_id,
                product_id=product_id,
                feasible=False,
                utility_score=0.0,
                scores={},
                rejection_reason=rejection_reason,
                proposed_price=price,
                round=round_num,
            )

        # ── Soft scoring components ──────────────────────────────────────
        scores: dict[str, float] = {
            "price": max(0.0, 1.0 - price / max_budget) if max_budget > 0 else 0.0,
            "spatial": min(1.0, width / room_width) if room_width > 0 else 0.0,
            "style": self._style_score(style, preferred_style),
            "availability": AVAILABILITY_SCORES.get(availability, _DEFAULT_AVAILABILITY_SCORE),
        }

        utility = sum(self._weights[k] * scores[k] for k in self.COMPONENTS)

        logger.debug(
            "Evaluated %s: feasible=True, utility=%.4f, scores=%s",
            product_id,
            utility,
            scores,
        )

        return ProposalEvaluation(
            agent_id=agent_id,
            product_id=product_id,
            feasible=True,
            utility_score=round(utility, 4),
            scores={k: round(v, 4) for k, v in scores.items()},
            proposed_price=price,
            round=round_num,
        )

    def _evaluate_with_utility_model(
        self,
        proposal: dict[str, Any],
        requirements: dict[str, Any],
    ) -> ProposalEvaluation:
        """Utility mode: validate, constraint-check and score via the optimizer."""
        agent_id = proposal.get("agent_id", "unknown")
        product = dict(proposal.get("product") or {})
        product_id = proposal.get("product_id") or product.get("product_id") or "unknown"
        offered = proposal.get("proposed_price", product.get("price"))
        round_num = int(proposal.get("negotiation_round", 0) or 0)
        reported = _finite_or_none(proposal.get("self_reported_utility"))

        # Score what is actually on offer: the negotiated price, attributed
        # to the bidder the coordinator is talking to.
        record = {**product, "product_id": product_id, "price": offered,
                  "source_agent_id": agent_id}
        req = UserRequirements.from_cnp(requirements)
        evaluation = self._optimizer.evaluate_candidate(record, req, require_complete=True)  # CNP records must be complete
        price = _finite_or_none(offered) or 0.0

        if not evaluation.eligible:
            return ProposalEvaluation(
                agent_id=agent_id,
                product_id=product_id,
                feasible=False,
                utility_score=0.0,
                rejection_reason="; ".join(evaluation.reasons),
                proposed_price=price,
                round=round_num,
                reported_utility=reported,
            )

        breakdown = evaluation.breakdown
        return ProposalEvaluation(
            agent_id=agent_id,
            product_id=product_id,
            feasible=True,
            utility_score=breakdown.normalised_utility,
            scores=dict(breakdown.components),
            proposed_price=price,
            round=round_num,
            quality_score=breakdown.components["quality"],
            utility_breakdown=breakdown.model_dump(),
            reported_utility=reported,
        )

    @staticmethod
    def _style_score(style: str, preferred_style: str) -> float:
        if not preferred_style:
            return 1.0  # no preference → every style is acceptable
        if not style:
            return 0.0  # unknown style can't be credited as a match
        if style == preferred_style:
            return 1.0
        if style in preferred_style or preferred_style in style:
            return 0.5
        return 0.0

    def rank(
        self,
        evaluations: list[ProposalEvaluation],
    ) -> list[ProposalEvaluation]:
        """Rank feasible evaluations by utility score (descending).

        Deterministic tie-breaking (same order as the 3.2 optimizer):
        higher quality score, lower proposed price, product ID A→Z, then
        earlier round.  In legacy mode the quality score is 0 for every
        proposal, so ties fall through to price.
        Infeasible proposals are excluded from the ranking.
        """
        feasible = [e for e in evaluations if e.feasible]
        feasible.sort(key=lambda e: (
            -round(e.utility_score, 6),
            -round(e.quality_score, 6),
            e.proposed_price,
            e.product_id,
            e.round,
        ))
        return feasible


def _finite_or_none(value: Any) -> float | None:
    """``float(value)`` if it is a finite real number (not bool), else ``None``."""
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None
