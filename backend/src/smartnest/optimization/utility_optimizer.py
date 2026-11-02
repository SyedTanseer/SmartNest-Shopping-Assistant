"""Deterministic utility optimizer (Assignment 3.2).

Algorithm — exhaustive evaluation, which is exact and cheap for catalogue
sizes in the tens or hundreds:

    1. Inspect every candidate (input order is preserved in the report).
    2. Validate its mandatory fields       → "invalid" if malformed.
    3. Check the hard constraints          → "rejected" with reasons.
    4. Compute the normalised utility U of every eligible candidate.
    5. Rank eligible candidates by the deterministic key
           (higher U, higher quality Q, lower price, product_id A→Z,
            source agent A→Z).
    6. Select rank 1, and verify it against every eligible candidate.

Room limits are optional.  When the user gave some, a candidate whose size
is not stated is neither rejected nor claimed to fit: it is "unverified".
The same holds for a candidate whose availability is unknown (confirmed out of
stock is rejected) or whose seating capacity is not stated while the user needs
more than one seat.  Unverified candidates are scored on the data they have
(missing components are excluded, the other weights are redistributed) and
ranked in their own list, ``OptimizationResult.unverified``; they are never
``selected``.  Each candidate states exactly what could not be verified
(``gaps``), its fit label, its availability state and an information
completeness indicator.

Because every candidate is evaluated and the ranking is a total order,
the selected candidate is provably the highest-utility feasible option
in the evaluated set.  If no candidate is feasible, there is no
recommendation (``selected`` is ``None``) — never a "best of the
infeasible".

The optimizer is pure computation: no sockets, no CNP state, no database,
no LLM.  :meth:`UtilityOptimizer.optimize_async` runs it in a worker
thread so an async web backend (Assignment 5.2) is never blocked.
"""

from __future__ import annotations

import asyncio
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict

from smartnest.optimization.constraint_validator import (
    AVAILABILITY_LABELS,
    ConstraintValidator,
    ProductCandidate,
    SpatialAssessment,
    Status,
    UserRequirements,
)
from smartnest.optimization.scoring_config import COMPONENTS
from smartnest.optimization.utility_model import SCORE_PRECISION, SYMBOLS, UtilityBreakdown, UtilityModel


class CandidateEvaluation(BaseModel):
    """Everything known about one candidate after evaluation."""

    model_config = ConfigDict(frozen=True)

    product_id: str
    source_agent_id: str | None = None
    name: str | None = None
    status: Status
    reasons: list[str] = []
    price: float | None = None
    breakdown: UtilityBreakdown | None = None
    rank: int | None = None      # position in ``ranking`` (eligible) or in ``unverified``
    spatial: SpatialAssessment | None = None
    availability: str | None = None   # "in_stock" / "out_of_stock" / "unknown"
    gaps: list[str] = []              # for an unverified product: "fit", "availability" and/or "seating"

    @property
    def eligible(self) -> bool:
        return self.status == "eligible"

    @property
    def unverified(self) -> bool:
        """Satisfies every checkable constraint but its fit against the room limits is not verified."""
        return self.status == "unverified"

    @property
    def availability_label(self) -> str | None:
        """"In stock" / "Out of stock" / "Availability unknown - check stock on the retailer's listing"."""
        return AVAILABILITY_LABELS.get(self.availability) if self.availability else None

    @property
    def completeness(self) -> float | None:
        """Information completeness of the score in [0, 1] (``None`` if not scored)."""
        return self.breakdown.completeness if self.breakdown else None

    @property
    def unavailable(self) -> list[str]:
        """Attributes that were needed for scoring but not stated."""
        return list(self.breakdown.unavailable) if self.breakdown else []

    @property
    def fit_label(self) -> str | None:
        """"Fit verified" / "Fit not verified" / "Dimensionally incompatible" / "No room limits specified"."""
        return self.spatial.label if self.spatial is not None else None

    @property
    def utility(self) -> float | None:
        """U on the 0–100 scale (``None`` if not eligible)."""
        return self.breakdown.total_utility if self.breakdown else None


class OptimizationResult(BaseModel):
    """Serialisable outcome of one optimisation run."""

    requirements: UserRequirements
    weights: dict[str, float]
    total_inspected: int
    eligible_count: int
    rejected_count: int  # candidates that fail a constraint or are malformed (not eligible, not unverified)
    invalid_count: int   # subset of rejected_count with malformed records
    evaluations: list[CandidateEvaluation]  # input order
    ranking: list[CandidateEvaluation]      # eligible only (fit verified, or no limits given), best first
    selected: CandidateEvaluation | None
    verified_optimal: bool
    explanation: str
    unverified_count: int = 0
    unverified: list[CandidateEvaluation] = []  # fit not verified, ranked among themselves; never selected

    @property
    def has_recommendation(self) -> bool:
        return self.selected is not None

    @property
    def rejected(self) -> list[CandidateEvaluation]:
        """Candidates that failed a constraint or were malformed (unverified ones are not rejected)."""
        return [e for e in self.evaluations if not e.eligible and not e.unverified]

    @property
    def best_unverified(self) -> CandidateEvaluation | None:
        """The top-ranked candidate that is not fully verified (never a confirmed recommendation)."""
        return self.unverified[0] if self.unverified else None

    def unverified_for(self, gap: str) -> list[CandidateEvaluation]:
        """Unverified candidates whose ``gaps`` include ``gap`` ("fit", "availability" or "seating")."""
        return [e for e in self.unverified if gap in e.gaps]

    @property
    def incompatible(self) -> list[CandidateEvaluation]:
        """Rejected because a stated size exceeds a room limit the user gave."""
        return [e for e in self.evaluations if e.spatial is not None and e.spatial.status == "incompatible"]

    def by_fit(self) -> dict[str, list[CandidateEvaluation]]:
        """Candidates grouped by how their fit is reported (empty groups omitted)."""
        groups: dict[str, list[CandidateEvaluation]] = {}
        for e in self.evaluations:
            if e.fit_label is not None:
                groups.setdefault(e.fit_label, []).append(e)
        return groups


def ranking_key(evaluation: CandidateEvaluation) -> tuple:
    """Deterministic total order: U ↓, information completeness ↓, Q ↓, price ↑, product_id ↑, source ↑.

    Completeness only breaks exact utility ties.  Q is a sort key here, not a score: a product with no rating sorts
    after one with a rating at an otherwise equal position (it is not treated as having Q = 0).
    """
    b = evaluation.breakdown
    return (
        -round(b.total_utility, SCORE_PRECISION),
        -round(b.completeness, SCORE_PRECISION),
        -round(b.components.get("quality", -1.0), SCORE_PRECISION),
        evaluation.price,
        evaluation.product_id,
        evaluation.source_agent_id or "",
    )


class UtilityOptimizer:
    """Evaluates, filters, ranks and selects product candidates.

    Args:
        model: The utility model (default configuration if omitted).
        validator: The hard-constraint validator; defaults to one that
            shares the model's configuration.
    """

    def __init__(
        self,
        model: UtilityModel | None = None,
        validator: ConstraintValidator | None = None,
    ) -> None:
        self._model = model or UtilityModel()
        self._validator = validator or ConstraintValidator(self._model.config)

    @property
    def model(self) -> UtilityModel:
        return self._model

    @property
    def validator(self) -> ConstraintValidator:
        return self._validator

    # ── Single candidate ─────────────────────────────────────────────────

    def evaluate_candidate(self, record: Any, requirements: UserRequirements, *,
                           require_complete: bool = False) -> CandidateEvaluation:
        """Validate, constraint-check and (if eligible or unverified) score one candidate.

        ``require_complete`` is the strict legacy mode used by CNP negotiations (see ``validate_record``).
        """
        result = self._validator.evaluate(record, requirements, require_complete=require_complete)
        cand = result.candidate
        if result.status in ("rejected", "invalid"):
            return CandidateEvaluation(
                product_id=result.product_id,
                source_agent_id=result.source_agent_id,
                name=cand.name if cand else None,
                status=result.status,
                reasons=list(result.reasons),
                price=cand.price if cand else None,
                spatial=result.spatial,
                availability=result.availability,
            )
        return CandidateEvaluation(
            product_id=cand.product_id,
            source_agent_id=cand.source_agent_id,
            name=cand.name,
            status=result.status,           # "eligible" or "unverified"
            reasons=list(result.reasons),   # for unverified: why the fit cannot be checked
            price=cand.price,
            breakdown=self._model.score(cand, requirements),
            spatial=result.spatial,
            availability=result.availability,
            gaps=list(result.gaps),
        )

    # ── Whole set ─────────────────────────────────────────────────────────

    def optimize(
        self,
        candidates: Iterable[Any],
        requirements: UserRequirements | dict[str, Any],
    ) -> OptimizationResult:
        """Exhaustively evaluate ``candidates`` and select the best feasible one.

        Raises:
            InvalidRequirementsError: If ``requirements`` are malformed.
        """
        if not isinstance(requirements, UserRequirements):
            requirements = UserRequirements(**requirements)

        evaluations: list[CandidateEvaluation] = []
        seen: set[tuple[str, str]] = set()
        for record in list(candidates):
            evaluation = self.evaluate_candidate(record, requirements)
            key = (evaluation.source_agent_id or "", evaluation.product_id)
            if evaluation.status != "invalid" and key in seen:
                evaluation = evaluation.model_copy(update={
                    "status": "invalid",
                    "breakdown": None,
                    "reasons": [f"invalid record: duplicate product {evaluation.product_id} "
                                f"from {evaluation.source_agent_id}"],
                })
            seen.add(key)
            evaluations.append(evaluation)

        eligible = sorted((e for e in evaluations if e.eligible), key=ranking_key)
        ranking = [e.model_copy(update={"rank": i}) for i, e in enumerate(eligible, start=1)]
        unverified_sorted = sorted((e for e in evaluations if e.unverified), key=ranking_key)
        unverified = [e.model_copy(update={"rank": i}) for i, e in enumerate(unverified_sorted, start=1)]
        rank_of = {(e.source_agent_id, e.product_id): e.rank for e in ranking + unverified}
        evaluations = [
            e.model_copy(update={"rank": rank_of.get((e.source_agent_id, e.product_id))})
            if (e.eligible or e.unverified) else e
            for e in evaluations
        ]

        selected = ranking[0] if ranking else None
        verified = selected is not None and all(
            selected.utility >= e.utility for e in ranking
        )

        invalid = sum(1 for e in evaluations if e.status == "invalid")
        return OptimizationResult(
            requirements=requirements,
            weights=self._model.weights,
            total_inspected=len(evaluations),
            eligible_count=len(ranking),
            rejected_count=len(evaluations) - len(ranking) - len(unverified),
            invalid_count=invalid,
            evaluations=evaluations,
            ranking=ranking,
            selected=selected,
            verified_optimal=verified,
            explanation=self._explain(ranking, len(evaluations), unverified),
            unverified_count=len(unverified),
            unverified=unverified,
        )

    async def optimize_async(
        self,
        candidates: Iterable[Any],
        requirements: UserRequirements | dict[str, Any],
    ) -> OptimizationResult:
        """Run :meth:`optimize` in a worker thread (non-blocking for async apps)."""
        return await asyncio.to_thread(self.optimize, list(candidates), requirements)

    # ── Explanation ──────────────────────────────────────────────────────

    @staticmethod
    def _explain(ranking: list[CandidateEvaluation], inspected: int,
                 unverified: list[CandidateEvaluation] | None = None) -> str:
        pending = ""
        if unverified:
            names = {"fit": "fit against the room limits", "availability": "stock", "seating": "seating capacity"}
            gaps = [names[g] for g in ("fit", "seating", "availability") if any(g in e.gaps for e in unverified)]
            pending = (f" {len(unverified)} other candidate(s) satisfy every constraint that could be checked "
                       f"but their {' and '.join(gaps)} {'is' if len(gaps) == 1 else 'are'} not verified; they are "
                       f"listed separately and are not recommended as verified"
                       f"{' (check stock on the retailer' + chr(39) + 's listing)' if 'availability' in ' '.join(g for e in unverified for g in e.gaps) else ''}.")
        if not ranking:
            return (f"No feasible candidate among {inspected} inspected — "
                    f"no recommendation is made." + pending)
        best = ranking[0]
        head = (f"{best.product_id} ({best.source_agent_id}) has the highest utility "
                f"U = {best.utility:.2f} of {len(ranking)} feasible candidate(s) "
                f"out of {inspected} inspected.")
        info = f" {best.breakdown.completeness_note}" if best.breakdown.completeness < 1 else ""
        if len(ranking) == 1:
            return head + " It is the only candidate that satisfies every hard constraint." + info + pending

        runner = ranking[1]
        margin = best.utility - runner.utility
        diffs = {
            k: best.breakdown.contributions[k] - runner.breakdown.contributions[k]
            for k in COMPONENTS
            if k in best.breakdown.contributions and k in runner.breakdown.contributions
        }
        ahead = sorted(((v, k) for k, v in diffs.items() if v > 1e-9), reverse=True)
        behind = sorted((v, k) for k, v in diffs.items() if v < -1e-9)

        def fmt(items):
            return ", ".join(f"{k} ({SYMBOLS[k]}) {v:+.2f}" for v, k in items)

        text = (f"{head} It beats the runner-up {runner.product_id} "
                f"(U = {runner.utility:.2f}) by {margin:.2f} points.")
        if ahead:
            text += f" Gains over the runner-up: {fmt(ahead)}."
        if behind:
            text += f" Trails it on: {fmt(behind)}."
        if abs(margin) <= 10 ** -SCORE_PRECISION:
            c_best, c_run = best.breakdown.completeness, runner.breakdown.completeness
            q_best, q_run = best.breakdown.components.get("quality"), runner.breakdown.components.get("quality")
            if c_best != c_run:
                rule = "more complete information"
            elif q_best != q_run:
                rule = "higher quality score"
            elif best.price != runner.price:
                rule = "lower price"
            else:
                rule = "product ID order"
            text += f" Utilities are tied; it wins the tie-break on {rule}."
        return text + info + pending


__all__ = [
    "CandidateEvaluation",
    "OptimizationResult",
    "ProductCandidate",
    "UtilityOptimizer",
    "ranking_key",
]
