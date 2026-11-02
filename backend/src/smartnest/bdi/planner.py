"""The BDI planner: turns desires into intentions, deterministically.

For every kind of goal there is a *goal handler* that knows

    - how to **plan** it from the current beliefs, and
    - whether a given belief change **could affect** it
      (so unrelated updates never trigger replanning).

The standard handlers are

``select_product``
    Reads the user's requirements from beliefs, builds the candidate
    list from the product catalogues, overlays what the agent has
    *learned since* (availability and price from environmental events),
    and runs the existing :class:`~smartnest.optimization.UtilityOptimizer`.
    The highest-utility feasible product becomes an :class:`Intention`
    whose preconditions are the optimizer's hard constraints expressed
    over beliefs — so the agent can later notice, from beliefs alone,
    that the choice is no longer valid.  Nothing about the winner is
    hardcoded: it comes from the data and the utility equation.

``maintain_budget``
    A maintenance goal.  Checks that the committed price and the
    recorded expenditure stay within the total budget.

Planning is pure Python; no LLM, network or database is involved.

Belief conventions (subject ``"user"`` holds the request)::

    user.total_budget      user.sofa_budget       user.max_width_cm
    user.max_depth_cm      user.min_seating       user.preferred_style
    user.preferred_colour  user.estimated_expenditure
    <product_id>.availability / .price / .width_cm / .depth_cm / .seating_capacity
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from smartnest.bdi.beliefs import BeliefBase, BeliefChange
from smartnest.bdi.conditions import Condition
from smartnest.bdi.desires import Desire
from smartnest.bdi.intentions import Action, Intention
from smartnest.optimization import (
    InvalidRequirementsError,
    OptimizationResult,
    UserRequirements,
    UtilityOptimizer,
)

USER = "user"
CATALOGUE_SOURCE = "catalogue"

BUDGET_PREDICATES = ("sofa_budget", "total_budget")
REQUIREMENT_PREDICATES = (
    "sofa_budget", "total_budget", "max_width_cm", "max_depth_cm", "max_height_cm",
    "min_seating", "preferred_style", "preferred_colour",
)
_POSITIVE_NUMBERS = ("sofa_budget", "total_budget", "max_width_cm", "max_depth_cm", "max_height_cm")
_TEXT = ("preferred_style", "preferred_colour")


# ── Value validation for user.* beliefs ─────────────────────────────────────

def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def normalise_requirement(predicate: str, value: Any) -> tuple[Any, str | None]:
    """Validate/normalise a ``user.<predicate>`` value.

    Returns ``(value, None)`` if usable, ``(None, reason)`` if not.  Applies to
    the known request predicates and ``estimated_expenditure``; other
    predicates pass through unchanged.
    """
    if predicate in _POSITIVE_NUMBERS:
        if not _is_number(value) or value <= 0:
            return None, f"{predicate} must be a finite number greater than 0, got {value!r}"
        return value, None
    if predicate == "min_seating":
        if not _is_number(value) or value < 1 or float(value) != int(value):
            return None, f"min_seating must be a whole number of at least 1, got {value!r}"
        return int(value), None
    if predicate in _TEXT:
        if not isinstance(value, str) or not value.strip():
            return None, f"{predicate} must be a non-empty string, got {value!r}"
        return value.strip().lower(), None
    if predicate == "estimated_expenditure":
        if not _is_number(value) or value < 0:
            return None, f"estimated_expenditure must be a finite number of at least 0, got {value!r}"
        return value, None
    return value, None


def effective_budget(sofa_budget: Any, total_budget: Any) -> float | None:
    """The budget the sofa must fit: the smaller of the known, valid budgets."""
    known = [v for v in (sofa_budget, total_budget) if _is_number(v) and v > 0]
    return min(known) if known else None


# ── Candidate sources ───────────────────────────────────────────────────────

class CandidateSource(Protocol):
    """Anything that can list product records for the optimizer.

    Records use the schema of ``smartnest.optimization.validate_record``.
    Real retailer APIs (Assignment 4.2) only need to implement this.
    """

    def get_candidates(self) -> list[dict[str, Any]]: ...


class CatalogCandidateSource:
    """Candidates from one or more ``ProductCatalog``-style objects.

    Each catalogue needs ``get_all_products()`` and an ``agent_id``.
    """

    def __init__(self, catalogs: Sequence[Any]) -> None:
        self._catalogs = list(catalogs)

    def get_candidates(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for catalog in self._catalogs:
            for product in catalog.get_all_products():
                record = dict(product)
                record.setdefault("source_agent_id", getattr(catalog, "agent_id", "catalogue"))
                records.append(record)
        return records


# ── Planning results ────────────────────────────────────────────────────────

@dataclass
class PlanOutcome:
    """What a goal handler concluded for one desire."""

    desire_id: str
    status: str  # planned | blocked | waiting | satisfied | violated
    explanation: str = ""
    intention: Intention | None = None
    optimization: OptimizationResult | None = None
    missing: list[str] = field(default_factory=list)


class GoalHandler(Protocol):
    goal_type: str

    def plan(self, desire: Desire, beliefs: BeliefBase, live_intentions: Sequence[Intention]) -> PlanOutcome: ...

    def is_affected_by(self, desire: Desire, change: BeliefChange, beliefs: BeliefBase,
                       held_product_ids: set[str]) -> tuple[bool, str]: ...


# ── select_product ──────────────────────────────────────────────────────────

class SelectProductHandler:
    """Chooses the highest-utility feasible product with the UtilityOptimizer."""

    goal_type = "select_product"

    def __init__(self, optimizer: UtilityOptimizer, candidate_source: CandidateSource) -> None:
        self.optimizer = optimizer
        self.source = candidate_source

    @property
    def available_states(self) -> list[str]:
        return sorted(self.optimizer.model.config.available_states)

    # Requirements ----------------------------------------------------------

    def requirements(self, beliefs: BeliefBase) -> tuple[UserRequirements | None, list[str]]:
        """Build optimizer requirements from beliefs, or list what is missing."""
        missing: list[str] = []
        budget = effective_budget(beliefs.value(USER, "sofa_budget"), beliefs.value(USER, "total_budget"))
        if budget is None:
            missing.append("user.sofa_budget or user.total_budget")
        # Room limits (max_width_cm / max_depth_cm / max_height_cm) are optional: only limits the user
        # gave are enforced.  But a limit that WAS given and has since been invalidated (e.g. the room
        # is being remeasured) is unknown, not absent: the agent waits instead of silently dropping it.
        for predicate in ("max_width_cm", "max_depth_cm", "max_height_cm"):
            if beliefs.value(USER, predicate) is None and beliefs.get(USER, predicate, include_invalid=True) is not None:
                missing.append(f"user.{predicate}")
        if beliefs.value(USER, "min_seating") is None:
            missing.append("user.min_seating")
        if missing:
            return None, missing
        try:
            requirements = UserRequirements(
                budget=budget,
                max_width_cm=beliefs.value(USER, "max_width_cm"),
                max_depth_cm=beliefs.value(USER, "max_depth_cm"),
                max_height_cm=beliefs.value(USER, "max_height_cm"),
                min_seating=beliefs.value(USER, "min_seating"),
                preferred_style=beliefs.value(USER, "preferred_style"),
                preferred_colour=beliefs.value(USER, "preferred_colour"),
            )
        except InvalidRequirementsError as exc:
            return None, [f"valid requirements ({exc})"]
        return requirements, []

    # Candidates ------------------------------------------------------------

    def candidates(self, beliefs: BeliefBase) -> list[dict[str, Any]]:
        """Catalogue records, overlaid with availability/price learned from events."""
        records: list[dict[str, Any]] = []
        for raw in self.source.get_candidates():
            record = dict(raw)
            product_id = record.get("product_id")
            if isinstance(product_id, str):
                for predicate in ("availability", "price"):
                    belief = beliefs.get(product_id, predicate)
                    if belief is not None and belief.source != CATALOGUE_SOURCE:
                        record[predicate] = belief.value
            records.append(record)
        return records

    def _catalogue_index(self) -> dict[str, dict[str, Any]]:
        try:
            return {r["product_id"]: r for r in self.source.get_candidates()
                    if isinstance(r.get("product_id"), str)}
        except Exception:
            return {}

    def known_product_ids(self) -> set[str]:
        return set(self._catalogue_index())

    # Planning --------------------------------------------------------------

    def plan(self, desire: Desire, beliefs: BeliefBase, live_intentions: Sequence[Intention]) -> PlanOutcome:
        requirements, missing = self.requirements(beliefs)
        if requirements is None:
            return PlanOutcome(desire.desire_id, "waiting", missing=missing,
                               explanation="cannot plan yet; missing: " + ", ".join(missing))

        records = self.candidates(beliefs)
        if not records:
            # No data yet (e.g. a live source that has not been refreshed) is not the same as
            # "nothing is feasible": wait for candidates instead of blocking the goal.
            return PlanOutcome(desire.desire_id, "waiting", missing=["product candidates"],
                               explanation="no product candidates are available yet")
        result = self.optimizer.optimize(records, requirements)
        if result.selected is None:
            return PlanOutcome(desire.desire_id, "blocked", optimization=result,
                               explanation=self._why_blocked(result))

        selected = result.selected
        record = next(r for r in records
                      if r.get("product_id") == selected.product_id
                      and r.get("source_agent_id") == selected.source_agent_id)
        intention = self._build_intention(desire, selected, record, beliefs)
        return PlanOutcome(desire.desire_id, "planned", intention=intention, optimization=result,
                           explanation=result.explanation)

    @staticmethod
    def _why_blocked(result: OptimizationResult) -> str:
        reasons = [f"{e.product_id}: {'; '.join(e.reasons)}" for e in result.evaluations[:6]]
        more = f" (+{len(result.evaluations) - 6} more)" if len(result.evaluations) > 6 else ""
        text = (f"none of the {result.total_inspected} candidates is feasible"
                + (": " + " | ".join(reasons) + more if reasons else ""))
        if result.unverified_count:
            text += (f". {result.unverified_count} candidate(s) satisfy every constraint that could be checked "
                     f"but their fit against the room limits is not verified (size not stated)")
        return text

    def _record_attributes(self, beliefs: BeliefBase, record: dict[str, Any]) -> None:
        """Remember the chosen product's attributes so preconditions can refer to beliefs."""
        pid = record["product_id"]
        for predicate in ("availability", "price", "width_cm", "depth_cm", "height_cm", "seating_capacity"):
            if record.get(predicate) is not None:
                beliefs.ensure(pid, predicate, record[predicate], CATALOGUE_SOURCE)

    def _build_intention(self, desire: Desire, selected: Any, record: dict[str, Any],
                         beliefs: BeliefBase) -> Intention:
        self._record_attributes(beliefs, record)
        pid = selected.product_id

        availability = Condition(subject=pid, predicate="availability", operator="in",
                                 value=self.available_states, description="product is available")
        budget_conditions = [
            Condition(subject=pid, predicate="price", operator="lte", ref_subject=USER,
                      ref_predicate=name, description=f"price within {name}")
            for name in BUDGET_PREDICATES if beliefs.has(USER, name)
        ]
        # Room limits are optional: a fit condition exists only for a limit the user gave (an invalidated
        # limit still counts as given, so the intention is suspended until it is known again).
        fit_conditions = [
            Condition(subject=pid, predicate=f"{axis}_cm", operator="lte", ref_subject=USER,
                      ref_predicate=f"max_{axis}_cm", description=f"fits the room {axis}")
            for axis in ("width", "depth", "height")
            if beliefs.get(USER, f"max_{axis}_cm", include_invalid=True) is not None
        ]
        fit_conditions.append(
            Condition(subject=pid, predicate="seating_capacity", operator="gte", ref_subject=USER,
                      ref_predicate="min_seating", description="enough seats"))
        hard_constraints = [availability, *budget_conditions, *fit_conditions]

        def copy(items: Sequence[Condition]) -> list[Condition]:
            return [c.model_copy(deep=True) for c in items]

        actions = [
            Action(kind="check_constraints",
                   description=f"Re-check availability, budget and room fit of {pid}",
                   preconditions=copy(hard_constraints)),
            Action(kind="recommend_product",
                   description=f"Recommend {selected.name or pid} to the user",
                   params={"product_id": pid}, preconditions=copy([availability])),
            Action(kind="reserve_product",
                   description=f"Reserve {pid} once the user confirms",
                   params={"product_id": pid}, manual=True,
                   preconditions=copy([availability, *budget_conditions])),
        ]
        return Intention(
            desire_id=desire.desire_id,
            description=f"Obtain {selected.name or pid} ({pid}) from {selected.source_agent_id}",
            actions=actions,
            preconditions=copy(hard_constraints),
            params={
                "product_id": pid,
                "source_agent_id": selected.source_agent_id,
                "name": selected.name,
                "price": selected.price,
                "utility": selected.utility,
            },
            rationale=f"Highest utility U={selected.utility:.2f} among feasible candidates",
        )

    # Relevance -------------------------------------------------------------

    def is_affected_by(self, desire: Desire, change: BeliefChange, beliefs: BeliefBase,
                       held_product_ids: set[str]) -> tuple[bool, str]:
        subject, predicate = change.subject, change.predicate

        if subject == USER:
            if predicate in BUDGET_PREDICATES:
                other = "total_budget" if predicate == "sofa_budget" else "sofa_budget"
                other_value = beliefs.value(USER, other)
                before = effective_budget(change.old_value, other_value) if predicate == "sofa_budget" \
                    else effective_budget(other_value, change.old_value)
                after = effective_budget(change.new_value, other_value) if predicate == "sofa_budget" \
                    else effective_budget(other_value, change.new_value)
                if before == after:
                    return False, f"{predicate} changed but the effective sofa budget is still {after}"
                return True, f"the effective sofa budget changed from {before} to {after}"
            if predicate in REQUIREMENT_PREDICATES:
                return True, f"the user's {predicate} changed"
            return False, f"user.{predicate} does not affect the choice of a product"

        if predicate in ("availability", "price"):
            record = self._catalogue_index().get(subject)
            if record is None:
                return False, f"{subject} is not a candidate product"
            if subject in held_product_ids:
                return True, f"{subject} is the product currently chosen"
            # With no earlier belief, the catalogue is what the agent had been assuming.
            old = change.old_value if change.old_value is not None else record.get(predicate)
            new = change.new_value
            if predicate == "availability":
                if new in self.available_states and old not in self.available_states:
                    return True, f"{subject} became available and might beat the current choice"
                return False, (f"{subject} is not the chosen product and this availability change "
                               f"cannot improve the current choice")
            if _is_number(new) and (not _is_number(old) or new < old):
                return True, f"{subject} got cheaper and might beat the current choice"
            return False, f"{subject} is not the chosen product and a price rise cannot improve the current choice"

        return False, f"{subject}.{predicate} does not concern this goal"


# ── maintain_budget ─────────────────────────────────────────────────────────

class MaintainBudgetHandler:
    """Monitors that committed price and recorded expenditure stay within the total budget."""

    goal_type = "maintain_budget"

    def plan(self, desire: Desire, beliefs: BeliefBase, live_intentions: Sequence[Intention]) -> PlanOutcome:
        total = beliefs.value(USER, "total_budget")
        if not _is_number(total) or total <= 0:
            return PlanOutcome(desire.desire_id, "waiting", missing=["user.total_budget"],
                               explanation="no total budget is known")
        committed = sum(i.params["price"] for i in live_intentions if _is_number(i.params.get("price")))
        spent = beliefs.value(USER, "estimated_expenditure")
        problems = []
        if committed > total:
            problems.append(f"committed price {committed:,.0f} exceeds the total budget {total:,.0f}")
        if _is_number(spent) and spent > total:
            problems.append(f"recorded expenditure {spent:,.0f} exceeds the total budget {total:,.0f}")
        if problems:
            return PlanOutcome(desire.desire_id, "violated", explanation="; ".join(problems))
        spent_text = f", recorded expenditure {spent:,.0f}" if _is_number(spent) else ""
        return PlanOutcome(desire.desire_id, "satisfied",
                           explanation=f"committed {committed:,.0f}{spent_text} within total budget {total:,.0f}")

    def is_affected_by(self, desire: Desire, change: BeliefChange, beliefs: BeliefBase,
                       held_product_ids: set[str]) -> tuple[bool, str]:
        if change.subject == USER and change.predicate in ("total_budget", "estimated_expenditure"):
            return True, f"user.{change.predicate} is monitored by this goal"
        if change.predicate == "price" and change.subject in held_product_ids:
            return True, f"the price of the committed product {change.subject} changed"
        return False, "not monitored by this goal"


# ── Planner ─────────────────────────────────────────────────────────────────

class BDIPlanner:
    """Dispatches desires to the goal handler for their ``goal_type``."""

    def __init__(self, handlers: Sequence[GoalHandler]) -> None:
        self._handlers: dict[str, GoalHandler] = {h.goal_type: h for h in handlers}

    @classmethod
    def default(cls, optimizer: UtilityOptimizer, candidate_source: CandidateSource) -> "BDIPlanner":
        """The standard planner: product selection plus budget maintenance."""
        return cls([SelectProductHandler(optimizer, candidate_source), MaintainBudgetHandler()])

    def supports(self, goal_type: str) -> bool:
        return goal_type in self._handlers

    def handler_for(self, desire: Desire) -> GoalHandler:
        try:
            return self._handlers[desire.goal_type]
        except KeyError:
            raise ValueError(f"no goal handler for goal type '{desire.goal_type}'") from None

    def plan(self, desire: Desire, beliefs: BeliefBase, live_intentions: Sequence[Intention]) -> PlanOutcome:
        return self.handler_for(desire).plan(desire, beliefs, live_intentions)

    def is_affected_by(self, desire: Desire, change: BeliefChange, beliefs: BeliefBase,
                       held_product_ids: set[str]) -> tuple[bool, str]:
        return self.handler_for(desire).is_affected_by(desire, change, beliefs, held_product_ids)
