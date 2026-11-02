"""AdaptiveSession: live information flowing through knowledge, validation and BDI.

This is the glue layer.  It is the only place that knows about all the
parts, and it keeps their responsibilities apart:

    Gemini            turns the user's words into *proposed* requirements (and back into prose)
    validation        ``normalise_requirement`` + the optimizer's validators check every value
    product discovery fetches real listings; the normalizer rejects incomplete ones
    knowledge store   records what was learned, where from, and when
    BDI agent         holds beliefs, decides, commits, and reconsiders when things change
    UtilityOptimizer  (inside the BDI planner) is the only thing that ranks products

Gemini never overrides a hard constraint and never produces a score: what
it extracts is validated, applied as beliefs, and then judged by the same
optimizer as everything else.

Safeguard against silent loosening: a model-extracted value may **never relax a
limit that is already on file** (a bigger budget, width or depth, or fewer seats).
Such a change is held as *pending* and reported; it is applied only when the user
explicitly confirms it (:meth:`AdaptiveSession.confirm_pending`).  Tightening a
limit, stating a limit for the first time, and changing soft preferences
(style, colour) apply immediately.

Flow of a request (``request``):
    extract requirements -> remember them -> apply as beliefs -> discover products
    -> remember listings -> BDI deliberates (UtilityOptimizer) -> recommend + explain

Flow of new information (``refresh_products`` / ``ingest_event`` / ``chat``):
    detect what changed -> events -> BDI updates beliefs and reconsiders only what is
    affected -> new recommendation if the intention changed.

Failures in a stage (Gemini down, discovery failing, knowledge store
unavailable) are reported in the turn's ``notes`` and never crash the
session; the agent keeps its last good state.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from smartnest.bdi import BDIAgent, normalise_requirement, select_sofa_desire, stay_within_budget_desire
from smartnest.bdi.planner import REQUIREMENT_PREDICATES, effective_budget
from smartnest.core.models import ActionLog, EnvironmentalEvent
from smartnest.integrations.gemini import (
    GeminiError,
    GeminiService,
    PlanChangeRequest,
    RecommendationExplanation,
    RecommendationFacts,
    RequirementsExtraction,
)
from smartnest.integrations.products import (
    DiscoveryResult,
    LiveCandidateSource,
    ProductDiscoveryError,
    ProductQuery,
    detect_changes,
)
from smartnest.knowledge import (
    KnowledgeKind,
    KnowledgeStore,
    KnowledgeStoreError,
    Provenance,
    Relationship,
    make_record,
)
from smartnest.knowledge.models import KnowledgeError

logger = logging.getLogger(__name__)

SELECT = "des_select_sofa"
#: Requirement fields that change what should be searched for.
QUERY_FIELDS = ("sofa_budget", "total_budget", "min_seating", "preferred_style", "preferred_colour",
                "max_width_cm", "max_depth_cm")


@dataclass
class Recommendation:
    """The product the agent currently recommends, with everything needed to show it."""

    product_id: str
    name: str
    price: float
    utility: float
    retailer: str | None
    url: str | None
    image_url: str | None
    verified_at: str | None
    intention_id: str
    explanation: RecommendationExplanation
    facts: RecommendationFacts
    listing_url: str | None = None      # the best usable link: the retailer page, else the Google Shopping page
    url_type: str | None = None         # "retailer" | "google_shopping" (``url`` is only ever a retailer link)
    availability: str | None = None     # "in_stock" for a confirmed recommendation
    completeness: float | None = None   # information completeness of the score, 0-1


@dataclass
class SessionTurn:
    """What one interaction did."""

    kind: str                                   # request | change | refresh | event
    user_message: str | None = None
    extraction: RequirementsExtraction | None = None
    change_request: PlanChangeRequest | None = None
    applied: dict[str, Any] = field(default_factory=dict)
    rejected: dict[str, str] = field(default_factory=dict)
    discovery: DiscoveryResult | None = None
    action_logs: list[ActionLog] = field(default_factory=list)
    recommendation: Recommendation | None = None
    changed_recommendation: bool = False
    needs: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    pending: dict[str, tuple[Any, Any]] = field(default_factory=dict)   # field -> (current, requested)
    #: Candidates that are NOT fully verified (stock, fit or seating unconfirmed): never a confirmed recommendation.
    unverified_options: list[dict[str, Any]] = field(default_factory=list)


#: Limits where a *larger* value is looser (budgets, sizes) ...
_LOOSER_WHEN_HIGHER = ("sofa_budget", "total_budget", "max_width_cm", "max_depth_cm")
#: ... and where a *smaller* value is looser.
_LOOSER_WHEN_LOWER = ("min_seating",)


def loosens(name: str, old: Any, new: Any) -> bool:
    """Whether changing the hard limit ``name`` from ``old`` to ``new`` makes it easier to satisfy."""
    if old is None or new is None:
        return False
    if name in _LOOSER_WHEN_HIGHER:
        return new > old
    if name in _LOOSER_WHEN_LOWER:
        return new < old
    return False


class AdaptiveSession:
    """Coordinates Gemini, live product discovery, knowledge and the BDI agent.

    Args:
        agent: The BDI agent (its planner's candidate source should be ``source``).
        gemini: The Gemini service (extraction and explanation only).
        source: The live candidate source the agent's planner reads from.
        knowledge: Optional knowledge store; if omitted nothing is persisted.
        user_id / project_id: Entities that knowledge records refer to.
        explain: Ask Gemini to explain recommendations (a template is used otherwise).
    """

    def __init__(self, *, agent: BDIAgent, gemini: GeminiService, source: LiveCandidateSource,
                 knowledge: KnowledgeStore | None = None, user_id: str = "user", project_id: str = "project",
                 explain: bool = True) -> None:
        self.agent = agent
        self.gemini = gemini
        self.source = source
        self.knowledge = knowledge
        self.user_id = user_id
        self.project_id = project_id
        self._explain = explain
        self.current: Recommendation | None = None
        self.pending_changes: dict[str, Any] = {}
        self._requirements_record_id: str | None = None
        self._recommendation_entity = f"agent:{agent.agent_id}/recommendation"

        existing = {d.goal_type for d in agent.desires.all()}
        if "select_product" not in existing:
            agent.add_desire(select_sofa_desire())
        if "maintain_budget" not in existing:
            agent.add_desire(stay_within_budget_desire())

    # ══ Public interactions ═══════════════════════════════════════════════

    def known_requirements(self) -> dict[str, Any]:
        """Requirements already recorded for this user (e.g. room limits from the home project)."""
        entity = f"user:{self.user_id}"
        record = self._safe(lambda: self.knowledge.latest(KnowledgeKind.REQUIREMENTS, entity)
                            if self.knowledge else None, None, [])
        if record is None:
            return {}
        self._requirements_record_id = record.knowledge_id
        return record.typed().belief_values()

    def request(self, message: str) -> SessionTurn:
        """A new request in the user's words: extract, validate, discover, decide, explain."""
        turn = SessionTurn(kind="request", user_message=message)
        known = self.known_requirements()

        # 1. Gemini proposes requirements (validated by its schema).
        try:
            extraction = self.gemini.extract_requirements(message)
        except GeminiError as exc:
            turn.notes.append(f"Could not understand the request: {exc}")
            return self._finish(turn)
        turn.extraction = extraction
        stated = extraction.stated_fields()

        # 2. Validate again with the deterministic rules; remember; apply as beliefs.
        valid, turn.rejected = self._validate_values(stated)
        existing = {**known, **self._current_requirements()}
        valid = self._hold_loosening(existing, valid, turn)
        self._remember_requirements(valid, extraction.confidence, message, turn)

        # 3. Discover products for the merged requirements, then let the agent decide once.
        merged = {**known, **self._current_requirements(), **valid}
        self._discover(self._query_for(merged), turn)
        turn.applied, _ = self._apply(merged, source="user", turn=turn)
        if not turn.action_logs:  # nothing reacted (e.g. every belief already held): look at the new data anyway
            log = self.agent.deliberate("request received")
            if log is not None:
                turn.action_logs.append(log)
        return self._finish(turn)

    def chat(self, message: str) -> SessionTurn:
        """A follow-up in the user's words: change preferences, accept, or ask something."""
        turn = SessionTurn(kind="change", user_message=message)
        current = self._current_requirements()
        try:
            change = self.gemini.interpret_change_request(message, current)
        except GeminiError as exc:
            turn.notes.append(f"Could not understand the message: {exc}")
            return self._finish(turn)
        turn.change_request = change

        if change.clarification_needed:
            turn.notes.append(f"Needs clarification: {change.clarification_needed}")
            return self._finish(turn)
        if change.intent == "confirm_choice":
            if self.agent.confirm(SELECT):
                turn.notes.append("Choice confirmed and reserved.")
            else:
                turn.notes.append("There is nothing to confirm yet.")
            return self._finish(turn)
        if change.intent != "modify_requirements" or not change.changes:
            turn.notes.append(f"No change to the plan (intent: {change.intent}).")
            return self._finish(turn)

        # Split the model's proposal into validated sets and clears.
        sets: dict[str, Any] = {}
        clears: list[str] = []
        for item in change.changes:
            if item.action == "clear":
                clears.append(item.field)
                continue
            if item.value is None:
                turn.rejected[item.field] = "no value was given"
                continue
            value, error = normalise_requirement(item.field, item.value)
            if error:
                turn.rejected[item.field] = error
            else:
                sets[item.field] = value

        sets = self._hold_loosening(current, sets, turn)
        proposed = {k: v for k, v in current.items() if k not in clears}
        proposed.update(sets)
        # New information needs a fresh look at the market *before* the agent decides.
        if any(self._differs(current.get(f), proposed.get(f)) for f in QUERY_FIELDS):
            self._discover(self._query_for(proposed), turn)

        for field_name in clears:
            self.agent.invalidate_belief("user", field_name, "cleared by the user")
        turn.applied, _ = self._apply(sets, source="user", turn=turn)
        self._remember_requirements({**sets}, change.confidence, message, turn, removed=clears)
        return self._finish(turn)

    def _hold_loosening(self, existing: dict[str, Any], values: dict[str, Any], turn: SessionTurn) -> dict[str, Any]:
        """Split off changes that would relax a limit already on file; they wait for confirmation."""
        allowed: dict[str, Any] = {}
        for name, value in values.items():
            if loosens(name, existing.get(name), value):
                self.pending_changes[name] = value
                turn.pending[name] = (existing[name], value)
                turn.notes.append(
                    f"{name} would change from {existing[name]:g} to {value:g}, which relaxes a limit you already "
                    f"set; say so explicitly (confirm_pending) to apply it.")
            else:
                allowed[name] = value
        return allowed

    def confirm_pending(self) -> SessionTurn:
        """Apply the held changes after the user has explicitly confirmed them."""
        turn = SessionTurn(kind="change")
        if not self.pending_changes:
            turn.notes.append("There is nothing waiting for confirmation.")
            return self._finish(turn)
        values, self.pending_changes = dict(self.pending_changes), {}
        proposed = {**self._current_requirements(), **values}
        self._discover(self._query_for(proposed), turn)
        turn.applied, _ = self._apply(values, source="user", turn=turn)
        self._remember_requirements(values, 1.0, "confirmed by the user", turn)
        return self._finish(turn)

    def refresh_products(self) -> SessionTurn:
        """Look at the market again; feed any differences to the agent as events."""
        turn = SessionTurn(kind="refresh")
        before = self.source.current_listings
        known_ids = {r["product_id"] for r in before}
        self._discover(self.source.query, turn)
        after = self.source.current_listings
        for event in detect_changes(before, after):
            self._observe(event, turn)
            log = self.agent.handle_event(event)
            if log is not None:
                turn.action_logs.append(log)
        if {r["product_id"] for r in after} - known_ids:
            log = self.agent.deliberate("new products were discovered")
            if log is not None:
                turn.action_logs.append(log)
        return self._finish(turn)

    def ingest_event(self, event: EnvironmentalEvent) -> SessionTurn:
        """New information from elsewhere (a price monitor, a retailer feed, ...)."""
        turn = SessionTurn(kind="event")
        self._observe(event, turn)
        log = self.agent.handle_event(event)
        if log is not None:
            turn.action_logs.append(log)
        return self._finish(turn)

    # ══ Requirements ═════════════════════════════════════════════════════

    def _current_requirements(self) -> dict[str, Any]:
        return {p: v for p in REQUIREMENT_PREDICATES if (v := self.agent.beliefs.value("user", p)) is not None}

    @staticmethod
    def _differs(a: Any, b: Any) -> bool:
        return a != b

    @staticmethod
    def _validate_values(values: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
        valid: dict[str, Any] = {}
        rejected: dict[str, str] = {}
        for name, raw in values.items():
            value, error = normalise_requirement(name, raw)
            if error:
                rejected[name] = error
            else:
                valid[name] = value
        return valid, rejected

    def _apply(self, values: dict[str, Any], source: str, turn: SessionTurn | None = None
               ) -> tuple[dict[str, Any], dict[str, str]]:
        valid, rejected = self._validate_values(values)
        if turn is not None:
            turn.rejected.update(rejected)
        if valid:
            log = self.agent.set_requirements(valid, source=source)
            if log is not None and turn is not None:
                turn.action_logs.append(log)
        return valid, rejected

    def _query_for(self, requirements: dict[str, Any]) -> ProductQuery:
        budget = effective_budget(requirements.get("sofa_budget"), requirements.get("total_budget"))
        return ProductQuery(
            category="sofa",
            max_price=budget,
            min_seating=requirements.get("min_seating"),
            style=requirements.get("preferred_style"),
            colour=requirements.get("preferred_colour"),
            max_width_cm=requirements.get("max_width_cm"),
            max_depth_cm=requirements.get("max_depth_cm"),
        )

    # ══ Discovery ════════════════════════════════════════════════════════

    def _discover(self, query: ProductQuery | None, turn: SessionTurn) -> None:
        if query is None:
            turn.notes.append("No product query is available yet.")
            return
        try:
            result = self.source.refresh(query)
        except ProductDiscoveryError as exc:
            turn.notes.append(f"Product discovery failed; keeping the previous listings: {exc}")
            return
        turn.discovery = result
        self._remember_products(result, turn)

    # ══ Recommendation ═══════════════════════════════════════════════════

    def _finish(self, turn: SessionTurn) -> SessionTurn:
        previous = self.current
        self.current = self._build_recommendation(previous)
        turn.recommendation = self.current
        turn.unverified_options = self._unverified_options()
        turn.changed_recommendation = (
            (previous.product_id, previous.price) != (self.current.product_id, self.current.price)
            if previous and self.current else previous is not self.current)
        if self.current is None:
            outcome = self.agent.last_outcomes.get(SELECT)
            if outcome is not None and outcome.missing:
                turn.needs = list(outcome.missing)
            elif outcome is not None and outcome.status == "blocked":
                turn.notes.append(f"No feasible product: {outcome.explanation}")
        elif turn.changed_recommendation or previous is None:
            self._remember_recommendation(self.current, turn)
        return turn

    def _build_recommendation(self, previous: Recommendation | None) -> Recommendation | None:
        intention = self.agent.current_intention(SELECT)
        if intention is None or intention.status.value not in ("active", "completed"):
            return None
        outcome = self.agent.last_outcomes.get(SELECT)
        params = intention.params
        listing = next((r for r in self.source.current_listings if r["product_id"] == params["product_id"]), {})

        facts = RecommendationFacts(
            product_id=params["product_id"], product_name=params.get("name") or params["product_id"],
            price=float(params["price"]), utility=float(params["utility"]),
            retailer=listing.get("retailer"), url=listing.get("url"),
            url_type=listing.get("listing_url_type") or listing.get("url_type"),
            budget=effective_budget(self.agent.beliefs.value("user", "sofa_budget"),
                                    self.agent.beliefs.value("user", "total_budget")),
            requirements={k: v for k, v in self._current_requirements().items() if k != "total_budget"},
        )
        if outcome is not None and outcome.optimization is not None:
            result = outcome.optimization
            facts.feasible_count, facts.inspected_count = result.eligible_count, result.total_inspected
            if result.selected is not None:
                facts.component_scores = dict(result.selected.breakdown.components)
                facts.fit_status = result.selected.fit_label
                facts.availability_status = result.selected.availability_label
                breakdown = result.selected.breakdown
                facts.completeness_percent = round(breakdown.completeness * 100, 1)
                facts.unavailable_attributes = list(breakdown.unavailable)
            if len(result.ranking) > 1:
                facts.runner_up_name = result.ranking[1].name
                facts.runner_up_utility = result.ranking[1].utility
            facts.rejected_examples = [f"{e.name or e.product_id}: {e.reasons[0]}" for e in result.rejected[:3] if e.reasons]

        if previous is not None and (previous.product_id, previous.price, previous.utility) == \
                (facts.product_id, facts.price, facts.utility):
            explanation = previous.explanation  # nothing changed: do not ask the model again
        elif self._explain:
            explanation = self.gemini.explain_recommendation(facts)
        else:
            explanation = self.gemini.template_explanation(facts)
        return Recommendation(
            product_id=facts.product_id, name=facts.product_name, price=facts.price, utility=facts.utility,
            retailer=facts.retailer, url=facts.url, image_url=listing.get("image_url"),
            verified_at=listing.get("verified_at"), intention_id=intention.intention_id,
            explanation=explanation, facts=facts,
            listing_url=listing.get("listing_url") or listing.get("url"),
            url_type=listing.get("listing_url_type") or ("retailer" if listing.get("url") else None),
            availability=outcome.optimization.selected.availability if outcome and outcome.optimization
            and outcome.optimization.selected else None,
            completeness=facts.completeness_percent / 100 if facts.completeness_percent is not None else None)

    def _unverified_options(self, limit: int = 5) -> list[dict[str, Any]]:
        """The best candidates that could not be fully verified, with exactly what is unverified.

        Never presented as confirmed: each says which of fit / seating / availability is unconfirmed.
        """
        outcome = self.agent.last_outcomes.get(SELECT)
        result = outcome.optimization if outcome is not None else None
        if result is None or not result.unverified:
            return []
        listings = {(r.get("source_agent_id"), r["product_id"]): r for r in self.source.current_listings}
        options = []
        for e in result.unverified[:limit]:
            listing = listings.get((e.source_agent_id, e.product_id), {})
            options.append({
                "product_id": e.product_id, "name": e.name, "price": e.price, "utility": e.utility,
                "completeness": e.completeness, "unavailable": e.unavailable, "gaps": list(e.gaps),
                "availability": e.availability, "availability_label": e.availability_label,
                "fit_label": e.fit_label, "why": list(e.reasons),
                "merchant": listing.get("merchant") or listing.get("retailer"),
                "listing_url": listing.get("listing_url") or listing.get("url"),
                "url_type": listing.get("listing_url_type") or listing.get("url_type"),
            })
        return options

    # ══ Knowledge ════════════════════════════════════════════════════════

    def _safe(self, action, default, notes: list[str]):
        """Run a knowledge operation; on failure record a note and carry on."""
        try:
            return action()
        except (KnowledgeStoreError, KnowledgeError) as exc:
            notes.append(f"Knowledge was not saved: {exc}")
            logger.warning("Knowledge operation failed: %s", exc)
            return default

    def _remember_requirements(self, values: dict[str, Any], confidence: float, message: str, turn: SessionTurn,
                               removed: list[str] | None = None) -> None:
        if self.knowledge is None or (not values and not removed):
            return
        provenance = Provenance(source="gemini", source_type="llm", method="requirement_extraction",
                                notes=f"from the user's message: {message[:200]}",
                                inferred_fields=[])
        entity = f"user:{self.user_id}"

        def save() -> None:
            record = self.knowledge.latest(KnowledgeKind.REQUIREMENTS, entity)
            changes = dict(values)
            for name in removed or []:
                changes[name] = None
            if record is None:
                created = self.knowledge.create(KnowledgeKind.REQUIREMENTS, entity, changes, provenance, confidence,
                                                [Relationship(relation="part_of", target_id=self.project_id,
                                                              target_kind="home_project")])
                self._requirements_record_id = created.knowledge_id
            else:
                self._requirements_record_id = record.knowledge_id
                self.knowledge.update(record.knowledge_id, changes, provenance, confidence)

        self._safe(save, None, turn.notes)

    def _remember_products(self, result: DiscoveryResult, turn: SessionTurn) -> None:
        if self.knowledge is None:
            return

        def save() -> None:
            for listing in result.listings:
                knowledge_id = f"prd_{listing['product_id']}"
                attributes = {
                    "product_id": listing["product_id"], "name": listing["name"], "category": listing.get("category"),
                    "price": listing["price"], "currency": "INR", "width_cm": listing["width_cm"],
                    "depth_cm": listing["depth_cm"], "height_cm": listing.get("height_cm"),
                    "seating_capacity": listing["seating_capacity"], "material": listing["material"],
                    "colour": listing["colour"], "style": listing["style"], "availability": listing["availability"],
                    "rating": listing["rating"], "review_count": listing["review_count"],
                    "retailer": listing.get("retailer"), "url": listing.get("url"), "image_url": listing.get("image_url"),
                    "url_type": listing.get("url_type"), "product_page_url": listing.get("product_page_url"),
                    "verified_at": listing["verified_at"],
                }
                attributes = {k: v for k, v in attributes.items() if v is not None}
                provenance = Provenance(source=result.provider_id, source_type="web", url=listing.get("url") or listing.get("product_page_url"),
                                        retrieved_at=result.retrieved_at,
                                        method=listing.get("discovery_method", "google_search_grounding"),
                                        inferred_fields=list(listing.get("inferred_fields", [])))
                existing = self.knowledge.get(knowledge_id)
                if existing is None:
                    rel = ([Relationship(relation="derived_from", target_id=self._requirements_record_id,
                                         target_kind="requirements")] if self._requirements_record_id else [])
                    self.knowledge.save(make_record(KnowledgeKind.PRODUCT_SPEC, f"product:{listing['product_id']}",
                                                    attributes, provenance, 0.8, rel, knowledge_id=knowledge_id))
                elif existing.valid:
                    self.knowledge.update(knowledge_id, attributes, provenance)

        self._safe(save, None, turn.notes)

    def _observe(self, event: EnvironmentalEvent, turn: SessionTurn) -> None:
        if self.knowledge is None:
            return
        payload = event.payload or {}
        subject = str(payload.get("product_id") or payload.get("product_name") or event.source)

        def save() -> None:
            self.knowledge.create(
                KnowledgeKind.OBSERVATION, f"product:{subject}",
                {"observation_type": str(event.event_type), "subject": subject, "details": dict(payload),
                 "observed_at": event.timestamp},
                Provenance(source=event.source, source_type="event", retrieved_at=event.timestamp,
                           method="environmental_event"))

        self._safe(save, None, turn.notes)

    def _remember_recommendation(self, rec: Recommendation, turn: SessionTurn) -> None:
        if self.knowledge is None:
            return

        def save() -> None:
            data = {"product_id": rec.product_id, "name": rec.name, "price": rec.price, "utility": rec.utility,
                    "url": rec.url, "retailer": rec.retailer, "intention_id": rec.intention_id,
                    "explanation": rec.explanation.text}
            attributes = {"agent_id": self.agent.agent_id, "topic": "recommendation",
                          "statement": f"Recommend {rec.name} at {rec.price:,.0f} INR", "data": data}
            provenance = Provenance(source=f"bdi:{self.agent.agent_id}", source_type="agent",
                                    method="utility_optimization")
            rels = [Relationship(relation="about", target_id=f"prd_{rec.product_id}", target_kind="product_spec")]
            record = self.knowledge.latest(KnowledgeKind.AGENT_KNOWLEDGE, self._recommendation_entity)
            if record is None:
                self.knowledge.create(KnowledgeKind.AGENT_KNOWLEDGE, self._recommendation_entity, attributes,
                                      provenance, 1.0, rels)
            else:
                self.knowledge.update(record.knowledge_id, attributes, provenance)
                stored = self.knowledge.get(record.knowledge_id)
                stored.relationships = rels
                self.knowledge.save(stored)

        self._safe(save, None, turn.notes)
