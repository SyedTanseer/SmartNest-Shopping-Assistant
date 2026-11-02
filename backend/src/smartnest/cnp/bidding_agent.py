"""CNP Bidding Agent (Product Research Agent).

Each bidding agent:
    1. Runs a ``RemoteAgentServer`` to receive CFP messages.
    2. Examines its own independent ``ProductCatalog``.
    3. Generates proposals based on actual catalogue entries and the
       constraints announced so far.
    4. Revises proposals in later rounds:
         - *Constraint feedback* (e.g. a newly revealed ``max_width``)
           changes which products are eligible, so the agent may switch
           product.
         - *Offer feedback* tells the agent whether its offer currently
           leads.  A trailing agent concedes on price by
           ``concession_step_pct`` per round, never beyond its own
           ``max_discount_pct`` floor.  A leading agent holds its price.
    5. Verifies awards against the offers it actually made (negotiation
       ID, task ID, product and price) before executing and reporting
       INFORM_DONE.

State is kept per negotiation, so one agent can take part in several
negotiations (from several coordinators) without mixing them up.  All
state access is guarded by a lock because the server runs each
connection in its own thread.

The agent's decisions depend entirely on its own catalogue data, its own
pricing strategy and the coordinator's messages — nothing is hardcoded.

Utility-based selection (Assignment 3.2):
    Give the agent a ``UtilityOptimizer`` and, whenever the CFP states
    every constraint the utility model needs (budget, width, depth,
    seating), the agent runs the optimizer over its own catalogue and
    proposes its highest-utility feasible product.  The proposal then
    carries ``self_reported_utility`` and ``utility_breakdown`` computed
    at the *offered* price.  The coordinator never trusts these values —
    it recalculates — so they are informative only.  When a constraint
    is still unannounced (e.g. width in round 1 of progressive
    disclosure), the agent falls back to its 3.1 heuristic.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

from smartnest.cnp.product_catalog import ProductCatalog
from smartnest.communication.message_router import MessageRouter
from smartnest.communication.remote_agent_server import RemoteAgentServer
from smartnest.core.models import MessageEnvelope, MessageType
from smartnest.optimization import (
    InvalidRequirementsError,
    UserRequirements,
    UtilityOptimizer,
)

logger = logging.getLogger(__name__)


@dataclass
class _NegotiationRecord:
    """Everything one bidding agent remembers about one negotiation."""

    negotiation_id: str
    task_id: str | None
    coordinator: str
    current_round: int = 0
    offers: list[dict[str, Any]] = field(default_factory=list)
    discounts: dict[str, float] = field(default_factory=dict)  # product_id → discount
    status: str = "open"  # open | awarded | rejected | cancelled | closed
    awarded: bool = False
    award_details: dict[str, Any] = field(default_factory=dict)
    final_result: dict[str, Any] = field(default_factory=dict)


class BiddingAgent:
    """A CNP product research agent with its own independent catalogue.

    Args:
        agent_id: Unique agent identifier.
        catalog: The agent's independent ``ProductCatalog``.
        host: Listen address (default "127.0.0.1").
        port: Listen port (default 0 = OS-assigned).
        max_discount_pct: Largest total discount this agent will ever
            give off list price (default 0.05 = 5%).
        concession_step_pct: Discount added each time the agent is told
            its offer is not leading (default 0.025 = 2.5%).
    """

    def __init__(
        self,
        agent_id: str,
        catalog: ProductCatalog,
        host: str = "127.0.0.1",
        port: int = 0,
        max_discount_pct: float = 0.05,
        concession_step_pct: float = 0.025,
        utility_optimizer: UtilityOptimizer | None = None,
    ) -> None:
        if not 0.0 <= max_discount_pct < 1.0:
            raise ValueError("max_discount_pct must be in [0, 1)")
        if concession_step_pct < 0.0:
            raise ValueError("concession_step_pct must be non-negative")

        self.agent_id = agent_id
        self._catalog = catalog
        self._host = host
        self._max_discount = max_discount_pct
        self._concession_step = concession_step_pct
        self._optimizer = utility_optimizer

        self._lock = threading.RLock()
        self._negotiations: dict[str, _NegotiationRecord] = {}
        self._latest_negotiation_id: str | None = None

        # Build router and server
        self._router = MessageRouter(agent_id=agent_id)
        self._router.register(MessageType.CFP, self._handle_cfp)
        self._router.register(MessageType.AWARD, self._handle_award)
        self._router.register(MessageType.REJECT_PROPOSAL, self._handle_reject_proposal)
        self._router.register(MessageType.INFORM_RESULT, self._handle_inform_result)
        self._router.register(MessageType.CANCEL, self._handle_cancel)

        self._server = RemoteAgentServer(
            router=self._router,
            host=host,
            port=port,
            agent_id=agent_id,
        )

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the agent's server (bound and listening when this returns)."""
        self._server.start()

    def stop(self, timeout: float = 3.0) -> None:
        """Stop the agent's server."""
        self._server.stop(timeout=timeout)

    @property
    def is_running(self) -> bool:
        return self._server.is_running

    @property
    def actual_port(self) -> int:
        """The port the server is listening on (OS-assigned if 0 was given)."""
        return self._server.port

    @property
    def host(self) -> str:
        return self._host

    # ── State ────────────────────────────────────────────────────────────

    def _latest(self) -> _NegotiationRecord | None:
        if self._latest_negotiation_id is None:
            return None
        return self._negotiations.get(self._latest_negotiation_id)

    @property
    def awarded(self) -> bool:
        """Whether this agent won its most recent negotiation."""
        with self._lock:
            record = self._latest()
            return bool(record and record.awarded)

    @property
    def award_details(self) -> dict[str, Any]:
        """Award details of the most recent negotiation (empty if not won)."""
        with self._lock:
            record = self._latest()
            return dict(record.award_details) if record else {}

    def negotiation_status(self, negotiation_id: str) -> str | None:
        """Status of a negotiation this agent took part in, or ``None``."""
        with self._lock:
            record = self._negotiations.get(negotiation_id)
            return record.status if record else None

    def offers_for(self, negotiation_id: str) -> list[dict[str, Any]]:
        """All offers this agent made in a negotiation (copies)."""
        with self._lock:
            record = self._negotiations.get(negotiation_id)
            return [dict(o) for o in record.offers] if record else []

    @property
    def catalog(self) -> ProductCatalog:
        return self._catalog

    @property
    def max_discount_pct(self) -> float:
        return self._max_discount

    # ── Message handlers ──────────────────────────────────────────────────

    def _handle_cfp(self, message: MessageEnvelope) -> MessageEnvelope:
        """Handle a Call for Proposals.

        Applies the announced requirements plus any constraints revealed
        in later rounds, picks a product from the agent's own catalogue,
        and prices it according to the agent's concession strategy.
        """
        payload = message.payload or {}
        requirements = payload.get("requirements", {}) or {}
        updated_constraints = payload.get("updated_constraints", {}) or {}
        offer_feedback = payload.get("offer_feedback") or {}
        round_num = int(payload.get("round", 1) or 1)
        negotiation_id = payload.get("negotiation_id") or f"anon_{message.task_id}"

        with self._lock:
            record = self._negotiations.get(negotiation_id)
            if record is None:
                record = _NegotiationRecord(
                    negotiation_id=negotiation_id,
                    task_id=message.task_id,
                    coordinator=message.sender,
                )
                self._negotiations[negotiation_id] = record
            record.current_round = round_num
            self._latest_negotiation_id = negotiation_id

        logger.info(
            "%s received CFP round %d (neg=%s)", self.agent_id, round_num, negotiation_id
        )

        effective_req = self._effective_requirements(requirements, updated_constraints)
        utility_req = self._utility_requirements(effective_req)

        if utility_req is not None:
            # 3.2: pick our own highest-utility feasible product.
            best_product = self._best_by_utility(utility_req)
            eligible = [best_product] if best_product else []
        else:
            eligible = self._catalog.find_eligible_products(
                max_budget=effective_req["max_budget"],
                room_width=effective_req["room_width"],
                preferred_style=effective_req["preferred_style"],
                seating_capacity=effective_req["seating_capacity"],
            )

        if not eligible:
            logger.info("%s has no eligible products in round %d", self.agent_id, round_num)
            return MessageEnvelope(
                sender=self.agent_id,
                receiver=message.sender,
                message_type=MessageType.FAILURE,
                task_id=message.task_id,
                payload={
                    "reason": "no_eligible_products",
                    "negotiation_id": negotiation_id,
                    "round": round_num,
                },
            )

        if utility_req is None:
            # The agent's own 3.1 selection heuristic — independent of the
            # coordinator's utility, which evaluates the proposal separately.
            best_product = self._select_best_product(eligible, effective_req)

        with self._lock:
            discount = self._next_discount(record, best_product["product_id"], offer_feedback)
            record.discounts[best_product["product_id"]] = discount
            proposal = self._build_proposal(
                best_product, round_num, negotiation_id, discount, effective_req
            )
            if utility_req is not None:
                self._attach_self_assessment(proposal, best_product, utility_req)
            record.offers.append(dict(proposal))

        logger.info(
            "%s proposing %s (₹%s, %s cm, discount %.1f%%) in round %d",
            self.agent_id,
            best_product["product_id"],
            proposal["proposed_price"],
            best_product["width_cm"],
            discount * 100,
            round_num,
        )

        return MessageEnvelope(
            sender=self.agent_id,
            receiver=message.sender,
            message_type=MessageType.PROPOSAL,
            task_id=message.task_id,
            payload=proposal,
        )

    def _handle_award(self, message: MessageEnvelope) -> MessageEnvelope:
        """Handle a contract award — verify it against our offers, then execute."""
        payload = message.payload or {}
        negotiation_id = payload.get("negotiation_id")
        product_id = payload.get("product_id")
        awarded_price = payload.get("proposed_price")

        logger.info(
            "%s received AWARD for %s at ₹%s (neg=%s)",
            self.agent_id,
            product_id,
            awarded_price,
            negotiation_id,
        )

        with self._lock:
            record = self._negotiations.get(negotiation_id) if negotiation_id else None
            problem = self._verify_award(record, message.task_id, product_id, awarded_price)
            if problem:
                logger.warning("%s refusing award: %s", self.agent_id, problem)
                return MessageEnvelope(
                    sender=self.agent_id,
                    receiver=message.sender,
                    message_type=MessageType.FAILURE,
                    task_id=message.task_id,
                    payload={
                        "reason": "award_does_not_match_proposal",
                        "detail": problem,
                        "negotiation_id": negotiation_id,
                    },
                )

            offer = self._matching_offer(record, product_id, awarded_price)
            final_price = offer["proposed_price"]
            record.awarded = True
            record.status = "awarded"
            record.award_details = {
                **payload,
                "task_id": message.task_id,
                "final_price": final_price,
            }

        # Execute the (simulated) task
        logger.info("%s executing task for product %s", self.agent_id, product_id)

        return MessageEnvelope(
            sender=self.agent_id,
            receiver=message.sender,
            message_type=MessageType.INFORM_DONE,
            task_id=message.task_id,
            payload={
                "negotiation_id": negotiation_id,
                "product_id": product_id,
                "final_price": final_price,
                "status": "executed",
                "agent_id": self.agent_id,
            },
        )

    def _handle_reject_proposal(self, message: MessageEnvelope) -> None:
        """The coordinator chose another bidder's proposal."""
        payload = message.payload or {}
        with self._lock:
            record = self._negotiations.get(payload.get("negotiation_id", ""))
            if record is not None and not record.awarded:
                record.status = "rejected"
        logger.info(
            "%s proposal rejected: %s", self.agent_id, payload.get("reason", "not selected")
        )
        return None

    def _handle_inform_result(self, message: MessageEnvelope) -> None:
        """Handle the final result notification."""
        payload = message.payload or {}
        with self._lock:
            record = self._negotiations.get(payload.get("negotiation_id", ""))
            if record is not None:
                record.final_result = dict(payload)
                if record.status == "open":
                    record.status = "closed"
        logger.info(
            "%s received INFORM_RESULT: winner=%s", self.agent_id, payload.get("winner")
        )
        return None  # No response needed

    def _handle_cancel(self, message: MessageEnvelope) -> None:
        """Handle negotiation cancellation."""
        payload = message.payload or {}
        with self._lock:
            record = self._negotiations.get(payload.get("negotiation_id", ""))
            if record is not None:
                record.status = "cancelled"
        logger.info("%s received CANCEL: reason=%s", self.agent_id, payload.get("reason"))
        return None

    # ── Decision helpers ───────────────────────────────────────────────────

    @staticmethod
    def _effective_requirements(
        requirements: dict[str, Any],
        updated_constraints: dict[str, Any],
    ) -> dict[str, Any]:
        """Merge announced requirements with constraints revealed later."""
        room_width = requirements.get("room_width")
        if "max_width" in updated_constraints:
            room_width = updated_constraints["max_width"]
        max_budget = updated_constraints.get("max_budget", requirements.get("max_budget", 0))
        return {
            "room_depth": updated_constraints.get("max_depth", requirements.get("room_depth")),
            "preferred_colour": requirements.get("preferred_colour"),
            "max_budget": max_budget or 0,
            "room_width": room_width,  # None → width not announced yet
            "preferred_style": updated_constraints.get(
                "preferred_style", requirements.get("preferred_style", "")
            ) or "",
            "seating_capacity": updated_constraints.get(
                "seating_capacity", requirements.get("seating_capacity")
            ),
        }

    @property
    def utility_optimizer(self) -> UtilityOptimizer | None:
        return self._optimizer

    def _catalog_records(self) -> list[dict[str, Any]]:
        """Catalogue entries attributed to this agent (we are the source)."""
        return [{**p, "source_agent_id": self.agent_id} for p in self._catalog.get_all_products()]

    def _utility_requirements(self, effective_req: dict[str, Any]) -> UserRequirements | None:
        """3.2 requirements, or ``None`` if no optimizer / constraints incomplete."""
        if self._optimizer is None:
            return None
        try:
            return UserRequirements.from_cnp(effective_req)
        except InvalidRequirementsError:
            return None  # e.g. width not announced yet → 3.1 heuristic

    def _best_by_utility(self, req: UserRequirements) -> dict[str, Any] | None:
        """Our highest-utility feasible catalogue product (``None`` if none)."""
        result = self._optimizer.optimize(self._catalog_records(), req)
        if result.selected is None:
            return None
        return self._catalog.get_product(result.selected.product_id)

    def evaluate_own_catalog(self, requirements: UserRequirements | dict[str, Any]):
        """Run the optimizer over this agent's own catalogue (no networking).

        Returns the full :class:`OptimizationResult`, so the agent can see
        its highest-utility feasible options and why others were rejected.
        """
        if self._optimizer is None:
            raise RuntimeError(f"{self.agent_id} has no utility optimizer")
        if not isinstance(requirements, UserRequirements):
            requirements = UserRequirements.from_cnp(requirements)
        return self._optimizer.optimize(self._catalog_records(), requirements)

    def _attach_self_assessment(
        self,
        proposal: dict[str, Any],
        product: dict[str, Any],
        req: UserRequirements,
    ) -> None:
        """Add our own utility estimate at the offered price (informative only)."""
        offered = {**product, "price": proposal["proposed_price"], "source_agent_id": self.agent_id}
        evaluation = self._optimizer.evaluate_candidate(offered, req, require_complete=True)  # CNP records must be complete
        if evaluation.breakdown is not None:
            proposal["self_reported_utility"] = evaluation.breakdown.normalised_utility
            proposal["utility_breakdown"] = evaluation.breakdown.model_dump()

    def _next_discount(
        self,
        record: _NegotiationRecord,
        product_id: str,
        offer_feedback: dict[str, Any],
    ) -> float:
        """Decide the discount for this round's offer on ``product_id``.

        Concede one step only when the coordinator asks for improved
        offers and this agent's offer is not the leading one.  Never go
        beyond ``max_discount_pct``.
        """
        current = record.discounts.get(product_id, 0.0)
        asked_to_improve = offer_feedback.get("request") == "improve_offer"
        leading = bool(offer_feedback.get("you_are_leading"))
        if asked_to_improve and not leading:
            return min(self._max_discount, current + self._concession_step)
        return current

    def _select_best_product(
        self,
        eligible: list[dict[str, Any]],
        requirements: dict[str, Any],
    ) -> dict[str, Any]:
        """Select the best product from eligible ones.

        When the room width is known: prefer products that use the width
        well (60%) while staying cheap (40%).  When it isn't known yet:
        pick the best-value (cheapest) product.  Ties → lower product_id
        for determinism.  This is the agent's own decision, independent
        of the coordinator's utility.
        """
        room_width = requirements.get("room_width")
        budget = requirements.get("max_budget") or 0

        def price_score(p: dict[str, Any]) -> float:
            return 1.0 - (p["price"] / budget) if budget > 0 else 0.0

        def score(p: dict[str, Any]) -> float:
            if room_width:
                return 0.6 * (p["width_cm"] / room_width) + 0.4 * price_score(p)
            return price_score(p)

        return max(sorted(eligible, key=lambda p: p["product_id"]), key=score)

    def _build_proposal(
        self,
        product: dict[str, Any],
        round_num: int,
        negotiation_id: str,
        discount: float,
        requirements: dict[str, Any],
    ) -> dict[str, Any]:
        """Build a proposal payload from a product and pricing decision."""
        list_price = product["price"]
        proposed_price = int(round(list_price * (1.0 - discount)))
        room_width = requirements.get("room_width")
        style = requirements.get("preferred_style", "")

        compatibility = {
            "within_budget": proposed_price <= requirements.get("max_budget", 0),
            # None = the coordinator hasn't announced a room width yet.
            "fits_room": (product["width_cm"] <= room_width) if room_width else None,
            "style_match": (product.get("style", "").lower() == style.lower()) if style else None,
        }
        known = [v for v in compatibility.values() if v is not None]
        feasibility_status = (
            "infeasible" if not all(known)
            else "feasible" if len(known) == len(compatibility)
            else "unverified"
        )

        explanation = (
            f"Proposing {product['name']} — ₹{proposed_price:,}"
            + (f" ({discount * 100:.1f}% off ₹{list_price:,})" if discount else "")
            + f", {product['width_cm']} cm, {product.get('style', 'unspecified')} style."
        )

        return {
            "agent_id": self.agent_id,
            "negotiation_id": negotiation_id,
            "product_id": product["product_id"],
            "product": dict(product),
            "list_price": list_price,
            "discount_pct": round(discount, 4),
            "proposed_price": proposed_price,
            "compatibility": compatibility,
            "feasibility_status": feasibility_status,
            "explanation": explanation,
            "negotiation_round": round_num,
        }

    @staticmethod
    def _matching_offer(
        record: _NegotiationRecord,
        product_id: Any,
        price: Any,
    ) -> dict[str, Any] | None:
        """Latest offer in ``record`` for ``product_id`` (and ``price``, if given)."""
        for offer in reversed(record.offers):
            if offer["product_id"] != product_id:
                continue
            if price is not None and offer["proposed_price"] != price:
                continue
            return offer
        return None

    def _verify_award(
        self,
        record: _NegotiationRecord | None,
        task_id: str | None,
        product_id: Any,
        price: Any,
    ) -> str:
        """Return why an award is invalid, or ``""`` if it can be accepted."""
        if record is None:
            return "unknown negotiation"
        if record.status in ("cancelled",):
            return "negotiation was cancelled"
        if record.task_id is not None and task_id != record.task_id:
            return f"task_id {task_id} does not match {record.task_id}"
        if self._matching_offer(record, product_id, price) is None:
            return f"no offer for {product_id}" + (f" at ₹{price}" if price is not None else "")
        return ""
