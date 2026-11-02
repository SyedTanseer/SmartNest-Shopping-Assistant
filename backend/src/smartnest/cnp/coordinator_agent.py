"""CNP Coordinator Agent.

The coordinator owns the whole negotiation lifecycle and policy:

    1. Announces requirements via CFP to all bidders.
    2. Collects proposals (CFPs go to every bidder first, then replies
       are gathered, so bidders work in parallel).
    3. Evaluates proposals with a pluggable ``ProposalEvaluator``.
    4. Decides whether to run another round, and what to say in it:
         - **Constraint feedback** — if proposals were rejected for a
           hard constraint the bidders weren't told about (e.g. width),
           the next CFP reveals it as ``updated_constraints``.
         - **Offer feedback** — once a feasible offer exists, each bidder
           is told whether its offer leads and asked to improve, so
           trailing bidders can concede on price.
    5. Stops when the best offer stops improving (after at least
       ``min_rounds``), or at ``max_rounds``.
    6. Awards the best offer from any round.  If the winner refuses the
       award, the next-best offer is tried.
    7. Receives INFORM_DONE, sends REJECT_PROPOSAL to the other bidders
       and INFORM_RESULT to everyone, then terminates.

Constraint disclosure:
    With ``reveal_constraints_progressively=True`` (default) the first CFP
    announces what the buyer wants (item, budget, style, seating) but not
    the room width, which is the coordinator's private site knowledge.  It
    is revealed only when bidders propose products that violate it.  With
    ``False``, every constraint is announced up front and rounds after
    the first are pure price negotiation.

The coordinator is a client-side component: it connects to each bidder's
``RemoteAgentServer`` using ``SocketTransport``.  State is tracked with a
``NegotiationStateMachine`` and every step is recorded in ``message_log``.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from smartnest.cnp.negotiation_state import (
    NegotiationState,
    NegotiationStateMachine,
)
from smartnest.cnp.proposal_evaluator import ProposalEvaluation, ProposalEvaluator
from smartnest.communication.message_codec import CodecError
from smartnest.communication.socket_transport import (
    SocketTransport,
    TransportError,
)
from smartnest.core.models import MessageEnvelope, MessageType

logger = logging.getLogger(__name__)

# Minimum utility gain that counts as "the offer improved".
_IMPROVEMENT_EPSILON = 1e-6


@dataclass
class NegotiationResult:
    """Outcome of :meth:`CoordinatorAgent.run_negotiation`."""

    success: bool
    negotiation_id: str
    task_id: str
    rounds: int
    winner: ProposalEvaluation | None = None
    reason: str = ""


class CoordinatorAgent:
    """Coordinates a CNP negotiation for a furniture requirement.

    Args:
        agent_id: Unique identifier (default "coordinator").
        evaluator: Optional custom ``ProposalEvaluator`` (Assignment 3.2 hook).
        max_rounds: Maximum CFP rounds (default 3).
        min_rounds: Rounds to run before an award is allowed (default 2) —
            the coordinator never awards the very first acceptable offer.
        reveal_constraints_progressively: See module docstring.
        response_timeout: Seconds to wait for each bidder's reply.
    """

    def __init__(
        self,
        agent_id: str = "coordinator",
        evaluator: ProposalEvaluator | None = None,
        max_rounds: int = 3,
        min_rounds: int = 2,
        reveal_constraints_progressively: bool = True,
        response_timeout: float = 10.0,
    ) -> None:
        if max_rounds < 1:
            raise ValueError("max_rounds must be at least 1")
        if not 1 <= min_rounds <= max_rounds:
            raise ValueError("min_rounds must be between 1 and max_rounds")

        self.agent_id = agent_id
        self._evaluator = evaluator or ProposalEvaluator()
        self._max_rounds = max_rounds
        self._min_rounds = min_rounds
        self._progressive = reveal_constraints_progressively
        self._response_timeout = response_timeout

        # Active negotiation state
        self._negotiation_id: str | None = None
        self._task_id: str | None = None
        self._state_machine: NegotiationStateMachine | None = None
        self._requirements: dict[str, Any] = {}
        self._bidders: dict[str, str] = {}  # agent_id → address "host:port"

        # Per-negotiation data
        self._round_proposals: dict[int, list[dict[str, Any]]] = {}
        self._round_evaluations: dict[int, list[ProposalEvaluation]] = {}
        self._current_round: int = 0
        self._revealed_constraints: dict[str, Any] = {}
        self._failed_bidders: set[str] = set()
        self._best: ProposalEvaluation | None = None
        self._result: NegotiationResult | None = None

        # Transports to bidding agents (agent_id → transport)
        self._transports: dict[str, SocketTransport] = {}

        # Full message log for timeline
        self._message_log: list[dict[str, Any]] = []
        self._log_seq = 0

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def state(self) -> NegotiationState | None:
        if self._state_machine:
            return self._state_machine.current_state
        return None

    @property
    def message_log(self) -> list[dict[str, Any]]:
        return list(self._message_log)

    @property
    def current_round(self) -> int:
        return self._current_round

    @property
    def max_rounds(self) -> int:
        return self._max_rounds

    @property
    def min_rounds(self) -> int:
        return self._min_rounds

    @property
    def negotiation_id(self) -> str | None:
        return self._negotiation_id

    @property
    def task_id(self) -> str | None:
        return self._task_id

    @property
    def state_history(self) -> list[dict[str, Any]]:
        if self._state_machine:
            return self._state_machine.history
        return []

    @property
    def round_evaluations(self) -> dict[int, list[ProposalEvaluation]]:
        """Evaluations per round (copy)."""
        return {r: list(evs) for r, evs in self._round_evaluations.items()}

    @property
    def round_proposals(self) -> dict[int, list[dict[str, Any]]]:
        """Raw proposals per round (copy)."""
        return {r: [dict(p) for p in ps] for r, ps in self._round_proposals.items()}

    @property
    def revealed_constraints(self) -> dict[str, Any]:
        return dict(self._revealed_constraints)

    @property
    def failed_bidders(self) -> set[str]:
        """Bidders dropped from the negotiation after a transport error."""
        return set(self._failed_bidders)

    @property
    def best_offer(self) -> ProposalEvaluation | None:
        """Best feasible offer seen so far in this negotiation."""
        return self._best

    @property
    def result(self) -> NegotiationResult | None:
        return self._result

    @property
    def evaluator(self) -> ProposalEvaluator:
        """The coordinator's own evaluator (``mode`` is "legacy" or "utility")."""
        return self._evaluator

    @property
    def requirements(self) -> dict[str, Any]:
        return dict(self._requirements)

    def _active_bidders(self) -> list[str]:
        return [
            a for a in self._bidders
            if a in self._transports and a not in self._failed_bidders
        ]

    # ── Setup ────────────────────────────────────────────────────────────

    def add_bidder(self, agent_id: str, host: str, port: int) -> None:
        """Register a bidding agent by its network address."""
        self._bidders[agent_id] = f"{host}:{port}"
        logger.info("Added bidder %s at %s:%d", agent_id, host, port)

    def set_requirements(
        self,
        task_description: str,
        max_budget: float,
        room_width: float,
        preferred_style: str = "",
        seating_capacity: int = 1,
        room_depth: float | None = None,
        preferred_colour: str | None = None,
    ) -> None:
        """Set the furniture requirements for this negotiation.

        ``room_depth`` and ``preferred_colour`` are optional (Assignment 3.2).
        A utility-mode evaluator needs ``room_depth`` to score spatial fit.
        """
        self._requirements = {
            "task_description": task_description,
            "max_budget": max_budget,
            "room_width": room_width,
            "preferred_style": preferred_style.lower(),
            "seating_capacity": seating_capacity,
        }
        if room_depth is not None:
            self._requirements["room_depth"] = room_depth
        if preferred_colour:
            self._requirements["preferred_colour"] = preferred_colour.lower()

    # ── Negotiation lifecycle ────────────────────────────────────────────

    def start_negotiation(self) -> str:
        """Initialize a new negotiation in the CREATED state.

        Returns the negotiation ID.
        """
        if not self._requirements:
            raise RuntimeError("set_requirements() must be called before start_negotiation()")
        self._negotiation_id = f"neg_{uuid.uuid4().hex[:8]}"
        self._task_id = f"task_{uuid.uuid4().hex[:8]}"
        self._state_machine = NegotiationStateMachine(self._negotiation_id)
        self._current_round = 0
        self._round_proposals.clear()
        self._round_evaluations.clear()
        self._revealed_constraints.clear()
        self._failed_bidders.clear()
        self._best = None
        self._result = None
        self._message_log.clear()
        self._log_seq = 0
        self._log("Negotiation created", "system", self._negotiation_id)
        return self._negotiation_id

    def connect_to_bidders(self) -> None:
        """Open a TCP connection to every registered bidder not yet connected.

        A bidder that can't be reached is excluded from the negotiation.

        Raises:
            TransportError: If no bidder at all could be reached.
        """
        for agent_id, address in self._bidders.items():
            if agent_id in self._transports:
                continue
            host, port_str = address.rsplit(":", 1)
            try:
                transport = SocketTransport.connect(
                    host, int(port_str), timeout=self._response_timeout
                )
            except TransportError as exc:
                logger.error("Failed to connect to %s: %s", agent_id, exc)
                self._failed_bidders.add(agent_id)
                self._log(f"[CONNECT] {agent_id} unreachable: {exc}", "error", agent_id)
                continue
            self._transports[agent_id] = transport
            logger.info("Connected to bidder %s at %s", agent_id, address)

        if not self._active_bidders():
            raise TransportError("Could not connect to any bidder")

    def run_negotiation(self) -> NegotiationResult:
        """Run the complete negotiation according to the coordinator's policy.

        Calls :meth:`start_negotiation` and :meth:`connect_to_bidders` if
        needed, runs CFP rounds until the stopping rule fires, then awards
        the best offer or reports failure.

        Stopping rule (checked after each round):
            - ``max_rounds`` reached → stop.
            - fewer than ``min_rounds`` run → continue.
            - no feasible offer yet and nothing new to reveal → stop (more
              rounds can't change anything).
            - a feasible offer exists but this round didn't improve on it
              → stop (offers have converged).
            - otherwise → continue.
        """
        if self._state_machine is None or self.state != NegotiationState.CREATED:
            self.start_negotiation()
        self.connect_to_bidders()  # no-op for bidders already connected

        new_constraints: dict[str, Any] = {}
        offer_feedback: dict[str, dict[str, Any]] | None = None

        while True:
            evaluations = self.run_cfp_round(
                updated_constraints=new_constraints or None,
                offer_feedback=offer_feedback,
            )
            round_num = self._current_round

            round_best = self.get_best_proposal(evaluations)
            improved = round_best is not None and (
                self._best is None
                or round_best.utility_score > self._best.utility_score + _IMPROVEMENT_EPSILON
            )
            if improved:
                self._best = round_best

            new_constraints = self._derive_constraints(evaluations)
            offer_feedback = self._offer_feedback() if self._best else None

            if not self._active_bidders():
                self._log("All bidders dropped out", "decision", "all")
                break
            if round_num >= self._max_rounds:
                self._log(f"Max rounds ({self._max_rounds}) reached", "decision", "all")
                break
            if round_num < self._min_rounds:
                self._log(
                    f"Round {round_num} < minimum {self._min_rounds} — continuing",
                    "decision",
                    "all",
                )
                continue
            if self._best is None and not new_constraints:
                self._log("No feasible offer and nothing left to reveal", "decision", "all")
                break
            if self._best is not None and not improved:
                self._log("Best offer did not improve — offers converged", "decision", "all")
                break
            self._log(
                f"Round {round_num} improved the best offer — requesting better offers",
                "decision",
                "all",
            )

        if self._best is None:
            reason = f"No feasible proposal after {self._current_round} round(s)"
            self.report_failure(reason)
            self._result = NegotiationResult(
                success=False,
                negotiation_id=self._negotiation_id,
                task_id=self._task_id,
                rounds=self._current_round,
                reason=reason,
            )
            return self._result

        # Award the best offer; if refused, fall back to the next best.
        for candidate in self._award_candidates():
            if self.award_contract(candidate):
                self._result = NegotiationResult(
                    success=True,
                    negotiation_id=self._negotiation_id,
                    task_id=self._task_id,
                    rounds=self._current_round,
                    winner=candidate,
                )
                return self._result

        reason = "Every award candidate refused or was unreachable"
        self.report_failure(reason)
        self._result = NegotiationResult(
            success=False,
            negotiation_id=self._negotiation_id,
            task_id=self._task_id,
            rounds=self._current_round,
            reason=reason,
        )
        return self._result

    def run_cfp_round(
        self,
        updated_constraints: dict[str, Any] | None = None,
        offer_feedback: dict[str, dict[str, Any]] | None = None,
    ) -> list[ProposalEvaluation]:
        """Execute one CFP → proposal collection → evaluation round.

        Args:
            updated_constraints: Constraints to add from this round on
                (e.g. ``{"max_width": 210}``).  They accumulate: later
                rounds keep announcing every constraint revealed so far.
            offer_feedback: Optional per-bidder feedback, keyed by bidder
                ID, e.g. ``{"you_are_leading": False, "request": "improve_offer"}``.

        Returns:
            List of ``ProposalEvaluation`` for this round.
        """
        if self._state_machine is None:
            raise RuntimeError("start_negotiation() must be called first")

        self._current_round += 1
        round_num = self._current_round

        current = self._state_machine.current_state
        if current == NegotiationState.CREATED:
            self._state_machine.transition(NegotiationState.CFP_SENT)
        else:
            # EVALUATING → NEGOTIATING → CFP_SENT (next round)
            if current != NegotiationState.NEGOTIATING:
                self._state_machine.transition(NegotiationState.NEGOTIATING)
            self._state_machine.transition(NegotiationState.CFP_SENT)

        newly_revealed: dict[str, Any] = {}
        if updated_constraints:
            newly_revealed = {
                k: v for k, v in updated_constraints.items()
                if self._revealed_constraints.get(k) != v
            }
            self._revealed_constraints.update(updated_constraints)

        base_payload: dict[str, Any] = {
            "task_description": self._requirements["task_description"],
            "negotiation_id": self._negotiation_id,
            "round": round_num,
            "requirements": self._announced_requirements(),
        }
        if self._revealed_constraints:
            base_payload["updated_constraints"] = dict(self._revealed_constraints)
            base_payload["feedback"] = self._build_feedback(self._revealed_constraints)

        # ── Send CFP to every active bidder first ─────────────────────────
        self._state_machine.transition(NegotiationState.COLLECTING_PROPOSALS)
        self._round_proposals[round_num] = []
        awaiting: list[str] = []

        for agent_id in self._active_bidders():
            payload = dict(base_payload)
            if offer_feedback and agent_id in offer_feedback:
                payload["offer_feedback"] = dict(offer_feedback[agent_id])
            cfp_msg = MessageEnvelope(
                sender=self.agent_id,
                receiver=agent_id,
                message_type=MessageType.CFP,
                task_id=self._task_id,
                payload=payload,
            )
            try:
                self._transports[agent_id].send_message(cfp_msg)
            except TransportError as exc:
                self._drop_bidder(agent_id, f"CFP send failed: {exc}")
                continue
            awaiting.append(agent_id)
            details: dict[str, Any] = {}
            if newly_revealed:
                details["new_constraints"] = newly_revealed
            if "offer_feedback" in payload:
                details["offer_feedback"] = payload["offer_feedback"]
            self._log(f"[ROUND {round_num}] CFP → {agent_id}", "cfp_sent", agent_id, details)

        # ── Then collect each reply ───────────────────────────────────────
        for agent_id in awaiting:
            try:
                reply = self._transports[agent_id].receive_message()
            except (TransportError, CodecError) as exc:
                self._drop_bidder(agent_id, f"no reply: {exc}")
                continue

            if reply.task_id != self._task_id:
                self._log(
                    f"[ROUND {round_num}] {agent_id} → reply for wrong task {reply.task_id}",
                    "unexpected",
                    agent_id,
                )
            elif reply.message_type == MessageType.PROPOSAL:
                proposal = dict(reply.payload)
                # Trust the transport identity and our own round counter,
                # not what the bidder wrote in the payload.
                proposal["agent_id"] = agent_id
                proposal["negotiation_round"] = round_num
                self._round_proposals[round_num].append(proposal)
                price = proposal.get("proposed_price")
                price_text = f" at ₹{price:,}" if isinstance(price, (int, float)) else ""
                self._log(
                    f"[ROUND {round_num}] {agent_id} → PROPOSAL: "
                    f"{proposal.get('product_id', '?')}{price_text}",
                    "proposal_received",
                    agent_id,
                    {
                        "product_id": proposal.get("product_id"),
                        "width_cm": proposal.get("product", {}).get("width_cm"),
                        "list_price": proposal.get("list_price"),
                        "proposed_price": proposal.get("proposed_price"),
                        "discount_pct": proposal.get("discount_pct"),
                    },
                )
            elif reply.message_type == MessageType.FAILURE:
                self._log(
                    f"[ROUND {round_num}] {agent_id} → No proposal "
                    f"({reply.payload.get('reason', 'unknown')})",
                    "no_proposal",
                    agent_id,
                    dict(reply.payload),
                )
            else:
                self._log(
                    f"[ROUND {round_num}] {agent_id} → unexpected {reply.message_type}",
                    "unexpected",
                    agent_id,
                )

        # ── Evaluate ──────────────────────────────────────────────────────
        self._state_machine.transition(NegotiationState.EVALUATING)
        evaluations: list[ProposalEvaluation] = []

        for proposal in self._round_proposals[round_num]:
            # The coordinator always recalculates utility itself, from the
            # product attributes and the price actually offered.  Any score
            # the bidder reported is only compared, never trusted.
            evaluation = self._evaluator.evaluate(proposal, self._requirements)
            evaluations.append(evaluation)

            if evaluation.feasible:
                details: dict[str, Any] = {
                    "utility": evaluation.utility_score,
                    "scores": evaluation.scores,
                }
                if evaluation.utility_breakdown:
                    details["breakdown"] = evaluation.utility_breakdown
                self._log(
                    f"  {evaluation.agent_id} candidate {evaluation.product_id} "
                    f"(₹{evaluation.proposed_price:,}) — utility={evaluation.utility_score:.4f}",
                    "proposal_feasible",
                    evaluation.agent_id,
                    details,
                )
                gap = evaluation.utility_discrepancy
                if gap is not None and abs(gap) > 1e-4:
                    self._log(
                        f"  {evaluation.agent_id} reported utility "
                        f"{evaluation.reported_utility:.4f} but the coordinator calculated "
                        f"{evaluation.utility_score:.4f} — using the coordinator's own score",
                        "utility_discrepancy",
                        evaluation.agent_id,
                        {"reported": evaluation.reported_utility,
                         "recalculated": evaluation.utility_score,
                         "difference": gap},
                    )
            else:
                self._log(
                    f"  {evaluation.agent_id} candidate {evaluation.product_id} rejected — "
                    f"{evaluation.rejection_reason}",
                    "proposal_rejected",
                    evaluation.agent_id,
                    {"reason": evaluation.rejection_reason},
                )

        self._round_evaluations[round_num] = evaluations
        return evaluations

    def check_agreement(self, evaluations: list[ProposalEvaluation]) -> bool:
        """Check if any feasible proposal exists."""
        return any(e.feasible for e in evaluations)

    def get_best_proposal(self, evaluations: list[ProposalEvaluation]) -> ProposalEvaluation | None:
        """Return the highest-utility feasible proposal, or None."""
        ranked = self._evaluator.rank(evaluations)
        return ranked[0] if ranked else None

    def award_contract(self, winner: ProposalEvaluation) -> bool:
        """Award ``winner``; on INFORM_DONE notify everyone and terminate.

        Returns:
            ``True`` if the winner confirmed with INFORM_DONE (negotiation
            COMPLETED → TERMINATED).  ``False`` if it refused or was
            unreachable — the state returns to NEGOTIATING so another
            candidate can be awarded.
        """
        self._state_machine.transition(NegotiationState.AWARDED)
        agent_id = winner.agent_id
        transport = self._transports.get(agent_id)

        award_msg = MessageEnvelope(
            sender=self.agent_id,
            receiver=agent_id,
            message_type=MessageType.AWARD,
            task_id=self._task_id,
            payload={
                "negotiation_id": self._negotiation_id,
                "product_id": winner.product_id,
                "proposed_price": winner.proposed_price,
                "utility_score": winner.utility_score,
                "round": winner.round,
            },
        )
        self._log(
            f"[AWARD] → {agent_id}: {winner.product_id} at ₹{winner.proposed_price:,} "
            f"(round {winner.round}, utility {winner.utility_score:.4f})",
            "award_sent",
            agent_id,
        )

        reply: MessageEnvelope | None = None
        if transport is None or agent_id in self._failed_bidders:
            self._log(f"[AWARD] {agent_id} is no longer reachable", "award_failed", agent_id)
        else:
            try:
                transport.send_message(award_msg)
                reply = transport.receive_message()
            except (TransportError, CodecError) as exc:
                self._drop_bidder(agent_id, f"award failed: {exc}")

        confirmed = (
            reply is not None
            and reply.message_type == MessageType.INFORM_DONE
            and reply.task_id == self._task_id
            and reply.payload.get("product_id") == winner.product_id
        )
        if not confirmed:
            if reply is not None:
                self._log(
                    f"[AWARD] {agent_id} refused: "
                    f"{reply.payload.get('detail') or reply.payload.get('reason') or reply.message_type}",
                    "award_refused",
                    agent_id,
                    dict(reply.payload),
                )
            self._state_machine.transition(NegotiationState.NEGOTIATING)
            return False

        self._state_machine.transition(NegotiationState.EXECUTING)
        self._log(f"[EXECUTION] {agent_id} → INFORM_DONE", "inform_done", agent_id, dict(reply.payload))
        self._state_machine.transition(NegotiationState.COMPLETED)

        result_payload = {
            "negotiation_id": self._negotiation_id,
            "winner": agent_id,
            "product_id": winner.product_id,
            "final_price": reply.payload.get("final_price", winner.proposed_price),
            "utility_score": winner.utility_score,
            "rounds": self._current_round,
            "status": "completed",
        }

        for other in self._active_bidders():
            if other == agent_id:
                continue
            self._send_quietly(other, MessageType.REJECT_PROPOSAL, {
                "negotiation_id": self._negotiation_id,
                "reason": f"Another proposal was selected ({agent_id})",
            })
            self._log(f"[REJECT] → {other}", "reject_sent", other)

        for other in self._active_bidders():
            self._send_quietly(other, MessageType.INFORM_RESULT, result_payload)
            self._log(f"[CONSENSUS] → {other}: INFORM_RESULT", "inform_result", other, result_payload)

        self._state_machine.transition(NegotiationState.TERMINATED)
        self._log("Negotiation successfully terminated.", "terminated", "all")
        return True

    def report_failure(self, reason: str) -> None:
        """Report that no agreement could be reached and notify bidders."""
        self._state_machine.transition(NegotiationState.FAILED)
        self._log(f"No agreement: {reason}", "failed", "all", {"reason": reason})

        for agent_id in self._active_bidders():
            self._send_quietly(agent_id, MessageType.CANCEL, {
                "negotiation_id": self._negotiation_id,
                "reason": reason,
            })
            self._log(f"[CANCEL] → {agent_id}", "cancel_sent", agent_id)

        self._state_machine.transition(NegotiationState.TERMINATED)
        self._log("Negotiation terminated — no agreement.", "terminated", "all")

    def close_all(self) -> None:
        """Close all transport connections."""
        for transport in self._transports.values():
            transport.close()
        self._transports.clear()

    # ── Policy helpers ────────────────────────────────────────────────────

    def _announced_requirements(self) -> dict[str, Any]:
        """Requirements included in every CFP.

        In progressive mode the room width stays private until revealed
        through ``updated_constraints``.
        """
        announced = dict(self._requirements)
        if self._progressive:
            announced.pop("room_width", None)
        return announced

    def _derive_constraints(self, evaluations: list[ProposalEvaluation]) -> dict[str, Any]:
        """Constraints to reveal next round, based on why proposals failed.

        Only constraints not yet announced are returned, so an empty dict
        means more rounds can't tell bidders anything new.
        """
        # rejection keyword → (constraint to reveal, requirement it comes from)
        rules = {
            "width": ("max_width", "room_width"),
            "budget": ("max_budget", "max_budget"),
            "seating": ("seating_capacity", "seating_capacity"),
        }
        announced = self._announced_requirements()
        new: dict[str, Any] = {}
        for ev in evaluations:
            if ev.feasible:
                continue
            reason = ev.rejection_reason.lower()
            for keyword, (constraint, source) in rules.items():
                if keyword not in reason:
                    continue
                already_known = (
                    constraint in self._revealed_constraints
                    or announced.get(source) is not None
                )
                if not already_known:
                    new[constraint] = self._requirements[source]
        return new

    def _offer_feedback(self) -> dict[str, dict[str, Any]]:
        """Per-bidder feedback asking for better offers."""
        best = self._best
        return {
            agent_id: {
                "request": "improve_offer",
                "you_are_leading": agent_id == best.agent_id,
                "best_utility": best.utility_score,
            }
            for agent_id in self._active_bidders()
        }

    def _award_candidates(self) -> list[ProposalEvaluation]:
        """All feasible offers from every round, best first, one per offer."""
        everything = [e for evs in self._round_evaluations.values() for e in evs]
        seen: set[tuple[str, str, float]] = set()
        ordered: list[ProposalEvaluation] = []
        for ev in self._evaluator.rank(everything):
            key = (ev.agent_id, ev.product_id, ev.proposed_price)
            if key in seen:
                continue
            seen.add(key)
            ordered.append(ev)
        return ordered

    def _drop_bidder(self, agent_id: str, reason: str) -> None:
        self._failed_bidders.add(agent_id)
        transport = self._transports.pop(agent_id, None)
        if transport is not None:
            transport.close()
        logger.warning("Dropping bidder %s: %s", agent_id, reason)
        self._log(f"[ROUND {self._current_round}] {agent_id} dropped — {reason}", "error", agent_id)

    def _send_quietly(self, agent_id: str, message_type: MessageType, payload: dict[str, Any]) -> None:
        """Send a one-way notification; a failure only drops that bidder."""
        transport = self._transports.get(agent_id)
        if transport is None:
            return
        try:
            transport.send_message(MessageEnvelope(
                sender=self.agent_id,
                receiver=agent_id,
                message_type=message_type,
                task_id=self._task_id,
                payload=payload,
            ))
        except TransportError as exc:
            self._drop_bidder(agent_id, f"{message_type.value} send failed: {exc}")

    def _build_feedback(self, constraints: dict[str, Any]) -> str:
        """Build a human-readable feedback string from revealed constraints."""
        parts: list[str] = []
        if "max_width" in constraints:
            parts.append(f"Only products with width ≤ {constraints['max_width']} cm are acceptable")
        if "max_budget" in constraints:
            parts.append(f"Budget ceiling is ₹{constraints['max_budget']:,}")
        if "seating_capacity" in constraints:
            parts.append(f"At least {constraints['seating_capacity']} seats are required")
        if "preferred_style" in constraints:
            parts.append(f"Style must be {constraints['preferred_style']}")
        return ". ".join(parts) + "." if parts else "Please reconsider your available alternatives."

    def _log(
        self,
        message: str,
        event_type: str,
        agent: str = "",
        details: dict[str, Any] | None = None,
    ) -> None:
        self._log_seq += 1
        self._message_log.append({
            "seq": self._log_seq,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": message,
            "event_type": event_type,
            "agent": agent,
            "round": self._current_round,
            "details": details or {},
        })
        logger.info(message)
