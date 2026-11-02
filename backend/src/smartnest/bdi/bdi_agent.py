"""The BDI agent.

``BDIAgent`` extends the existing :class:`~smartnest.agents.base_agent.BaseAgent`
(identity, ``state``, ``action_log``, logging, the ``handle_event``
contract), so it plugs into the same :class:`EventStream` the reflex agent
reads.  The CNP agents are not touched.

Deliberation cycle (``handle_event`` and the ``set_*`` methods):

    1. **Perceive** — map the environmental event to belief updates.
       Updates that change nothing are dropped; malformed ones are
       rejected and recorded.
    2. **Filter** — ask each wanted desire's goal handler whether the real
       changes can affect it.  If nothing is affected and no committed
       intention became invalid, the agent does *not* replan; it records
       why.
    3. **Check intentions** — evaluate every live intention's preconditions
       against the new beliefs.  An intention whose preconditions fail is
       recognised as invalid and suspended.
    4. **Reconsider desires** — in priority order (conflicting desires
       resolved first), re-plan each pursuable desire.  The select-product
       handler runs the existing UtilityOptimizer.
    5. **Revise intentions** — adopt a new intention, keep the current one
       (re-confirmed, figures refreshed), or replace it, recording the
       complete before/after state and an explanation.
    6. **Act** — execute automatic steps of committed intentions; steps
       needing outside confirmation wait.
    7. **Persist** — save beliefs, desires, intentions and history if
       anything changed.

All decisions are deterministic Python.  Persistence goes through an
injected :class:`AgentStateStore`; this module knows nothing about
databases.  A failed load or save never crashes the agent: it is
recorded as a transition, and after a failed *load* saving is disabled so
an existing record can never be overwritten by a blank state.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Sequence

from smartnest.agents.base_agent import BaseAgent
from smartnest.bdi import transitions as T
from smartnest.bdi.beliefs import BeliefBase, BeliefChange
from smartnest.bdi.desires import Desire, DesireManager, DesireStatus
from smartnest.bdi.intentions import (
    ActionExecutor,
    Intention,
    IntentionManager,
    IntentionStatus,
    default_executor,
)
from smartnest.bdi.planner import (
    CATALOGUE_SOURCE,
    USER,
    BDIPlanner,
    PlanOutcome,
    normalise_requirement,
)
from smartnest.bdi.state_store import AgentSnapshot, AgentStateStore, StateStoreError
from smartnest.bdi.transitions import Transition
from smartnest.core.events import EventStream
from smartnest.core.models import ActionLog, ActionResultType, EnvironmentalEvent, EventType

logger = logging.getLogger(__name__)

DEFAULT_MAX_HISTORY = 200


def _normalise_availability(raw: Any) -> str | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    value = raw.strip().lower().replace(" ", "_").replace("-", "_")
    return "in_stock" if value == "available" else value


class BDIAgent(BaseAgent):
    """A deterministic belief-desire-intention agent.

    Args:
        agent_id: Stable identity; also the key under which state is persisted.
        planner: The :class:`BDIPlanner` that turns desires into intentions.
        state_store: Optional persistence.  If given, state is loaded on
            construction (when ``auto_load``) and saved after every change.
        name: Optional human-readable name.
        executor: Callable running an intention's actions
            (default: records that the step was performed).
        auto_load: Load saved state when the agent is created.
        max_history: How many transitions to keep.
    """

    def __init__(
        self,
        agent_id: str,
        planner: BDIPlanner,
        state_store: AgentStateStore | None = None,
        name: str | None = None,
        executor: ActionExecutor = default_executor,
        auto_load: bool = True,
        max_history: int = DEFAULT_MAX_HISTORY,
    ) -> None:
        super().__init__(agent_id=agent_id, name=name)
        if max_history < 1:
            raise ValueError("max_history must be at least 1")
        self.planner = planner
        self.beliefs = BeliefBase()
        self.desires = DesireManager()
        self.intentions = IntentionManager()
        self._transitions: list[Transition] = []
        self._store = state_store
        self._executor = executor
        self._max_history = max_history
        self._lock = threading.RLock()
        self._dirty = False
        self._persistence_enabled = state_store is not None
        self._last_waiting: dict[str, tuple[str, ...]] = {}
        #: Latest planning outcome per desire (in memory only): lets callers explain
        #: a decision with the optimizer's own figures instead of recomputing them.
        self.last_outcomes: dict[str, PlanOutcome] = {}
        self.state_restored = False

        if state_store is not None and auto_load:
            self.load_state()
        self._sync_state()

    # ══ Public API ═══════════════════════════════════════════════════════

    @property
    def transitions(self) -> list[Transition]:
        """The agent's explained history, oldest first (a copy)."""
        with self._lock:
            return list(self._transitions)

    @property
    def last_transition(self) -> Transition | None:
        with self._lock:
            return self._transitions[-1] if self._transitions else None

    def current_intention(self, desire_id: str | None = None) -> Intention | None:
        """The live intention for ``desire_id`` (default: the first live intention)."""
        with self._lock:
            if desire_id is not None:
                return self.intentions.live_for(desire_id)
            live = self.intentions.live()
            return live[0] if live else None

    # ── Desires ──────────────────────────────────────────────────────────

    def add_desire(self, desire: Desire) -> Desire:
        with self._lock:
            self.desires.add(desire)
            self._record(T.DESIRE_ADDED, f"Added desire '{desire.description}' (priority {desire.priority})",
                         after={"desire_id": desire.desire_id, "priority": desire.priority,
                                "goal_type": desire.goal_type, "status": desire.status.value})
            self._commit()
            return desire

    def drop_desire(self, desire_id: str, reason: str = "dropped by request") -> bool:
        with self._lock:
            desire = self.desires.get(desire_id)
            if desire is None:
                return False
            changed = self.desires.drop(desire_id, reason)
            for intention in self.intentions.live():
                if intention.desire_id == desire_id:
                    self.intentions.cancel(intention.intention_id, f"desire dropped: {reason}")
                    self._record(T.INTENTION_CANCELLED, f"Cancelled intention {intention.intention_id}: its desire was dropped",
                                 before=self._summary(intention), explanation=reason)
            if changed:
                self._record(T.DESIRE_STATUS, f"Dropped desire '{desire.description}'", explanation=reason)
            self._reconsider_conflicts()
            self._commit()
            return changed

    def set_desire_priority(self, desire_id: str, priority: int) -> bool:
        """Change a desire's priority and re-run conflict resolution and planning."""
        with self._lock:
            changed = self.desires.set_priority(desire_id, priority)
            if changed:
                self._record(T.DESIRE_STATUS, f"Priority of {desire_id} set to {priority}")
                self._deliberate(trigger=f"priority of {desire_id} changed", reasons=["desire priorities changed"])
            return changed

    # ── Beliefs ──────────────────────────────────────────────────────────

    def set_requirement(self, predicate: str, value: Any, source: str = "user") -> ActionLog | None:
        """Set one ``user.<predicate>`` belief (e.g. ``"sofa_budget"``) and react to it."""
        return self.set_requirements({predicate: value}, source=source)

    def set_requirements(self, values: dict[str, Any], source: str = "user") -> ActionLog | None:
        """Set several ``user.*`` beliefs at once, then react **once**.

        Raises:
            ValueError: If a value is invalid (nothing is changed).
        """
        updates = []
        for predicate, raw in values.items():
            value, error = normalise_requirement(predicate, raw)
            if error:
                raise ValueError(error)
            updates.append((USER, predicate, value))
        with self._lock:
            return self._perceive(updates, [], source, trigger=f"user input ({', '.join(values)})")

    def set_belief(self, subject: str, predicate: str, value: Any, source: str = "user") -> ActionLog | None:
        """Record any belief and react to it."""
        if subject == USER:
            value, error = normalise_requirement(predicate, value)
            if error:
                raise ValueError(error)
        with self._lock:
            return self._perceive([(subject, predicate, value)], [], source, trigger=f"belief {subject}.{predicate}")

    def invalidate_belief(self, subject: str, predicate: str, reason: str = "") -> ActionLog | None:
        """Stop trusting a belief; it then counts as unknown."""
        with self._lock:
            change = self.beliefs.invalidate(subject, predicate, reason, source="agent")
            return self._react([change] if change else [], [], trigger=f"invalidated {subject}.{predicate}")

    # ── Events and deliberation ──────────────────────────────────────────

    def handle_event(self, event: EnvironmentalEvent) -> ActionLog | None:
        """React to an environmental event (the :class:`BaseAgent` contract).

        Returns an :class:`ActionLog` when the agent's intentions changed
        (or a goal became blocked); ``None`` when the event did not
        require any change.
        """
        if event is None:
            self.logger.warning("Received None event; skipping.")
            return None
        with self._lock:
            updates, problems = self._updates_from_event(event)
            trigger = f"event {event.event_id} ({event.event_type}) from {event.source}"
            if not updates and not problems:
                self._record(T.UPDATE_IGNORED, f"Event type '{event.event_type}' carries nothing the BDI agent tracks",
                             trigger=trigger)
                self._commit()
                return None
            return self._perceive(updates, problems, event.source, trigger=trigger, event_id=event.event_id)

    def process_events(self, stream: EventStream, timeout: float = 0.0) -> list[ActionLog]:
        """Drain ``stream``, handling each event.  Returns the action logs produced."""
        logs: list[ActionLog] = []
        while True:
            event = stream.next_event(timeout=timeout)
            if event is None:
                return logs
            log = self.handle_event(event)
            if log is not None:
                logs.append(log)

    def deliberate(self, trigger: str = "manual deliberation") -> ActionLog | None:
        """Reconsider every desire now, regardless of what changed (e.g. at startup)."""
        with self._lock:
            return self._deliberate(trigger=trigger, reasons=[trigger])

    # ── Intentions ───────────────────────────────────────────────────────

    def confirm(self, desire_id: str | None = None) -> bool:
        """Give the outside confirmation for a live intention's manual step.

        Runs the manual action (e.g. reserving the product).  When the
        intention completes, its desire is marked ACHIEVED.

        Returns ``True`` if the confirmed step ran.
        """
        with self._lock:
            intention = self.current_intention(desire_id)
            if intention is None:
                return False
            outcome = self.intentions.confirm(intention.intention_id, self.beliefs, self._executor)
            if outcome.status != "done":
                self._record(T.INTENTION_PROGRESS, f"Confirmation of {intention.intention_id} did not run: {outcome.status}",
                             explanation=outcome.detail)
                self._commit()
                return False
            self._record(T.INTENTION_PROGRESS, f"Confirmed and executed '{outcome.kind}' for {intention.intention_id}",
                         before=None, after=self._summary(intention))
            if intention.status == IntentionStatus.COMPLETED:
                self.desires.achieve(intention.desire_id, f"completed via {intention.intention_id}")
                self._record(T.DESIRE_ACHIEVED, f"Desire {intention.desire_id} achieved",
                             after=self._summary(intention),
                             explanation=f"{intention.params.get('product_id')} has been reserved")
            self._commit()
            return True

    def cancel_intention(self, intention_id: str, reason: str = "cancelled by request") -> bool:
        with self._lock:
            intention = self.intentions.get(intention_id)
            if intention is None or not self.intentions.cancel(intention_id, reason):
                return False
            self._record(T.INTENTION_CANCELLED, f"Cancelled intention {intention_id}",
                         before=self._summary(intention), explanation=reason)
            self._commit()
            return True

    # ── Persistence ──────────────────────────────────────────────────────

    def load_state(self) -> bool:
        """Load saved state from the store.

        Returns ``True`` if state was restored.  A missing record starts
        fresh (saving stays enabled).  A store error or corrupted record
        also starts fresh, but **disables saving** so the existing record
        is never overwritten; call ``load_state()`` again to retry.
        """
        with self._lock:
            if self._store is None:
                return False
            try:
                snapshot = self._store.load(self.agent_id)
                if snapshot is None:
                    self._persistence_enabled = True
                    self._record(T.STATE_NOT_FOUND, f"No saved state for '{self.agent_id}'; starting fresh")
                    return False
                self._restore(snapshot)
            except (StateStoreError, ValueError, TypeError) as exc:
                self._persistence_enabled = False
                self._record(T.PERSISTENCE_ERROR, "Could not load saved state; running in memory only",
                             explanation=f"{exc}. Saving is disabled so the stored record is not overwritten.")
                return False
            self._persistence_enabled = True
            self.state_restored = True
            self._record(T.STATE_RESTORED,
                         f"Restored state saved {snapshot.saved_at.isoformat()}: "
                         f"{len(self.beliefs)} beliefs, {len(self.desires)} desires, "
                         f"{len(self.intentions)} intentions, {len(snapshot.transitions)} transitions")
            self._sync_state()
            return True

    def save_state(self) -> bool:
        """Save now, even if nothing changed.  Returns ``True`` on success."""
        with self._lock:
            return self._save(force=True)

    def reset(self) -> None:
        """Clear the agent's in-memory state (the stored record is untouched until the next save)."""
        with self._lock:
            super().reset()
            self.beliefs.clear()
            self.desires.clear()
            self.intentions.clear()
            self._transitions.clear()
            self._last_waiting.clear()
            self.last_outcomes.clear()
            self._dirty = False

    # ══ Perception ═══════════════════════════════════════════════════════

    def _updates_from_event(self, event: EnvironmentalEvent) -> tuple[list[tuple[str, str, Any]], list[str]]:
        """Translate an event into ``(subject, predicate, value)`` updates plus rejected fields."""
        payload = event.payload if isinstance(event.payload, dict) else {}
        kind = event.event_type
        updates: list[tuple[str, str, Any]] = []
        problems: list[str] = []

        def user(predicate: str, raw: Any) -> None:
            value, error = normalise_requirement(predicate, raw)
            if error:
                problems.append(error)
            else:
                updates.append((USER, predicate, value))

        def first(*keys: str) -> Any:
            return next((payload[k] for k in keys if payload.get(k) is not None), None)

        if kind == EventType.PRODUCT_AVAILABILITY.value:
            subject = first("product_id", "product_name")
            availability = _normalise_availability(payload.get("availability"))
            if not isinstance(subject, str) or availability is None:
                problems.append("availability event needs a product and an availability value")
            else:
                updates.append((subject, "availability", availability))
        elif kind == EventType.PRICE_CHANGE.value:
            subject = first("product_id", "product_name")
            price = payload.get("new_price")
            valid_price = isinstance(price, (int, float)) and not isinstance(price, bool) and price > 0 and price == price
            if not isinstance(subject, str) or not valid_price:
                problems.append(f"price event needs a product and a positive new_price, got {price!r}")
            else:
                updates.append((subject, "price", price))
        elif kind == EventType.BUDGET_UPDATE.value:
            seen = False
            for predicate in ("total_budget", "sofa_budget", "estimated_expenditure"):
                if predicate in payload:
                    seen = True
                    user(predicate, payload[predicate])
            if not seen:
                problems.append("budget event carried no total_budget, sofa_budget or estimated_expenditure")
        elif kind == EventType.ROOM_UPDATE.value:
            width = first("max_width_cm", "room_width", "width_cm")
            depth = first("max_depth_cm", "room_depth", "depth_cm")
            height = first("max_height_cm", "room_height", "height_cm")
            if width is None and depth is None and height is None:
                problems.append("room event carried no width, depth or height")
            if height is not None:
                user("max_height_cm", height)
            if width is not None:
                user("max_width_cm", width)
            if depth is not None:
                user("max_depth_cm", depth)
        elif kind == EventType.PREFERENCE_UPDATE.value:
            found = False
            for predicate, aliases in (("preferred_style", ("preferred_style", "style")),
                                       ("preferred_colour", ("preferred_colour", "preferred_color", "colour", "color")),
                                       ("min_seating", ("min_seating", "seating_capacity"))):
                raw = first(*aliases)
                if raw is not None:
                    found = True
                    user(predicate, raw)
            if not found:
                problems.append("preference event carried no style, colour or seating")
        return updates, problems

    def _perceive(self, updates: Sequence[tuple[str, str, Any]], problems: list[str], source: str,
                  trigger: str, event_id: str | None = None) -> ActionLog | None:
        changes: list[BeliefChange] = []
        for subject, predicate, value in updates:
            try:
                change = self.beliefs.update(subject, predicate, value, source=source)
            except ValueError as exc:
                problems.append(str(exc))
                continue
            if change is not None:
                changes.append(change)
        return self._react(changes, problems, trigger, event_id)

    def _react(self, changes: list[BeliefChange], problems: list[str], trigger: str,
               event_id: str | None = None) -> ActionLog | None:
        """Decide whether the changes require reconsideration; if so, do it."""
        for problem in problems:
            self._record(T.UPDATE_REJECTED, f"Rejected update: {problem}", trigger=trigger)
        if not changes:
            self._commit()
            return None

        for change in changes:
            self._record(T.BELIEF_UPDATED, f"Belief updated: {change.describe()}", trigger=trigger,
                         before={"value": change.old_value}, after={"value": change.new_value},
                         details={"subject": change.subject, "predicate": change.predicate,
                                  "kind": change.kind, "source": change.source})

        affected, ignoring = self._relevance(changes)
        invalid = self._invalid_intentions()
        targets = set(affected) | {intention.desire_id for intention, _ in invalid}
        if not targets:
            self._record(T.UPDATE_IGNORED, "No replanning needed",
                         trigger=trigger, explanation="; ".join(dict.fromkeys(ignoring)) or "nothing depends on this belief")
            self._commit()
            return None
        reasons = [r for rs in affected.values() for r in rs] or ["a committed intention is no longer valid"]
        return self._deliberate(trigger, reasons, event_id, targets=targets)

    def _relevance(self, changes: list[BeliefChange]) -> tuple[dict[str, list[str]], list[str]]:
        """Ask each wanted desire's handler whether the changes could affect it.

        Returns ``({desire_id: reasons}, ignore_reasons)``.  Only the desires
        in the first mapping need to be re-planned.
        """
        held = {i.params.get("product_id") for i in self.intentions.live() if i.params.get("product_id")}
        affected: dict[str, list[str]] = {}
        ignoring: list[str] = []
        wanted = [d for d in self.desires.all() if d.status in (DesireStatus.ACTIVE, DesireStatus.BLOCKED)
                  and self.planner.supports(d.goal_type)]
        for change in changes:
            hit = False
            for desire in wanted:
                is_affected, reason = self.planner.is_affected_by(desire, change, self.beliefs, held)
                if is_affected:
                    affected.setdefault(desire.desire_id, []).append(reason)
                    hit = True
                else:
                    ignoring.append(reason)
            if not wanted and not hit:
                ignoring.append(f"{change.subject}.{change.predicate} concerns no wanted desire")
        return affected, ignoring

    def _invalid_intentions(self) -> list[tuple[Intention, Any]]:
        found = []
        for intention in self.intentions.live():
            if intention.status in (IntentionStatus.ACTIVE, IntentionStatus.PENDING):
                validation = self.intentions.validate(intention, self.beliefs)
                if not validation.valid:
                    found.append((intention, validation))
        return found

    # ══ Deliberation ═════════════════════════════════════════════════════

    def _deliberate(self, trigger: str, reasons: list[str], event_id: str | None = None,
                    targets: set[str] | None = None) -> ActionLog | None:
        """Reconsider desires.

        ``targets`` limits re-planning to the desires actually affected by
        the change (``None`` = reconsider everything).  Maintenance goals are
        always re-checked at the end, because they depend on the final
        commitments; that check is cheap and records nothing unless a goal's
        status changes.
        """
        cycle: list[Transition] = []

        # 1. Recognise invalid intentions.
        for intention, validation in self._invalid_intentions():
            before = self._summary(intention)
            self.intentions.suspend(intention.intention_id, validation.explain())
            cycle.append(self._record(
                T.INTENTION_INVALIDATED,
                f"Intention {intention.intention_id} ({intention.params.get('product_id')}) is no longer valid",
                trigger=trigger, before=before, after=self._summary(intention),
                explanation=validation.explain(),
                details={"failed_conditions": [f.explanation for f in validation.failures]}))

        # 2. Resolve conflicts between desires.
        cycle.extend(self._reconsider_conflicts(trigger))

        # 3. Reconsider desires in priority order.
        pursuable = self.desires.pursuable(self.beliefs)
        waiting = self.desires.waiting_for_conditions(self.beliefs)
        replan = [d for d in pursuable if d.kind == "achievement" and (targets is None or d.desire_id in targets)]
        replan += [d for d in pursuable if d.kind == "maintenance"]
        achieve = " > ".join(f"{d.desire_id}(p{d.priority})" for d in replan if d.kind == "achievement") or "none"
        monitor = ", ".join(f"{d.desire_id}(p{d.priority})" for d in replan if d.kind == "maintenance")
        listing = f"goals by priority: {achieve}" + (f"; then re-checking maintenance goals: {monitor}" if monitor else "")
        untouched = [d.desire_id for d in pursuable if d not in replan]
        notes = [f"waiting: {d.desire_id} ({'; '.join(r.explanation for r in unmet)})" for d, unmet in waiting]
        if untouched:
            notes.append(f"not affected, left as is: {', '.join(untouched)}")
        self._record(T.DESIRES_RECONSIDERED, f"Reconsidering desires: {listing}",
                     trigger=trigger,
                     explanation="; ".join(dict.fromkeys(reasons)) + (" | " + " | ".join(notes) if notes else ""),
                     details={"order": [d.desire_id for d in replan]})

        committed: list[Intention] = []
        for desire in replan:
            try:
                outcome = self.planner.plan(desire, self.beliefs, self.intentions.live())
            except ValueError as exc:  # unknown goal type
                cycle.append(self._record(T.DESIRE_STATUS, f"Cannot pursue {desire.desire_id}", trigger=trigger,
                                          explanation=str(exc)))
                continue
            except Exception as exc:  # a planning bug must not take the agent down
                logger.exception("Planning failed for %s", desire.desire_id)
                cycle.append(self._record(T.DESIRE_STATUS, f"Planning failed for {desire.desire_id}",
                                          trigger=trigger, explanation=f"{type(exc).__name__}: {exc}"))
                continue
            self.last_outcomes[desire.desire_id] = outcome
            intention = self._apply_outcome(desire, outcome, trigger, cycle)
            if intention is not None:
                committed.append(intention)

        # 4. Act: run automatic steps of committed intentions.
        for intention in committed:
            outcomes = self.intentions.run(intention.intention_id, self.beliefs, self._executor)
            done = [o.kind for o in outcomes if o.status == "done"]
            failed = [o for o in outcomes if o.status in ("failed", "precondition_failed")]
            waiting_for = [o.kind for o in outcomes if o.status == "awaiting_confirmation"]
            if done or failed:
                text = f"Executed: {', '.join(done) or 'nothing'}"
                if waiting_for:
                    text += f"; awaiting confirmation for: {', '.join(waiting_for)}"
                if failed:
                    text += f"; problem: {failed[0].detail}"
                self._record(T.INTENTION_PROGRESS, f"{intention.intention_id}: {text}", trigger=trigger,
                             after=self._summary(intention))

        self._commit()
        log = self._to_action_log(cycle, event_id)
        if log is not None:
            self.record_action(log)
        return log

    def _reconsider_conflicts(self, trigger: str = "") -> list[Transition]:
        recorded: list[Transition] = []
        for desire, message in self.desires.resolve_conflicts():
            recorded.append(self._record(T.DESIRE_STATUS, f"Desire {desire.desire_id} {message}", trigger=trigger,
                                         after={"desire_id": desire.desire_id, "status": desire.status.value}))
            if desire.status == DesireStatus.SUSPENDED:
                for intention in self.intentions.live():
                    if intention.desire_id == desire.desire_id:
                        self.intentions.suspend(intention.intention_id, f"desire suspended: {message}")
        return recorded

    def _apply_outcome(self, desire: Desire, outcome: PlanOutcome, trigger: str,
                       cycle: list[Transition]) -> Intention | None:
        """Turn one planning outcome into belief-consistent state changes.

        Returns the intention that should now be executed, if any.
        """
        live = self.intentions.live_for(desire.desire_id)

        if outcome.status == "waiting":
            key = tuple(outcome.missing)
            if self._last_waiting.get(desire.desire_id) != key:
                self._last_waiting[desire.desire_id] = key
                cycle.append(self._record(T.DESIRE_WAITING, f"Desire {desire.desire_id} is waiting for information",
                                          trigger=trigger, explanation=outcome.explanation))
            return None
        self._last_waiting.pop(desire.desire_id, None)

        if outcome.status in ("satisfied", "violated"):
            status = DesireStatus.ACTIVE if outcome.status == "satisfied" else DesireStatus.BLOCKED
            status_changed = desire.status != status
            self.desires.set_status(desire.desire_id, status, outcome.explanation)
            if status_changed:
                kind = T.DESIRE_UNBLOCKED if status == DesireStatus.ACTIVE else T.DESIRE_BLOCKED
                cycle.append(self._record(kind, f"Desire {desire.desire_id} is {outcome.status}",
                                          trigger=trigger, explanation=outcome.explanation,
                                          after={"desire_id": desire.desire_id, "status": status.value}))
            return None

        if outcome.status == "blocked":
            before = self._summary(live) if live else None
            if live is not None:
                self.intentions.cancel(live.intention_id, "no feasible alternative exists")
            changed = self.desires.set_status(desire.desire_id, DesireStatus.BLOCKED, outcome.explanation)
            if changed or live is not None:
                cycle.append(self._record(
                    T.DESIRE_BLOCKED,
                    f"No feasible option for '{desire.description}'; no recommendation is made",
                    trigger=trigger, before=before, after=None, explanation=outcome.explanation,
                    details=self._optimization_details(outcome)))
            return None

        # planned
        new = outcome.intention
        assert new is not None
        if desire.status == DesireStatus.BLOCKED and self.desires.set_status(
                desire.desire_id, DesireStatus.ACTIVE, "a feasible option exists again"):
            cycle.append(self._record(T.DESIRE_UNBLOCKED, f"Desire {desire.desire_id} can be pursued again",
                                      trigger=trigger))

        if live is not None and self._same_choice(live, new):
            return self._retain(live, new, outcome, trigger, cycle)
        if live is not None:
            return self._replace(live, new, outcome, trigger, cycle)

        self.intentions.add(new)
        self.intentions.activate(new.intention_id)
        cycle.append(self._record(
            T.INTENTION_ADOPTED, f"Adopted intention {new.intention_id}: {new.description}",
            trigger=trigger, before=None, after=self._summary(new),
            explanation=outcome.explanation, details=self._optimization_details(outcome)))
        return new

    @staticmethod
    def _same_choice(a: Intention, b: Intention) -> bool:
        return (a.params.get("product_id") == b.params.get("product_id")
                and a.params.get("source_agent_id") == b.params.get("source_agent_id"))

    def _retain(self, live: Intention, new: Intention, outcome: PlanOutcome, trigger: str,
                cycle: list[Transition]) -> Intention | None:
        before = self._summary(live)
        old_price, old_utility = live.params.get("price"), live.params.get("utility")
        live.refresh_from(new)
        resumed = False
        if live.status == IntentionStatus.SUSPENDED:
            resumed = self.intentions.resume(live.intention_id, self.beliefs, "re-planning confirmed this choice")
        notes = [f"Re-planning chose the same product ({live.params.get('product_id')})."]
        if old_price != live.params.get("price"):
            notes.append(f"Price changed {old_price} -> {live.params.get('price')}.")
        if old_utility != live.params.get("utility"):
            notes.append(f"Utility changed {old_utility} -> {live.params.get('utility')}.")
        if resumed:
            notes.append("The intention was suspended and has been resumed.")
        notes.append(outcome.explanation)
        cycle.append(self._record(
            T.INTENTION_RETAINED,
            f"Kept intention {live.intention_id} ({live.params.get('product_id')}) after reconsideration",
            trigger=trigger, before=before, after=self._summary(live), explanation=" ".join(notes),
            details=self._optimization_details(outcome)))
        return live if live.status == IntentionStatus.ACTIVE else None

    def _replace(self, old: Intention, new: Intention, outcome: PlanOutcome, trigger: str,
                 cycle: list[Transition]) -> Intention:
        validation = self.intentions.validate(old, self.beliefs)
        before = self._summary(old)
        reason = validation.explain() if not validation.valid else "a higher-utility option exists"
        self.intentions.replace(old.intention_id, new, reason=reason)

        old_id = old.params.get("product_id")
        why_old = (f"{old_id} was dropped because: {validation.explain()}." if not validation.valid
                   else f"{old_id} is still feasible but a better option now exists.")
        explanation = " ".join([why_old, self._old_product_note(old, outcome),
                                f"The UtilityOptimizer then selected {new.params.get('product_id')}: {outcome.explanation}"])
        cycle.append(self._record(
            T.INTENTION_REPLACED,
            f"Replaced intention {old.intention_id} ({old_id}) with {new.intention_id} "
            f"({new.params.get('product_id')})",
            trigger=trigger, before=before, after=self._summary(new), explanation=explanation,
            details=self._optimization_details(outcome)))
        return new

    @staticmethod
    def _old_product_note(old: Intention, outcome: PlanOutcome) -> str:
        result = outcome.optimization
        if result is None:
            return ""
        pid, agent = old.params.get("product_id"), old.params.get("source_agent_id")
        evaluation = next((e for e in result.evaluations
                           if e.product_id == pid and e.source_agent_id == agent), None)
        if evaluation is None:
            return f"{pid} is no longer offered."
        if not evaluation.eligible:
            return f"The optimizer rejects {pid}: {'; '.join(evaluation.reasons)}."
        return f"The optimizer ranks {pid} #{evaluation.rank} (U={evaluation.utility:.2f})."

    @staticmethod
    def _optimization_details(outcome: PlanOutcome) -> dict[str, Any]:
        result = outcome.optimization
        if result is None:
            return {}
        return {
            "candidates_inspected": result.total_inspected,
            "eligible": result.eligible_count,
            "rejected": result.rejected_count,
            "fit_not_verified": result.unverified_count,
            "ranking": [{"rank": e.rank, "product_id": e.product_id, "source_agent_id": e.source_agent_id,
                         "price": e.price, "utility": e.utility} for e in result.ranking[:5]],
            "rejected_candidates": [{"product_id": e.product_id, "reasons": e.reasons}
                                    for e in result.rejected[:8]],
        }

    # ══ Bookkeeping ══════════════════════════════════════════════════════

    @staticmethod
    def _summary(intention: Intention | None) -> dict[str, Any] | None:
        if intention is None:
            return None
        return {
            "intention_id": intention.intention_id,
            "status": intention.status.value,
            "product_id": intention.params.get("product_id"),
            "source_agent_id": intention.params.get("source_agent_id"),
            "price": intention.params.get("price"),
            "utility": intention.params.get("utility"),
        }

    def _record(self, kind: str, summary: str, trigger: str = "", before: dict[str, Any] | None = None,
                after: dict[str, Any] | None = None, explanation: str = "",
                details: dict[str, Any] | None = None) -> Transition:
        transition = Transition(kind=kind, trigger=trigger, summary=summary, before=before, after=after,
                                explanation=explanation, details=details or {})
        self._transitions.append(transition)
        if len(self._transitions) > self._max_history:
            del self._transitions[: len(self._transitions) - self._max_history]
        self._dirty = True
        self.logger.info("%s | %s%s", kind, summary, f" | {explanation}" if explanation else "")
        return transition

    def _to_action_log(self, cycle: list[Transition], event_id: str | None) -> ActionLog | None:
        order = [
            (T.DESIRE_BLOCKED, "BDI_BLOCKED", "No feasible option", ActionResultType.WARNING),
            (T.INTENTION_REPLACED, "BDI_REVISED", "Intention revised", ActionResultType.INFO),
            (T.INTENTION_ADOPTED, "BDI_ADOPTED", "Intention adopted", ActionResultType.APPROVAL),
            (T.INTENTION_RETAINED, "BDI_RETAINED", "Intention retained", ActionResultType.INFO),
        ]
        for kind, rule_id, rule_name, result_type in order:
            transition = next((t for t in cycle if t.kind == kind), None)
            if transition is not None:
                return ActionLog(
                    event_id=event_id or "manual",
                    rule_id=rule_id,
                    rule_name=rule_name,
                    action_description=transition.summary,
                    result_type=result_type,
                    result_message=transition.explanation or transition.summary,
                    result_details={"before": transition.before, "after": transition.after,
                                    **transition.details},
                )
        return None

    def _sync_state(self) -> None:
        """Mirror a compact view of BDI state into ``BaseAgent.state``."""
        self.state.update({
            "beliefs": self.beliefs.as_dict(),
            "desires": [{"desire_id": d.desire_id, "status": d.status.value, "priority": d.priority}
                        for d in self.desires.all()],
            "intentions": [self._summary(i) for i in self.intentions.live()],
            "transition_count": len(self._transitions),
        })

    def _commit(self) -> None:
        self._sync_state()
        self._save()

    # ── Snapshot / restore ───────────────────────────────────────────────

    def _snapshot(self) -> AgentSnapshot:
        return AgentSnapshot(
            agent_id=self.agent_id,
            beliefs=self.beliefs.to_list(),
            desires=self.desires.to_list(),
            intentions=self.intentions.to_list(),
            transitions=[t.model_dump(mode="json") for t in self._transitions],
        )

    def _restore(self, snapshot: AgentSnapshot) -> None:
        beliefs, desires, intentions = BeliefBase(), DesireManager(), IntentionManager()
        beliefs.load(snapshot.beliefs)
        desires.load(snapshot.desires)
        intentions.load(snapshot.intentions)
        transitions = [Transition.model_validate(t) for t in snapshot.transitions]
        # Only replace the live state once everything validated.
        self.beliefs, self.desires, self.intentions = beliefs, desires, intentions
        self._transitions = transitions[-self._max_history:]

    def _save(self, force: bool = False) -> bool:
        if self._store is None or not self._persistence_enabled or not (self._dirty or force):
            return False
        try:
            self._store.save(self._snapshot())
        except Exception as exc:  # persistence must never crash the agent
            self._record(T.PERSISTENCE_ERROR, "Could not save state; continuing in memory",
                         explanation=f"{type(exc).__name__}: {exc}")
            self._dirty = True  # retry with the next change
            return False
        self._dirty = False
        return True
