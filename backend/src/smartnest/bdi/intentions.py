"""Intentions: plans the agent has committed to.

An intention is a committed, ordered plan for one desire.  It carries

    - **preconditions** — conditions over beliefs under which the
      commitment stays valid (for a product choice: the product is
      available, within budget and fits the room).  If any stops holding,
      the intention is *invalid* and the agent must reconsider.
    - **actions** — ordered steps, each with its own preconditions and a
      status.  Steps marked ``manual`` wait for an outside confirmation
      (e.g. the user agreeing to reserve the product).

Lifecycle::

    PENDING ──activate──► ACTIVE ──all actions done──► COMPLETED
                            │  ▲
                   suspend  ▼  │ resume
                          SUSPENDED
    any non-terminal ──cancel──► CANCELLED
    any non-terminal ──replace─► REPLACED  (and a new intention takes over)

``COMPLETED``, ``CANCELLED`` and ``REPLACED`` are terminal.

The :class:`IntentionManager` performs those transitions, validates
intentions against the current beliefs, and executes actions through an
injected *executor* callable — so this module has no knowledge of
sockets, databases or any outside system.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Callable, Iterable, Literal

from pydantic import BaseModel, Field

from smartnest.bdi.common import new_id, utc_now
from smartnest.bdi.conditions import BeliefReader, Condition, ConditionResult


class IntentionStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    REPLACED = "replaced"


class ActionStatus(str, Enum):
    PENDING = "pending"
    DONE = "done"
    FAILED = "failed"


LIVE_STATUSES = (IntentionStatus.PENDING, IntentionStatus.ACTIVE, IntentionStatus.SUSPENDED)
TERMINAL_STATUSES = (IntentionStatus.COMPLETED, IntentionStatus.CANCELLED, IntentionStatus.REPLACED)


class Action(BaseModel):
    """One step of a plan."""

    action_id: str = Field(default_factory=lambda: new_id("act"))
    kind: str
    description: str = ""
    params: dict[str, Any] = Field(default_factory=dict)
    preconditions: list[Condition] = Field(default_factory=list)
    manual: bool = False
    status: ActionStatus = ActionStatus.PENDING
    result: dict[str, Any] = Field(default_factory=dict)
    executed_at: datetime | None = None


class Intention(BaseModel):
    """A committed plan for one desire."""

    intention_id: str = Field(default_factory=lambda: new_id("int"))
    desire_id: str
    description: str
    actions: list[Action] = Field(default_factory=list)
    preconditions: list[Condition] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""
    status: IntentionStatus = IntentionStatus.PENDING
    status_reason: str = ""
    replaced_by: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @property
    def is_live(self) -> bool:
        return self.status in LIVE_STATUSES

    def pending_actions(self) -> list[Action]:
        return [a for a in self.actions if a.status == ActionStatus.PENDING]

    def refresh_from(self, newer: "Intention") -> None:
        """Adopt a re-planned intention's conditions and figures, keeping progress made.

        Used when re-planning reaches the same choice: executed steps stay
        executed, while preconditions, price and utility are brought up to date.
        """
        self.preconditions = [c.model_copy() for c in newer.preconditions]
        fresh = {a.kind: a for a in newer.actions}
        for action in self.pending_actions():
            if action.kind in fresh:
                action.preconditions = [c.model_copy() for c in fresh[action.kind].preconditions]
        self.params = dict(newer.params)
        self.rationale = newer.rationale
        self.updated_at = utc_now()


class ValidationResult(BaseModel):
    """Whether an intention's preconditions still hold."""

    valid: bool
    failures: list[ConditionResult] = Field(default_factory=list)

    def explain(self) -> str:
        return "; ".join(f.explanation for f in self.failures) if self.failures else "all preconditions hold"


class ActionOutcome(BaseModel):
    """Result of trying to execute the next action of an intention."""

    intention_id: str
    action_id: str | None = None
    kind: str | None = None
    status: Literal["done", "failed", "precondition_failed", "awaiting_confirmation", "nothing_to_do"]
    detail: str = ""
    result: dict[str, Any] = Field(default_factory=dict)


ActionExecutor = Callable[[Action, Intention], dict[str, Any]]


def default_executor(action: Action, intention: Intention) -> dict[str, Any]:
    """Records that the step was performed; SmartNest has no outside effects to trigger yet."""
    return {"performed": action.kind, "product_id": intention.params.get("product_id")}


class IntentionManager:
    """Creates, validates, executes, suspends, cancels and replaces intentions."""

    def __init__(self) -> None:
        self._intentions: dict[str, Intention] = {}

    # ── Reading ──────────────────────────────────────────────────────────

    def get(self, intention_id: str) -> Intention | None:
        return self._intentions.get(intention_id)

    def all(self) -> list[Intention]:
        return sorted(self._intentions.values(), key=lambda i: i.created_at)

    def live(self) -> list[Intention]:
        return [i for i in self.all() if i.is_live]

    def live_for(self, desire_id: str) -> Intention | None:
        """The desire's current (non-terminal) intention, if any — the most recent one."""
        matching = [i for i in self.live() if i.desire_id == desire_id]
        return matching[-1] if matching else None

    def history_for(self, desire_id: str) -> list[Intention]:
        return [i for i in self.all() if i.desire_id == desire_id]

    def __len__(self) -> int:
        return len(self._intentions)

    # ── Creation and lifecycle ───────────────────────────────────────────

    def add(self, intention: Intention) -> Intention:
        if intention.intention_id in self._intentions:
            raise ValueError(f"intention {intention.intention_id} already exists")
        self._intentions[intention.intention_id] = intention
        return intention

    def activate(self, intention_id: str, reason: str = "committed") -> bool:
        return self._move(intention_id, IntentionStatus.ACTIVE, reason, from_=(IntentionStatus.PENDING,))

    def suspend(self, intention_id: str, reason: str = "") -> bool:
        return self._move(intention_id, IntentionStatus.SUSPENDED, reason,
                          from_=(IntentionStatus.PENDING, IntentionStatus.ACTIVE))

    def resume(self, intention_id: str, beliefs: BeliefReader, reason: str = "preconditions hold again") -> bool:
        """Reactivate a suspended intention — only if its preconditions hold now."""
        intention = self._require(intention_id)
        if intention.status != IntentionStatus.SUSPENDED or not self.validate(intention, beliefs).valid:
            return False
        return self._move(intention_id, IntentionStatus.ACTIVE, reason, from_=(IntentionStatus.SUSPENDED,))

    def cancel(self, intention_id: str, reason: str = "") -> bool:
        return self._move(intention_id, IntentionStatus.CANCELLED, reason, from_=LIVE_STATUSES)

    def replace(self, old_id: str, new: Intention, reason: str = "") -> Intention:
        """Retire ``old_id`` as REPLACED and commit ``new`` in its place."""
        old = self._require(old_id)
        if not old.is_live:
            raise ValueError(f"intention {old_id} is {old.status.value} and cannot be replaced")
        self.add(new)
        old.replaced_by = new.intention_id
        self._move(old_id, IntentionStatus.REPLACED, reason, from_=LIVE_STATUSES)
        self.activate(new.intention_id, f"replaces {old_id}")
        return new

    def _move(self, intention_id: str, status: IntentionStatus, reason: str,
              from_: Iterable[IntentionStatus]) -> bool:
        intention = self._require(intention_id)
        if intention.status not in tuple(from_):
            return False
        intention.status = status
        intention.status_reason = reason
        intention.updated_at = utc_now()
        return True

    # ── Validation ───────────────────────────────────────────────────────

    @staticmethod
    def validate(intention: Intention, beliefs: BeliefReader) -> ValidationResult:
        """Check the intention's own preconditions and those of its pending actions."""
        conditions = list(intention.preconditions)
        for action in intention.pending_actions():
            conditions.extend(action.preconditions)
        seen: set[str] = set()
        failures: list[ConditionResult] = []
        for condition in conditions:
            result = condition.evaluate(beliefs)
            if not result.ok and result.explanation not in seen:
                seen.add(result.explanation)
                failures.append(result)
        return ValidationResult(valid=not failures, failures=failures)

    # ── Execution ────────────────────────────────────────────────────────

    def execute_next(
        self,
        intention_id: str,
        beliefs: BeliefReader,
        executor: ActionExecutor = default_executor,
        allow_manual: bool = False,
    ) -> ActionOutcome:
        """Run the next pending action of an ACTIVE intention.

        - If the intention is no longer valid it is suspended and nothing runs.
        - A ``manual`` action only runs when ``allow_manual`` is true (i.e.
          once the outside confirmation has been given).
        - When the last action finishes the intention becomes COMPLETED.
        """
        intention = self._require(intention_id)
        if intention.status != IntentionStatus.ACTIVE:
            return ActionOutcome(intention_id=intention_id, status="nothing_to_do",
                                 detail=f"intention is {intention.status.value}")
        pending = intention.pending_actions()
        if not pending:
            self._move(intention_id, IntentionStatus.COMPLETED, "all actions done", from_=(IntentionStatus.ACTIVE,))
            return ActionOutcome(intention_id=intention_id, status="nothing_to_do", detail="no pending actions")

        action = pending[0]
        validation = self.validate(intention, beliefs)
        if not validation.valid:
            self.suspend(intention_id, validation.explain())
            return ActionOutcome(intention_id=intention_id, action_id=action.action_id, kind=action.kind,
                                 status="precondition_failed", detail=validation.explain())
        if action.manual and not allow_manual:
            return ActionOutcome(intention_id=intention_id, action_id=action.action_id, kind=action.kind,
                                 status="awaiting_confirmation",
                                 detail=action.description or "waiting for confirmation")
        try:
            result = executor(action, intention)
        except Exception as exc:  # a failing step must not crash the agent
            action.status = ActionStatus.FAILED
            action.result = {"error": f"{type(exc).__name__}: {exc}"}
            self.suspend(intention_id, f"action {action.kind} failed: {exc}")
            return ActionOutcome(intention_id=intention_id, action_id=action.action_id, kind=action.kind,
                                 status="failed", detail=str(exc))

        action.status = ActionStatus.DONE
        action.result = dict(result or {})
        action.executed_at = utc_now()
        intention.updated_at = utc_now()
        if not intention.pending_actions():
            self._move(intention_id, IntentionStatus.COMPLETED, "all actions done", from_=(IntentionStatus.ACTIVE,))
        return ActionOutcome(intention_id=intention_id, action_id=action.action_id, kind=action.kind,
                             status="done", result=action.result)

    def run(self, intention_id: str, beliefs: BeliefReader,
            executor: ActionExecutor = default_executor) -> list[ActionOutcome]:
        """Execute automatic actions in order until a manual step, failure or the end."""
        outcomes: list[ActionOutcome] = []
        while True:
            outcome = self.execute_next(intention_id, beliefs, executor, allow_manual=False)
            if outcome.status == "nothing_to_do":
                break
            outcomes.append(outcome)
            if outcome.status != "done":
                break
        return outcomes

    def confirm(self, intention_id: str, beliefs: BeliefReader,
                executor: ActionExecutor = default_executor) -> ActionOutcome:
        """Give the outside confirmation: run the pending manual action (and anything after it)."""
        outcome = self.execute_next(intention_id, beliefs, executor, allow_manual=True)
        if outcome.status == "done":
            self.run(intention_id, beliefs, executor)
        return outcome

    # ── Persistence ──────────────────────────────────────────────────────

    def to_list(self) -> list[dict[str, Any]]:
        return [i.model_dump(mode="json") for i in self._intentions.values()]

    def load(self, items: Iterable[Intention | dict[str, Any]]) -> None:
        restored = {}
        for item in items:
            intention = item if isinstance(item, Intention) else Intention.model_validate(item)
            restored[intention.intention_id] = intention
        self._intentions = restored

    def clear(self) -> None:
        self._intentions.clear()

    def _require(self, intention_id: str) -> Intention:
        intention = self._intentions.get(intention_id)
        if intention is None:
            raise KeyError(f"unknown intention {intention_id}")
        return intention
