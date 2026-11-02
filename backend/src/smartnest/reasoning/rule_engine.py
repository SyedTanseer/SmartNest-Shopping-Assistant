"""Generic, reusable rule engine.

The ``RuleEngine`` is intentionally decoupled from SmartNest business
logic.  It operates on abstract ``Rule`` objects whose conditions and
actions are plain callables.  This means the engine can be reused for
any domain, not just home furnishing.

Execution semantics:
    1. Rules are evaluated in priority order (lower number = higher priority).
    2. The first rule whose condition returns ``True`` is triggered.
    3. Only one rule fires per event (single-shot reflex behaviour).
    4. If no rule matches, the engine logs that no action was taken.
    5. Malformed events are caught and logged without crashing.

This is a *reflex* agent pattern: condition → action with no internal
state tracking.  BDI and utility-based reasoning will be layered on top
in future assignments.
"""

from __future__ import annotations

import logging
from typing import Callable

from smartnest.core.models import ActionLog, ActionResultType, EnvironmentalEvent

logger = logging.getLogger(__name__)

# Type alias for condition and action callables.
ConditionFunc = Callable[[EnvironmentalEvent], bool]
ActionFunc = Callable[[EnvironmentalEvent], ActionLog]


class Rule:
    """A single condition-action rule.

    Attributes:
        rule_id:     Unique identifier.
        name:        Human-readable name.
        priority:    Lower number = higher priority.
        condition:   Callable that takes an event and returns bool.
        action:      Callable that takes an event and returns an ActionLog.
        explanation: Human-readable description of what the rule does.
        enabled:     Whether the rule is active (can be toggled at runtime).
    """

    def __init__(
        self,
        rule_id: str,
        name: str,
        priority: int,
        condition: ConditionFunc,
        action: ActionFunc,
        explanation: str = "",
    ) -> None:
        self.rule_id = rule_id
        self.name = name
        self.priority = priority
        self.condition = condition
        self.action = action
        self.explanation = explanation
        self.enabled = True

    def evaluate(self, event: EnvironmentalEvent) -> bool:
        """Return True if this rule's condition matches the event."""
        if not self.enabled:
            return False
        try:
            return self.condition(event)
        except Exception:
            logger.exception(
                "Condition error in rule %s for event %s", self.rule_id, event.event_id
            )
            return False

    def execute(self, event: EnvironmentalEvent) -> ActionLog:
        """Execute the action associated with this rule."""
        logger.debug("Executing action for rule %s on event %s", self.rule_id, event.event_id)
        try:
            return self.action(event)
        except Exception as exc:
            logger.exception(
                "Action error in rule %s for event %s", self.rule_id, event.event_id
            )
            return ActionLog(
                event_id=event.event_id,
                rule_id=self.rule_id,
                rule_name=self.name,
                action_description=f"Action failed: {exc}",
                result_type=ActionResultType.ERROR,
                result_message=f"Rule {self.rule_id} raised an exception: {exc}",
            )

    def __repr__(self) -> str:
        return f"<Rule id={self.rule_id} name={self.name} priority={self.priority}>"


class RuleEngine:
    """A reusable rule engine that evaluates rules against events.

    Rules are stored in a list and sorted by priority on evaluation.
    New rules can be registered at runtime via :meth:`register`.
    """

    def __init__(self) -> None:
        self._rules: list[Rule] = []

    # ── Registration ────────────────────────────────────────────────────────

    def register(self, rule: Rule) -> None:
        """Add a rule to the engine."""
        existing_ids = {r.rule_id for r in self._rules}
        if rule.rule_id in existing_ids:
            raise ValueError(f"Duplicate rule_id: {rule.rule_id}")
        self._rules.append(rule)
        self._rules.sort(key=lambda r: r.priority)
        logger.debug("Registered rule %s (priority=%d)", rule.rule_id, rule.priority)

    def unregister(self, rule_id: str) -> None:
        """Remove a rule by its identifier."""
        self._rules = [r for r in self._rules if r.rule_id != rule_id]

    def enable(self, rule_id: str) -> None:
        """Enable a rule by its identifier."""
        for r in self._rules:
            if r.rule_id == rule_id:
                r.enabled = True
                logger.debug("Enabled rule %s", rule_id)
                return

    def disable(self, rule_id: str) -> None:
        """Disable a rule by its identifier."""
        for r in self._rules:
            if r.rule_id == rule_id:
                r.enabled = False
                logger.debug("Disabled rule %s", rule_id)
                return

    # ── Introspection ─────────────────────────────────────────────────────

    @property
    def rules(self) -> list[Rule]:
        """Return a copy of the registered rules, sorted by priority."""
        return list(self._rules)

    # ── Evaluation ────────────────────────────────────────────────────────

    def evaluate(self, event: EnvironmentalEvent) -> ActionLog | None:
        """Evaluate all rules against an event.

        Returns the ActionLog from the first matching rule, or ``None``
        if no rule matches.  Malformed events are caught so they don't
        crash the simulation.
        """
        # Guard against None or invalid events
        if event is None:
            logger.warning("Received None event — skipping")
            return None

        if not isinstance(event, EnvironmentalEvent):
            logger.warning(
                "Received non-EnvironmentalEvent (%s) — skipping",
                type(event).__name__,
            )
            return None

        logger.info(
            "Evaluating %d rules against event %s (%s)",
            len(self._rules),
            event.event_id,
            event.event_type,
        )

        try:
            for rule in self._rules:
                matched = rule.evaluate(event)
                if matched:
                    logger.info(
                        "Rule triggered — %s (id=%s, priority=%d)",
                        rule.name,
                        rule.rule_id,
                        rule.priority,
                    )
                    return rule.execute(event)
        except Exception as exc:
            logger.exception("Unexpected error evaluating event %s: %s", event.event_id, exc)
            return ActionLog(
                event_id=event.event_id,
                rule_id="engine",
                rule_name="RuleEngine",
                action_description="Engine-level error",
                result_type=ActionResultType.ERROR,
                result_message=f"Evaluation failed: {exc}",
            )

        logger.info("No rule matched event %s", event.event_id)
        return None
