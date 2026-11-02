"""Reflex agent implementation.

A ``ReflexAgent`` extends ``BaseAgent`` by wiring it to a ``RuleEngine``.
On each event it:
    1. Logs the incoming event.
    2. Delegates evaluation to the rule engine.
    3. Records the resulting action (if any) to its internal log.

The agent has no internal memory of past events — it reacts only to
the current observation.  Memory, beliefs and planning will be layered
on top in later BDI and utility-based assignments.
"""

from __future__ import annotations

import logging

from smartnest.agents.base_agent import BaseAgent
from smartnest.core.models import ActionLog, EnvironmentalEvent
from smartnest.reasoning.rule_engine import RuleEngine

logger = logging.getLogger(__name__)


class ReflexAgent(BaseAgent):
    """A simple reflex agent that delegates event evaluation to a RuleEngine.

    Args:
        agent_id: Unique agent identifier.
        rule_engine: The ``RuleEngine`` instance to use.
        name: Optional human-readable name.
    """

    def __init__(
        self,
        agent_id: str,
        rule_engine: RuleEngine,
        name: str | None = None,
    ) -> None:
        super().__init__(agent_id=agent_id, name=name or agent_id)
        self.rule_engine = rule_engine

    # ── BaseAgent implementation ─────────────────────────────────────────

    def handle_event(self, event: EnvironmentalEvent) -> ActionLog | None:
        """Observe an event and trigger the appropriate rule.

        Args:
            event: The environmental event to process.

        Returns:
            An ``ActionLog`` if a rule fired, ``None`` otherwise.
        """
        if event is None:
            self.logger.warning("Received None event; skipping.")
            return None

        self.logger.info(
            "Observing event %s | type=%s | source=%s",
            event.event_id,
            event.event_type,
            event.source,
        )

        action_log = self.rule_engine.evaluate(event)

        if action_log:
            self.record_action(action_log)
        else:
            self.logger.info("No rule triggered for event %s", event.event_id)

        return action_log
