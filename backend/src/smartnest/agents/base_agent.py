"""Base agent abstraction for SmartNest.

All agents — reflex, BDI, utility-based, etc. — will inherit from
``BaseAgent`` so they share a common identity, state container and
logging interface.  This keeps the architecture consistent when new
agent types are introduced in later assignments.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from smartnest.core.models import ActionLog, EnvironmentalEvent


class BaseAgent(ABC):
    """Abstract base class for all SmartNest agents.

    Each agent has:
        - A unique ``agent_id`` and human-readable ``name``.
        - An internal ``state`` dictionary (agent beliefs in future BDI).
        - An ``action_log`` recording every action it has taken.
        - A logger scoped to the agent's name.

    Subclasses must implement :meth:`handle_event`.
    """

    def __init__(self, agent_id: str, name: str | None = None) -> None:
        self.agent_id: str = agent_id
        self.name: str = name or agent_id
        self.state: dict[str, Any] = {}
        self.action_log: list[ActionLog] = []
        self.logger = logging.getLogger(f"smartnest.agent.{self.name}")

    @abstractmethod
    def handle_event(self, event: EnvironmentalEvent) -> ActionLog | None:
        """Process an environmental event and optionally return an action log.

        Args:
            event: The environmental event to process.

        Returns:
            An ``ActionLog`` if an action was triggered, ``None`` otherwise.
        """
        ...

    def record_action(self, log: ActionLog) -> None:
        """Append an action log to the agent's history."""
        self.action_log.append(log)
        self.logger.info(
            "Action logged — rule=%s result=%s message=%s",
            log.rule_name,
            log.result_type,
            log.result_message,
        )

    def reset(self) -> None:
        """Clear agent state and action log."""
        self.state.clear()
        self.action_log.clear()
        self.logger.debug("Agent %s reset", self.name)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} id={self.agent_id} name={self.name}>"
