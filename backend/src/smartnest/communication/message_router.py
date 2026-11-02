"""Message routing and handler dispatch.

The ``MessageRouter`` dispatches incoming ``MessageEnvelope`` objects to
registered handlers keyed by ``MessageType``.  Each handler is a callable
that receives the incoming message and returns an optional response
message (or ``None`` if no response is needed).

This keeps routing logic separate from transport and business logic.
New handlers can be registered at runtime without modifying the router
or the server.

Design notes:
    - The router does **not** execute arbitrary code from the network.
      Only explicitly registered handlers are called.
    - Unsupported message types produce a ``FAILURE`` response rather
      than being silently ignored.
    - The router is transport-agnostic — it operates on
      ``MessageEnvelope`` objects, not raw bytes.
"""

from __future__ import annotations

import logging
from typing import Callable

from smartnest.core.models import MessageEnvelope, MessageType

logger = logging.getLogger(__name__)

# Type alias: handler takes an incoming MessageEnvelope and returns an
# optional outgoing MessageEnvelope as the response.
MessageHandler = Callable[[MessageEnvelope], MessageEnvelope | None]


class MessageRouter:
    """Routes ``MessageEnvelope`` objects to registered handlers by type.

    Handlers are keyed by ``MessageType`` value (string).  When a message
    arrives, the router looks up the handler for its ``message_type`` and
    invokes it.  If no handler is registered, the router returns a
    ``FAILURE`` response.
    """

    def __init__(self, agent_id: str = "router") -> None:
        self._handlers: dict[str, MessageHandler] = {}
        self._agent_id = agent_id

    # ── Registration ──────────────────────────────────────────────────────

    def register(self, message_type: MessageType, handler: MessageHandler) -> None:
        """Register a handler for a message type.

        Args:
            message_type: The ``MessageType`` to handle.
            handler: Callable that takes an incoming envelope and returns
                     an optional response envelope.
        """
        key = message_type.value
        self._handlers[key] = handler
        logger.debug("Router registered handler for %s", key)

    def unregister(self, message_type: MessageType) -> None:
        """Remove a handler by message type."""
        self._handlers.pop(message_type.value, None)

    @property
    def registered_types(self) -> list[str]:
        """Return a list of registered message type strings."""
        return list(self._handlers.keys())

    # ── Dispatch ─────────────────────────────────────────────────────────

    def route(self, message: MessageEnvelope) -> MessageEnvelope | None:
        """Dispatch a message to its registered handler.

        Args:
            message: The incoming ``MessageEnvelope``.

        Returns:
            The handler's response ``MessageEnvelope``, or ``None`` if
            the handler chose not to respond.  If no handler is
            registered, a ``FAILURE`` envelope is returned.
        """
        msg_type = message.message_type
        handler = self._handlers.get(msg_type)

        if handler is None:
            logger.warning(
                "No handler registered for message type %s (from %s)",
                msg_type,
                message.sender,
            )
            return MessageEnvelope(
                sender=self._agent_id,
                receiver=message.sender,
                message_type=MessageType.FAILURE,
                task_id=message.task_id,
                payload={
                    "reason": "unsupported_message_type",
                    "unsupported_type": msg_type,
                },
            )

        logger.debug(
            "Routing message %s (type=%s, task=%s) to handler",
            message.message_id,
            msg_type,
            message.task_id,
        )

        try:
            return handler(message)
        except Exception as exc:
            logger.exception(
                "Handler for %s raised an exception: %s", msg_type, exc
            )
            return MessageEnvelope(
                sender=self._agent_id,
                receiver=message.sender,
                message_type=MessageType.FAILURE,
                task_id=message.task_id,
                payload={
                    "reason": "handler_exception",
                    "error": str(exc),
                },
            )
