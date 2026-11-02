"""Tests for the MessageRouter — handler dispatch, unsupported types, errors.

Verifies:
    - Registered handlers are called for matching message types.
    - Unregistered types return a FAILURE response.
    - Handler exceptions are caught and returned as FAILURE.
    - Register/unregister lifecycle.
    - Task ID correlation passes through the router.
"""

from __future__ import annotations

import pytest

from smartnest.communication.message_router import MessageRouter
from smartnest.core.models import MessageEnvelope, MessageType


def _make_msg(msg_type: MessageType = MessageType.INFORM, **kw) -> MessageEnvelope:
    defaults = dict(
        sender="agent_a",
        receiver="agent_b",
        message_type=msg_type,
        task_id="task_001",
        payload={},
    )
    defaults.update(kw)
    return MessageEnvelope(**defaults)


class TestRegistration:
    def test_register_handler(self):
        router = MessageRouter()
        router.register(MessageType.INFORM, lambda m: None)
        assert "inform" in router.registered_types

    def test_unregister_handler(self):
        router = MessageRouter()
        router.register(MessageType.INFORM, lambda m: None)
        router.unregister(MessageType.INFORM)
        assert "inform" not in router.registered_types

    def test_multiple_types_registered(self):
        router = MessageRouter()
        router.register(MessageType.INFORM, lambda m: None)
        router.register(MessageType.STATE_SYNC, lambda m: None)
        router.register(MessageType.TASK_REQUEST, lambda m: None)
        assert len(router.registered_types) == 3


class TestDispatch:
    def test_handler_called_with_message(self):
        router = MessageRouter()
        received: list[MessageEnvelope] = []

        def handler(msg):
            received.append(msg)
            return None

        router.register(MessageType.INFORM, handler)
        msg = _make_msg(MessageType.INFORM)
        router.route(msg)
        assert len(received) == 1
        assert received[0] is msg

    def test_handler_returns_response(self):
        router = MessageRouter()
        reply = _make_msg(MessageType.ACK, sender="agent_b", receiver="agent_a")
        router.register(MessageType.STATE_SYNC, lambda m: reply)
        msg = _make_msg(MessageType.STATE_SYNC)
        response = router.route(msg)
        assert response is reply

    def test_task_id_preserved_in_response(self):
        router = MessageRouter()

        def handler(msg):
            return MessageEnvelope(
                sender="agent_b",
                receiver=msg.sender,
                message_type=MessageType.ACK,
                task_id=msg.task_id,
                payload={},
            )

        router.register(MessageType.STATE_SYNC, handler)
        msg = _make_msg(MessageType.STATE_SYNC, task_id="correlation_123")
        response = router.route(msg)
        assert response.task_id == "correlation_123"


class TestUnsupportedType:
    def test_unsupported_type_returns_failure(self):
        router = MessageRouter()
        msg = _make_msg(MessageType.INFORM)
        response = router.route(msg)
        assert response is not None
        assert response.message_type == "failure"
        assert response.payload["reason"] == "unsupported_message_type"
        assert response.payload["unsupported_type"] == "inform"

    def test_unsupported_type_preserves_task_id(self):
        router = MessageRouter()
        msg = _make_msg(MessageType.CFP, task_id="task_xyz")
        response = router.route(msg)
        assert response.task_id == "task_xyz"


class TestHandlerException:
    def test_handler_exception_returns_failure(self):
        router = MessageRouter()

        def bad_handler(msg):
            raise RuntimeError("handler crashed")

        router.register(MessageType.INFORM, bad_handler)
        msg = _make_msg(MessageType.INFORM)
        response = router.route(msg)
        assert response is not None
        assert response.message_type == "failure"
        assert "handler_exception" in response.payload["reason"]
        assert "handler crashed" in response.payload["error"]

    def test_handler_returning_none_is_none(self):
        """If a handler returns None, the router returns None (no response)."""
        router = MessageRouter()
        router.register(MessageType.INFORM, lambda m: None)
        msg = _make_msg(MessageType.INFORM)
        response = router.route(msg)
        assert response is None
