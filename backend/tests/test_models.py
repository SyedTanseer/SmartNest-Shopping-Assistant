"""Tests for core data models.

Verify serialization, defaults and validation of Pydantic models.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from smartnest.core.models import (
    ActionLog,
    ActionResultType,
    EnvironmentalEvent,
    EventType,
    MessageEnvelope,
    MessageType,
)


class TestEnvironmentalEvent:
    def test_defaults_populated(self):
        event = EnvironmentalEvent(
            event_type=EventType.BUDGET_UPDATE,
            payload={"total_budget": 100},
        )
        assert event.event_id.startswith("evt_")
        assert event.source == "environment"
        assert event.timestamp is not None
        assert event.payload["total_budget"] == 100

    def test_serializable_to_dict(self):
        event = EnvironmentalEvent(
            event_type=EventType.BUDGET_UPDATE,
            payload={"x": 1},
        )
        d = event.model_dump()
        assert d["event_type"] == "budget_update"
        assert d["payload"]["x"] == 1
        assert "timestamp" in d

    def test_json_serializable(self):
        """model_dump_json should produce valid JSON string."""
        event = EnvironmentalEvent(
            event_type=EventType.PRODUCT_VALIDATION,
            payload={"price": 1000},
        )
        json_str = event.model_dump_json()
        assert isinstance(json_str, str)
        assert "product_validation" in json_str

    def test_custom_event_id(self):
        event = EnvironmentalEvent(
            event_id="custom_123",
            event_type=EventType.BUDGET_UPDATE,
        )
        assert event.event_id == "custom_123"


class TestActionLog:
    def test_action_log_creation(self):
        log = ActionLog(
            event_id="evt_1",
            rule_id="R001",
            rule_name="Test Rule",
            action_description="Test action",
            result_type=ActionResultType.WARNING,
            result_message="Test warning",
        )
        assert log.log_id.startswith("log_")
        assert log.result_type == "warning"

    def test_action_log_serializable(self):
        log = ActionLog(
            event_id="evt_1",
            rule_id="R001",
            rule_name="Test",
            action_description="desc",
            result_type=ActionResultType.INFO,
            result_message="msg",
            result_details={"key": "value"},
        )
        d = log.model_dump()
        assert d["result_details"]["key"] == "value"


class TestMessageEnvelope:
    def test_message_envelope_defaults(self):
        msg = MessageEnvelope(
            sender="agent_a",
            receiver="agent_b",
            message_type=MessageType.INFORM,
        )
        assert msg.message_id.startswith("msg_")
        assert msg.schema_version == "1.0"
        assert msg.signature is None
        assert msg.task_id is None

    def test_message_envelope_serializable(self):
        msg = MessageEnvelope(
            sender="a",
            receiver="b",
            message_type=MessageType.CFP,
            task_id="task_1",
            payload={"proposal": 100},
        )
        d = msg.model_dump()
        assert d["sender"] == "a"
        assert d["message_type"] == "call_for_proposals"
        assert d["task_id"] == "task_1"

    def test_cnp_message_types_available(self):
        """Verify Contract Net Protocol message types are defined."""
        assert MessageType.CFP.value == "call_for_proposals"
        assert MessageType.PROPOSAL.value == "proposal"
        assert MessageType.AWARD.value == "award"
        assert MessageType.INFORM_DONE.value == "inform_done"
        assert MessageType.INFORM_RESULT.value == "inform_result"

    def test_assignment_2_2_message_types_available(self):
        """Verify Assignment 2.2 message types are defined."""
        assert MessageType.STATE_SYNC.value == "state_sync"
        assert MessageType.TASK_REQUEST.value == "task_request"
        assert MessageType.ACK.value == "ack"

    def test_state_sync_envelope_round_trip(self):
        """STATE_SYNC message should round-trip through serialization."""
        msg = MessageEnvelope(
            sender="agent_a",
            receiver="agent_b",
            message_type=MessageType.STATE_SYNC,
            task_id="sync_001",
            payload={"project_id": "HOME_001", "budget": 200000},
        )
        d = msg.model_dump()
        assert d["message_type"] == "state_sync"
        assert d["task_id"] == "sync_001"
        assert d["payload"]["project_id"] == "HOME_001"
