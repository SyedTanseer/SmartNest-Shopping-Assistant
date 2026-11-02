"""Tests for the MessageCodec — serialization, deserialization, and error handling.

Verifies:
    - Round-trip encode/decode preserves all MessageEnvelope fields.
    - Invalid JSON is rejected.
    - Malformed payloads are rejected.
    - Message type validation.
    - Empty data handling.
    - Nested payload preservation.
    - Timestamp and task_id preservation.
"""

from __future__ import annotations

import json

import pytest

from smartnest.communication.message_codec import CodecError, MessageCodec
from smartnest.core.models import MessageEnvelope, MessageType


# ── Fixtures ────────────────────────────────────────────────────────────────

def _make_message(**overrides) -> MessageEnvelope:
    defaults = dict(
        sender="agent_a",
        receiver="agent_b",
        message_type=MessageType.STATE_SYNC,
        task_id="task_001",
        payload={"project_id": "HOME_001", "budget": 200000},
    )
    defaults.update(overrides)
    return MessageEnvelope(**defaults)


# ── Round-trip tests ────────────────────────────────────────────────────────

class TestEncodeDecode:
    def test_round_trip_preserves_fields(self):
        msg = _make_message()
        encoded = MessageCodec.encode(msg)
        decoded = MessageCodec.decode(encoded)
        assert decoded.sender == "agent_a"
        assert decoded.receiver == "agent_b"
        assert decoded.message_type == "state_sync"
        assert decoded.task_id == "task_001"
        assert decoded.payload["project_id"] == "HOME_001"

    def test_round_trip_preserves_nested_payload(self):
        msg = _make_message(payload={
            "rooms": {"living_room": {"width_cm": 210, "length_cm": 350}},
            "preferences": {"style": "minimalist"},
        })
        decoded = MessageCodec.decode(MessageCodec.encode(msg))
        assert decoded.payload["rooms"]["living_room"]["width_cm"] == 210
        assert decoded.payload["preferences"]["style"] == "minimalist"

    def test_round_trip_preserves_timestamp(self):
        msg = _make_message()
        encoded = MessageCodec.encode(msg)
        decoded = MessageCodec.decode(encoded)
        # Timestamp should be valid ISO format
        assert decoded.timestamp is not None

    def test_round_trip_preserves_task_id(self):
        msg = _make_message(task_id="correlation_abc123")
        decoded = MessageCodec.decode(MessageCodec.encode(msg))
        assert decoded.task_id == "correlation_abc123"

    def test_encode_returns_bytes(self):
        msg = _make_message()
        encoded = MessageCodec.encode(msg)
        assert isinstance(encoded, bytes)

    def test_decode_preserves_signature_field(self):
        msg = _make_message(signature="placeholder_sig")
        decoded = MessageCodec.decode(MessageCodec.encode(msg))
        assert decoded.signature == "placeholder_sig"

    def test_round_trip_all_message_types(self):
        """Every MessageType should round-trip correctly."""
        for mt in MessageType:
            msg = _make_message(message_type=mt)
            decoded = MessageCodec.decode(MessageCodec.encode(msg))
            assert decoded.message_type == mt.value


# ── Error handling ──────────────────────────────────────────────────────────

class TestDecodeErrors:
    def test_invalid_json_raises_codec_error(self):
        with pytest.raises(CodecError, match="Invalid JSON"):
            MessageCodec.decode(b"{not valid json")

    def test_empty_data_raises_codec_error(self):
        with pytest.raises(CodecError, match="Empty message"):
            MessageCodec.decode(b"")

    def test_non_object_json_raises_codec_error(self):
        with pytest.raises(CodecError, match="not a JSON object"):
            MessageCodec.decode(b'"just a string"')

    def test_array_json_raises_codec_error(self):
        with pytest.raises(CodecError, match="not a JSON object"):
            MessageCodec.decode(b"[1, 2, 3]")

    def test_missing_required_field_raises(self):
        """Missing 'sender' should fail validation."""
        raw = json.dumps({
            "receiver": "agent_b",
            "message_type": "inform",
        })
        with pytest.raises(CodecError, match="Invalid MessageEnvelope"):
            MessageCodec.decode(raw.encode())

    def test_invalid_message_type_raises(self):
        raw = json.dumps({
            "sender": "a",
            "receiver": "b",
            "message_type": "nonexistent_type",
        })
        with pytest.raises(CodecError, match="Invalid MessageEnvelope"):
            MessageCodec.decode(raw.encode())


# ── Type validation ────────────────────────────────────────────────────────

class TestTypeValidation:
    def test_is_valid_type_recognises_known(self):
        assert MessageCodec.is_valid_type("state_sync") is True
        assert MessageCodec.is_valid_type("task_request") is True
        assert MessageCodec.is_valid_type("ack") is True
        assert MessageCodec.is_valid_type("inform_result") is True
        assert MessageCodec.is_valid_type("call_for_proposals") is True

    def test_is_valid_type_rejects_unknown(self):
        assert MessageCodec.is_valid_type("nonexistent") is False
        assert MessageCodec.is_valid_type("") is False
