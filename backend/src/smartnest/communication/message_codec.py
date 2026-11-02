"""Message serialization and deserialization for TCP transport.

Uses JSON as the wire format.  Every message is a validated
``MessageEnvelope`` — there is no raw or unstructured wire data.

The codec is intentionally stateless so it can be reused by any agent
or transport implementation without shared mutable state.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import ValidationError

from smartnest.core.models import MessageEnvelope, MessageType

logger = logging.getLogger(__name__)

# Maximum accepted message size: 16 MiB.  Oversized frames are rejected
# to prevent memory exhaustion attacks.
MAX_MESSAGE_SIZE: int = 16 * 1024 * 1024


class CodecError(Exception):
    """Raised when a message cannot be encoded or decoded."""


class MessageCodec:
    """Serialize ``MessageEnvelope`` to/from JSON bytes.

    All methods are static so the codec can be used as a drop-in utility
    without instantiation.  An instance is also provided for convenience.
    """

    @staticmethod
    def encode(message: MessageEnvelope) -> bytes:
        """Serialize a ``MessageEnvelope`` to JSON bytes (UTF-8).

        Args:
            message: The envelope to serialize.

        Returns:
            UTF-8 JSON bytes.

        Raises:
            CodecError: If serialization fails.
        """
        try:
            json_str = message.model_dump_json()
            return json_str.encode("utf-8")
        except Exception as exc:
            raise CodecError(f"Failed to encode message: {exc}") from exc

    @staticmethod
    def decode(data: bytes) -> MessageEnvelope:
        """Deserialize JSON bytes into a validated ``MessageEnvelope``.

        Args:
            data: UTF-8 JSON bytes.

        Returns:
            A validated ``MessageEnvelope``.

        Raises:
            CodecError: If the JSON is invalid or the payload doesn't
                       conform to the ``MessageEnvelope`` schema.
        """
        if not data:
            raise CodecError("Empty message data")

        try:
            raw = json.loads(data)
        except json.JSONDecodeError as exc:
            raise CodecError(f"Invalid JSON: {exc}") from exc

        if not isinstance(raw, dict):
            raise CodecError("Message root is not a JSON object")

        try:
            return MessageEnvelope.model_validate(raw)
        except ValidationError as exc:
            raise CodecError(f"Invalid MessageEnvelope: {exc}") from exc

    @staticmethod
    def is_valid_type(message_type: str) -> bool:
        """Check if a string is a recognised ``MessageType`` value."""
        return message_type in {t.value for t in MessageType}
