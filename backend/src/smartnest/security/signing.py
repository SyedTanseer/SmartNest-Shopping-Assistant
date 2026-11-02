"""Signing and verifying ``MessageEnvelope`` objects with Ed25519.

What is signed
--------------
A fixed domain-separation prefix followed by the canonical JSON of **every envelope field except the signature**:
``message_id, sender, receiver, message_type, task_id, payload, timestamp, schema_version``.  Changing any of them -
the sender, the receiver, the command type, one nested payload value, the timestamp - invalidates the signature.

Canonical form: keys sorted, no insignificant whitespace, UTF-8, ``NaN``/``Infinity`` refused, timestamps rendered as
UTC ISO-8601 with microseconds.  The same envelope therefore produces the same bytes before and after a trip through
JSON, so a message verifies on the receiving side.

What a signature gives and does not give
----------------------------------------
Authenticity (the holder of the sender's private key produced it) and integrity (it was not modified).  It does **not**
encrypt anything, and by itself it does not stop replay of a captured message: that is the job of
:class:`~smartnest.security.replay.ReplayGuard`.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from smartnest.core.models import MessageEnvelope
from smartnest.security.errors import SigningError
from smartnest.security.keys import SIGNATURE_LENGTH, AgentKeyPair, verify_raw
from smartnest.security.registry import TrustedAgent, TrustedAgentRegistry

DOMAIN = b"smartnest.message.v1\n"

# Verification outcome codes (also used in the audit log).
OK = "ok"
UNSIGNED = "unsigned"
MALFORMED_SIGNATURE = "malformed_signature"
UNKNOWN_SENDER = "unknown_sender"
REVOKED_SENDER = "revoked_sender"
BAD_SIGNATURE = "bad_signature"


def _utc_iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds")


def canonical_bytes(envelope: MessageEnvelope) -> bytes:
    """The exact bytes that are signed: domain prefix + canonical JSON of all fields except ``signature``."""
    body: dict[str, Any] = {
        "message_id": envelope.message_id,
        "sender": envelope.sender,
        "receiver": envelope.receiver,
        "message_type": envelope.message_type,
        "task_id": envelope.task_id,
        "payload": envelope.payload,
        "timestamp": _utc_iso(envelope.timestamp),
        "schema_version": envelope.schema_version,
    }
    try:
        text = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise SigningError(f"the message cannot be canonicalised: {exc}") from None
    return DOMAIN + text.encode("utf-8")


def sign_envelope(envelope: MessageEnvelope, keypair: AgentKeyPair) -> MessageEnvelope:
    """A copy of ``envelope`` carrying the sender's signature.

    Raises:
        SigningError: if the envelope names a different sender than the key pair's agent (no signing on someone
            else's behalf), or cannot be canonicalised.
    """
    if envelope.sender != keypair.agent_id:
        raise SigningError(f"cannot sign as {envelope.sender!r} with the key of {keypair.agent_id!r}")
    signature = keypair.sign(canonical_bytes(envelope))
    return envelope.model_copy(update={"signature": base64.b64encode(signature).decode("ascii")})


@dataclass(frozen=True)
class VerificationResult:
    """Outcome of verifying a message's signature and the sender's identity."""

    ok: bool
    code: str                      # OK or one of the failure codes above
    detail: str = ""
    agent: TrustedAgent | None = None

    def __bool__(self) -> bool:
        return self.ok


def verify_envelope(envelope: MessageEnvelope, registry: TrustedAgentRegistry) -> VerificationResult:
    """Check that ``envelope`` was signed by the (trusted, non-revoked) agent it names as its sender.

    The sender's public key comes only from the registry - never from the message itself.
    """
    if not envelope.signature:
        return VerificationResult(False, UNSIGNED, "the message carries no signature")
    try:
        signature = base64.b64decode(envelope.signature, validate=True)
    except Exception:
        return VerificationResult(False, MALFORMED_SIGNATURE, "the signature is not valid base64")
    if len(signature) != SIGNATURE_LENGTH:
        return VerificationResult(False, MALFORMED_SIGNATURE, f"the signature is not {SIGNATURE_LENGTH} bytes")

    agent = registry.get(envelope.sender)
    if agent is None:
        return VerificationResult(False, UNKNOWN_SENDER, f"{envelope.sender!r} is not a trusted agent")
    if agent.revoked:
        return VerificationResult(False, REVOKED_SENDER, f"the key of {envelope.sender!r} has been revoked", agent)
    try:
        data = canonical_bytes(envelope)
    except SigningError as exc:
        return VerificationResult(False, BAD_SIGNATURE, str(exc), agent)
    if not verify_raw(agent.public_key(), signature, data):
        return VerificationResult(False, BAD_SIGNATURE, "the signature does not match the message and sender", agent)
    return VerificationResult(True, OK, "", agent)
