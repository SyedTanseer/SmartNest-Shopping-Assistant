"""``SecureMessageRouter``: authenticate and authorize every message before any handler sees it.

It wraps an ordinary :class:`~smartnest.communication.message_router.MessageRouter` and has the same ``route(message)``
method, so it can be handed to the existing :class:`RemoteAgentServer` without changing the server or the transport.

Checks, in this order (the first failure rejects the message and **no handler runs**):

    1. addressed to this agent                    -> else  wrong_receiver   (a message meant for someone else is not ours)
    2. carries a well-formed signature            -> else  unsigned / malformed_signature
    3. the sender is a trusted, non-revoked agent -> else  unknown_sender / revoked_sender
    4. the signature verifies against the sender's registered public key -> else bad_signature (forged or modified)
    5. fresh and not seen before                  -> else  stale / from_the_future / replay
    6. the sender's roles allow this message type -> else  not_authorized
    7. only now: dispatch to the wrapped router

The reply to the sender is signed with this agent's own key, so the caller can verify it too.  What a rejected peer is
told is deliberately coarse (``authentication_failed``, ``message_rejected``, ``not_authorized``); the precise code and
reason go to the audit log only, so an attacker cannot probe which check failed.
"""

from __future__ import annotations

import logging

from smartnest.communication.message_router import MessageRouter
from smartnest.core.models import MessageEnvelope, MessageType
from smartnest.security.audit import AuditLog
from smartnest.security.keys import AgentKeyPair
from smartnest.security.policy import AuthorizationPolicy
from smartnest.security.registry import TrustedAgentRegistry
from smartnest.security.replay import ReplayGuard
from smartnest.security.signing import sign_envelope, verify_envelope

logger = logging.getLogger(__name__)

WRONG_RECEIVER = "wrong_receiver"
NOT_AUTHORIZED = "not_authorized"

#: What the peer is told, by internal code.
_PUBLIC_REASON = {
    "unsigned": "authentication_failed", "malformed_signature": "authentication_failed",
    "unknown_sender": "authentication_failed", "revoked_sender": "authentication_failed",
    "bad_signature": "authentication_failed",
    WRONG_RECEIVER: "message_rejected", "stale": "message_rejected", "from_the_future": "message_rejected",
    "replay": "message_rejected", NOT_AUTHORIZED: "not_authorized",
}


class SecureMessageRouter:
    """A router that only dispatches authenticated, fresh, authorized messages.

    Args:
        identity: This agent's key pair (its id must be the id other agents address messages to).
        inner: The router that holds the real handlers.
        registry: The trusted agents (public keys + roles).
        policy: Role permissions (default: :class:`AuthorizationPolicy`).
        replay_guard: Freshness / uniqueness checks (default: 120 s window).
        audit: Where decisions are recorded (default: a new :class:`AuditLog`).
    """

    def __init__(self, identity: AgentKeyPair, inner: MessageRouter, registry: TrustedAgentRegistry, *,
                 policy: AuthorizationPolicy | None = None, replay_guard: ReplayGuard | None = None,
                 audit: AuditLog | None = None) -> None:
        self.identity = identity
        self._inner = inner
        self.registry = registry
        self.policy = policy or AuthorizationPolicy()
        self.replay_guard = replay_guard or ReplayGuard()
        self.audit = audit or AuditLog()

    # ── The router interface used by RemoteAgentServer ──────────────────

    def route(self, message: MessageEnvelope) -> MessageEnvelope | None:
        """Reject, or dispatch and sign the reply."""
        if message.receiver != self.identity.agent_id:
            return self._reject(message, WRONG_RECEIVER,
                                f"the message is addressed to {message.receiver!r}, not to {self.identity.agent_id!r}")

        verification = verify_envelope(message, self.registry)
        fingerprint = verification.agent.fingerprint if verification.agent else None
        if not verification.ok:
            return self._reject(message, verification.code, verification.detail, fingerprint)

        fresh, code, detail = self.replay_guard.check(message)   # only after the signature is known good
        if not fresh:
            return self._reject(message, code, detail, fingerprint)

        decision = self.policy.check(verification.agent, message)
        if not decision.allowed:
            return self._reject(message, NOT_AUTHORIZED, decision.reason, fingerprint)

        self.audit.record("accepted", "ok", "", message, fingerprint)
        response = self._inner.route(message)
        return self._sign_reply(response, message) if response is not None else None

    # ── Internals ────────────────────────────────────────────────────────

    def _reject(self, message: MessageEnvelope, code: str, detail: str, fingerprint: str | None = None) -> MessageEnvelope:
        self.audit.record("rejected", code, detail, message, fingerprint)
        failure = MessageEnvelope(
            sender=self.identity.agent_id, receiver=str(message.sender)[:128] or "unknown",
            message_type=MessageType.FAILURE, task_id=message.task_id,
            payload={"reason": _PUBLIC_REASON.get(code, "message_rejected")})
        return sign_envelope(failure, self.identity)

    def _sign_reply(self, response: MessageEnvelope, request: MessageEnvelope) -> MessageEnvelope:
        """Sign the handler's reply as this agent (the reply's sender is always our own identity)."""
        reply = response.model_copy(update={"sender": self.identity.agent_id, "receiver": request.sender,
                                            "signature": None})
        return sign_envelope(reply, self.identity)
