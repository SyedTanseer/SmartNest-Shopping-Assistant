"""Client side: send signed messages and verify the signed replies.

``SecureChannel`` signs outgoing messages with the caller's key and checks that the reply really comes from the trusted
server it addressed (signature valid, sender is the expected agent, not a replay of an old reply).  A reply that fails
these checks is returned together with a failed :class:`VerificationResult`, never silently trusted.
"""

from __future__ import annotations

from dataclasses import dataclass

from smartnest.communication.socket_transport import SocketTransport
from smartnest.core.models import MessageEnvelope
from smartnest.security.keys import AgentKeyPair
from smartnest.security.registry import TrustedAgentRegistry
from smartnest.security.replay import ReplayGuard
from smartnest.security.signing import VerificationResult, sign_envelope, verify_envelope


@dataclass(frozen=True)
class SecureReply:
    message: MessageEnvelope
    verification: VerificationResult

    @property
    def trusted(self) -> bool:
        return self.verification.ok


class SecureChannel:
    """One TCP connection that signs what it sends and verifies what it receives."""

    def __init__(self, identity: AgentKeyPair, registry: TrustedAgentRegistry, host: str, port: int,
                 *, expected_server: str, timeout: float = 10.0) -> None:
        self.identity = identity
        self.registry = registry
        self.expected_server = expected_server
        self._replay = ReplayGuard()
        self._transport = SocketTransport.connect(host, port, timeout=timeout)

    def request(self, message: MessageEnvelope, *, sign: bool = True) -> SecureReply:
        """Send ``message`` (signed as this agent unless ``sign=False``) and return the verified reply."""
        outgoing = sign_envelope(message, self.identity) if sign else message
        self._transport.send_message(outgoing)
        reply = self._transport.receive_message()
        verification = verify_envelope(reply, self.registry)
        if verification.ok and reply.sender != self.expected_server:
            verification = VerificationResult(False, "unexpected_sender",
                                              f"expected a reply from {self.expected_server!r}, got {reply.sender!r}")
        if verification.ok and reply.receiver != self.identity.agent_id:
            verification = VerificationResult(False, "wrong_receiver", "the reply is addressed to someone else")
        if verification.ok:
            fresh, code, detail = self._replay.check(reply)
            if not fresh:
                verification = VerificationResult(False, code, detail)
        return SecureReply(reply, verification)

    def close(self) -> None:
        self._transport.close()

    def __enter__(self) -> "SecureChannel":
        return self

    def __exit__(self, *args) -> None:
        self.close()
