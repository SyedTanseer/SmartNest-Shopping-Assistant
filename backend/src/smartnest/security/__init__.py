"""Agent authentication and message integrity (Assignment 5.1).

    keys.py           Ed25519 key pairs, secure storage, fingerprints
    signing.py        canonical signing and verification of MessageEnvelope
    registry.py       trusted agents: public keys and roles (public data only)
    policy.py         role-based authorization of message types and task types
    replay.py         freshness window and one-time message ids
    audit.py          audit trail of accepted and rejected messages
    secure_router.py  SecureMessageRouter: authenticate + authorize before any handler runs
    channel.py        SecureChannel: sign requests, verify replies

Signatures give authenticity and integrity.  They do not encrypt: anyone on the wire can still read a message.
This package depends only on ``core`` and the router/transport interfaces; no other layer depends on it.
"""

from smartnest.security.audit import AuditEvent, AuditLog
from smartnest.security.channel import SecureChannel, SecureReply
from smartnest.security.errors import KeyManagementError, RegistryError, SecurityError, SigningError
from smartnest.security.keys import AgentKeyPair, fingerprint_of
from smartnest.security.policy import COMMAND_TYPES, AuthorizationPolicy, Decision
from smartnest.security.registry import TrustedAgent, TrustedAgentRegistry
from smartnest.security.replay import ReplayGuard
from smartnest.security.secure_router import SecureMessageRouter
from smartnest.security.signing import VerificationResult, canonical_bytes, sign_envelope, verify_envelope

__all__ = [
    "COMMAND_TYPES", "AgentKeyPair", "AuditEvent", "AuditLog", "AuthorizationPolicy", "Decision", "KeyManagementError",
    "RegistryError", "ReplayGuard", "SecureChannel", "SecureMessageRouter", "SecureReply", "SecurityError",
    "SigningError", "TrustedAgent", "TrustedAgentRegistry", "VerificationResult", "canonical_bytes",
    "fingerprint_of", "sign_envelope", "verify_envelope",
]
