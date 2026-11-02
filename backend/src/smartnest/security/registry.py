"""The registry of trusted agents: who may be believed, with which public key and which roles.

The registry holds **public keys only**, so it can be stored and shared freely (JSON, MongoDB).  A receiver decides
whom to trust by what is in its registry - never by what a message claims about itself.

Rules enforced here:
  * one agent id has one key; replacing it needs ``replace=True`` (a deliberate rotation);
  * one public key cannot be registered under two ids (no aliasing);
  * a revoked agent stays in the registry, marked revoked, so its messages are rejected with a clear reason.
"""

from __future__ import annotations

import base64
import json
import threading
from dataclasses import dataclass
from typing import Iterable

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from smartnest.security.errors import RegistryError
from smartnest.security.keys import (
    AgentKeyPair,
    fingerprint_of,
    public_key_from_b64,
    public_key_from_bytes,
    validate_agent_id,
)



@dataclass(frozen=True)
class TrustedAgent:
    """A trusted agent: id, public key (base64) and roles (what the authorization policy keys on)."""

    agent_id: str
    public_key_b64: str
    roles: frozenset[str] = frozenset()
    revoked: bool = False

    def public_key(self) -> Ed25519PublicKey:
        return public_key_from_b64(self.public_key_b64)

    @property
    def fingerprint(self) -> str:
        return fingerprint_of(base64.b64decode(self.public_key_b64))


class TrustedAgentRegistry:
    """Thread-safe store of :class:`TrustedAgent` entries."""

    def __init__(self) -> None:
        self._agents: dict[str, TrustedAgent] = {}
        self._lock = threading.RLock()

    # ── Registration ─────────────────────────────────────────────────────

    def add(self, agent_id: str, public_key: "AgentKeyPair | bytes | str", roles: Iterable[str] = (),
            *, replace: bool = False) -> TrustedAgent:
        """Trust ``agent_id`` with this public key (an ``AgentKeyPair``, 32 raw bytes, or base64 text)."""
        agent_id = validate_agent_id(agent_id)
        if isinstance(public_key, AgentKeyPair):
            raw = public_key.public_key_bytes
        elif isinstance(public_key, (bytes, bytearray)):
            raw = bytes(public_key)
            public_key_from_bytes(raw)
        elif isinstance(public_key, str):
            public_key_from_b64(public_key)          # strict: raises KeyManagementError on bad base64 / wrong length
            raw = base64.b64decode(public_key, validate=True)
        else:
            raise RegistryError("public_key must be an AgentKeyPair, raw bytes or base64 text")
        b64 = base64.b64encode(raw).decode("ascii")
        role_set = frozenset(str(r) for r in roles)
        with self._lock:
            existing = self._agents.get(agent_id)
            if existing is not None and not replace:
                raise RegistryError(f"{agent_id!r} is already registered; use replace=True to rotate its key")
            for other_id, other in self._agents.items():
                if other_id != agent_id and other.public_key_b64 == b64:
                    raise RegistryError(f"that public key is already registered for {other_id!r}")
            agent = TrustedAgent(agent_id, b64, role_set, False)
            self._agents[agent_id] = agent
            return agent

    def revoke(self, agent_id: str) -> bool:
        """Mark an agent's key as revoked.  Returns False if the agent is unknown."""
        with self._lock:
            agent = self._agents.get(agent_id)
            if agent is None:
                return False
            self._agents[agent_id] = TrustedAgent(agent.agent_id, agent.public_key_b64, agent.roles, True)
            return True

    def remove(self, agent_id: str) -> bool:
        with self._lock:
            return self._agents.pop(agent_id, None) is not None

    # ── Lookup ───────────────────────────────────────────────────────────

    def get(self, agent_id: object) -> TrustedAgent | None:
        if not isinstance(agent_id, str):
            return None
        with self._lock:
            return self._agents.get(agent_id)

    def agents(self) -> list[TrustedAgent]:
        with self._lock:
            return list(self._agents.values())

    def __contains__(self, agent_id: object) -> bool:
        return self.get(agent_id) is not None

    def __len__(self) -> int:
        with self._lock:
            return len(self._agents)

    # ── Persistence (public data only) ───────────────────────────────────

    def to_json(self) -> str:
        with self._lock:
            data = [{"agent_id": a.agent_id, "public_key": a.public_key_b64, "roles": sorted(a.roles),
                     "revoked": a.revoked} for a in self._agents.values()]
        return json.dumps({"version": 1, "agents": data}, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "TrustedAgentRegistry":
        try:
            data = json.loads(text)
            entries = data["agents"]
        except Exception:
            raise RegistryError("not a valid registry document") from None
        registry = cls()
        for entry in entries:
            try:
                agent = registry.add(entry["agent_id"], entry["public_key"], entry.get("roles", ()))
            except RegistryError:
                raise
            except Exception as exc:
                raise RegistryError(f"invalid registry entry: {type(exc).__name__}") from None
            if entry.get("revoked"):
                registry.revoke(agent.agent_id)
        return registry
