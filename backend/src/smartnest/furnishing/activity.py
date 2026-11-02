"""The agent activity log and the signed message bus that feeds it.

``AgentBus`` is how the furnishing agents talk to each other.  Every request and reply is a real
``MessageEnvelope`` signed with Ed25519 (Assignment 5.1) and checked by a ``SecureMessageRouter`` before any handler
runs, so the activity monitor shows *authenticated* traffic, including the messages that were refused.  The bus is
in-process (a direct call into each agent's router), the same router the TCP server uses, so moving an agent to
another machine means giving it a ``RemoteAgentServer`` and a ``SecureChannel`` - no agent code changes.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from smartnest.communication.message_router import MessageRouter
from smartnest.core.models import MessageEnvelope, MessageType
from smartnest.security import (
    AgentKeyPair,
    AuditLog,
    SecureMessageRouter,
    TrustedAgentRegistry,
    sign_envelope,
    verify_envelope,
)


@dataclass
class ActivityEvent:
    seq: int
    time: str
    agent: str
    kind: str                 # message | decision | info | warning | replan
    summary: str
    peer: str | None = None
    task: str | None = None
    verified: bool | None = None       # for messages: did the signature check pass
    duration_ms: float | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in ("seq", "time", "agent", "kind", "summary", "peer", "task",
                                              "verified", "duration_ms", "details")}


class ActivityLog:
    """Thread-safe, bounded, append-only list of what the agents did."""

    def __init__(self, max_events: int = 2000) -> None:
        self._events: list[ActivityEvent] = []
        self._lock = threading.Lock()
        self._seq = 0
        self._max = max_events

    def record(self, agent: str, kind: str, summary: str, **kw: Any) -> ActivityEvent:
        with self._lock:
            self._seq += 1
            event = ActivityEvent(self._seq, datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                                  agent, kind, summary, **kw)
            self._events.append(event)
            if len(self._events) > self._max:
                del self._events[: len(self._events) - self._max]
            return event

    def events(self, since: int = 0) -> list[ActivityEvent]:
        with self._lock:
            return [e for e in self._events if e.seq > since]

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)


Handler = Callable[[str, dict[str, Any]], dict[str, Any]]


class AgentBus:
    """Registers agents and delivers signed task requests between them."""

    COORDINATOR = "coordinator"

    def __init__(self, log: ActivityLog) -> None:
        self.log = log
        self.registry = TrustedAgentRegistry()
        self.audit = AuditLog()
        self._keys: dict[str, AgentKeyPair] = {}
        self._routers: dict[str, SecureMessageRouter] = {}
        self._keys[self.COORDINATOR] = AgentKeyPair.generate(self.COORDINATOR)
        self.registry.add(self.COORDINATOR, self._keys[self.COORDINATOR], roles=["coordinator"])

    def register(self, agent_id: str, handler: Handler) -> None:
        """Add a worker agent whose ``handler(task_type, payload)`` returns a JSON-able dict."""
        key = AgentKeyPair.generate(agent_id)
        self._keys[agent_id] = key
        self.registry.add(agent_id, key, roles=["worker"])
        inner = MessageRouter(agent_id)

        def on_task(message: MessageEnvelope) -> MessageEnvelope:
            task = str(message.payload.get("task_type"))
            try:
                result = handler(task, message.payload.get("data") or {})
                return MessageEnvelope(sender=agent_id, receiver=message.sender, task_id=message.task_id,
                                       message_type=MessageType.INFORM_RESULT, payload={"result": result})
            except Exception as exc:  # a failing agent reports failure instead of crashing the coordinator
                return MessageEnvelope(sender=agent_id, receiver=message.sender, task_id=message.task_id,
                                       message_type=MessageType.FAILURE,
                                       payload={"reason": f"{type(exc).__name__}: {exc}"})

        inner.register(MessageType.TASK_REQUEST, on_task)
        self._routers[agent_id] = SecureMessageRouter(key, inner, self.registry, audit=self.audit)

    @property
    def agent_ids(self) -> list[str]:
        return [self.COORDINATOR, *self._routers]

    def call(self, receiver: str, task_type: str, data: dict[str, Any], *, sender: str | None = None,
             summary: str | None = None) -> dict[str, Any]:
        """Send a signed task request from ``sender`` (default: the coordinator) and return the verified result."""
        sender = sender or self.COORDINATOR
        started = time.perf_counter()
        request = MessageEnvelope(sender=sender, receiver=receiver, message_type=MessageType.TASK_REQUEST,
                                  payload={"task_type": task_type, "data": data})
        signed = sign_envelope(request, self._keys[sender])
        reply = self._routers[receiver].route(signed)
        elapsed = (time.perf_counter() - started) * 1000
        verification = verify_envelope(reply, self.registry) if reply is not None else None
        ok = bool(verification and verification.ok and reply.message_type == MessageType.INFORM_RESULT.value)
        self.log.record(sender, "message", summary or f"asked {receiver} to {task_type}", peer=receiver,
                        task=task_type, verified=bool(verification and verification.ok), duration_ms=round(elapsed, 2))
        if not ok:
            reason = (reply.payload.get("reason") if reply is not None else None) or "no reply"
            self.log.record(receiver, "warning", f"{task_type} failed: {reason}", peer=sender, task=task_type,
                            verified=bool(verification and verification.ok))
            raise AgentCallError(receiver, task_type, str(reason))
        return reply.payload["result"]


class AgentCallError(RuntimeError):
    def __init__(self, agent: str, task: str, reason: str) -> None:
        super().__init__(f"{agent} could not {task}: {reason}")
        self.agent, self.task, self.reason = agent, task, reason
