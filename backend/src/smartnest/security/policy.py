"""Authorization: which authenticated agent may send which kind of message.

Authentication ("this really is agent X") is not authorization ("agent X may order this").  A trusted but low-privilege
agent must not be able to issue commands.  The policy maps **roles** (stored in the trusted-agent registry) to the
message types they may send, and optionally restricts which ``task_type`` values a role may request.

Default roles
-------------
    coordinator   may send anything, including every command
    worker        may report and answer: INFORM, ACK, QUERY, PROPOSAL, REJECT_PROPOSAL, INFORM_DONE, INFORM_RESULT, FAILURE
    observer      may only ask and acknowledge: QUERY, INFORM, ACK

``COMMAND_TYPES`` are the messages that make a receiver *do* something or change its state: task requests, state
synchronisation, calls for proposals, awards and acceptances, cancellations and generic requests.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from smartnest.core.models import MessageEnvelope, MessageType
from smartnest.security.registry import TrustedAgent

COMMAND_TYPES: frozenset[str] = frozenset({
    MessageType.TASK_REQUEST.value, MessageType.STATE_SYNC.value, MessageType.CFP.value, MessageType.AWARD.value,
    MessageType.ACCEPT_PROPOSAL.value, MessageType.CANCEL.value, MessageType.REQUEST.value,
})

_REPORTING = frozenset({
    MessageType.INFORM.value, MessageType.ACK.value, MessageType.QUERY.value, MessageType.PROPOSAL.value,
    MessageType.REJECT_PROPOSAL.value, MessageType.INFORM_DONE.value, MessageType.INFORM_RESULT.value,
    MessageType.FAILURE.value,
})

ALL_TYPES: frozenset[str] = frozenset(t.value for t in MessageType)

DEFAULT_ROLE_PERMISSIONS: dict[str, frozenset[str]] = {
    "coordinator": ALL_TYPES,
    "worker": _REPORTING,
    "observer": frozenset({MessageType.QUERY.value, MessageType.INFORM.value, MessageType.ACK.value}),
}


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.allowed


@dataclass
class AuthorizationPolicy:
    """Role-based permissions.

    Attributes:
        permissions: role -> message types that role may send.
        allowed_tasks: optional role -> the ``task_type`` values it may request (a role absent here is unrestricted
            for tasks it is otherwise allowed to request).
    """

    permissions: dict[str, frozenset[str]] = field(default_factory=lambda: dict(DEFAULT_ROLE_PERMISSIONS))
    allowed_tasks: dict[str, frozenset[str]] = field(default_factory=dict)

    def check(self, agent: TrustedAgent, message: MessageEnvelope) -> Decision:
        """Whether ``agent`` (already authenticated) may send ``message``."""
        mtype = message.message_type
        granted = [role for role in agent.roles if mtype in self.permissions.get(role, frozenset())]
        if not granted:
            kind = "command" if mtype in COMMAND_TYPES else "message type"
            roles = ", ".join(sorted(agent.roles)) or "none"
            return Decision(False, f"{agent.agent_id!r} (roles: {roles}) may not send the {kind} {mtype!r}")
        if mtype == MessageType.TASK_REQUEST.value:
            task = message.payload.get("task_type") if isinstance(message.payload, dict) else None
            restricted = [role for role in granted if role in self.allowed_tasks]
            if restricted and not any(task in self.allowed_tasks[role] for role in restricted) \
                    and len(restricted) == len(granted):
                return Decision(False, f"{agent.agent_id!r} may not request the task {task!r}")
        return Decision(True)
