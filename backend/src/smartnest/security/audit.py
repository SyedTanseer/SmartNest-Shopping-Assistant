"""Security audit trail: every accepted and every rejected message, with the reason.

Events never contain key material or the message payload (a payload may hold personal data); they record who, what
type, the outcome and a short reason.  Events are kept in a bounded in-memory list *and* written to the
``smartnest.security.audit`` logger (INFO for accepted, WARNING for rejected) so they show up in runtime logs.
"""

from __future__ import annotations

import logging
import threading
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timezone

logger = logging.getLogger("smartnest.security.audit")


@dataclass(frozen=True)
class AuditEvent:
    timestamp: datetime
    outcome: str                 # "accepted" | "rejected"
    code: str                    # "ok" or the rejection code
    detail: str
    sender: str
    receiver: str
    message_type: str
    message_id: str
    key_fingerprint: str | None = None

    def format(self) -> str:
        tag = "ACCEPTED" if self.outcome == "accepted" else "REJECTED"
        extra = f" - {self.detail}" if self.detail else ""
        return (f"{tag} {self.code}: {self.message_type} from {self.sender!r} to {self.receiver!r} "
                f"(message {self.message_id})" + (f" [key {self.key_fingerprint}]" if self.key_fingerprint else "") + extra)


class AuditLog:
    def __init__(self, max_events: int = 10_000) -> None:
        self._events: deque[AuditEvent] = deque(maxlen=max_events)
        self._lock = threading.Lock()

    def record(self, outcome: str, code: str, detail: str, message, key_fingerprint: str | None = None) -> AuditEvent:
        event = AuditEvent(datetime.now(timezone.utc), outcome, code, detail, str(message.sender), str(message.receiver),
                           str(message.message_type), str(message.message_id), key_fingerprint)
        with self._lock:
            self._events.append(event)
        (logger.info if outcome == "accepted" else logger.warning)(event.format())
        return event

    @property
    def events(self) -> list[AuditEvent]:
        with self._lock:
            return list(self._events)

    def rejected(self) -> list[AuditEvent]:
        return [e for e in self.events if e.outcome == "rejected"]

    def accepted(self) -> list[AuditEvent]:
        return [e for e in self.events if e.outcome == "accepted"]

    def summary(self) -> dict[str, int]:
        """Counts by ``outcome:code``, e.g. ``{"accepted:ok": 2, "rejected:bad_signature": 1}``."""
        return dict(Counter(f"{e.outcome}:{e.code}" for e in self.events))
