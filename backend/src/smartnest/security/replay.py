"""Replay protection: a valid signature on an old message must not be accepted again.

A signature proves who wrote a message, not when it was sent or that it is new.  An attacker who records a legitimate
signed command could resend it.  :class:`ReplayGuard` therefore enforces:

  * **freshness**: the message timestamp must be within ``window`` seconds of now (and not more than ``max_skew``
    seconds in the future, to tolerate small clock differences);
  * **uniqueness**: each ``(sender, message_id)`` is accepted once inside the window.

Call it only **after** the signature has been verified, so unauthenticated garbage can never fill the memory of seen
ids.  Memory is bounded (``max_entries``); entries older than the window are forgotten because the freshness check
rejects those messages anyway.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Callable

from smartnest.core.models import MessageEnvelope

FRESH = "ok"
STALE = "stale"
FUTURE = "from_the_future"
REPLAY = "replay"


class ReplayGuard:
    def __init__(self, window_seconds: float = 120.0, max_skew_seconds: float = 30.0, max_entries: int = 50_000,
                 clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc)) -> None:
        if window_seconds <= 0 or max_skew_seconds < 0 or max_entries < 1:
            raise ValueError("window must be positive, skew non-negative, max_entries at least 1")
        self._window = window_seconds
        self._skew = max_skew_seconds
        self._max = max_entries
        self._clock = clock
        self._seen: "OrderedDict[tuple[str, str], datetime]" = OrderedDict()
        self._lock = threading.Lock()

    def check(self, message: MessageEnvelope) -> tuple[bool, str, str]:
        """``(accepted, code, detail)``.  An accepted message is remembered, so a second copy is a replay."""
        now = self._clock()
        sent = message.timestamp if message.timestamp.tzinfo else message.timestamp.replace(tzinfo=timezone.utc)
        age = (now - sent).total_seconds()
        if age > self._window:
            return False, STALE, f"the message is {age:.0f} s old (the window is {self._window:g} s)"
        if age < -self._skew:
            return False, FUTURE, f"the message is dated {-age:.0f} s in the future"
        key = (message.sender, message.message_id)
        with self._lock:
            self._expire(now)
            if key in self._seen:
                return False, REPLAY, "this message id was already accepted"
            self._seen[key] = sent
            while len(self._seen) > self._max:
                self._seen.popitem(last=False)
        return True, FRESH, ""

    def _expire(self, now: datetime) -> None:
        while self._seen:
            key, sent = next(iter(self._seen.items()))
            if (now - sent).total_seconds() > self._window:
                self._seen.popitem(last=False)
            else:
                break

    def __len__(self) -> int:
        with self._lock:
            return len(self._seen)
