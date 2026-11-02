"""Session registry: many users' conversations, optionally persisted.

A session is stored as ``{request, events, shortlist, history}``.  Because planning is deterministic, replaying the
request and the events rebuilds the plan exactly, so the database never needs to hold (or trust) a stale copy of
prices or picks, and a server restart loses nothing.  Persistence goes through the ``Repository`` interface (an
in-memory one in tests, MongoDB Atlas ``home_projects`` in production); a storage failure is logged and never breaks
a request.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from typing import Any, Callable

from smartnest.furnishing import FurnishingCoordinator, FurnishingRequest, PlanError
from smartnest.furnishing.assistant import HomeSession

logger = logging.getLogger(__name__)


class SessionNotFound(KeyError):
    pass


class SessionRegistry:
    def __init__(self, make_coordinator: Callable[[], FurnishingCoordinator], *, repository: Any = None,
                 max_sessions: int = 200) -> None:
        self._make = make_coordinator
        self._repo = repository
        self._sessions: "OrderedDict[str, HomeSession]" = OrderedDict()
        self._max = max_sessions
        self._lock = threading.RLock()

    def create(self) -> HomeSession:
        session = HomeSession(self._make())
        with self._lock:
            self._sessions[session.id] = session
            while len(self._sessions) > self._max:
                self._sessions.popitem(last=False)
        return session

    def get(self, session_id: str) -> HomeSession:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is not None:
                self._sessions.move_to_end(session_id)
                return session
        session = self._load(session_id)
        if session is None:
            raise SessionNotFound(session_id)
        with self._lock:
            self._sessions[session_id] = session
        return session

    def __len__(self) -> int:
        return len(self._sessions)

    # ── Persistence ──────────────────────────────────────────────────────

    def save(self, session: HomeSession) -> None:
        if self._repo is None:
            return
        doc = {
            "request": None if session.request is None else {
                "total_budget": session.request.total_budget, "bedrooms": session.request.bedrooms,
                "style": session.request.style, "colour": session.request.colour,
                "room_dimensions": session.request.room_dimensions, "scope": list(session.request.scope),
                "audience": session.request.audience, "age": session.request.age,
                "themes": list(session.request.themes), "items": list(session.request.items)},
            "events": session.events, "shortlist": session.coordinator.shortlist, "history": session.history[-100:],
            "draft": session.draft, "created_at": session.created_at,
        }
        try:
            self._repo.upsert(session.id, doc)
        except Exception as exc:
            logger.warning("could not persist session %s: %s", session.id, type(exc).__name__)

    def _load(self, session_id: str) -> HomeSession | None:
        if self._repo is None:
            return None
        try:
            doc = self._repo.get(session_id)
        except Exception as exc:
            logger.warning("could not load session %s: %s", session_id, type(exc).__name__)
            return None
        if not doc:
            return None
        session = HomeSession(self._make(), session_id)
        session.created_at = doc.get("created_at", session.created_at)
        session.history = list(doc.get("history") or [])
        session.draft = dict(doc.get("draft") or {})
        req = doc.get("request")
        try:
            if req:
                session.start(FurnishingRequest(
                    req["total_budget"], req.get("bedrooms"), req.get("style"), req.get("colour"),
                    req.get("room_dimensions"), tuple(req.get("scope") or ()), req.get("audience"), req.get("age"),
                    tuple(req.get("themes") or ()), tuple(req.get("items") or ())))
                for event in doc.get("events") or []:
                    session.apply(event)
                for pid in doc.get("shortlist") or []:
                    try:
                        session.coordinator.toggle_shortlist(pid)
                    except PlanError:
                        pass
        except Exception as exc:                    # a document that no longer replays is treated as missing
            logger.warning("session %s could not be rebuilt: %s", session_id, type(exc).__name__)
            return None
        return session
