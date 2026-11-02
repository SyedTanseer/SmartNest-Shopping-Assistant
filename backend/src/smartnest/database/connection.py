"""Centralised MongoDB connection management.

One :class:`MongoConnection` owns one ``MongoClient``.  A ``MongoClient``
is thread-safe and maintains its own connection pool, so a single
instance is meant to be shared by the whole process — the background
workers, the CNP agents and the future FastAPI app all use the same one
via :func:`get_connection`.  Do not create a client per request or per
thread.

Lifecycle::

    conn = get_connection()          # built lazily from backend/.env
    conn.connect()                   # create client + ping (fail fast at startup)
    repo = BaseRepository(conn.collection(Collections.PRODUCTS))
    ...
    close_connection()               # at shutdown (FastAPI lifespan, atexit, ...)

``connect()`` is optional: the client is also created on first use.  The
driver connects lazily, so creating the manager never blocks on the
network.

Security: every error raised from here is passed through the settings'
redaction, and chained driver exceptions are suppressed (``from None``)
so a traceback can never print the connection string.

PyMongo is synchronous.  From async code (FastAPI) call it with
``await asyncio.to_thread(...)`` / ``run_in_threadpool`` so the event
loop is not blocked.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import ConfigurationError, PyMongoError

from smartnest.database.errors import (
    DatabaseConfigError,
    DatabaseConnectionError,
    DatabaseError,
)
from smartnest.database.settings import DatabaseSettings, load_settings

logger = logging.getLogger(__name__)

ClientFactory = Callable[..., Any]


@dataclass(frozen=True)
class HealthStatus:
    """Result of a connection health check."""

    ok: bool
    database: str
    latency_ms: float | None = None
    error: str | None = None


class MongoConnection:
    """Owns the shared ``MongoClient`` for one set of settings.

    Args:
        settings: Validated database settings.
        client_factory: Callable building the client from
            ``(uri, **options)``.  Defaults to ``pymongo.MongoClient``;
            tests inject a fake.
    """

    def __init__(
        self,
        settings: DatabaseSettings,
        client_factory: ClientFactory | None = None,
    ) -> None:
        self._settings = settings
        self._factory: ClientFactory = client_factory or MongoClient
        self._lock = threading.RLock()
        self._client: Any = None
        self._closed = False

    # ── State ────────────────────────────────────────────────────────────

    @property
    def settings(self) -> DatabaseSettings:
        return self._settings

    @property
    def is_connected(self) -> bool:
        """Whether a client currently exists (not whether the server is reachable)."""
        with self._lock:
            return self._client is not None

    @property
    def is_closed(self) -> bool:
        with self._lock:
            return self._closed

    # ── Client creation ──────────────────────────────────────────────────

    def _options(self) -> dict[str, Any]:
        s = self._settings
        return {
            "appName": s.app_name,
            "serverSelectionTimeoutMS": s.server_selection_timeout_ms,
            "connectTimeoutMS": s.connect_timeout_ms,
            "socketTimeoutMS": s.socket_timeout_ms,
            "maxPoolSize": s.max_pool_size,
            "minPoolSize": s.min_pool_size,
            "tz_aware": True,  # datetimes come back timezone-aware (UTC), like our models
            "uuidRepresentation": "standard",
        }

    def _create_client(self) -> Any:
        try:
            client = self._factory(self._settings.uri, **self._options())
        except (ConfigurationError, ValueError, TypeError) as exc:
            raise DatabaseConfigError(
                f"Invalid MongoDB configuration ({type(exc).__name__}): "
                f"{self._settings.redact(str(exc))}"
            ) from None
        except PyMongoError as exc:
            raise DatabaseConnectionError(
                f"Could not create MongoDB client ({type(exc).__name__}): "
                f"{self._settings.redact(str(exc))}"
            ) from None
        logger.info("MongoDB client created for database '%s'", self._settings.database)
        return client

    def _ensure_client(self) -> Any:
        with self._lock:
            if self._closed:
                raise DatabaseConnectionError(
                    "MongoDB connection is closed; call connect() to open it again"
                )
            if self._client is None:
                self._client = self._create_client()
            return self._client

    def _discard_client(self) -> None:
        """Drop the client without marking the manager closed (used after a failed connect)."""
        with self._lock:
            client, self._client = self._client, None
        self._close_quietly(client)

    @staticmethod
    def _close_quietly(client: Any) -> None:
        if client is None:
            return
        try:
            client.close()
        except Exception:  # never let cleanup mask the original problem
            logger.warning("Error while closing MongoDB client", exc_info=False)

    # ── Public API ───────────────────────────────────────────────────────

    @property
    def client(self) -> Any:
        """The shared ``MongoClient`` (created on first access)."""
        return self._ensure_client()

    @property
    def database(self) -> Database:
        """The configured database."""
        return self.client[self._settings.database]

    def collection(self, name: str) -> Collection:
        """A collection in the configured database.

        ``name`` may be a :class:`~smartnest.database.collections.Collections`
        member or any non-empty string.
        """
        collection_name = str(name)
        if not collection_name.strip():
            raise DatabaseConfigError("collection name must be a non-empty string")
        return self.database[collection_name]

    def connect(self, verify: bool = True) -> Any:
        """Create the client (re-opening after :meth:`close`) and optionally ping.

        With ``verify=True`` an unreachable or unauthorised server raises
        :class:`DatabaseConnectionError` immediately and the half-open
        client is discarded, so a later ``connect()`` starts clean.
        """
        with self._lock:
            self._closed = False
            client = self._ensure_client()
        if verify:
            try:
                self.ping()
            except DatabaseConnectionError:
                self._discard_client()
                raise
        return client

    def ping(self) -> float:
        """Round-trip a ``ping`` command to the server.

        Returns:
            Latency in milliseconds.

        Raises:
            DatabaseConnectionError: If the server cannot be reached or
                rejects the credentials.
        """
        client = self._ensure_client()
        start = time.perf_counter()
        try:
            client.admin.command("ping")
        except PyMongoError as exc:
            raise DatabaseConnectionError(
                f"MongoDB ping failed ({type(exc).__name__}): {self._settings.redact(str(exc))}"
            ) from None
        return (time.perf_counter() - start) * 1000.0

    def check_health(self) -> HealthStatus:
        """Like :meth:`ping` but never raises; suited to a health endpoint."""
        try:
            latency = self.ping()
        except DatabaseError as exc:
            return HealthStatus(ok=False, database=self._settings.database, error=str(exc))
        return HealthStatus(ok=True, database=self._settings.database, latency_ms=round(latency, 2))

    def is_healthy(self) -> bool:
        return self.check_health().ok

    def close(self) -> None:
        """Close the client and its pool.  Safe to call more than once."""
        with self._lock:
            client, self._client = self._client, None
            was_open = not self._closed
            self._closed = True
        self._close_quietly(client)
        if was_open and client is not None:
            logger.info("MongoDB connection closed")

    # ── Context manager ──────────────────────────────────────────────────

    def __enter__(self) -> "MongoConnection":
        self.connect(verify=True)
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        state = "closed" if self.is_closed else ("connected" if self.is_connected else "idle")
        return f"<MongoConnection {self._settings.host}/{self._settings.database} {state}>"


# ── Process-wide shared connection ──────────────────────────────────────────

_shared: MongoConnection | None = None
_shared_lock = threading.Lock()


def get_connection(
    settings: DatabaseSettings | None = None,
    client_factory: ClientFactory | None = None,
) -> MongoConnection:
    """The process-wide shared :class:`MongoConnection`.

    Created on first call from ``settings`` (default: :func:`load_settings`).
    Later calls return the same object — ``settings`` and ``client_factory``
    only take effect when it is first created (or re-created after
    :func:`close_connection`).  Thread-safe.
    """
    global _shared
    with _shared_lock:
        if _shared is None or _shared.is_closed:
            _shared = MongoConnection(settings or load_settings(), client_factory)
        return _shared


def close_connection() -> None:
    """Close and forget the shared connection (call at application shutdown)."""
    global _shared
    with _shared_lock:
        connection, _shared = _shared, None
    if connection is not None:
        connection.close()
