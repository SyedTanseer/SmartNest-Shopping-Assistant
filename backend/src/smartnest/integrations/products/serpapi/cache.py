"""MongoDB-backed cache for SerpApi responses (saves paid searches).

Entries live in the ``search_cache`` collection through the repository
interface (so tests use :class:`InMemoryRepository`, production Atlas).
Each entry records when it was fetched; the cache never decides on its own
that old data is good — callers get the entry *and* whether it is fresh:

    fresh   age <= TTL      served without calling SerpApi
    stale   age >  TTL      returned only so a caller can fall back to it when SerpApi
                            fails; it is always flagged as stale

The key is a hash of the request parameters **without** the API key, so no
secret is ever stored.  The payload is kept as a JSON string, which avoids
MongoDB field-name restrictions on third-party data.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from smartnest.integrations.products.models import utc_now

logger = logging.getLogger(__name__)

SEARCH_CACHE_COLLECTION = "search_cache"


@dataclass(frozen=True)
class CacheEntry:
    payload: dict[str, Any]
    fetched_at: datetime
    fresh: bool

    @property
    def stale(self) -> bool:
        return not self.fresh


def cache_key(kind: str, params: dict[str, Any]) -> str:
    """Stable key for a request; ``api_key`` is excluded even if passed by mistake."""
    clean = {k: v for k, v in sorted(params.items()) if k != "api_key"}
    digest = hashlib.sha256(json.dumps([kind, clean], sort_keys=True).encode("utf-8")).hexdigest()[:32]
    return f"serp_{kind}_{digest}"


class SearchCache:
    """Cache of SerpApi responses over a ``Repository``.

    Args:
        repository: Document store with ``get``/``upsert`` (Repository interface).
        ttl: How long an entry stays fresh.
        clock: Injectable time source.

    Cache failures never break discovery: a repository error is logged and treated as a miss.
    """

    def __init__(self, repository: Any, *, ttl: timedelta = timedelta(hours=6),
                 clock: Callable[[], datetime] = utc_now) -> None:
        if ttl <= timedelta(0):
            raise ValueError("ttl must be positive")
        self._repo = repository
        self._ttl = ttl
        self._clock = clock
        self.hits = 0
        self.misses = 0

    def get(self, kind: str, params: dict[str, Any]) -> CacheEntry | None:
        key = cache_key(kind, params)
        try:
            doc = self._repo.get(key)
        except Exception as exc:  # a broken cache must not break discovery
            logger.warning("search cache read failed: %s", type(exc).__name__)
            return None
        if not doc:
            return None
        try:
            fetched = datetime.fromisoformat(doc["fetched_at"])
            if fetched.tzinfo is None:
                fetched = fetched.replace(tzinfo=timezone.utc)
            payload = json.loads(doc["payload_json"])
        except (KeyError, TypeError, ValueError):
            logger.warning("search cache entry %s is unreadable; ignoring it", key)
            return None
        if not isinstance(payload, dict):
            return None
        return CacheEntry(payload, fetched, fresh=self._clock() - fetched <= self._ttl)

    def put(self, kind: str, params: dict[str, Any], payload: dict[str, Any]) -> datetime:
        key = cache_key(kind, params)
        now = self._clock()
        document = {"kind": kind, "params": {k: v for k, v in params.items() if k != "api_key"},
                    "fetched_at": now.isoformat(), "expires_at": (now + self._ttl).isoformat(),
                    "payload_json": json.dumps(payload)}
        try:
            self._repo.upsert(key, document)
        except Exception as exc:
            logger.warning("search cache write failed: %s", type(exc).__name__)
        return now
