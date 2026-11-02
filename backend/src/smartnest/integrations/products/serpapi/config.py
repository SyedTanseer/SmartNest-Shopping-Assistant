"""SerpApi configuration, loaded from ``backend/.env`` and the environment.

    SERPAPI_API_KEY             required
    SERPAPI_TIMEOUT_SECONDS     optional, default 20
    SERPAPI_MAX_RESULTS         optional, default 10  (shopping results considered per search)
    SERPAPI_MAX_DETAIL_CALLS    optional, default 3   (product-details lookups per discovery run)
    SERPAPI_CACHE_TTL_HOURS     optional, default 6   (how long a cached search stays fresh)
    SERPAPI_LIFETIME_LIMIT      optional hard cap on TOTAL requests ever made with this checkout; the word
                                ``unlimited`` removes the cap ON PURPOSE (an absent value is not the same: scripts
                                that spend requests refuse to run, because it may mean the .env was not loaded)
                                (counted in backend/.serpapi_usage.json; unset = no local cap)

Real environment variables win over ``.env``.  The key is stored with
``repr=False`` and every error message is redacted, so it never reaches
logs, tracebacks or test output.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from dotenv import dotenv_values

from smartnest.integrations.gemini.errors import redact_secrets

#: backend/.env (this file is backend/src/smartnest/integrations/products/serpapi/config.py).
DEFAULT_ENV_FILE = Path(__file__).resolve().parents[5] / ".env"
ENDPOINT = "https://serpapi.com/search.json"

_PLACEHOLDERS = {"your_api_key", "your-api-key", "changeme", "<api_key>", "xxx", "todo", "<your_serpapi_api_key>"}


class SerpApiConfigError(Exception):
    """SerpApi is not configured (or configured with an invalid value)."""


@dataclass(frozen=True)
class SerpApiSettings:
    """Validated SerpApi settings.  Build with :func:`load_serpapi_settings`."""

    api_key: str = field(repr=False)
    timeout_seconds: float = 20.0
    max_results: int = 10
    max_detail_calls: int = 3
    cache_ttl_hours: float = 6.0
    lifetime_limit: int | None = None
    uncapped: bool = False              # the cap was removed deliberately (SERPAPI_LIFETIME_LIMIT=unlimited)
    usage_file: Path = Path(__file__).resolve().parents[5] / ".serpapi_usage.json"
    country: str = "in"
    language: str = "en"
    endpoint: str = ENDPOINT

    def __post_init__(self) -> None:
        key = (self.api_key or "").strip()
        if not key:
            raise SerpApiConfigError("SERPAPI_API_KEY is not set; add it to backend/.env")
        if key.lower() in _PLACEHOLDERS or key.startswith("<") or any(c.isspace() for c in key):
            raise SerpApiConfigError("SERPAPI_API_KEY still looks like a placeholder; put the real key in backend/.env")
        object.__setattr__(self, "api_key", key)
        if isinstance(self.timeout_seconds, bool) or not 1 <= self.timeout_seconds <= 120:
            raise SerpApiConfigError("SERPAPI_TIMEOUT_SECONDS must be between 1 and 120")
        if isinstance(self.max_results, bool) or not 1 <= self.max_results <= 100:
            raise SerpApiConfigError("SERPAPI_MAX_RESULTS must be between 1 and 100")
        if isinstance(self.max_detail_calls, bool) or not 0 <= self.max_detail_calls <= 20:
            raise SerpApiConfigError("SERPAPI_MAX_DETAIL_CALLS must be between 0 and 20")
        if self.lifetime_limit is not None and (isinstance(self.lifetime_limit, bool) or self.lifetime_limit < 0):
            raise SerpApiConfigError("SERPAPI_LIFETIME_LIMIT must be a whole number >= 0")
        if isinstance(self.cache_ttl_hours, bool) or not 0 < self.cache_ttl_hours <= 24 * 30:
            raise SerpApiConfigError("SERPAPI_CACHE_TTL_HOURS must be greater than 0 (and at most 720)")

    def redact(self, text: str) -> str:
        return redact_secrets(text, (self.api_key,))

    def __str__(self) -> str:
        return f"SerpApiSettings(country={self.country}, max_results={self.max_results}, api_key=set)"


def load_serpapi_settings(env_file: str | Path | None = None,
                          environ: Mapping[str, str] | None = None) -> SerpApiSettings:
    """Load settings; raises :class:`SerpApiConfigError` for a missing/placeholder key or bad value."""
    path = Path(env_file) if env_file is not None else DEFAULT_ENV_FILE
    file_values = dotenv_values(path) if path.is_file() else {}
    env = os.environ if environ is None else environ

    def get(name: str) -> str | None:
        value = env.get(name)
        return value if value is not None and value.strip() else file_values.get(name)

    def number(name: str, default: float, cast):
        raw = get(name)
        if raw is None or not str(raw).strip():
            return default
        try:
            return cast(str(raw).strip())
        except ValueError:
            raise SerpApiConfigError(f"{name} must be a number") from None

    uncapped = str(get("SERPAPI_LIFETIME_LIMIT") or "").strip().lower() in ("unlimited", "none", "off", "no")

    return SerpApiSettings(
        api_key=get("SERPAPI_API_KEY") or "",
        timeout_seconds=number("SERPAPI_TIMEOUT_SECONDS", 20.0, float),
        max_results=number("SERPAPI_MAX_RESULTS", 10, int),
        max_detail_calls=number("SERPAPI_MAX_DETAIL_CALLS", 3, int),
        cache_ttl_hours=number("SERPAPI_CACHE_TTL_HOURS", 6.0, float),
        lifetime_limit=None if uncapped else number("SERPAPI_LIFETIME_LIMIT", None, int),
        uncapped=uncapped,
    )


def serpapi_configured(env_file: str | Path | None = None, environ: Mapping[str, str] | None = None) -> bool:
    """Whether a usable key is configured (never raises, never reveals the key)."""
    try:
        load_serpapi_settings(env_file, environ)
    except SerpApiConfigError:
        return False
    return True
