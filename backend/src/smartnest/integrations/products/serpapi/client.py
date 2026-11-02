"""Thin HTTP client for SerpApi (``httpx``), with explicit error handling.

Every failure is a :class:`SerpApiError` subclass whose message is
redacted (the API key travels as a query parameter, so URLs from the
HTTP library are never put in messages).  No automatic retries: a retry
can consume another paid search, and 401/429 cannot succeed on retry.

    401 / 403            -> SerpApiAuthError      (bad or missing key)
    429                  -> SerpApiQuotaError     (rate limit or searches used up)
    timeout              -> SerpApiTimeoutError
    5xx / network        -> SerpApiError
    non-JSON / wrong shape -> SerpApiResponseError

SerpApi also reports some problems with HTTP 200 and an ``"error"`` field;
those are mapped too.  "Google hasn't returned any results" is *not* a
failure: it is an empty result, returned as ``{"_empty": True}``.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

import httpx

from smartnest.integrations.gemini.errors import redact_secrets
from smartnest.integrations.products.serpapi.config import SerpApiSettings


class _RedactKeyFilter(logging.Filter):
    """httpx logs every request URL at INFO, and SerpApi takes the key as a query parameter.

    Without this filter the key would appear in any log that captures ``httpx`` output.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            return True
        cleaned = redact_secrets(message)
        if cleaned != message:
            record.msg, record.args = cleaned, ()
        return True


for _name in ("httpx", "httpcore"):
    _logger = logging.getLogger(_name)
    if not any(isinstance(f, _RedactKeyFilter) for f in _logger.filters):
        _logger.addFilter(_RedactKeyFilter())


class SerpApiError(Exception):
    """Base class for every error raised by the SerpApi layer."""


class SerpApiAuthError(SerpApiError):
    """HTTP 401/403: the API key is missing, invalid or not allowed."""


class SerpApiQuotaError(SerpApiError):
    """HTTP 429: rate limited or out of searches."""


class SerpApiTimeoutError(SerpApiError):
    """The request timed out."""


class SerpApiResponseError(SerpApiError):
    """The response was not the JSON object that was expected."""


#: The limit used when no cap is configured: far beyond any real use, so the ledger only counts.
UNCAPPED = 10 ** 9


class UsageLedger:
    """A persistent, hard cap on the number of SerpApi requests this checkout may ever make.

    The count is written to disk *before* each request (so a crash can only over-count, never
    under-count) and a request beyond the cap is refused locally, without contacting SerpApi.
    """

    def __init__(self, path: Path, limit: int) -> None:
        self._path, self.limit = Path(path), limit
        self._lock = threading.Lock()

    @property
    def used(self) -> int:
        try:
            return int(json.loads(self._path.read_text(encoding="utf-8")).get("used", 0))
        except (OSError, ValueError, AttributeError, TypeError):
            return 0

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)

    def reserve(self) -> None:
        with self._lock:
            used = self.used
            if used >= self.limit:
                raise SerpApiQuotaError(
                    f"local SerpApi budget exhausted ({used} of {self.limit} requests used); "
                    "no request was sent. Raise SERPAPI_LIFETIME_LIMIT to allow more.")
            self._path.write_text(json.dumps({"used": used + 1, "limit": self.limit}), encoding="utf-8")


class SerpApiClient:
    """Calls ``https://serpapi.com/search.json``.

    Args:
        settings: Validated settings (key, timeout, endpoint, country, language).
        transport: Optional ``httpx`` transport, so tests can serve canned responses.
    """

    def __init__(self, settings: SerpApiSettings, *, transport: httpx.BaseTransport | None = None) -> None:
        self._settings = settings
        self._transport = transport
        # A configured cap enforces and counts.  A cap removed ON PURPOSE (``unlimited``) still counts, so "requests used"
        # stays true; its limit is just unreachable.  Plain default settings (nothing configured, e.g. in tests) touch no file.
        if settings.lifetime_limit is not None:
            self.ledger = UsageLedger(settings.usage_file, settings.lifetime_limit)
        elif settings.uncapped:
            self.ledger = UsageLedger(settings.usage_file, UNCAPPED)
        else:
            self.ledger = None
        self.calls = 0  # number of HTTP requests made (each may cost one SerpApi search)

    # ── Public operations ────────────────────────────────────────────────

    def search(self, query: str) -> dict[str, Any]:
        """Google Shopping search (``engine=google_shopping``, ``gl``/``hl`` from settings)."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must not be empty")
        return self._get({"engine": "google_shopping", "q": query.strip(),
                          "gl": self._settings.country, "hl": self._settings.language})

    def product_details(self, page_token: str) -> dict[str, Any]:
        """Retailer offers and features for one product (``engine=google_immersive_product``)."""
        if not isinstance(page_token, str) or not page_token.strip():
            raise ValueError("page_token must not be empty")
        return self._get({"engine": "google_immersive_product", "page_token": page_token.strip(),
                          "gl": self._settings.country, "hl": self._settings.language})

    # ── Internals ────────────────────────────────────────────────────────

    def _get(self, params: dict[str, str]) -> dict[str, Any]:
        s = self._settings
        if self.ledger is not None:
            self.ledger.reserve()  # raises before any network access once the cap is reached
        self.calls += 1
        try:
            with httpx.Client(timeout=s.timeout_seconds, transport=self._transport) as http:
                response = http.get(s.endpoint, params={**params, "api_key": s.api_key})
        except httpx.TimeoutException:
            raise SerpApiTimeoutError(f"SerpApi did not answer within {s.timeout_seconds:g} s") from None
        except httpx.HTTPError as exc:
            raise SerpApiError(f"SerpApi request failed ({type(exc).__name__})") from None

        body = self._json(response)
        message = s.redact(str(body.get("error", ""))) if isinstance(body, dict) else ""
        status = response.status_code
        if status in (401, 403):
            raise SerpApiAuthError(f"SerpApi rejected the API key (HTTP {status})")
        if status == 429:
            raise SerpApiQuotaError(f"SerpApi rate limit or quota reached (HTTP 429){': ' + message if message else ''}")
        if status >= 500:
            raise SerpApiError(f"SerpApi server error (HTTP {status})")
        if status >= 400:
            raise SerpApiError(f"SerpApi returned HTTP {status}{': ' + message if message else ''}")
        if not isinstance(body, dict):
            raise SerpApiResponseError("SerpApi response is not a JSON object")
        if message:
            lowered = message.lower()
            if "hasn't returned any results" in lowered or "has not returned any results" in lowered:
                return {"_empty": True, "error": message}
            if "invalid api key" in lowered or "api key" in lowered and "missing" in lowered:
                raise SerpApiAuthError("SerpApi rejected the API key")
            if "run out of searches" in lowered or "limit" in lowered and "search" in lowered:
                raise SerpApiQuotaError(f"SerpApi quota reached: {message}")
            raise SerpApiError(f"SerpApi reported an error: {message}")
        return body

    @staticmethod
    def _json(response: httpx.Response) -> Any:
        try:
            return response.json()
        except (json.JSONDecodeError, ValueError):
            if response.status_code >= 400:
                return None  # the status code is reported by the caller
            raise SerpApiResponseError("SerpApi response is not valid JSON") from None
