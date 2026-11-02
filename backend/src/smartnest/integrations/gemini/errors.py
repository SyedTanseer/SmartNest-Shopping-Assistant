"""Gemini integration errors and secret redaction.

Every failure of the Gemini layer is a :class:`GeminiError`, so callers
never handle SDK exceptions.  Messages never contain the API key: all
text passes through :func:`redact_secrets`.
"""

from __future__ import annotations

import re
from typing import Iterable

MASK = "***"
_MAX_LENGTH = 400

# Strings that look like Google API keys, even if they are not the configured one.
_GOOGLE_KEY = re.compile(r"AIza[0-9A-Za-z_\-]{20,}")
_KEY_PARAM = re.compile(r"([?&](?:key|api_key|apikey)=)[^&\s\"']+", re.IGNORECASE)


def redact_secrets(text: str, secrets: Iterable[str] = ()) -> str:
    """Mask the configured secrets and anything key-shaped; truncate long text."""
    for secret in secrets:
        if secret and len(secret) >= 8:
            text = text.replace(secret, MASK)
    text = _GOOGLE_KEY.sub(MASK, text)
    text = _KEY_PARAM.sub(rf"\1{MASK}", text)
    if len(text) > _MAX_LENGTH:
        text = text[:_MAX_LENGTH] + "..."
    return text


class GeminiError(Exception):
    """Base class for every error raised by the Gemini layer."""


class GeminiConfigError(GeminiError, ValueError):
    """Missing or invalid Gemini configuration (e.g. no API key)."""


class GeminiAPIError(GeminiError):
    """The Gemini API returned an error or could not be reached."""

    def __init__(self, message: str, status: int | None = None, retryable: bool = False) -> None:
        super().__init__(message)
        self.status = status
        self.retryable = retryable


class GeminiTimeoutError(GeminiAPIError):
    """The request exceeded its time limit."""

    def __init__(self, message: str = "Gemini request timed out") -> None:
        super().__init__(message, status=None, retryable=True)


class GeminiBlockedError(GeminiError):
    """The response was blocked or empty (safety filters, no candidates)."""


class GeminiResponseError(GeminiError):
    """The model's output could not be turned into the requested validated structure."""
