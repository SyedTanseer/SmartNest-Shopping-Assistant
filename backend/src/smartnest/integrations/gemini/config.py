"""Gemini configuration, loaded from ``backend/.env`` and the environment.

    GEMINI_API_KEY   required
    GEMINI_MODEL     optional (default below)
    GEMINI_TIMEOUT_SECONDS   optional (default 60)

Real environment variables win over ``.env``.  The key is stored with
``repr=False`` so it never appears in ``repr()``, logs or tracebacks, and
validation errors never include it.

The default model is the one the current Gemini API documentation
recommends as its fast, low-cost general model.  Models are retired over
time; set ``GEMINI_MODEL`` in ``.env`` to change it without code changes.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from dotenv import dotenv_values

from smartnest.integrations.gemini.errors import GeminiConfigError

DEFAULT_MODEL = "gemini-3.5-flash-lite"
DEFAULT_TIMEOUT_SECONDS = 60.0

#: backend/.env (this file lives in backend/src/smartnest/integrations/gemini/).
DEFAULT_ENV_FILE = Path(__file__).resolve().parents[4] / ".env"

_MODEL_NAME = re.compile(r"^[a-z0-9][a-z0-9.\-_]{1,80}$")
_PLACEHOLDERS = {"your_api_key", "your-api-key", "changeme", "<api_key>", "xxx", "todo"}


@dataclass(frozen=True)
class GeminiSettings:
    """Validated Gemini settings.  Build with :func:`load_gemini_settings`."""

    api_key: str = field(repr=False)
    model: str = DEFAULT_MODEL
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        key = (self.api_key or "").strip()
        if not key:
            raise GeminiConfigError("GEMINI_API_KEY is not set; add it to backend/.env")
        if key.lower() in _PLACEHOLDERS or key.startswith("<") or any(c.isspace() for c in key):
            raise GeminiConfigError("GEMINI_API_KEY still looks like a placeholder; put the real key in backend/.env")
        object.__setattr__(self, "api_key", key)
        if not isinstance(self.model, str) or not _MODEL_NAME.match(self.model):
            raise GeminiConfigError("GEMINI_MODEL must look like a model code, e.g. gemini-3.5-flash-lite")
        if isinstance(self.timeout_seconds, bool) or not isinstance(self.timeout_seconds, (int, float)) \
                or not 1 <= self.timeout_seconds <= 600:
            raise GeminiConfigError("GEMINI_TIMEOUT_SECONDS must be a number between 1 and 600")

    @property
    def secret_fragments(self) -> tuple[str, ...]:
        return (self.api_key,)

    def redact(self, text: str) -> str:
        from smartnest.integrations.gemini.errors import redact_secrets
        return redact_secrets(text, self.secret_fragments)

    def __str__(self) -> str:
        return f"GeminiSettings(model={self.model}, api_key=set)"


def load_gemini_settings(
    env_file: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> GeminiSettings:
    """Load settings from ``.env`` (default ``backend/.env``) and the environment.

    Raises:
        GeminiConfigError: if the key is missing/placeholder or a value is invalid.
    """
    path = Path(env_file) if env_file is not None else DEFAULT_ENV_FILE
    file_values: dict[str, str | None] = dict(dotenv_values(path, encoding="utf-8-sig")) if path.is_file() else {}
    env = os.environ if environ is None else environ

    def get(name: str) -> str | None:
        value = env.get(name)
        return value if value is not None and value.strip() else file_values.get(name)

    raw_timeout = get("GEMINI_TIMEOUT_SECONDS")
    try:
        timeout = float(raw_timeout) if raw_timeout else DEFAULT_TIMEOUT_SECONDS
    except ValueError:
        raise GeminiConfigError("GEMINI_TIMEOUT_SECONDS must be a number") from None
    return GeminiSettings(
        api_key=get("GEMINI_API_KEY") or "",
        model=(get("GEMINI_MODEL") or DEFAULT_MODEL).strip(),
        timeout_seconds=timeout,
    )


def gemini_configured(env_file: str | Path | None = None, environ: Mapping[str, str] | None = None) -> bool:
    """Whether a usable key is configured (never raises, never reveals the key)."""
    try:
        load_gemini_settings(env_file, environ)
    except GeminiConfigError:
        return False
    return True
