"""Database configuration, loaded from ``backend/.env`` and the environment.

Variables:

    MONGODB_URI                          required  mongodb:// or mongodb+srv:// connection string
    MONGODB_DATABASE                     optional  database name (default ``smartnest``)
    MONGODB_SERVER_SELECTION_TIMEOUT_MS  optional  how long to wait for a server (default 5000)
    MONGODB_MAX_POOL_SIZE                optional  connection-pool ceiling (default 50)

Precedence: real environment variables win over values in ``.env``, so a
deployment can override a developer's file.  The ``.env`` file is read
without touching ``os.environ``.

Security rules enforced here:
    - The URI is stored with ``repr=False`` so it never appears in
      ``repr()``, logs or tracebacks of the settings object.
    - Validation errors never contain the URI or any part of it.
    - ``redacted_uri`` / ``host`` give a safe description for display.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping
from urllib.parse import unquote

from dotenv import dotenv_values

from smartnest.database.errors import DatabaseConfigError, redact_text

DEFAULT_DATABASE = "smartnest"
DEFAULT_APP_NAME = "smartnest"
DEFAULT_SERVER_SELECTION_TIMEOUT_MS = 5000
DEFAULT_CONNECT_TIMEOUT_MS = 5000
DEFAULT_SOCKET_TIMEOUT_MS = 20000
DEFAULT_MAX_POOL_SIZE = 50

#: ``backend/.env`` (settings.py lives in backend/src/smartnest/database/).
DEFAULT_ENV_FILE = Path(__file__).resolve().parents[3] / ".env"

_VALID_SCHEMES = ("mongodb://", "mongodb+srv://")
# MongoDB forbids these characters in database names.
_INVALID_DB_CHARS = re.compile(r'[/\\. "$*<>:|?\x00]')
_MAX_DB_NAME_LENGTH = 63


@dataclass(frozen=True)
class DatabaseSettings:
    """Validated MongoDB settings.  Construct via :func:`load_settings`."""

    uri: str = field(repr=False)
    database: str = DEFAULT_DATABASE
    app_name: str = DEFAULT_APP_NAME
    server_selection_timeout_ms: int = DEFAULT_SERVER_SELECTION_TIMEOUT_MS
    connect_timeout_ms: int = DEFAULT_CONNECT_TIMEOUT_MS
    socket_timeout_ms: int = DEFAULT_SOCKET_TIMEOUT_MS
    max_pool_size: int = DEFAULT_MAX_POOL_SIZE
    min_pool_size: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.uri, str) or not self.uri.strip():
            raise DatabaseConfigError("MONGODB_URI is not set")
        uri = self.uri.strip()
        if uri.startswith("<") or "<" in uri or ">" in uri:
            raise DatabaseConfigError(
                "MONGODB_URI still contains a placeholder such as <db_password>; "
                "replace it with the real value in backend/.env"
            )
        if not uri.lower().startswith(_VALID_SCHEMES):
            raise DatabaseConfigError("MONGODB_URI must start with mongodb:// or mongodb+srv://")
        if any(ch.isspace() for ch in uri):
            raise DatabaseConfigError("MONGODB_URI must not contain whitespace")
        object.__setattr__(self, "uri", uri)

        self._validate_database_name(self.database)
        for name in ("server_selection_timeout_ms", "connect_timeout_ms",
                     "socket_timeout_ms", "max_pool_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise DatabaseConfigError(f"{name} must be a positive integer")
        if isinstance(self.min_pool_size, bool) or not isinstance(self.min_pool_size, int) \
                or not 0 <= self.min_pool_size <= self.max_pool_size:
            raise DatabaseConfigError("min_pool_size must be between 0 and max_pool_size")
        if not isinstance(self.app_name, str) or not self.app_name.strip():
            raise DatabaseConfigError("app_name must be a non-empty string")

    @staticmethod
    def _validate_database_name(name: object) -> None:
        if not isinstance(name, str) or not name.strip():
            raise DatabaseConfigError("MONGODB_DATABASE must be a non-empty string")
        if len(name) > _MAX_DB_NAME_LENGTH:
            raise DatabaseConfigError(f"MONGODB_DATABASE must be at most {_MAX_DB_NAME_LENGTH} characters")
        if _INVALID_DB_CHARS.search(name):
            raise DatabaseConfigError('MONGODB_DATABASE contains an invalid character (/\\. "$*<>:|?)')

    # ── Safe descriptions ─────────────────────────────────────────────────

    @property
    def host(self) -> str:
        """Host part of the URI (no credentials, no path or options)."""
        remainder = self.uri.split("://", 1)[1].rsplit("@", 1)[-1]
        return re.split(r"[/?]", remainder, maxsplit=1)[0]

    @property
    def redacted_uri(self) -> str:
        """The URI with credentials masked, safe to print or log."""
        scheme = self.uri.split("://", 1)[0]
        return f"{scheme}://***@{self.host}/"

    @property
    def secret_fragments(self) -> tuple[str, ...]:
        """Substrings (password, ``user:password``) to mask in any message."""
        remainder = self.uri.split("://", 1)[1]
        if "@" not in remainder:
            return ()
        userinfo = remainder.rsplit("@", 1)[0]
        fragments = {userinfo, unquote(userinfo)}
        if ":" in userinfo:
            password = userinfo.split(":", 1)[1]
            fragments.update({password, unquote(password)})
        return tuple(sorted(fragments, key=len, reverse=True))

    def redact(self, text: str) -> str:
        """Mask this configuration's secrets (and any URI credentials) in ``text``."""
        return redact_text(text, self.secret_fragments)

    def __str__(self) -> str:
        return f"DatabaseSettings(host={self.host}, database={self.database})"


def _int_setting(raw: str | None, name: str, default: int) -> int:
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        raise DatabaseConfigError(f"{name} must be an integer") from None


def load_settings(
    env_file: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> DatabaseSettings:
    """Load and validate database settings.

    Args:
        env_file: Path to a ``.env`` file.  Defaults to ``backend/.env``;
            a missing file is fine (the environment alone may be enough).
        environ: Environment mapping that overrides the file (defaults to
            ``os.environ``).  Injectable for tests.

    Raises:
        DatabaseConfigError: If the URI is missing or any value is invalid.
            The message never contains the URI.
    """
    path = Path(env_file) if env_file is not None else DEFAULT_ENV_FILE
    file_values: dict[str, str | None] = {}
    if path.is_file():
        file_values = dict(dotenv_values(path, encoding="utf-8-sig"))
    env = os.environ if environ is None else environ

    def get(name: str) -> str | None:
        value = env.get(name)
        if value is not None and value.strip():
            return value
        return file_values.get(name)

    return DatabaseSettings(
        uri=get("MONGODB_URI") or "",
        database=(get("MONGODB_DATABASE") or DEFAULT_DATABASE).strip(),
        server_selection_timeout_ms=_int_setting(
            get("MONGODB_SERVER_SELECTION_TIMEOUT_MS"),
            "MONGODB_SERVER_SELECTION_TIMEOUT_MS",
            DEFAULT_SERVER_SELECTION_TIMEOUT_MS,
        ),
        max_pool_size=_int_setting(
            get("MONGODB_MAX_POOL_SIZE"), "MONGODB_MAX_POOL_SIZE", DEFAULT_MAX_POOL_SIZE
        ),
    )
