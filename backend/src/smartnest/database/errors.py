"""Database error types and credential-safe error translation.

Callers (agents, workers, the future API) should never have to import
PyMongo exceptions.  Everything the database layer raises is a
:class:`DatabaseError`:

    DatabaseError
    ├── DatabaseConfigError      bad / missing configuration  (also a ValueError)
    ├── DatabaseConnectionError  cannot reach or authenticate with the server
    ├── DatabaseOperationError   a read/write failed
    │   └── DuplicateDocumentError   unique key (``_id``) already exists
    └── InvalidDocumentError     caller passed an unusable document / id / filter

Security: error messages must never contain the connection string or
password.  :func:`redact_text` masks ``user:password@`` in any
``mongodb://`` / ``mongodb+srv://`` URI, and masks any extra secret
fragments it is given (e.g. the bare password), before a message is
built.
"""

from __future__ import annotations

import re
from typing import Iterable

from pymongo.errors import (
    BulkWriteError,
    ConnectionFailure,
    DuplicateKeyError,
    PyMongoError,
)

MASK = "***"

# mongodb://<user>:<password>@host  (also mongodb+srv://)  →  credentials masked
_URI_CREDENTIALS = re.compile(r"(mongodb(?:\+srv)?://)[^@/\s]+@", re.IGNORECASE)

# Secrets shorter than this are not masked individually (they would mangle
# ordinary words); the URI pattern above still masks them inside URIs.
_MIN_SECRET_LENGTH = 4

_MAX_MESSAGE_LENGTH = 300


class DatabaseError(Exception):
    """Base class for every error raised by the database layer."""


class DatabaseConfigError(DatabaseError, ValueError):
    """Missing or invalid database configuration."""


class DatabaseConnectionError(DatabaseError):
    """The MongoDB server could not be reached, or refused the connection."""


class DatabaseOperationError(DatabaseError):
    """A database read or write failed."""


class DuplicateDocumentError(DatabaseOperationError):
    """A document with the same ``_id`` already exists."""


class InvalidDocumentError(DatabaseError, ValueError):
    """The caller supplied an invalid document, identifier or filter."""


def redact_text(text: str, secrets: Iterable[str] = ()) -> str:
    """Mask credentials in ``text`` and truncate it to a sane length.

    Args:
        text: Arbitrary message text (e.g. ``str(exception)``).
        secrets: Extra substrings to mask wherever they appear
            (e.g. the password and the full ``user:password`` pair).
    """
    text = _URI_CREDENTIALS.sub(rf"\1{MASK}@", text)
    for secret in secrets:
        if secret and len(secret) >= _MIN_SECRET_LENGTH:
            text = text.replace(secret, MASK)
    if len(text) > _MAX_MESSAGE_LENGTH:
        text = text[:_MAX_MESSAGE_LENGTH] + "..."
    return text


def translate_error(
    exc: BaseException,
    action: str,
    secrets: Iterable[str] = (),
) -> DatabaseError:
    """Convert a PyMongo (or other) exception into a :class:`DatabaseError`.

    The result's message names the failed ``action`` and carries a
    credential-redacted description.  Duplicate-key errors deliberately
    omit the server text, which would echo the duplicated value.
    """
    if isinstance(exc, DuplicateKeyError) or _is_duplicate_bulk_error(exc):
        return DuplicateDocumentError(f"{action} failed: a document with this _id already exists")
    detail = redact_text(str(exc), secrets)
    if isinstance(exc, ConnectionFailure):  # includes server-selection timeouts
        return DatabaseConnectionError(f"{action} failed: cannot reach MongoDB ({type(exc).__name__}): {detail}")
    if isinstance(exc, PyMongoError):
        return DatabaseOperationError(f"{action} failed ({type(exc).__name__}): {detail}")
    return DatabaseOperationError(f"{action} failed ({type(exc).__name__}): {detail}")


def _is_duplicate_bulk_error(exc: BaseException) -> bool:
    if not isinstance(exc, BulkWriteError):
        return False
    errors = (exc.details or {}).get("writeErrors", [])
    return any(err.get("code") == 11000 for err in errors)
