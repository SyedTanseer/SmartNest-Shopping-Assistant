"""SmartNest persistence layer (MongoDB Atlas).

    settings.py      configuration from backend/.env (credential-safe)
    connection.py    one shared, thread-safe MongoClient + health check
    collections.py   established collection names
    repositories/    Repository interface and reusable CRUD implementation
    errors.py        DatabaseError hierarchy (no raw PyMongo errors escape)

Independent of the agents, the CNP/utility code, Gemini and the UI.  No
existing application data is stored yet.
"""

from smartnest.database.collections import ALL_COLLECTIONS, Collections
from smartnest.database.connection import (
    HealthStatus,
    MongoConnection,
    close_connection,
    get_connection,
)
from smartnest.database.errors import (
    DatabaseConfigError,
    DatabaseConnectionError,
    DatabaseError,
    DatabaseOperationError,
    DuplicateDocumentError,
    InvalidDocumentError,
    redact_text,
)
from smartnest.database.repositories import (
    BaseRepository,
    Document,
    InMemoryRepository,
    Repository,
)
from smartnest.database.settings import DatabaseSettings, load_settings

__all__ = [
    "ALL_COLLECTIONS",
    "BaseRepository",
    "Collections",
    "DatabaseConfigError",
    "DatabaseConnectionError",
    "DatabaseError",
    "DatabaseOperationError",
    "DatabaseSettings",
    "Document",
    "DuplicateDocumentError",
    "HealthStatus",
    "InMemoryRepository",
    "InvalidDocumentError",
    "MongoConnection",
    "Repository",
    "close_connection",
    "get_connection",
    "load_settings",
    "redact_text",
]
