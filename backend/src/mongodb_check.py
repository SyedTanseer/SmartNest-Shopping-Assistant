#!/usr/bin/env python3
"""SmartNest — MongoDB Atlas connectivity check.

Run from the ``backend`` directory:

    python src/mongodb_check.py

Steps:
    1. Load configuration from backend/.env (and the environment).
    2. Connect to Atlas.
    3. Ping the server.
    4. Insert a temporary test document.
    5. Retrieve it and verify its contents.
    6. Delete it and verify it is gone.
    7. Close the connection.

The test document lives in a scratch collection (not one of SmartNest's
real collections).  The document is always deleted, and the scratch
collection is dropped again if this run created it and it is empty, so
the database is left as it was found.

Credentials are never printed: only the host name and database name are
shown, and every error message is passed through credential redaction.
Exit status: 0 if every step succeeded, 1 otherwise.
"""

from __future__ import annotations

import sys
import uuid
from typing import Callable

from smartnest.database import (
    BaseRepository,
    DatabaseError,
    MongoConnection,
    load_settings,
)

SCRATCH_COLLECTION = "_smartnest_connectivity_check"


def _safe_error(exc: BaseException, connection: MongoConnection | None) -> str:
    text = f"{type(exc).__name__}: {exc}"
    return connection.settings.redact(text) if connection is not None else text


def run(
    connection: MongoConnection | None = None,
    out: Callable[[str], None] = print,
) -> int:
    """Run the check and return the process exit status.

    Args:
        connection: Connection to use.  By default one is built from
            ``backend/.env``.  Injectable for tests.
        out: Output function (defaults to ``print``).
    """
    out("SmartNest MongoDB Atlas connectivity check")
    out("=" * 50)

    repo: BaseRepository | None = None
    doc_id: str | None = None
    created_scratch = False
    ok = True

    try:
        # 1. Configuration
        if connection is None:
            try:
                connection = MongoConnection(load_settings())
            except DatabaseError as exc:
                out(f"[1/7] FAIL  configuration: {exc}")
                return 1
        settings = connection.settings
        out(f"[1/7] OK    configuration loaded  (host={settings.host}, database={settings.database})")

        # 2. Connect
        connection.connect(verify=False)
        out("[2/7] OK    client created (connection pool ready)")

        # 3. Ping
        latency = connection.ping()
        out(f"[3/7] OK    ping succeeded        ({latency:.0f} ms)")

        # 4. Insert
        scratch_existed = SCRATCH_COLLECTION in connection.database.list_collection_names()
        created_scratch = not scratch_existed
        repo = BaseRepository(connection.collection(SCRATCH_COLLECTION), id_prefix="check")
        token = uuid.uuid4().hex
        doc_id = repo.create({"purpose": "connectivity check", "token": token, "nested": {"n": 1}})
        out(f"[4/7] OK    inserted test document ({doc_id})")

        # 5. Retrieve and verify
        fetched = repo.get(doc_id)
        if fetched is None:
            raise DatabaseError("inserted document was not found")
        expected = {"purpose": "connectivity check", "token": token, "nested": {"n": 1}}
        if any(fetched.get(k) != v for k, v in expected.items()) or fetched["_id"] != doc_id:
            raise DatabaseError("retrieved document does not match what was inserted")
        if fetched["created_at"].tzinfo is None:
            raise DatabaseError("timestamps are not timezone-aware")
        out("[5/7] OK    retrieved document and verified its contents")

        # 6. Delete and verify
        if not repo.delete(doc_id):
            raise DatabaseError("delete reported that nothing was deleted")
        if repo.get(doc_id) is not None:
            raise DatabaseError("document still present after delete")
        doc_id = None
        out("[6/7] OK    deleted test document and verified it is gone")

    except DatabaseError as exc:
        ok = False
        out(f"FAIL  {_safe_error(exc, connection)}")
    except Exception as exc:  # unexpected: still never print credentials
        ok = False
        out(f"FAIL  unexpected {_safe_error(exc, connection)}")
    finally:
        # Cleanup runs on success and failure alike.
        if repo is not None and doc_id is not None:
            try:
                if repo.delete(doc_id):
                    out("      cleaned up leftover test document")
            except Exception as exc:
                out(f"      WARNING: could not remove test document {doc_id}: {_safe_error(exc, connection)}")
        if repo is not None and created_scratch and connection is not None:
            try:
                if repo.count() == 0:
                    connection.database.drop_collection(SCRATCH_COLLECTION)
            except Exception as exc:
                out(f"      WARNING: could not drop scratch collection: {_safe_error(exc, connection)}")
        # 7. Close (always, success or failure)
        if connection is not None:
            connection.close()
            if ok:
                out("[7/7] OK    connection closed")

    out("=" * 50)
    out("RESULT: SUCCESS - MongoDB Atlas is reachable and read/write/delete work" if ok
        else "RESULT: FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(run())
