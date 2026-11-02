#!/usr/bin/env python3
"""Smallest possible check that the MongoDB connection string works.

Stores one random value, reads it back, and reports PASS or FAIL.

    python src/mongodb_smoke_test.py          # store, retrieve, then clean up
    python src/mongodb_smoke_test.py --keep   # leave the value in Atlas so you can see it there

The value goes into the 'smoke_test' collection of the database named in
backend/.env (MONGODB_DATABASE).  Credentials are never printed.
"""

from __future__ import annotations

import sys
import uuid

from smartnest.database import BaseRepository, DatabaseError, MongoConnection, load_settings


def main(keep: bool = False) -> int:
    try:
        settings = load_settings()
        print(f"Connecting to {settings.host}, database '{settings.database}' ...")
        with MongoConnection(settings) as connection:  # connects and pings
            repo = BaseRepository(connection.collection("smoke_test"), id_prefix="smoke")

            value = uuid.uuid4().hex
            doc_id = repo.create({"value": value})
            print(f"stored    : {value}  (document id {doc_id})")

            found = repo.get(doc_id)
            retrieved = found["value"] if found else None
            print(f"retrieved : {retrieved}")

            if keep:
                print(f"kept in Atlas: {settings.database}.smoke_test  (id {doc_id})")
            else:
                repo.delete(doc_id)
                if repo.count() == 0:
                    connection.database.drop_collection("smoke_test")

            if retrieved == value:
                print("PASS - the connection string works: the value was stored and read back.")
                return 0
            print("FAIL - the retrieved value does not match what was stored.")
            return 1
    except DatabaseError as exc:  # messages are already credential-safe
        print(f"FAIL - {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main(keep="--keep" in sys.argv[1:]))
