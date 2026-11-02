"""Optional integration tests against a live MongoDB Atlas cluster.

Skipped by default (they need network access and real credentials).  Run:

    PowerShell:   $env:SMARTNEST_RUN_ATLAS_TESTS = "1"; python -m pytest -m atlas
    bash:         SMARTNEST_RUN_ATLAS_TESTS=1 python -m pytest -m atlas

Every test works in a uniquely named scratch collection that is dropped
afterwards; SmartNest's real collections are never touched.
"""

from __future__ import annotations

import os
import threading
import uuid
from datetime import timezone

import pytest

from smartnest.database import (
    BaseRepository,
    DatabaseError,
    DuplicateDocumentError,
    InvalidDocumentError,
    MongoConnection,
    load_settings,
)

pytestmark = [
    pytest.mark.atlas,
    pytest.mark.skipif(
        os.environ.get("SMARTNEST_RUN_ATLAS_TESTS") != "1",
        reason="set SMARTNEST_RUN_ATLAS_TESTS=1 to run tests against MongoDB Atlas",
    ),
]


@pytest.fixture(scope="module")
def connection():
    conn = MongoConnection(load_settings())
    conn.connect(verify=True)
    yield conn
    conn.close()


@pytest.fixture()
def repo(connection):
    name = f"_smartnest_test_{uuid.uuid4().hex[:10]}"
    repository = BaseRepository(connection.collection(name), id_prefix="t")
    yield repository
    connection.database.drop_collection(name)


class TestAtlasConnection:
    def test_ping_and_health(self, connection):
        assert connection.ping() > 0
        status = connection.check_health()
        assert status.ok and status.latency_ms is not None

    def test_client_is_shared(self, connection):
        assert connection.client is connection.client


class TestAtlasCrud:
    def test_full_roundtrip(self, repo):
        doc_id = repo.create({"name": "Sofa", "price": 27499, "tags": ["a", "b"]})
        doc = repo.get(doc_id)
        assert doc["name"] == "Sofa" and doc["tags"] == ["a", "b"]
        assert doc["created_at"].tzinfo is not None and doc["created_at"].utcoffset() == timezone.utc.utcoffset(None)

        assert repo.update(doc_id, {"price": 25000}) is True
        assert repo.get(doc_id)["price"] == 25000
        assert repo.exists(doc_id) and repo.count() == 1

        assert repo.delete(doc_id) is True
        assert repo.get(doc_id) is None and repo.delete(doc_id) is False

    def test_duplicate_id_raises(self, repo):
        repo.create({"_id": "same"})
        with pytest.raises(DuplicateDocumentError):
            repo.create({"_id": "same"})

    def test_create_many_find_sort_limit_skip(self, repo):
        repo.create_many([{"_id": f"p{i}", "price": i * 10, "kind": "sofa" if i % 2 else "lamp"}
                          for i in range(1, 6)])
        assert [d["_id"] for d in repo.find({"kind": "sofa"}, sort=[("price", -1)])] == ["p5", "p3", "p1"]
        assert [d["_id"] for d in repo.find(sort=[("price", 1)], skip=1, limit=2)] == ["p2", "p3"]
        assert repo.count({"kind": "lamp"}) == 2

    def test_upsert_keeps_created_at(self, repo):
        assert repo.upsert("state", {"beliefs": {"budget": 100}}) is True
        first = repo.get("state")
        assert repo.upsert("state", {"beliefs": {"budget": 80}}) is False
        second = repo.get("state")
        assert second["beliefs"] == {"budget": 80} and second["created_at"] == first["created_at"]

    def test_update_many_and_delete_many(self, repo):
        repo.create_many([{"_id": "a", "k": 1}, {"_id": "b", "k": 1}, {"_id": "c", "k": 2}])
        assert repo.update_many({"k": 1}, {"seen": True}) == 2
        assert repo.delete_many({"k": 1}) == 2 and repo.count() == 1
        with pytest.raises(InvalidDocumentError):
            repo.delete_many({})

    def test_errors_are_database_errors(self, connection):
        repo = BaseRepository(connection.collection("_smartnest_never_created"))
        with pytest.raises(InvalidDocumentError):
            repo.get("")
        assert repo.get("missing") is None


class TestAtlasSharedClient:
    def test_worker_threads_share_one_client(self, repo):
        errors: list[Exception] = []

        def work(n: int):
            try:
                for i in range(5):
                    repo.create({"worker": n, "i": i})
            except Exception as exc:  # pragma: no cover - failure path
                errors.append(exc)

        threads = [threading.Thread(target=work, args=(n,)) for n in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert repo.count() == 30
