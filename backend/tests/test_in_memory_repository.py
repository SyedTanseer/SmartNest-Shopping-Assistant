"""Tests for InMemoryRepository: same rules as BaseRepository, no server."""

from __future__ import annotations

import threading

import pytest

from smartnest.database import (
    BaseRepository,
    DuplicateDocumentError,
    InMemoryRepository,
    InvalidDocumentError,
    Repository,
)
from smartnest.database.repositories import InMemoryCollection


@pytest.fixture()
def repo() -> InMemoryRepository:
    return InMemoryRepository("agent_states", id_prefix="agent")


class TestContract:
    def test_is_a_repository(self, repo):
        assert isinstance(repo, BaseRepository) and isinstance(repo, Repository) and repo.name == "agent_states"

    def test_create_get_update_delete(self, repo):
        doc_id = repo.create({"name": "x", "n": 1})
        assert doc_id.startswith("agent_")
        assert repo.get(doc_id)["name"] == "x"
        assert repo.update(doc_id, {"n": 2}) is True and repo.get(doc_id)["n"] == 2
        assert repo.exists(doc_id) and repo.count() == 1
        assert repo.delete(doc_id) is True and repo.get(doc_id) is None and repo.delete(doc_id) is False

    def test_upsert_keeps_created_at(self, repo):
        assert repo.upsert("a1", {"v": 1}) is True
        first = repo.get("a1")
        assert repo.upsert("a1", {"v": 2}) is False
        second = repo.get("a1")
        assert second["v"] == 2 and second["created_at"] == first["created_at"]
        assert second["updated_at"] >= first["updated_at"]

    def test_duplicates_and_invalid_input(self, repo):
        repo.create({"_id": "dup"})
        with pytest.raises(DuplicateDocumentError):
            repo.create({"_id": "dup"})
        with pytest.raises(InvalidDocumentError):
            repo.create({"_id": 5})
        with pytest.raises(InvalidDocumentError):
            repo.delete_many({})

    def test_find_sort_skip_limit(self, repo):
        repo.create_many([{"_id": f"p{i}", "price": 100 - i * 10, "kind": "a" if i % 2 else "b"} for i in range(5)])
        assert [d["_id"] for d in repo.find({"kind": "a"}, sort=[("price", 1)])] == ["p3", "p1"]
        assert [d["_id"] for d in repo.find(sort=[("price", -1)], skip=1, limit=2)] == ["p1", "p2"]

    def test_documents_are_copied_in_and_out(self, repo):
        original = {"_id": "c", "nested": {"a": [1]}}
        repo.create(original)
        original["nested"]["a"].append(2)
        fetched = repo.get("c")
        assert fetched["nested"]["a"] == [1]
        fetched["nested"]["a"].append(3)
        assert repo.get("c")["nested"]["a"] == [1]

    def test_bulk_duplicate(self, repo):
        repo.create({"_id": "a"})
        with pytest.raises(DuplicateDocumentError):
            repo.create_many([{"_id": "b"}, {"_id": "a"}])

    def test_instances_are_isolated(self):
        a, b = InMemoryRepository(), InMemoryRepository()
        a.create({"_id": "x"})
        assert b.count() == 0

    def test_collection_records_calls_for_assertions(self, repo):
        repo.create({"_id": "x"})
        repo.get("x")
        assert repo.collection.calls == ["insert_one", "find_one"]

    def test_thread_safe(self, repo):
        def work(n):
            for i in range(30):
                repo.create({"w": n, "i": i})
        threads = [threading.Thread(target=work, args=(n,)) for n in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert repo.count() == 180

    def test_collection_usable_directly(self):
        collection = InMemoryCollection("raw")
        collection.insert_one({"_id": 1, "v": "a"})
        assert collection.count_documents({"v": "a"}) == 1 and collection.count_documents({"v": "a"}, limit=1) == 1
        assert collection.find_one({"v": "z"}) is None
