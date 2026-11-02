"""Tests for the Repository interface and BaseRepository CRUD (fake collection)."""

from __future__ import annotations

import threading
from datetime import datetime, timezone

import pytest
from pydantic import BaseModel
from pymongo.errors import OperationFailure, ServerSelectionTimeoutError

from tests.fake_mongo import BrokenCollection, FakeCollection
from smartnest.core.models import EnvironmentalEvent, EventType
from smartnest.database import (
    BaseRepository,
    DatabaseConnectionError,
    DatabaseOperationError,
    DuplicateDocumentError,
    InvalidDocumentError,
    Repository,
)


@pytest.fixture()
def repo() -> BaseRepository:
    return BaseRepository(FakeCollection("products"), id_prefix="prd")


# ── Interface ────────────────────────────────────────────────────────────────

class TestInterface:
    def test_base_repository_implements_repository(self, repo):
        assert isinstance(repo, Repository)

    def test_interface_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            Repository()

    def test_in_memory_implementation_can_stand_in(self):
        """Agent code depends on Repository, so any implementation will do."""
        class Dict(Repository):
            def __init__(self): self.d = {}
            def create(self, document): self.d[document["_id"]] = document; return document["_id"]
            def create_many(self, documents): return [self.create(x) for x in documents]
            def get(self, document_id): return self.d.get(document_id)
            def find_one(self, filter=None): return next(iter(self.d.values()), None)
            def find(self, filter=None, *, sort=None, limit=0, skip=0, projection=None): return list(self.d.values())
            def count(self, filter=None): return len(self.d)
            def exists(self, document_id): return document_id in self.d
            def update(self, document_id, changes): return document_id in self.d
            def update_many(self, filter, changes): return 0
            def upsert(self, document_id, document): return True
            def delete(self, document_id): return self.d.pop(document_id, None) is not None
            def delete_many(self, filter, *, allow_all=False): return 0

        r = Dict()
        assert r.create({"_id": "a"}) == "a" and r.exists("a")

    def test_prefix_must_be_non_empty(self):
        with pytest.raises(ValueError):
            BaseRepository(FakeCollection(), id_prefix=" ")


# ── Create ───────────────────────────────────────────────────────────────────

class TestCreate:
    def test_generates_prefixed_string_id(self, repo):
        doc_id = repo.create({"name": "Sofa"})
        assert isinstance(doc_id, str) and doc_id.startswith("prd_") and len(doc_id) == 20

    def test_ids_are_unique(self, repo):
        assert len({repo.create({"n": i}) for i in range(100)}) == 100

    def test_keeps_supplied_id(self, repo):
        assert repo.create({"_id": "SOFA_A1", "name": "Sofa"}) == "SOFA_A1"
        assert repo.get("SOFA_A1")["name"] == "Sofa"

    def test_none_id_is_replaced_by_generated(self, repo):
        assert repo.create({"_id": None, "x": 1}).startswith("prd_")

    @pytest.mark.parametrize("bad", [123, "", "   ", ["a"]])
    def test_invalid_id_rejected(self, repo, bad):
        with pytest.raises(InvalidDocumentError):
            repo.create({"_id": bad})

    def test_timestamps_are_set_and_timezone_aware(self, repo):
        before = datetime.now(timezone.utc)
        doc = repo.get(repo.create({"x": 1}))
        assert doc["created_at"].tzinfo is not None
        assert before <= doc["created_at"] <= datetime.now(timezone.utc)
        assert doc["updated_at"] == doc["created_at"]

    def test_supplied_created_at_is_kept(self, repo):
        stamp = datetime(2020, 1, 1, tzinfo=timezone.utc)
        assert repo.get(repo.create({"created_at": stamp}))["created_at"] == stamp

    def test_input_document_is_not_mutated(self, repo):
        original = {"name": "Sofa"}
        repo.create(original)
        assert original == {"name": "Sofa"}

    def test_duplicate_id_raises(self, repo):
        repo.create({"_id": "dup"})
        with pytest.raises(DuplicateDocumentError):
            repo.create({"_id": "dup"})

    def test_accepts_pydantic_model(self, repo):
        event = EnvironmentalEvent(event_id="evt_1", event_type=EventType.PRICE_CHANGE,
                                   payload={"new_price": 5})
        repo.create({**event.model_dump(), "_id": event.event_id})
        stored = repo.get("evt_1")
        assert stored["event_type"] == "price_change" and stored["payload"] == {"new_price": 5}

    def test_pydantic_model_without_id_gets_generated_id(self, repo):
        class Item(BaseModel):
            name: str
        assert repo.get(repo.create(Item(name="lamp")))["name"] == "lamp"

    @pytest.mark.parametrize("bad", [None, "text", 5, ["a"]])
    def test_non_mapping_rejected(self, repo, bad):
        with pytest.raises(InvalidDocumentError):
            repo.create(bad)

    def test_create_many(self, repo):
        ids = repo.create_many([{"n": 1}, {"_id": "x", "n": 2}, {"n": 3}])
        assert len(ids) == 3 and ids[1] == "x" and repo.count() == 3

    def test_create_many_empty_is_noop(self, repo):
        assert repo.create_many([]) == []
        assert repo.collection.calls == []

    def test_create_many_duplicate(self, repo):
        repo.create({"_id": "a"})
        with pytest.raises(DuplicateDocumentError):
            repo.create_many([{"_id": "b"}, {"_id": "a"}])

    def test_create_many_validates_everything_before_writing(self, repo):
        with pytest.raises(InvalidDocumentError):
            repo.create_many([{"n": 1}, "oops"])
        assert repo.count() == 0


# ── Read ─────────────────────────────────────────────────────────────────────

class TestRead:
    @pytest.fixture()
    def filled(self, repo):
        repo.create_many([
            {"_id": "a", "kind": "sofa", "price": 30},
            {"_id": "b", "kind": "sofa", "price": 10},
            {"_id": "c", "kind": "lamp", "price": 20},
            {"_id": "d", "kind": "sofa", "price": 20},
        ])
        return repo

    def test_get_missing_returns_none(self, repo):
        assert repo.get("nope") is None

    @pytest.mark.parametrize("bad", ["", None, 5])
    def test_get_requires_string_id(self, repo, bad):
        with pytest.raises(InvalidDocumentError):
            repo.get(bad)

    def test_find_one(self, filled):
        assert filled.find_one({"kind": "lamp"})["_id"] == "c"
        assert filled.find_one({"kind": "table"}) is None

    def test_find_all(self, filled):
        assert len(filled.find()) == 4

    def test_find_with_filter(self, filled):
        assert {d["_id"] for d in filled.find({"kind": "sofa"})} == {"a", "b", "d"}

    def test_sort_limit_skip(self, filled):
        ids = [d["_id"] for d in filled.find(sort=[("price", 1), ("_id", 1)])]
        assert ids == ["b", "c", "d", "a"]
        assert [d["_id"] for d in filled.find(sort=[("price", -1)], limit=2)] == ["a", "c"] \
            or [d["_id"] for d in filled.find(sort=[("price", -1)], limit=2)][0] == "a"
        assert [d["_id"] for d in filled.find(sort=[("price", 1), ("_id", 1)], skip=1, limit=2)] == ["c", "d"]

    @pytest.mark.parametrize("kwargs", [{"limit": -1}, {"skip": -1}, {"limit": True}, {"limit": 1.5}])
    def test_invalid_limit_or_skip(self, filled, kwargs):
        with pytest.raises(InvalidDocumentError):
            filled.find(**kwargs)

    def test_invalid_filter_type(self, filled):
        with pytest.raises(InvalidDocumentError):
            filled.find(["not", "a", "mapping"])

    def test_count_and_exists(self, filled):
        assert filled.count() == 4 and filled.count({"kind": "sofa"}) == 3
        assert filled.exists("a") and not filled.exists("zzz")

    def test_reads_return_copies(self, filled):
        doc = filled.get("a")
        doc["price"] = 999
        assert filled.get("a")["price"] == 30


# ── Update / upsert ──────────────────────────────────────────────────────────

class TestUpdate:
    def test_update_sets_fields_and_bumps_updated_at(self, repo):
        doc_id = repo.create({"_id": "u", "price": 10, "stock": 5})
        before = repo.get(doc_id)
        assert repo.update(doc_id, {"price": 12}) is True
        after = repo.get(doc_id)
        assert after["price"] == 12 and after["stock"] == 5
        assert after["created_at"] == before["created_at"]
        assert after["updated_at"] >= before["updated_at"]

    def test_update_missing_returns_false(self, repo):
        assert repo.update("ghost", {"x": 1}) is False

    def test_update_cannot_change_id_or_use_operators(self, repo):
        repo.create({"_id": "u"})
        for bad in ({"_id": "other"}, {"$set": {"x": 1}}, {"$inc": {"n": 1}}):
            with pytest.raises(InvalidDocumentError):
                repo.update("u", bad)

    def test_update_empty_changes_rejected(self, repo):
        repo.create({"_id": "u"})
        with pytest.raises(InvalidDocumentError):
            repo.update("u", {})

    def test_update_ignores_created_at_changes(self, repo):
        repo.create({"_id": "u"})
        created = repo.get("u")["created_at"]
        repo.update("u", {"created_at": datetime(1999, 1, 1, tzinfo=timezone.utc), "x": 1})
        assert repo.get("u")["created_at"] == created

    def test_update_many(self, repo):
        repo.create_many([{"_id": "a", "k": 1}, {"_id": "b", "k": 1}, {"_id": "c", "k": 2}])
        assert repo.update_many({"k": 1}, {"flag": True}) == 2
        assert repo.get("a")["flag"] and not repo.get("c").get("flag")
        assert repo.update_many({"k": 99}, {"flag": True}) == 0

    def test_upsert_inserts_then_updates_keeping_created_at(self, repo):
        assert repo.upsert("state_1", {"beliefs": {"budget": 100}}) is True
        first = repo.get("state_1")
        assert first["beliefs"] == {"budget": 100} and first["_id"] == "state_1"
        assert repo.upsert("state_1", {"beliefs": {"budget": 80}}) is False
        second = repo.get("state_1")
        assert second["beliefs"] == {"budget": 80}
        assert second["created_at"] == first["created_at"]
        assert repo.count() == 1

    def test_upsert_ignores_id_inside_document(self, repo):
        repo.upsert("real", {"_id": "ignored", "x": 1})
        assert repo.get("real")["x"] == 1 and repo.get("ignored") is None


# ── Delete ───────────────────────────────────────────────────────────────────

class TestDelete:
    def test_delete(self, repo):
        repo.create({"_id": "d"})
        assert repo.delete("d") is True
        assert repo.get("d") is None
        assert repo.delete("d") is False

    def test_delete_many_with_filter(self, repo):
        repo.create_many([{"_id": "a", "k": 1}, {"_id": "b", "k": 1}, {"_id": "c", "k": 2}])
        assert repo.delete_many({"k": 1}) == 2
        assert repo.count() == 1

    def test_delete_many_refuses_empty_filter(self, repo):
        repo.create_many([{"_id": "a"}, {"_id": "b"}])
        with pytest.raises(InvalidDocumentError, match="every document"):
            repo.delete_many({})
        assert repo.count() == 2

    def test_delete_many_allow_all(self, repo):
        repo.create_many([{"_id": "a"}, {"_id": "b"}])
        assert repo.delete_many({}, allow_all=True) == 2


# ── Error handling ───────────────────────────────────────────────────────────

class TestErrorHandling:
    SECRET_URI = "mongodb+srv://demo_user:fakePass123@cluster.example.mongodb.net/"

    @pytest.mark.parametrize("call", [
        lambda r: r.create({"x": 1}),
        lambda r: r.get("a"),
        lambda r: r.find(),
        lambda r: r.count(),
        lambda r: r.update("a", {"x": 1}),
        lambda r: r.upsert("a", {"x": 1}),
        lambda r: r.delete("a"),
    ])
    def test_unreachable_server_is_connection_error(self, call):
        repo = BaseRepository(BrokenCollection(ServerSelectionTimeoutError("timed out")))
        with pytest.raises(DatabaseConnectionError):
            call(repo)

    def test_operation_failure_is_operation_error(self):
        repo = BaseRepository(BrokenCollection(OperationFailure("not authorized", 13)))
        with pytest.raises(DatabaseOperationError, match="not authorized"):
            repo.get("a")

    def test_no_raw_pymongo_exceptions_escape(self):
        repo = BaseRepository(BrokenCollection(OperationFailure("x", 1)))
        try:
            repo.count()
        except Exception as exc:
            assert type(exc).__module__.startswith("smartnest.database")

    def test_error_messages_redact_credentials(self):
        err = ServerSelectionTimeoutError(f"failed connecting to {self.SECRET_URI}")
        repo = BaseRepository(BrokenCollection(err))
        with pytest.raises(DatabaseConnectionError) as info:
            repo.get("a")
        assert "fakePass123" not in str(info.value) and "demo_user" not in str(info.value)

    def test_error_names_the_action_and_collection(self):
        repo = BaseRepository(BrokenCollection(OperationFailure("boom", 1)))
        with pytest.raises(DatabaseOperationError, match=r"delete on 'broken'"):
            repo.delete("a")

    def test_duplicate_message_does_not_echo_server_text(self):
        repo = BaseRepository(FakeCollection())
        repo.create({"_id": "secret-email@example.com"})
        with pytest.raises(DuplicateDocumentError) as info:
            repo.create({"_id": "secret-email@example.com"})
        assert "secret-email" not in str(info.value)


# ── Thread safety ────────────────────────────────────────────────────────────

class TestSharedUse:
    def test_concurrent_creates_from_worker_threads(self):
        repo = BaseRepository(FakeCollection(), id_prefix="w")
        ids: list[str] = []
        lock = threading.Lock()

        def work():
            for i in range(25):
                new_id = repo.create({"i": i})
                with lock:
                    ids.append(new_id)

        threads = [threading.Thread(target=work) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(ids) == 200 and len(set(ids)) == 200 and repo.count() == 200
