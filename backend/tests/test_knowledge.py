"""Tests for knowledge records and the knowledge store."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone

import pytest

from smartnest.database import InMemoryRepository
from smartnest.knowledge import (
    KnowledgeError,
    KnowledgeKind,
    KnowledgeRecord,
    KnowledgeStore,
    KnowledgeStoreError,
    Provenance,
    Relationship,
    RequirementsAttrs,
    make_record,
)
from tests.fake_mongo import BrokenCollection

USER = Provenance(source="user", source_type="user", method="chat")
LLM = Provenance(source="gemini", source_type="llm", method="requirement_extraction")


def req(**attrs) -> KnowledgeRecord:
    return make_record(KnowledgeKind.REQUIREMENTS, "user:u1", attrs or {"sofa_budget": 30000}, USER)


def product(**over) -> KnowledgeRecord:
    attrs = {"product_id": "P1", "name": "Sofa", "price": 25000, "width_cm": 200, "depth_cm": 90,
             "seating_capacity": 3, "rating": 4.2, "review_count": 10, "availability": "In_Stock",
             "url": "https://shop.example.com/p/1", "verified_at": datetime(2026, 10, 1, tzinfo=timezone.utc)}
    attrs.update(over)
    return make_record(KnowledgeKind.PRODUCT_SPEC, "product:P1", attrs,
                       Provenance(source="gemini_search", source_type="web", url="https://shop.example.com/p/1"))


# ── Creation and validation ─────────────────────────────────────────────────

class TestCreation:
    def test_every_record_has_identity_entity_attributes_provenance_confidence_timestamps(self):
        r = req(sofa_budget=30000, preferred_style=" Minimalist ")
        assert r.knowledge_id.startswith("req_") and r.entity == "user:u1"
        assert r.attributes["preferred_style"] == "minimalist" and r.attributes["sofa_budget"] == 30000
        assert r.provenance.source == "user" and r.provenance.retrieved_at.tzinfo is not None
        assert r.confidence == 1.0 and r.version == 1 and r.valid
        assert r.recorded_at.tzinfo is not None and r.modified_at.tzinfo is not None

    @pytest.mark.parametrize("kind,prefix", [
        (KnowledgeKind.REQUIREMENTS, "req_"), (KnowledgeKind.HOME_PROJECT, "prj_"), (KnowledgeKind.ROOM, "room_"),
        (KnowledgeKind.BUDGET, "bud_"), (KnowledgeKind.PRODUCT_SPEC, "prd_"),
        (KnowledgeKind.OBSERVATION, "obs_"), (KnowledgeKind.AGENT_KNOWLEDGE, "agk_"),
    ])
    def test_ids_are_unique_and_prefixed_by_kind(self, kind, prefix):
        attrs = {
            KnowledgeKind.REQUIREMENTS: {"sofa_budget": 1}, KnowledgeKind.HOME_PROJECT: {"name": "x"},
            KnowledgeKind.ROOM: {"name": "r", "width_cm": 100, "length_cm": 200},
            KnowledgeKind.BUDGET: {"total_budget": 100}, KnowledgeKind.PRODUCT_SPEC: {"product_id": "p", "name": "n"},
            KnowledgeKind.OBSERVATION: {"observation_type": "price_change", "subject": "p"},
            KnowledgeKind.AGENT_KNOWLEDGE: {"agent_id": "a", "topic": "t"},
        }[kind]
        ids = {make_record(kind, "e", attrs, USER).knowledge_id for _ in range(20)}
        assert len(ids) == 20 and all(i.startswith(prefix) for i in ids)

    def test_explicit_id_is_kept(self):
        assert make_record(KnowledgeKind.HOME_PROJECT, "p", {"name": "x"}, USER, knowledge_id="prj_custom").knowledge_id == "prj_custom"

    def test_relationships(self):
        room = make_record(KnowledgeKind.ROOM, "room:r", {"name": "r", "width_cm": 1, "length_cm": 2}, USER,
                           relationships=[Relationship(relation="part_of", target_id="prj_1", target_kind="home_project")])
        assert room.related("part_of")[0].target_id == "prj_1" and room.related("sold_by") == []

    @pytest.mark.parametrize("bad", [
        {"sofa_budget": -1}, {"sofa_budget": 0}, {"sofa_budget": math.nan}, {"max_width_cm": math.inf},
        {"min_seating": 0}, {"min_seating": True}, {"currency": "USD"}, {"unknown_field": 1},
    ])
    def test_invalid_requirements_rejected(self, bad):
        with pytest.raises(KnowledgeError):
            make_record(KnowledgeKind.REQUIREMENTS, "user:u1", bad, USER)

    @pytest.mark.parametrize("bad", [{"price": -5}, {"rating": 5.5}, {"rating": math.nan}, {"seating_capacity": 0},
                                     {"review_count": -1}, {"width_cm": 0}])
    def test_invalid_product_specs_rejected(self, bad):
        with pytest.raises(KnowledgeError):
            product(**bad)

    def test_product_spec_normalises_and_keeps_provenance_fields(self):
        p = product()
        assert p.attributes["availability"] == "in_stock" and p.provenance.url == "https://shop.example.com/p/1"
        assert p.attributes["verified_at"].startswith("2026-10-01")

    def test_room_budget_project_validation(self):
        with pytest.raises(KnowledgeError):
            make_record(KnowledgeKind.ROOM, "room:r", {"name": "r", "width_cm": -1, "length_cm": 2}, USER)
        with pytest.raises(KnowledgeError):
            make_record(KnowledgeKind.BUDGET, "b", {"total_budget": 100, "allocations": {"sofa": 80, "rug": 50}}, USER)
        with pytest.raises(KnowledgeError):
            make_record(KnowledgeKind.HOME_PROJECT, "p", {"name": ""}, USER)
        assert make_record(KnowledgeKind.BUDGET, "b", {"total_budget": 100, "allocations": {"sofa": 60}}, USER)

    @pytest.mark.parametrize("confidence", [-0.1, 1.1, math.nan, True])
    def test_confidence_must_be_between_zero_and_one(self, confidence):
        with pytest.raises(KnowledgeError):
            make_record(KnowledgeKind.REQUIREMENTS, "u", {"sofa_budget": 1}, USER, confidence=confidence)

    def test_entity_and_provenance_source_required(self):
        with pytest.raises(KnowledgeError):
            make_record(KnowledgeKind.REQUIREMENTS, "  ", {"sofa_budget": 1}, USER)
        with pytest.raises(ValueError):
            Provenance(source="  ", source_type="user")

    def test_typed_view_and_belief_values(self):
        r = req(sofa_budget=30000, max_width_cm=210, preferred_style="minimalist", notes="n")
        typed = r.typed()
        assert isinstance(typed, RequirementsAttrs)
        assert typed.belief_values() == {"sofa_budget": 30000.0, "max_width_cm": 210.0, "preferred_style": "minimalist"}


# ── Updates, history and invalidation ───────────────────────────────────────

class TestUpdates:
    def test_update_changes_attributes_bumps_version_and_keeps_history(self):
        r = req(sofa_budget=30000, preferred_style="minimalist")
        diff = r.update({"preferred_style": "contemporary"}, LLM, confidence=0.8)
        assert diff == {"preferred_style": ("minimalist", "contemporary")}
        assert r.version == 2 and r.attributes["preferred_style"] == "contemporary"
        assert r.provenance.source == "gemini" and r.confidence == 0.8
        assert r.history[0].version == 1 and r.history[0].previous == {"preferred_style": "minimalist"}
        assert r.history[0].replaced_by == "gemini"
        assert r.modified_at >= r.recorded_at

    def test_update_with_no_change_is_a_no_op(self):
        r = req(sofa_budget=30000)
        version, modified = r.version, r.modified_at
        assert r.update({"sofa_budget": 30000}, LLM) == {}
        assert (r.version, r.modified_at, r.history) == (version, modified, [])

    def test_none_removes_an_attribute(self):
        r = req(sofa_budget=30000, preferred_style="minimalist")
        assert r.update({"preferred_style": None}, USER) == {"preferred_style": ("minimalist", None)}
        assert "preferred_style" not in r.attributes

    def test_new_attribute_history_records_absence(self):
        r = req(sofa_budget=30000)
        r.update({"min_seating": 3}, USER)
        assert r.history[0].previous == {"min_seating": None}

    def test_invalid_update_leaves_the_record_untouched(self):
        r = req(sofa_budget=30000)
        before = r.model_dump()
        with pytest.raises(KnowledgeError):
            r.update({"sofa_budget": -5}, LLM)
        assert r.model_dump() == before

    def test_multiple_updates_accumulate_history(self):
        r = product()
        for price in (24000, 23000, 22000):
            r.update({"price": price}, USER)
        assert r.version == 4 and [h.previous["price"] for h in r.history] == [25000, 24000, 23000]

    def test_invalidation(self):
        r = req()
        assert r.invalidate("superseded by a new request") is True
        assert not r.valid and r.invalidated_at is not None and r.invalidation_reason == "superseded by a new request"
        assert r.invalidate("again") is False

    def test_invalidation_needs_a_reason(self):
        with pytest.raises(KnowledgeError):
            req().invalidate("  ")

    def test_invalidated_records_cannot_be_updated(self):
        r = req()
        r.invalidate("no longer true")
        with pytest.raises(KnowledgeError, match="invalidated"):
            r.update({"sofa_budget": 1}, USER)


class TestSerialisation:
    def test_round_trip_through_json(self):
        r = product()
        r.update({"price": 20000}, USER)
        r.relationships.append(Relationship(relation="derived_from", target_id="req_1"))
        document = json.loads(json.dumps(r.to_document()))
        back = KnowledgeRecord.from_document(document)
        assert back == r and back.history[0].previous == {"price": 25000.0}

    def test_corrupt_document_is_a_knowledge_error(self):
        with pytest.raises(KnowledgeError):
            KnowledgeRecord.from_document({"kind": "requirements", "entity": "x", "attributes": {"sofa_budget": -1}})
        with pytest.raises(KnowledgeError):
            KnowledgeRecord.from_document({"kind": "nonsense"})

    def test_repository_bookkeeping_fields_are_ignored(self):
        document = req().to_document() | {"_id": "x", "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-01T00:00:00Z"}
        assert KnowledgeRecord.from_document(document).kind == KnowledgeKind.REQUIREMENTS


# ── The store ───────────────────────────────────────────────────────────────

@pytest.fixture()
def store() -> KnowledgeStore:
    return KnowledgeStore(InMemoryRepository("knowledge", id_prefix="kn"))


class TestStore:
    def test_create_and_get(self, store):
        r = store.create(KnowledgeKind.REQUIREMENTS, "user:u1", {"sofa_budget": 30000}, USER, 0.9)
        got = store.get(r.knowledge_id)
        assert got == r and got.confidence == 0.9
        assert store.get("missing") is None

    def test_persisted_document_is_plain_json(self, store):
        r = store.create(KnowledgeKind.PRODUCT_SPEC, "product:P1",
                         {"product_id": "P1", "name": "Sofa", "verified_at": datetime(2026, 1, 1, tzinfo=timezone.utc)}, USER)
        document = store._repository.get(r.knowledge_id)
        for key in ("_id", "created_at", "updated_at"):
            document.pop(key)
        json.dumps(document)

    def test_update_persists_and_reports_the_diff(self, store):
        r = store.create(KnowledgeKind.REQUIREMENTS, "user:u1", {"sofa_budget": 30000}, USER)
        assert store.update(r.knowledge_id, {"sofa_budget": 25000}, LLM) == {"sofa_budget": (30000.0, 25000.0)}
        stored = store.get(r.knowledge_id)
        assert stored.version == 2 and stored.attributes["sofa_budget"] == 25000.0 and len(stored.history) == 1

    def test_unchanged_update_writes_nothing(self, store):
        r = store.create(KnowledgeKind.REQUIREMENTS, "user:u1", {"sofa_budget": 30000}, USER)
        calls = len(store._repository.collection.calls)
        assert store.update(r.knowledge_id, {"sofa_budget": 30000}, LLM) == {}
        assert not any(c in ("update_one", "insert_one") for c in store._repository.collection.calls[calls:])

    def test_timestamps_never_repeat(self):
        from smartnest.knowledge.models import utc_now
        stamps = [utc_now() for _ in range(2000)]
        assert all(a < b for a, b in zip(stamps, stamps[1:]))

    def test_find_filters_and_orders_newest_first(self, store):
        a = store.create(KnowledgeKind.REQUIREMENTS, "user:a", {"sofa_budget": 1}, USER)
        b = store.create(KnowledgeKind.REQUIREMENTS, "user:b", {"sofa_budget": 2}, USER)
        store.create(KnowledgeKind.ROOM, "room:r", {"name": "r", "width_cm": 1, "length_cm": 2}, USER)
        store.update(a.knowledge_id, {"sofa_budget": 5}, USER)  # a is now the most recently modified
        found = store.find(KnowledgeKind.REQUIREMENTS)
        assert [r.knowledge_id for r in found] == [a.knowledge_id, b.knowledge_id]
        assert [r.entity for r in store.find(entity="user:b")] == ["user:b"]
        assert len(store.find()) == 3 and store.count(KnowledgeKind.REQUIREMENTS) == 2

    def test_invalidated_records_are_hidden_unless_asked_for(self, store):
        r = store.create(KnowledgeKind.REQUIREMENTS, "user:u1", {"sofa_budget": 1}, USER)
        assert store.invalidate(r.knowledge_id, "old") is True
        assert store.invalidate(r.knowledge_id, "old") is False
        assert store.find(KnowledgeKind.REQUIREMENTS) == [] and store.latest(KnowledgeKind.REQUIREMENTS, "user:u1") is None
        assert len(store.find(KnowledgeKind.REQUIREMENTS, valid_only=False)) == 1
        assert store.count(KnowledgeKind.REQUIREMENTS) == 0 and store.count(KnowledgeKind.REQUIREMENTS, valid_only=False) == 1

    def test_latest(self, store):
        store.create(KnowledgeKind.REQUIREMENTS, "user:u1", {"sofa_budget": 1}, USER)
        newer = store.create(KnowledgeKind.REQUIREMENTS, "user:u1", {"sofa_budget": 2}, USER)
        assert store.latest(KnowledgeKind.REQUIREMENTS, "user:u1").knowledge_id == newer.knowledge_id

    def test_related_records_are_followed(self, store):
        project = store.create(KnowledgeKind.HOME_PROJECT, "project:p", {"name": "p"}, USER)
        room = store.create(KnowledgeKind.ROOM, "room:r", {"name": "r", "width_cm": 1, "length_cm": 2}, USER,
                            relationships=[Relationship(relation="part_of", target_id=project.knowledge_id)])
        assert [r.knowledge_id for r in store.related_to(room.knowledge_id, "part_of")] == [project.knowledge_id]
        assert store.related_to(project.knowledge_id) == []

    def test_unknown_record_operations(self, store):
        with pytest.raises(KnowledgeStoreError, match="unknown"):
            store.update("nope", {"a": 1}, USER)
        with pytest.raises(KnowledgeStoreError):
            store.invalidate("nope", "x")

    def test_delete(self, store):
        r = store.create(KnowledgeKind.REQUIREMENTS, "u", {"sofa_budget": 1}, USER)
        assert store.delete(r.knowledge_id) is True and store.delete(r.knowledge_id) is False

    def test_invalid_stored_documents_are_skipped_and_reported(self, store):
        good = store.create(KnowledgeKind.REQUIREMENTS, "user:u1", {"sofa_budget": 1}, USER)
        store._repository.upsert("req_bad", {"kind": "requirements", "entity": "user:u1", "valid": True,
                                             "attributes": {"sofa_budget": -9}})
        assert [r.knowledge_id for r in store.find(KnowledgeKind.REQUIREMENTS)] == [good.knowledge_id]
        assert store.skipped == ["req_bad"]
        with pytest.raises(KnowledgeStoreError):
            store.get("req_bad")


class TestStoreFailures:
    @pytest.fixture()
    def broken(self):
        return KnowledgeStore(_BrokenRepo())

    def test_every_operation_reports_store_errors(self, broken):
        record = req()
        for action in (lambda: broken.save(record), lambda: broken.get("x"), lambda: broken.find(),
                       lambda: broken.count(), lambda: broken.delete("x"), lambda: broken.latest(KnowledgeKind.ROOM, "e")):
            with pytest.raises(KnowledgeStoreError):
                action()

    def test_errors_carry_the_cause_and_the_record_id(self):
        record = req()
        with pytest.raises(KnowledgeStoreError) as info:
            KnowledgeStore(_BrokenRepo()).save(record)
        assert "database unreachable" in str(info.value) and record.knowledge_id in str(info.value)


class _BrokenRepo:
    def _boom(self, *a, **k):
        raise RuntimeError("database unreachable")

    upsert = get = find = delete = count = _boom
