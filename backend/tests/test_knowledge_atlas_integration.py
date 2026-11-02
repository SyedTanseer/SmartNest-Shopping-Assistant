"""Optional live test: knowledge persisted in MongoDB Atlas (collection ``knowledge``).

Skipped by default.  Run:

    PowerShell:   $env:SMARTNEST_RUN_ATLAS_TESTS = "1"; python -m pytest -m atlas
    bash:         SMARTNEST_RUN_ATLAS_TESTS=1 python -m pytest -m atlas

Each test uses a unique entity prefix and deletes exactly the records it created.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone

import pytest

from smartnest.database import BaseRepository, MongoConnection, load_settings
from smartnest.knowledge import (
    KNOWLEDGE_COLLECTION,
    KnowledgeKind,
    KnowledgeStore,
    Provenance,
    Relationship,
)

pytestmark = [
    pytest.mark.atlas,
    pytest.mark.skipif(os.environ.get("SMARTNEST_RUN_ATLAS_TESTS") != "1",
                       reason="set SMARTNEST_RUN_ATLAS_TESTS=1 to run tests against MongoDB Atlas"),
]

USER = Provenance(source="user", source_type="user", method="atlas test")


@pytest.fixture()
def atlas():
    connection = MongoConnection(load_settings())
    connection.connect(verify=True)
    repository = BaseRepository(connection.collection(KNOWLEDGE_COLLECTION), id_prefix="kn")
    created: list[str] = []

    class Tracking:
        def __init__(self, inner):
            self.inner = inner

        def upsert(self, document_id, document):
            created.append(document_id)
            return self.inner.upsert(document_id, document)

        def __getattr__(self, name):
            return getattr(self.inner, name)

    store = KnowledgeStore(Tracking(repository))
    prefix = f"atlas_test_{uuid.uuid4().hex[:8]}"
    yield store, prefix
    for knowledge_id in created:
        repository.delete(knowledge_id)
    connection.close()


class TestKnowledgeOnAtlas:
    def test_every_kind_round_trips(self, atlas):
        store, p = atlas
        project = store.create(KnowledgeKind.HOME_PROJECT, f"project:{p}", {"name": "Living room ₹ refresh"}, USER)
        samples = {
            KnowledgeKind.REQUIREMENTS: {"sofa_budget": 30000, "preferred_style": "minimalist", "min_seating": 3},
            KnowledgeKind.ROOM: {"name": "living room", "width_cm": 210, "length_cm": 350},
            KnowledgeKind.BUDGET: {"total_budget": 200000, "allocations": {"sofa": 30000}},
            KnowledgeKind.PRODUCT_SPEC: {"product_id": f"P_{p}", "name": "Sofa", "price": 24999, "rating": 4.3,
                                         "url": "https://shop.example.com/p/1",
                                         "verified_at": datetime(2026, 10, 31, 12, tzinfo=timezone.utc)},
            KnowledgeKind.OBSERVATION: {"observation_type": "price_change", "subject": f"P_{p}", "details": {"new_price": 1}},
            KnowledgeKind.AGENT_KNOWLEDGE: {"agent_id": "a", "topic": "recommendation", "data": {"x": [1, 2]}},
        }
        for kind, attrs in samples.items():
            saved = store.create(kind, f"{kind.value}:{p}", attrs, USER, 0.9,
                                 [Relationship(relation="part_of", target_id=project.knowledge_id, target_kind="home_project")])
            loaded = store.get(saved.knowledge_id)
            assert loaded == saved, kind
            assert loaded.provenance.retrieved_at.tzinfo is not None
        assert store.get(project.knowledge_id).attributes["name"] == "Living room ₹ refresh"

    def test_updates_keep_history_and_invalidation_hides_records(self, atlas):
        store, p = atlas
        record = store.create(KnowledgeKind.REQUIREMENTS, f"user:{p}", {"sofa_budget": 30000, "preferred_style": "minimalist"}, USER)
        assert store.update(record.knowledge_id, {"preferred_style": "contemporary"},
                            Provenance(source="gemini", source_type="llm")) == {"preferred_style": ("minimalist", "contemporary")}
        stored = store.get(record.knowledge_id)
        assert stored.version == 2 and stored.history[0].previous == {"preferred_style": "minimalist"}
        assert store.latest(KnowledgeKind.REQUIREMENTS, f"user:{p}").knowledge_id == record.knowledge_id

        assert store.invalidate(record.knowledge_id, "superseded") is True
        assert store.latest(KnowledgeKind.REQUIREMENTS, f"user:{p}") is None
        assert [r.knowledge_id for r in store.find(entity=f"user:{p}", valid_only=False)] == [record.knowledge_id]

    def test_queries_order_newest_first_and_follow_relationships(self, atlas):
        store, p = atlas
        project = store.create(KnowledgeKind.HOME_PROJECT, f"project:{p}", {"name": "p"}, USER)
        older = store.create(KnowledgeKind.ROOM, f"room:{p}", {"name": "a", "width_cm": 1, "length_cm": 2}, USER,
                             relationships=[Relationship(relation="part_of", target_id=project.knowledge_id)])
        newer = store.create(KnowledgeKind.ROOM, f"room:{p}", {"name": "b", "width_cm": 1, "length_cm": 2}, USER)
        assert [r.knowledge_id for r in store.find(KnowledgeKind.ROOM, f"room:{p}")] == [newer.knowledge_id, older.knowledge_id]
        assert [r.knowledge_id for r in store.related_to(older.knowledge_id, "part_of")] == [project.knowledge_id]

    def test_a_second_store_instance_sees_the_same_data(self, atlas):
        store, p = atlas
        record = store.create(KnowledgeKind.REQUIREMENTS, f"user:{p}", {"sofa_budget": 1234}, USER)
        with MongoConnection(load_settings()) as other:
            again = KnowledgeStore(BaseRepository(other.collection(KNOWLEDGE_COLLECTION), id_prefix="kn"))
            assert again.get(record.knowledge_id).attributes["sofa_budget"] == 1234
