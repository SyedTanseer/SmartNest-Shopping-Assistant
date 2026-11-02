"""Tests for BDI state persistence: saving, restoring, missing records and database failures."""

from __future__ import annotations

import json

import pytest
from pymongo.errors import ServerSelectionTimeoutError

from smartnest.bdi import (
    AgentSnapshot,
    BDIAgent,
    InMemoryAgentStateStore,
    IntentionStatus,
    RepositoryAgentStateStore,
    StateStoreError,
)
from smartnest.database import (
    BaseRepository,
    Collections,
    DatabaseConnectionError,
    InMemoryRepository,
)
from tests.bdi_support import (
    BUDGET,
    SELECT,
    chosen,
    event,
    in_stock,
    kinds_since,
    make_agent,
    make_planner,
    out_of_stock,
    reference_choice,
)
from smartnest.core.models import EventType as ET

AGENT_ID = "bdi_persist_test"


def repo_store():
    repository = InMemoryRepository(Collections.AGENT_STATES.value, id_prefix="agent")
    return RepositoryAgentStateStore(repository), repository


def restart(store, agent_id=AGENT_ID, **kw) -> BDIAgent:
    return BDIAgent(agent_id, make_planner(), state_store=store, **kw)


class FlakyStore:
    """A store whose load/save can be made to fail."""

    def __init__(self):
        self.inner = InMemoryAgentStateStore()
        self.fail_load = False
        self.fail_save = False
        self.saves = 0

    def load(self, agent_id):
        if self.fail_load:
            raise StateStoreError("database unreachable")
        return self.inner.load(agent_id)

    def save(self, snapshot):
        if self.fail_save:
            raise StateStoreError("database unreachable")
        self.saves += 1
        self.inner.save(snapshot)


# ── Saving ──────────────────────────────────────────────────────────────────

class TestSaving:
    def test_state_is_saved_after_meaningful_changes(self):
        store, repository = repo_store()
        agent = make_agent(store, agent_id=AGENT_ID)
        document = repository.get(AGENT_ID)
        assert document is not None and document["_id"] == AGENT_ID
        assert document["agent_id"] == AGENT_ID and document["agent_type"] == "bdi"
        assert len(document["beliefs"]) >= 7 and len(document["desires"]) == 2
        assert len(document["intentions"]) == 1 and document["transitions"]
        assert document["created_at"] and document["updated_at"]

    def test_saved_document_is_plain_json(self):
        store, repository = repo_store()
        make_agent(store, agent_id=AGENT_ID)
        document = repository.get(AGENT_ID)
        for key in ("_id", "created_at", "updated_at"):
            document.pop(key)
        assert json.loads(json.dumps(document)) == document

    def test_each_change_is_saved(self):
        store = InMemoryAgentStateStore()
        agent = make_agent(store, agent_id=AGENT_ID)
        saves = store.save_count
        agent.handle_event(out_of_stock("SOFA_A2"))
        assert store.save_count > saves
        assert store.load(AGENT_ID).intentions[-1]["params"]["product_id"] == chosen(agent)

    def test_unchanged_updates_are_not_saved(self):
        store = InMemoryAgentStateStore()
        agent = make_agent(store, agent_id=AGENT_ID)
        saves = store.save_count
        agent.set_requirement("sofa_budget", 30000)                       # same value
        agent.handle_event(event(ET.BUDGET_UPDATE, sofa_budget=30000))     # same value
        agent.handle_event(None)
        assert store.save_count == saves

    def test_ignored_updates_are_recorded_and_saved(self):
        store = InMemoryAgentStateStore()
        agent = make_agent(store, agent_id=AGENT_ID)
        saves = store.save_count
        agent.handle_event(out_of_stock("SOFA_A7"))
        assert store.save_count == saves + 1
        assert "update_ignored" in [t["kind"] for t in store.load(AGENT_ID).transitions]

    def test_repeated_saves_update_one_record_and_keep_created_at(self):
        store, repository = repo_store()
        agent = make_agent(store, agent_id=AGENT_ID)
        first = repository.get(AGENT_ID)
        agent.handle_event(out_of_stock("SOFA_A2"))
        second = repository.get(AGENT_ID)
        assert repository.count() == 1
        assert second["created_at"] == first["created_at"] and second["updated_at"] >= first["updated_at"]
        assert len(second["transitions"]) > len(first["transitions"])

    def test_save_state_forces_a_save(self):
        store = InMemoryAgentStateStore()
        agent = make_agent(store, agent_id=AGENT_ID)
        saves = store.save_count
        assert agent.save_state() is True and store.save_count == saves + 1

    def test_agents_with_different_ids_do_not_clash(self):
        store, repository = repo_store()
        make_agent(store, agent_id="agent_one")
        two = make_agent(store, agent_id="agent_two")
        two.handle_event(out_of_stock("SOFA_A2"))
        assert repository.count() == 2
        assert restart(store, "agent_one").current_intention(SELECT).params["product_id"] == "SOFA_A2"
        assert restart(store, "agent_two").current_intention(SELECT).params["product_id"] != "SOFA_A2"

    def test_agent_without_a_store_works_and_never_persists(self):
        agent = make_agent(None)
        agent.handle_event(out_of_stock("SOFA_A2"))
        assert chosen(agent) != "SOFA_A2" and agent.save_state() is False


# ── Restoring ───────────────────────────────────────────────────────────────

class TestRestoring:
    @pytest.fixture(params=["memory_store", "repository_store"])
    def store(self, request):
        return InMemoryAgentStateStore() if request.param == "memory_store" else repo_store()[0]

    def test_everything_is_restored(self, store):
        original = make_agent(store, agent_id=AGENT_ID)
        original.handle_event(out_of_stock("SOFA_A2"))
        original.handle_event(event(ET.BUDGET_UPDATE, sofa_budget=27000))
        expected_choice = chosen(original)

        restored = restart(store)

        assert restored.state_restored
        assert restored.beliefs.as_dict() == original.beliefs.as_dict()
        assert [d.model_dump() for d in restored.desires.all()] == [d.model_dump() for d in original.desires.all()]
        assert [i.model_dump() for i in restored.intentions.all()] == [i.model_dump() for i in original.intentions.all()]
        assert chosen(restored) == expected_choice
        assert [t.transition_id for t in restored.transitions[:-1]] == \
            [t.transition_id for t in original.transitions][:len(restored.transitions) - 1]
        assert restored.transitions[-1].kind == "state_restored"

    def test_intention_progress_survives(self, store):
        original = make_agent(store, agent_id=AGENT_ID)
        restored = restart(store)
        actions = restored.current_intention(SELECT).actions
        assert [a.status.value for a in actions] == ["done", "done", "pending"]
        assert restored.current_intention(SELECT).status == IntentionStatus.ACTIVE

    def test_belief_details_survive(self, store):
        original = make_agent(store, agent_id=AGENT_ID)
        original.handle_event(out_of_stock("SOFA_A2"))
        original.invalidate_belief("user", "preferred_colour", "unsure")
        restored = restart(store)
        belief = restored.beliefs.get("SOFA_A2", "availability")
        assert (belief.value, belief.source) == ("out_of_stock", "product_monitor")
        assert belief.timestamp == original.beliefs.get("SOFA_A2", "availability").timestamp
        invalid = restored.beliefs.get("user", "preferred_colour", include_invalid=True)
        assert not invalid.valid and invalid.invalidation_reason == "unsure"

    def test_restored_agent_keeps_working(self, store):
        make_agent(store, agent_id=AGENT_ID).handle_event(out_of_stock("SOFA_A2"))
        restored = restart(store)
        log = restored.handle_event(in_stock("SOFA_A2"))  # it remembers A2 had been out of stock
        assert log.rule_id == "BDI_REVISED" and chosen(restored) == "SOFA_A2"

    def test_restored_agent_remembers_what_it_already_knows(self, store):
        make_agent(store, agent_id=AGENT_ID).handle_event(out_of_stock("SOFA_A2"))
        restored = restart(store)
        before = len(restored.transitions)
        assert restored.handle_event(out_of_stock("SOFA_A2")) is None  # already known
        assert len(restored.transitions) == before

    def test_restored_agent_matches_the_optimizer_after_further_changes(self, store):
        make_agent(store, agent_id=AGENT_ID).handle_event(out_of_stock("SOFA_A2"))
        restored = restart(store)
        restored.handle_event(event(ET.ROOM_UPDATE, max_width_cm=195))
        assert chosen(restored) == reference_choice({"max_width_cm": 195}, {"SOFA_A2": "out_of_stock"})

    def test_a_completed_goal_stays_achieved(self, store):
        original = make_agent(store, agent_id=AGENT_ID)
        original.confirm(SELECT)
        restored = restart(store)
        assert restored.desires.get(SELECT).status.value == "achieved"
        assert restored.handle_event(out_of_stock("SOFA_A2")) is None

    def test_history_is_capped_on_restore(self, store):
        original = make_agent(store, agent_id=AGENT_ID)
        for i in range(20):
            original.set_belief("noise", "n", i)
        assert len(restart(store, max_history=5).transitions) <= 5

    def test_autoload_can_be_disabled(self, store):
        make_agent(store, agent_id=AGENT_ID)
        agent = restart(store, auto_load=False)
        assert not agent.state_restored and len(agent.beliefs) == 0
        assert agent.load_state() is True and agent.state_restored


# ── Missing records ─────────────────────────────────────────────────────────

class TestMissingRecord:
    def test_new_agent_starts_fresh(self):
        store, repository = repo_store()
        agent = restart(store)
        assert not agent.state_restored and len(agent.beliefs) == 0
        assert agent.transitions[0].kind == "state_not_found"
        assert repository.count() == 0  # nothing is written until there is something to say

    def test_saving_works_after_a_missing_record(self):
        store, repository = repo_store()
        agent = restart(store)
        agent.set_requirement("sofa_budget", 100)
        assert repository.get(AGENT_ID) is not None

    def test_load_state_without_a_store(self):
        assert make_agent(None).load_state() is False


# ── Database failures ───────────────────────────────────────────────────────

class TestLoadFailure:
    def test_agent_runs_in_memory_and_does_not_overwrite_the_stored_record(self):
        store = FlakyStore()
        make_agent(store, agent_id=AGENT_ID)
        stored = json.dumps(store.inner.load(AGENT_ID).to_document(), sort_keys=True)
        store.fail_load = True
        saves = store.saves

        agent = restart(store)
        assert not agent.state_restored
        assert agent.transitions[0].kind == "persistence_error"
        assert "Saving is disabled" in agent.transitions[0].explanation

        # The agent still works...
        agent.set_requirements({"total_budget": 1, "sofa_budget": 1})
        agent.set_requirement("max_width_cm", 100)
        # ...but it never wrote over the good record.
        store.fail_load = False
        assert store.saves == saves
        assert json.dumps(store.inner.load(AGENT_ID).to_document(), sort_keys=True) == stored

    def test_retrying_the_load_recovers_the_saved_state(self):
        store = FlakyStore()
        make_agent(store, agent_id=AGENT_ID)
        store.fail_load = True
        agent = restart(store)
        store.fail_load = False
        assert agent.load_state() is True
        assert agent.state_restored and chosen(agent) == "SOFA_A2"

    def test_repository_errors_become_state_store_errors(self):
        class Down:
            def get(self, document_id): raise DatabaseConnectionError("MongoDB ping failed")
            def upsert(self, document_id, document): raise DatabaseConnectionError("MongoDB ping failed")
            def delete(self, document_id): raise DatabaseConnectionError("MongoDB ping failed")

        store = RepositoryAgentStateStore(Down())
        with pytest.raises(StateStoreError, match="could not read"):
            store.load("x")
        with pytest.raises(StateStoreError, match="could not save"):
            store.save(AgentSnapshot(agent_id="x"))
        with pytest.raises(StateStoreError, match="could not delete"):
            store.delete("x")

    def test_agent_survives_a_real_database_error_type(self):
        from tests.fake_mongo import BrokenCollection
        broken = BaseRepository(BrokenCollection(ServerSelectionTimeoutError("down")), id_prefix="agent")
        agent = restart(RepositoryAgentStateStore(broken))
        agent.set_requirements({"total_budget": 1000})
        assert agent.beliefs.value("user", "total_budget") == 1000
        assert any(t.kind == "persistence_error" for t in agent.transitions)


class TestSaveFailure:
    def test_agent_keeps_working_and_records_the_error(self):
        store = FlakyStore()
        agent = make_agent(store, agent_id=AGENT_ID)
        store.fail_save = True
        agent.handle_event(out_of_stock("SOFA_A2"))
        assert chosen(agent) != "SOFA_A2"                                   # the decision was still made
        errors = [t for t in agent.transitions if t.kind == "persistence_error"]
        assert errors and "Could not save state" in errors[-1].summary

    def test_state_is_saved_once_the_database_is_back(self):
        store = FlakyStore()
        agent = make_agent(store, agent_id=AGENT_ID)
        store.fail_save = True
        agent.handle_event(out_of_stock("SOFA_A2"))
        store.fail_save = False
        agent.handle_event(event(ET.PREFERENCE_UPDATE, preferred_colour="grey"))
        restored = restart(store)
        assert chosen(restored) == chosen(agent)
        assert restored.beliefs.value("SOFA_A2", "availability") == "out_of_stock"
        assert restored.beliefs.value("user", "preferred_colour") == "grey"

    def test_unserialisable_belief_is_a_recorded_error_not_a_crash(self):
        store = InMemoryAgentStateStore()
        agent = make_agent(store, agent_id=AGENT_ID)
        agent.set_belief("thing", "blob", object())
        assert any(t.kind == "persistence_error" for t in agent.transitions)
        assert chosen(agent) == "SOFA_A2"


class TestCorruptedRecord:
    def corrupt(self, repository, **changes):
        document = repository.get(AGENT_ID)
        document.update(changes)
        repository.collection.docs[AGENT_ID].update(changes)
        return document

    @pytest.mark.parametrize("changes", [
        {"schema_version": 99},
        {"beliefs": "not a list"},
        {"beliefs": [{"subject": "only"}]},
        {"desires": [{"description": ""}]},
        {"transitions": [{"kind": "x"}]},
        {"agent_id": None},
    ])
    def test_corrupted_state_is_not_loaded_and_not_overwritten(self, changes):
        store, repository = repo_store()
        make_agent(store, agent_id=AGENT_ID)
        self.corrupt(repository, **changes)
        snapshot_before = dict(repository.collection.docs[AGENT_ID])

        agent = restart(store)

        assert not agent.state_restored
        assert agent.transitions[0].kind == "persistence_error"
        agent.set_requirement("sofa_budget", 12345)
        assert repository.collection.docs[AGENT_ID] == snapshot_before  # untouched

    def test_extra_repository_fields_are_tolerated(self):
        store, repository = repo_store()
        make_agent(store, agent_id=AGENT_ID)
        repository.collection.docs[AGENT_ID]["some_future_field"] = {"x": 1}
        assert restart(store).state_restored


# ── Snapshot model and independence ─────────────────────────────────────────

class TestSnapshotAndStores:
    def test_snapshot_round_trip(self):
        agent = make_agent(None, agent_id=AGENT_ID)
        snapshot = agent._snapshot()
        again = AgentSnapshot.from_document(json.loads(json.dumps(snapshot.to_document())))
        assert again.agent_id == AGENT_ID and again.beliefs == snapshot.beliefs

    def test_in_memory_store_copies_documents(self):
        store = InMemoryAgentStateStore()
        snapshot = AgentSnapshot(agent_id="a", beliefs=[{"subject": "x", "predicate": "y", "value": [1]}])
        store.save(snapshot)
        snapshot.beliefs[0]["value"].append(2)
        assert store.load("a").beliefs[0]["value"] == [1]
        assert store.delete("a") is True and store.delete("a") is False and store.load("a") is None

    def test_repository_store_delete(self):
        store, repository = repo_store()
        store.save(AgentSnapshot(agent_id="a"))
        assert repository.count() == 1 and store.delete("a") is True and repository.count() == 0

    def test_any_object_with_get_upsert_delete_can_back_the_store(self):
        class DictRepo:
            def __init__(self): self.docs = {}
            def get(self, i): return self.docs.get(i)
            def upsert(self, i, d): self.docs[i] = d; return True
            def delete(self, i): return self.docs.pop(i, None) is not None

        backend = DictRepo()
        agent = make_agent(RepositoryAgentStateStore(backend), agent_id=AGENT_ID)
        assert AGENT_ID in backend.docs
        assert restart(RepositoryAgentStateStore(backend)).state_restored
