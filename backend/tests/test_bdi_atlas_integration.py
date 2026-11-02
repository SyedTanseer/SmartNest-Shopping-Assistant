"""Optional live test: BDI state persisted in MongoDB Atlas (``agent_states`` collection).

Skipped by default.  Run:

    PowerShell:   $env:SMARTNEST_RUN_ATLAS_TESTS = "1"; python -m pytest -m atlas
    bash:         SMARTNEST_RUN_ATLAS_TESTS=1 python -m pytest -m atlas

The test uses a unique agent id in the real ``agent_states`` collection and
deletes only its own record afterwards.
"""

from __future__ import annotations

import os
import uuid

import pytest

from smartnest.bdi import BDIAgent, RepositoryAgentStateStore
from smartnest.database import BaseRepository, Collections, MongoConnection, load_settings
from tests.bdi_support import SELECT, chosen, in_stock, make_agent, make_planner, out_of_stock, reference_choice

pytestmark = [
    pytest.mark.atlas,
    pytest.mark.skipif(
        os.environ.get("SMARTNEST_RUN_ATLAS_TESTS") != "1",
        reason="set SMARTNEST_RUN_ATLAS_TESTS=1 to run tests against MongoDB Atlas",
    ),
]


@pytest.fixture()
def atlas_store():
    connection = MongoConnection(load_settings())
    connection.connect(verify=True)
    repository = BaseRepository(connection.collection(Collections.AGENT_STATES), id_prefix="agent")
    agent_id = f"bdi_atlas_test_{uuid.uuid4().hex[:10]}"
    yield RepositoryAgentStateStore(repository), repository, agent_id
    repository.delete(agent_id)
    connection.close()


class TestBdiOnAtlas:
    def test_state_is_saved_in_agent_states(self, atlas_store):
        store, repository, agent_id = atlas_store
        make_agent(store, agent_id=agent_id)
        document = repository.get(agent_id)
        assert document["_id"] == agent_id and document["agent_type"] == "bdi"
        assert len(document["desires"]) == 2 and len(document["intentions"]) == 1
        assert document["created_at"].tzinfo is not None

    def test_state_survives_a_restart(self, atlas_store):
        store, repository, agent_id = atlas_store
        original = make_agent(store, agent_id=agent_id)
        original.handle_event(out_of_stock("SOFA_A2"))
        expected = chosen(original)
        assert expected != "SOFA_A2"

        restored = BDIAgent(agent_id, make_planner(), state_store=store)

        assert restored.state_restored
        assert chosen(restored) == expected
        assert restored.beliefs.as_dict() == original.beliefs.as_dict()
        assert [d.model_dump() for d in restored.desires.all()] == [d.model_dump() for d in original.desires.all()]
        assert len(restored.transitions) >= len(original.transitions)

        log = restored.handle_event(in_stock("SOFA_A2"))
        assert log.rule_id == "BDI_REVISED" and chosen(restored) == "SOFA_A2" == reference_choice()

    def test_repeated_saves_update_one_record(self, atlas_store):
        store, repository, agent_id = atlas_store
        agent = make_agent(store, agent_id=agent_id)
        created = repository.get(agent_id)["created_at"]
        agent.handle_event(out_of_stock("SOFA_A2"))
        document = repository.get(agent_id)
        assert document["created_at"] == created and document["updated_at"] >= created
        assert repository.count({"_id": agent_id}) == 1

    def test_missing_record_starts_fresh(self, atlas_store):
        store, repository, agent_id = atlas_store
        agent = BDIAgent(agent_id, make_planner(), state_store=store)
        assert not agent.state_restored and agent.transitions[0].kind == "state_not_found"
        assert repository.get(agent_id) is None
