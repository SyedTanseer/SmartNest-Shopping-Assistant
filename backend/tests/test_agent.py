"""Tests for the ReflexAgent.

Verify that the agent correctly observes events, delegates to the rule
engine, records action logs, and handles edge cases.
"""

from __future__ import annotations

import pytest

from smartnest.agents.reflex_agent import ReflexAgent
from smartnest.core.models import (
    ActionResultType,
    EnvironmentalEvent,
    EventType,
)
from smartnest.reasoning.rule_engine import RuleEngine
from smartnest.reasoning.rules import all_smartnest_rules


@pytest.fixture()
def agent() -> ReflexAgent:
    """A ReflexAgent with all SmartNest rules registered."""
    engine = RuleEngine()
    for rule in all_smartnest_rules():
        engine.register(rule)
    return ReflexAgent(agent_id="test_agent", rule_engine=engine, name="TestReflex")


# ── Tests ──────────────────────────────────────────────────────────────────

class TestReflexAgentBasic:
    def test_agent_identity(self, agent: ReflexAgent):
        assert agent.agent_id == "test_agent"
        assert agent.name == "TestReflex"

    def test_initial_state_empty(self, agent: ReflexAgent):
        assert agent.state == {}
        assert agent.action_log == []

    def test_reset_clears_state(self, agent: ReflexAgent):
        agent.state["foo"] = "bar"
        agent.action_log.append(None)  # simulate an entry
        agent.reset()
        assert agent.state == {}
        assert agent.action_log == []


class TestReflexAgentHandling:
    def test_budget_exceeded_event(self, agent: ReflexAgent):
        event = EnvironmentalEvent(
            event_id="t1",
            event_type=EventType.BUDGET_UPDATE,
            source="test",
            payload={"total_budget": 200_000, "estimated_expenditure": 218_000},
        )
        log = agent.handle_event(event)
        assert log is not None
        assert log.rule_id == "R001_BUDGET_EXCEEDED"
        assert log.result_type == ActionResultType.WARNING
        assert len(agent.action_log) == 1

    def test_furniture_incompatibility_event(self, agent: ReflexAgent):
        event = EnvironmentalEvent(
            event_id="t2",
            event_type=EventType.FURNITURE_COMPATIBILITY,
            source="test",
            payload={"product_name": "Sofa", "product_width": 230, "room_width": 210},
        )
        log = agent.handle_event(event)
        assert log is not None
        assert log.rule_id == "R002_FURNITURE_INCOMPATIBLE"
        assert log.result_type == ActionResultType.REJECTION

    def test_product_unavailable_event(self, agent: ReflexAgent):
        event = EnvironmentalEvent(
            event_id="t3",
            event_type=EventType.PRODUCT_AVAILABILITY,
            source="test",
            payload={"product_name": "REF_001", "availability": "out_of_stock"},
        )
        log = agent.handle_event(event)
        assert log is not None
        assert log.rule_id == "R003_PRODUCT_UNAVAILABLE"
        assert log.result_type == ActionResultType.FLAG

    def test_missing_information_event(self, agent: ReflexAgent):
        event = EnvironmentalEvent(
            event_id="t4",
            event_type=EventType.MISSING_INFORMATION,
            source="test",
            payload={"missing_field": "room_dimensions"},
        )
        log = agent.handle_event(event)
        assert log is not None
        assert log.rule_id == "R004_MISSING_INFO"
        assert log.result_type == ActionResultType.REQUEST

    def test_valid_product_event(self, agent: ReflexAgent):
        event = EnvironmentalEvent(
            event_id="t5",
            event_type=EventType.PRODUCT_VALIDATION,
            source="test",
            payload={
                "product_name": "Dining table",
                "product_price": 24_999,
                "available_budget": 30_000,
                "fits_room": True,
            },
        )
        log = agent.handle_event(event)
        assert log is not None
        assert log.rule_id == "R005_VALID_PRODUCT"
        assert log.result_type == ActionResultType.APPROVAL

    def test_no_matching_rule_returns_none(self, agent: ReflexAgent):
        """An event with a valid type but no matching payload should return None."""
        event = EnvironmentalEvent(
            event_id="t6",
            event_type=EventType.BUDGET_UPDATE,
            source="test",
            payload={},  # missing required fields
        )
        log = agent.handle_event(event)
        assert log is None
        assert len(agent.action_log) == 0

    def test_none_event_returns_none(self, agent: ReflexAgent):
        log = agent.handle_event(None)
        assert log is None


class TestReflexAgentAllScenarios:
    def test_all_five_assignment_scenarios(self, agent: ReflexAgent):
        """Run all five Assignment 1.2 events and verify expected outcomes."""
        events = [
            EnvironmentalEvent(
                event_id="evt_001",
                event_type=EventType.BUDGET_UPDATE,
                source="budget_tracker",
                payload={"total_budget": 200_000, "estimated_expenditure": 218_000},
            ),
            EnvironmentalEvent(
                event_id="evt_002",
                event_type=EventType.FURNITURE_COMPATIBILITY,
                source="room_analyzer",
                payload={"product_name": "Three-seater sofa", "product_width": 230, "room_width": 210},
            ),
            EnvironmentalEvent(
                event_id="evt_003",
                event_type=EventType.PRODUCT_AVAILABILITY,
                source="product_catalog",
                payload={"product_name": "Refrigerator REF_001", "availability": "out_of_stock"},
            ),
            EnvironmentalEvent(
                event_id="evt_004",
                event_type=EventType.MISSING_INFORMATION,
                source="room_analyzer",
                payload={"missing_field": "room_dimensions"},
            ),
            EnvironmentalEvent(
                event_id="evt_005",
                event_type=EventType.PRODUCT_VALIDATION,
                source="product_catalog",
                payload={
                    "product_name": "Dining table",
                    "product_price": 24_999,
                    "available_budget": 30_000,
                    "fits_room": True,
                },
            ),
        ]

        expected_rules = [
            "R001_BUDGET_EXCEEDED",
            "R002_FURNITURE_INCOMPATIBLE",
            "R003_PRODUCT_UNAVAILABLE",
            "R004_MISSING_INFO",
            "R005_VALID_PRODUCT",
        ]
        expected_types = [
            ActionResultType.WARNING,
            ActionResultType.REJECTION,
            ActionResultType.FLAG,
            ActionResultType.REQUEST,
            ActionResultType.APPROVAL,
        ]

        for event, exp_rule, exp_type in zip(events, expected_rules, expected_types, strict=True):
            log = agent.handle_event(event)
            assert log is not None, f"No log for event {event.event_id}"
            assert log.rule_id == exp_rule, f"Expected {exp_rule}, got {log.rule_id}"
            assert log.result_type == exp_type, f"Expected {exp_type}, got {log.result_type}"

        assert len(agent.action_log) == 5
