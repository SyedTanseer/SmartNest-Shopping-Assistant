"""Tests for TaskHandlerRegistry and evaluate_furniture_fit.

Verifies:
    - STATE_SYNC handler stores state and returns ACK.
    - STATE_SYNC with missing project_id returns FAILURE.
    - TASK_REQUEST dispatches to registered evaluator.
    - TASK_REQUEST with unknown task returns FAILURE.
    - TASK_REQUEST before state sync returns FAILURE.
    - evaluate_furniture_fit produces correct results from received data.
    - Evaluation is not hardcoded — depends on actual state.
    - Task ID correlation.
"""

from __future__ import annotations

import pytest

from smartnest.communication.task_handlers import (
    TaskHandlerRegistry,
    evaluate_furniture_fit,
)
from smartnest.core.models import MessageEnvelope, MessageType


APARTMENT_STATE = {
    "project_id": "HOME_001",
    "budget": 200000,
    "remaining_budget": 30000,
    "rooms": {
        "living_room": {"width_cm": 210, "length_cm": 350},
        "bedroom": {"width_cm": 300, "length_cm": 350},
    },
    "preferences": {"style": "minimalist", "color": "neutral"},
}


def _sync_msg(state: dict | None = None) -> MessageEnvelope:
    return MessageEnvelope(
        sender="agent_a",
        receiver="agent_b",
        message_type=MessageType.STATE_SYNC,
        task_id="sync_001",
        payload=state if state is not None else APARTMENT_STATE,
    )


def _task_msg(payload: dict) -> MessageEnvelope:
    return MessageEnvelope(
        sender="agent_a",
        receiver="agent_b",
        message_type=MessageType.TASK_REQUEST,
        task_id="task_001",
        payload=payload,
    )


class TestStateSyncHandler:
    def test_state_sync_stores_and_returns_ack(self):
        registry = TaskHandlerRegistry(agent_id="agent_b")
        response = registry.handle_state_sync(_sync_msg())
        assert response.message_type == "ack"
        assert response.payload["status"] == "synchronized"
        assert response.payload["project_id"] == "HOME_001"
        assert registry.has_state()

    def test_state_stored_correctly(self):
        registry = TaskHandlerRegistry()
        registry.handle_state_sync(_sync_msg())
        state = registry.apartment_state
        assert state["project_id"] == "HOME_001"
        assert state["rooms"]["living_room"]["width_cm"] == 210
        assert state["remaining_budget"] == 30000

    def test_missing_project_id_returns_failure(self):
        registry = TaskHandlerRegistry()
        bad_state = {"budget": 100000}  # no project_id
        response = registry.handle_state_sync(_sync_msg(bad_state))
        assert response.message_type == "failure"
        assert response.payload["reason"] == "missing_project_id"

    def test_task_id_preserved_in_ack(self):
        registry = TaskHandlerRegistry()
        msg = _sync_msg()
        msg.task_id = "correlation_abc"
        response = registry.handle_state_sync(msg)
        assert response.task_id == "correlation_abc"


class TestTaskRequestHandler:
    def test_task_dispatched_to_evaluator(self):
        registry = TaskHandlerRegistry()
        registry.register_evaluator("evaluate_furniture_fit", evaluate_furniture_fit)
        registry.handle_state_sync(_sync_msg())

        payload = {
            "task_type": "evaluate_furniture_fit",
            "product": {
                "product_id": "SOFA_001",
                "width_cm": 230,
                "price": 24999,
                "target_room": "living_room",
            },
        }
        response = registry.handle_task_request(_task_msg(payload))
        assert response.message_type == "inform_result"
        result = response.payload["result"]
        assert result["fits"] is False
        assert result["width_excess_cm"] == 20
        assert result["within_budget"] is True

    def test_unknown_task_type_returns_failure(self):
        registry = TaskHandlerRegistry()
        registry.handle_state_sync(_sync_msg())
        response = registry.handle_task_request(_task_msg({"task_type": "unknown_task"}))
        assert response.message_type == "failure"
        assert response.payload["reason"] == "unknown_task_type"

    def test_missing_task_type_returns_failure(self):
        registry = TaskHandlerRegistry()
        registry.handle_state_sync(_sync_msg())
        response = registry.handle_task_request(_task_msg({}))
        assert response.message_type == "failure"
        assert response.payload["reason"] == "missing_task_type"

    def test_task_before_state_sync_returns_failure(self):
        registry = TaskHandlerRegistry()
        registry.register_evaluator("evaluate_furniture_fit", evaluate_furniture_fit)
        payload = {"task_type": "evaluate_furniture_fit", "product": {}}
        response = registry.handle_task_request(_task_msg(payload))
        assert response.message_type == "failure"
        assert response.payload["reason"] == "no_synchronized_state"

    def test_task_id_preserved_in_result(self):
        registry = TaskHandlerRegistry()
        registry.register_evaluator("evaluate_furniture_fit", evaluate_furniture_fit)
        registry.handle_state_sync(_sync_msg())
        msg = _task_msg({
            "task_type": "evaluate_furniture_fit",
            "product": {"product_id": "X", "width_cm": 100, "price": 5000},
        })
        msg.task_id = "task_correlation_999"
        response = registry.handle_task_request(msg)
        assert response.task_id == "task_correlation_999"


class TestEvaluateFurnitureFit:
    """Verify the evaluator produces data-dependent results."""

    def test_sofa_too_wide_rejected(self):
        state = dict(APARTMENT_STATE)
        result = evaluate_furniture_fit(state, {
            "task_type": "evaluate_furniture_fit",
            "product": {
                "product_id": "SOFA_001",
                "width_cm": 230,
                "price": 24999,
                "target_room": "living_room",
            },
        })
        assert result["fits"] is False
        assert result["width_excess_cm"] == 20
        assert result["within_budget"] is True
        assert "exceeds available room width" in result["recommendation"]

    def test_product_within_budget_and_fits_accepted(self):
        state = dict(APARTMENT_STATE)
        result = evaluate_furniture_fit(state, {
            "task_type": "evaluate_furniture_fit",
            "product": {
                "product_id": "CHAIR_001",
                "width_cm": 60,
                "price": 5000,
                "target_room": "living_room",
            },
        })
        assert result["fits"] is True
        assert result["within_budget"] is True
        assert "Accept" in result["recommendation"]

    def test_product_fits_but_over_budget_rejected(self):
        state = dict(APARTMENT_STATE)
        result = evaluate_furniture_fit(state, {
            "task_type": "evaluate_furniture_fit",
            "product": {
                "product_id": "EXPENSIVE_001",
                "width_cm": 100,
                "price": 50000,
                "target_room": "living_room",
            },
        })
        assert result["fits"] is True
        assert result["within_budget"] is False
        assert "exceeds the remaining budget" in result["recommendation"]

    def test_different_room_used(self):
        """If target_room is bedroom, the bedroom dimensions are used."""
        result = evaluate_furniture_fit(APARTMENT_STATE, {
            "task_type": "evaluate_furniture_fit",
            "product": {
                "product_id": "BED_001",
                "width_cm": 290,
                "price": 20000,
                "target_room": "bedroom",
            },
        })
        assert result["fits"] is True
        assert result["room_width_cm"] == 300

    def test_no_target_room_uses_first_room(self):
        """Without target_room, the first room in state is used."""
        result = evaluate_furniture_fit(APARTMENT_STATE, {
            "task_type": "evaluate_furniture_fit",
            "product": {
                "product_id": "X",
                "width_cm": 210,
                "price": 1000,
            },
        })
        # First room is living_room (width 210), so 210 <= 210 = fits
        assert result["fits"] is True
        assert result["room_width_cm"] == 210

    def test_no_rooms_returns_no_rooms_message(self):
        result = evaluate_furniture_fit({"remaining_budget": 10000}, {
            "task_type": "evaluate_furniture_fit",
            "product": {"product_id": "X", "width_cm": 100, "price": 5000},
        })
        assert result["fits"] is False
        assert "No rooms" in result["recommendation"]

    def test_result_depends_on_actual_state_not_hardcoded(self):
        """Same product, different room widths → different result."""
        narrow_state = {
            "project_id": "HOME_002",
            "remaining_budget": 50000,
            "rooms": {"room": {"width_cm": 200, "length_cm": 300}},
        }
        wide_state = {
            "project_id": "HOME_003",
            "remaining_budget": 50000,
            "rooms": {"room": {"width_cm": 300, "length_cm": 300}},
        }
        payload = {
            "task_type": "evaluate_furniture_fit",
            "product": {"product_id": "SOFA", "width_cm": 250, "price": 10000},
        }
        narrow_result = evaluate_furniture_fit(narrow_state, payload)
        wide_result = evaluate_furniture_fit(wide_state, payload)

        assert narrow_result["fits"] is False
        assert wide_result["fits"] is True
        assert narrow_result["room_width_cm"] == 200
        assert wide_result["room_width_cm"] == 300


class TestEvaluatorRegistration:
    def test_register_and_list_evaluators(self):
        registry = TaskHandlerRegistry()
        registry.register_evaluator("evaluate_furniture_fit", evaluate_furniture_fit)
        assert "evaluate_furniture_fit" in registry.registered_task_types

    def test_multiple_evaluators(self):
        registry = TaskHandlerRegistry()

        def dummy_evaluator(state, payload):
            return {"done": True}

        registry.register_evaluator("evaluate_furniture_fit", evaluate_furniture_fit)
        registry.register_evaluator("custom_task", dummy_evaluator)
        assert len(registry.registered_task_types) == 2


# ── Robustness fixes ─────────────────────────────────────────────────────────

class TestRoomSelectionAndValidation:
    def test_unknown_target_room_is_reported_not_substituted(self):
        result = evaluate_furniture_fit(APARTMENT_STATE, {
            "product": {"product_id": "X", "width_cm": 100, "price": 1000,
                        "target_room": "kitchen"},
        })
        assert result["fits"] is False
        assert result["error"] == "unknown_target_room"
        assert "kitchen" in result["recommendation"]
        assert set(result["available_rooms"]) == set(APARTMENT_STATE["rooms"])

    def test_default_room_is_named_in_result(self):
        result = evaluate_furniture_fit(APARTMENT_STATE, {
            "product": {"product_id": "X", "width_cm": 100, "price": 1000},
        })
        assert result["room_name"] == next(iter(APARTMENT_STATE["rooms"]))

    def test_depth_checked_against_room_length(self):
        state = {"remaining_budget": 50000,
                 "rooms": {"hall": {"width_cm": 300, "length_cm": 150}}}
        too_deep = evaluate_furniture_fit(state, {
            "product": {"product_id": "BED", "width_cm": 200, "depth_cm": 190, "price": 1000},
        })
        assert too_deep["fits"] is False
        assert too_deep["depth_excess_cm"] == 40
        assert "depth" in too_deep["recommendation"]

        fits = evaluate_furniture_fit(state, {
            "product": {"product_id": "BED", "width_cm": 200, "depth_cm": 140, "price": 1000},
        })
        assert fits["fits"] is True

    def test_state_sync_rejects_rooms_list(self):
        registry = TaskHandlerRegistry()
        msg = MessageEnvelope(
            sender="agent_a", receiver="agent_b",
            message_type=MessageType.STATE_SYNC,
            payload={"project_id": "P1", "rooms": [{"width_cm": 200}]},
        )
        response = registry.handle_state_sync(msg)
        assert response.message_type == "failure"
        assert response.payload["reason"] == "invalid_rooms"
        assert not registry.has_state()

    def test_state_sync_rejects_non_numeric_width(self):
        registry = TaskHandlerRegistry()
        msg = MessageEnvelope(
            sender="agent_a", receiver="agent_b",
            message_type=MessageType.STATE_SYNC,
            payload={"project_id": "P1", "rooms": {"hall": {"width_cm": "wide"}}},
        )
        response = registry.handle_state_sync(msg)
        assert response.payload["reason"] == "invalid_rooms"

    def test_evaluator_cannot_mutate_stored_state(self):
        registry = TaskHandlerRegistry()
        registry.handle_state_sync(MessageEnvelope(
            sender="a", receiver="b", message_type=MessageType.STATE_SYNC,
            payload={"project_id": "P1", "rooms": {"hall": {"width_cm": 200}}},
        ))

        def mutating(state, payload):
            state["project_id"] = "HACKED"
            state["rooms"]["hall"]["width_cm"] = 1
            return {}

        registry.register_evaluator("mutate", mutating)
        registry.handle_task_request(MessageEnvelope(
            sender="a", receiver="b", message_type=MessageType.TASK_REQUEST,
            payload={"task_type": "mutate"},
        ))
        assert registry.apartment_state["project_id"] == "P1"
        assert registry.apartment_state["rooms"]["hall"]["width_cm"] == 200
