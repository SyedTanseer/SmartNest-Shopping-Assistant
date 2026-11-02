"""Tests for the negotiation state machine.

Verifies:
    - Initial state is CREATED.
    - Valid transitions proceed correctly.
    - Invalid transitions raise InvalidStateTransition.
    - History records all transitions with timestamps.
    - Terminal state detection.
    - Full lifecycle traversal.
"""

from __future__ import annotations

import pytest

from smartnest.cnp.negotiation_state import (
    InvalidStateTransition,
    NegotiationState,
    NegotiationStateMachine,
)


class TestInitialState:
    def test_initial_state_is_created(self):
        sm = NegotiationStateMachine("neg_001")
        assert sm.current_state == NegotiationState.CREATED

    def test_history_has_one_entry(self):
        sm = NegotiationStateMachine("neg_001")
        assert len(sm.history) == 1
        assert sm.history[0]["state"] == "created"


class TestValidTransitions:
    def test_created_to_cfp_sent(self):
        sm = NegotiationStateMachine("neg_001")
        sm.transition(NegotiationState.CFP_SENT)
        assert sm.current_state == NegotiationState.CFP_SENT

    def test_cfp_sent_to_collecting(self):
        sm = NegotiationStateMachine("neg_001")
        sm.transition(NegotiationState.CFP_SENT)
        sm.transition(NegotiationState.COLLECTING_PROPOSALS)
        assert sm.current_state == NegotiationState.COLLECTING_PROPOSALS

    def test_full_successful_lifecycle(self):
        sm = NegotiationStateMachine("neg_001")
        sm.transition(NegotiationState.CFP_SENT)
        sm.transition(NegotiationState.COLLECTING_PROPOSALS)
        sm.transition(NegotiationState.EVALUATING)
        sm.transition(NegotiationState.AWARDED)
        sm.transition(NegotiationState.EXECUTING)
        sm.transition(NegotiationState.COMPLETED)
        sm.transition(NegotiationState.TERMINATED)
        assert sm.current_state == NegotiationState.TERMINATED
        assert sm.is_terminal()

    def test_full_failure_lifecycle(self):
        sm = NegotiationStateMachine("neg_001")
        sm.transition(NegotiationState.CFP_SENT)
        sm.transition(NegotiationState.COLLECTING_PROPOSALS)
        sm.transition(NegotiationState.EVALUATING)
        sm.transition(NegotiationState.FAILED)
        sm.transition(NegotiationState.TERMINATED)
        assert sm.is_terminal()

    def test_multi_round_transition(self):
        """EVALUATING → NEGOTIATING → CFP_SENT (next round)."""
        sm = NegotiationStateMachine("neg_001")
        sm.transition(NegotiationState.CFP_SENT)
        sm.transition(NegotiationState.COLLECTING_PROPOSALS)
        sm.transition(NegotiationState.EVALUATING)
        sm.transition(NegotiationState.NEGOTIATING)
        sm.transition(NegotiationState.CFP_SENT)
        assert sm.current_state == NegotiationState.CFP_SENT


class TestInvalidTransitions:
    def test_created_to_awarded_invalid(self):
        sm = NegotiationStateMachine("neg_001")
        with pytest.raises(InvalidStateTransition, match="Cannot transition"):
            sm.transition(NegotiationState.AWARDED)

    def test_completed_to_cfp_sent_invalid(self):
        sm = NegotiationStateMachine("neg_001")
        sm.transition(NegotiationState.CFP_SENT)
        sm.transition(NegotiationState.COLLECTING_PROPOSALS)
        sm.transition(NegotiationState.EVALUATING)
        sm.transition(NegotiationState.AWARDED)
        sm.transition(NegotiationState.EXECUTING)
        sm.transition(NegotiationState.COMPLETED)
        with pytest.raises(InvalidStateTransition):
            sm.transition(NegotiationState.CFP_SENT)

    def test_terminated_is_terminal(self):
        sm = NegotiationStateMachine("neg_001")
        sm.transition(NegotiationState.FAILED)
        sm.transition(NegotiationState.TERMINATED)
        with pytest.raises(InvalidStateTransition):
            sm.transition(NegotiationState.CFP_SENT)


class TestHistory:
    def test_history_records_all_transitions(self):
        sm = NegotiationStateMachine("neg_001")
        sm.transition(NegotiationState.CFP_SENT)
        sm.transition(NegotiationState.COLLECTING_PROPOSALS)
        assert len(sm.history) == 3  # CREATED + 2 transitions
        states = [h["state"] for h in sm.history]
        assert states == ["created", "cfp_sent", "collecting_proposals"]

    def test_history_has_timestamps(self):
        sm = NegotiationStateMachine("neg_001")
        sm.transition(NegotiationState.CFP_SENT)
        for entry in sm.history:
            assert "timestamp" in entry
            assert entry["negotiation_id"] == "neg_001"

    def test_can_transition_check(self):
        sm = NegotiationStateMachine("neg_001")
        assert sm.can_transition(NegotiationState.CFP_SENT) is True
        assert sm.can_transition(NegotiationState.AWARDED) is False


# ── Fix: refused award ───────────────────────────────────────────────────────

class TestAwardRefusalTransition:
    def test_awarded_can_return_to_negotiating(self):
        sm = NegotiationStateMachine("neg_r")
        for s in (NegotiationState.CFP_SENT, NegotiationState.COLLECTING_PROPOSALS,
                  NegotiationState.EVALUATING, NegotiationState.AWARDED,
                  NegotiationState.NEGOTIATING, NegotiationState.AWARDED,
                  NegotiationState.EXECUTING, NegotiationState.COMPLETED,
                  NegotiationState.TERMINATED):
            sm.transition(s)
        assert sm.is_terminal()

    def test_executing_cannot_return_to_negotiating(self):
        sm = NegotiationStateMachine("neg_x")
        for s in (NegotiationState.CFP_SENT, NegotiationState.COLLECTING_PROPOSALS,
                  NegotiationState.EVALUATING, NegotiationState.AWARDED,
                  NegotiationState.EXECUTING):
            sm.transition(s)
        with pytest.raises(InvalidStateTransition):
            sm.transition(NegotiationState.NEGOTIATING)
