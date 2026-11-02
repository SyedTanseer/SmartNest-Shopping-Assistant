"""Tests for the generic RuleEngine and Rule abstraction.

These tests verify engine mechanics (registration, priority ordering,
enable/disable, duplicate detection) using dummy rules so they don't
depend on SmartNest domain logic.
"""

from __future__ import annotations

import pytest

from smartnest.core.models import ActionLog, ActionResultType, EnvironmentalEvent, EventType
from smartnest.reasoning.rule_engine import Rule, RuleEngine


# ── Fixtures ────────────────────────────────────────────────────────────────

def _make_event(event_type: EventType = EventType.INFO if hasattr(EventType, "INFO") else EventType.BUDGET_UPDATE) -> EnvironmentalEvent:
    return EnvironmentalEvent(
        event_type=event_type,
        payload={},
    )


def _dummy_rule(rule_id: str, priority: int, matches: bool = True) -> Rule:
    """Create a minimal rule for testing engine mechanics."""
    def condition(event: EnvironmentalEvent) -> bool:
        return matches

    def action(event: EnvironmentalEvent) -> ActionLog:
        return ActionLog(
            event_id=event.event_id,
            rule_id=rule_id,
            rule_name=f"Dummy {rule_id}",
            action_description="Dummy action",
            result_type=ActionResultType.INFO,
            result_message="dummy",
        )

    return Rule(
        rule_id=rule_id,
        name=f"Dummy {rule_id}",
        priority=priority,
        condition=condition,
        action=action,
        explanation="test rule",
    )


# ── Tests ──────────────────────────────────────────────────────────────────

class TestRuleEngineRegistration:
    def test_register_single_rule(self):
        engine = RuleEngine()
        engine.register(_dummy_rule("R1", 10))
        assert len(engine.rules) == 1

    def test_register_multiple_rules_sorted_by_priority(self):
        engine = RuleEngine()
        engine.register(_dummy_rule("R3", 30))
        engine.register(_dummy_rule("R1", 10))
        engine.register(_dummy_rule("R2", 20))
        ids = [r.rule_id for r in engine.rules]
        assert ids == ["R1", "R2", "R3"]

    def test_duplicate_rule_id_raises(self):
        engine = RuleEngine()
        engine.register(_dummy_rule("R1", 10))
        with pytest.raises(ValueError, match="Duplicate rule_id"):
            engine.register(_dummy_rule("R1", 20))

    def test_unregister_rule(self):
        engine = RuleEngine()
        engine.register(_dummy_rule("R1", 10))
        engine.register(_dummy_rule("R2", 20))
        engine.unregister("R1")
        assert len(engine.rules) == 1
        assert engine.rules[0].rule_id == "R2"

    def test_enable_disable_rule(self):
        engine = RuleEngine()
        engine.register(_dummy_rule("R1", 10))
        engine.disable("R1")
        assert engine.rules[0].enabled is False
        engine.enable("R1")
        assert engine.rules[0].enabled is True


class TestRuleEngineEvaluation:
    def test_first_matching_rule_fires(self):
        engine = RuleEngine()
        engine.register(_dummy_rule("R1", 10, matches=False))
        engine.register(_dummy_rule("R2", 20, matches=True))
        log = engine.evaluate(_make_event())
        assert log is not None
        assert log.rule_id == "R2"

    def test_priority_determines_first_match(self):
        """If two rules match, the one with lower priority number fires first."""
        engine = RuleEngine()
        engine.register(_dummy_rule("R_high", 5, matches=True))
        engine.register(_dummy_rule("R_low", 50, matches=True))
        log = engine.evaluate(_make_event())
        assert log is not None
        assert log.rule_id == "R_high"

    def test_no_matching_rule_returns_none(self):
        engine = RuleEngine()
        engine.register(_dummy_rule("R1", 10, matches=False))
        log = engine.evaluate(_make_event())
        assert log is None

    def test_disabled_rule_does_not_fire(self):
        engine = RuleEngine()
        engine.register(_dummy_rule("R1", 10, matches=True))
        engine.disable("R1")
        log = engine.evaluate(_make_event())
        assert log is None

    def test_none_event_returns_none(self):
        engine = RuleEngine()
        engine.register(_dummy_rule("R1", 10))
        assert engine.evaluate(None) is None

    def test_non_event_object_returns_none(self):
        engine = RuleEngine()
        engine.register(_dummy_rule("R1", 10))
        assert engine.evaluate("not an event") is None


class TestRuleErrorHandling:
    def test_condition_exception_does_not_crash(self):
        """A rule whose condition raises should be treated as non-matching."""
        def bad_condition(event: EnvironmentalEvent) -> bool:
            raise RuntimeError("boom")

        def ok_action(event: EnvironmentalEvent) -> ActionLog:
            return ActionLog(
                event_id=event.event_id,
                rule_id="bad",
                rule_name="Bad Rule",
                action_description="should not fire",
                result_type=ActionResultType.INFO,
                result_message="should not happen",
            )

        engine = RuleEngine()
        engine.register(Rule(
            rule_id="R_bad",
            name="Bad Condition Rule",
            priority=10,
            condition=bad_condition,
            action=ok_action,
        ))
        log = engine.evaluate(_make_event())
        assert log is None  # condition raised → treated as non-match

    def test_action_exception_produces_error_log(self):
        """A rule whose action raises should produce an ERROR action log."""
        def good_condition(event: EnvironmentalEvent) -> bool:
            return True

        def bad_action(event: EnvironmentalEvent) -> ActionLog:
            raise RuntimeError("action failed")

        engine = RuleEngine()
        engine.register(Rule(
            rule_id="R_bad_action",
            name="Bad Action Rule",
            priority=10,
            condition=good_condition,
            action=bad_action,
        ))
        log = engine.evaluate(_make_event())
        assert log is not None
        assert log.result_type == ActionResultType.ERROR
        assert "action failed" in log.result_message
