"""Tests for beliefs and conditions."""

from __future__ import annotations

import json
import math

import pytest

from smartnest.bdi import BeliefBase, Condition


class TestBeliefUpdates:
    def test_new_belief_is_added(self):
        beliefs = BeliefBase()
        change = beliefs.update("user", "sofa_budget", 30000, source="user")
        assert change.kind == "added" and change.old_value is None and change.new_value == 30000
        assert beliefs.value("user", "sofa_budget") == 30000

    def test_belief_records_subject_predicate_value_source_timestamp(self):
        beliefs = BeliefBase()
        beliefs.update("SOFA_A2", "availability", "in_stock", source="catalogue")
        belief = beliefs.get("SOFA_A2", "availability")
        assert (belief.subject, belief.predicate, belief.value, belief.source) == \
            ("SOFA_A2", "availability", "in_stock", "catalogue")
        assert belief.timestamp.tzinfo is not None and belief.valid

    def test_changed_value_reports_old_and_new(self):
        beliefs = BeliefBase()
        beliefs.update("SOFA_A2", "availability", "in_stock", "catalogue")
        change = beliefs.update("SOFA_A2", "availability", "out_of_stock", "product_monitor")
        assert change.kind == "updated"
        assert (change.old_value, change.new_value, change.source) == ("in_stock", "out_of_stock", "product_monitor")
        assert beliefs.get("SOFA_A2", "availability").source == "product_monitor"

    def test_identical_update_is_not_a_change(self):
        beliefs = BeliefBase()
        beliefs.update("user", "max_width_cm", 210)
        stamp = beliefs.get("user", "max_width_cm").timestamp
        assert beliefs.update("user", "max_width_cm", 210, "someone else") is None
        assert beliefs.update("user", "max_width_cm", 210.0) is None  # numerically equal
        assert beliefs.get("user", "max_width_cm").timestamp == stamp

    @pytest.mark.parametrize("subject,predicate", [("", "p"), ("s", ""), ("  ", "p"), (None, "p"), ("s", 5)])
    def test_invalid_names_rejected(self, subject, predicate):
        with pytest.raises(ValueError):
            BeliefBase().update(subject, predicate, 1)

    def test_get_returns_copies(self):
        beliefs = BeliefBase()
        beliefs.update("user", "style", "minimalist")
        beliefs.get("user", "style").value = "hacked"
        assert beliefs.value("user", "style") == "minimalist"

    def test_unknown_belief(self):
        beliefs = BeliefBase()
        assert beliefs.get("x", "y") is None
        assert beliefs.value("x", "y") is None and beliefs.value("x", "y", "dflt") == "dflt"
        assert not beliefs.has("x", "y")


class TestInvalidation:
    def test_invalidated_belief_is_unknown(self):
        beliefs = BeliefBase()
        beliefs.update("user", "max_width_cm", 210)
        change = beliefs.invalidate("user", "max_width_cm", "room being remeasured")
        assert change.kind == "invalidated" and change.old_value == 210
        assert beliefs.value("user", "max_width_cm") is None
        assert beliefs.get("user", "max_width_cm") is None
        assert not beliefs.has("user", "max_width_cm")

    def test_invalid_belief_remains_inspectable(self):
        beliefs = BeliefBase()
        beliefs.update("user", "max_width_cm", 210)
        beliefs.invalidate("user", "max_width_cm", "stale")
        belief = beliefs.get("user", "max_width_cm", include_invalid=True)
        assert not belief.valid and belief.invalidation_reason == "stale" and belief.invalidated_at is not None
        assert belief.value == 210

    def test_invalidating_unknown_or_already_invalid_does_nothing(self):
        beliefs = BeliefBase()
        assert beliefs.invalidate("x", "y") is None
        beliefs.update("x", "y", 1)
        beliefs.invalidate("x", "y")
        assert beliefs.invalidate("x", "y") is None

    def test_update_revalidates(self):
        beliefs = BeliefBase()
        beliefs.update("x", "y", 1)
        beliefs.invalidate("x", "y")
        change = beliefs.update("x", "y", 1)  # same value, but it was invalid -> a real change
        assert change is not None and change.kind == "added"
        assert beliefs.value("x", "y") == 1 and beliefs.get("x", "y").valid

    def test_query_excludes_invalid_by_default(self):
        beliefs = BeliefBase()
        beliefs.update("a", "p", 1)
        beliefs.update("a", "q", 2)
        beliefs.invalidate("a", "q")
        assert [b.predicate for b in beliefs.query(subject="a")] == ["p"]
        assert len(beliefs.query(subject="a", include_invalid=True)) == 2


class TestEnsureAndQuery:
    def test_ensure_sets_missing(self):
        beliefs = BeliefBase()
        assert beliefs.ensure("p", "price", 100, "catalogue").kind == "added"

    def test_ensure_refreshes_own_source(self):
        beliefs = BeliefBase()
        beliefs.ensure("p", "price", 100, "catalogue")
        assert beliefs.ensure("p", "price", 90, "catalogue").new_value == 90

    def test_ensure_never_overwrites_other_source(self):
        beliefs = BeliefBase()
        beliefs.update("p", "price", 80, "price_monitor")
        assert beliefs.ensure("p", "price", 100, "catalogue") is None
        assert beliefs.value("p", "price") == 80

    def test_ensure_replaces_invalid(self):
        beliefs = BeliefBase()
        beliefs.update("p", "price", 80, "price_monitor")
        beliefs.invalidate("p", "price")
        assert beliefs.ensure("p", "price", 100, "catalogue") is not None

    def test_query_filters_and_as_dict(self):
        beliefs = BeliefBase()
        beliefs.update("user", "a", 1)
        beliefs.update("user", "b", 2)
        beliefs.update("p", "a", 3)
        assert len(beliefs.query(subject="user")) == 2
        assert len(beliefs.query(predicate="a")) == 2
        assert beliefs.as_dict() == {"user.a": 1, "user.b": 2, "p.a": 3}
        assert len(beliefs) == 3 and ("user", "a") in beliefs


class TestBeliefPersistence:
    def test_round_trip_is_json_compatible(self):
        beliefs = BeliefBase()
        beliefs.update("user", "style", "minimalist", "user")
        beliefs.update("p", "tags", ["a", "b"])
        beliefs.update("p", "old", 1)
        beliefs.invalidate("p", "old", "stale")
        data = json.loads(json.dumps(beliefs.to_list()))
        restored = BeliefBase()
        restored.load(data)
        assert restored.as_dict() == beliefs.as_dict()
        assert restored.get("p", "old", include_invalid=True).invalidation_reason == "stale"
        assert restored.get("user", "style").timestamp == beliefs.get("user", "style").timestamp

    def test_load_replaces_everything(self):
        beliefs = BeliefBase()
        beliefs.update("old", "x", 1)
        beliefs.load([])
        assert len(beliefs) == 0


class TestConditions:
    @staticmethod
    def beliefs(**values):
        b = BeliefBase()
        for key, value in values.items():
            subject, predicate = key.split("__")
            b.update(subject, predicate, value)
        return b

    @pytest.mark.parametrize("operator,value,actual,expected", [
        ("eq", 5, 5, True), ("eq", 5, 6, False), ("ne", 5, 6, True), ("ne", 5, 5, False),
        ("lt", 5, 4, True), ("lt", 5, 5, False), ("lte", 5, 5, True), ("lte", 5, 6, False),
        ("gt", 5, 6, True), ("gt", 5, 5, False), ("gte", 5, 5, True), ("gte", 5, 4, False),
        ("in", [1, 2], 2, True), ("in", [1, 2], 3, False),
        ("not_in", [1, 2], 3, True), ("not_in", [1, 2], 1, False),
    ])
    def test_operators(self, operator, value, actual, expected):
        condition = Condition(subject="s", predicate="p", operator=operator, value=value)
        assert condition.evaluate(self.beliefs(s__p=actual)).ok is expected

    def test_exists_and_not_exists(self):
        b = self.beliefs(s__p=1)
        assert Condition(subject="s", predicate="p", operator="exists").evaluate(b).ok
        assert not Condition(subject="s", predicate="q", operator="exists").evaluate(b).ok
        assert Condition(subject="s", predicate="q", operator="not_exists").evaluate(b).ok
        assert not Condition(subject="s", predicate="p", operator="not_exists").evaluate(b).ok

    def test_unknown_belief_fails_with_reason(self):
        result = Condition(subject="s", predicate="p", operator="eq", value=1).evaluate(BeliefBase())
        assert not result.ok and "s.p is unknown" in result.explanation

    def test_invalidated_belief_counts_as_unknown(self):
        b = self.beliefs(s__p=1)
        b.invalidate("s", "p")
        assert not Condition(subject="s", predicate="p", operator="eq", value=1).evaluate(b).ok

    def test_reference_to_another_belief(self):
        condition = Condition(subject="sofa", predicate="price", operator="lte",
                              ref_subject="user", ref_predicate="budget")
        assert condition.evaluate(self.beliefs(sofa__price=100, user__budget=100)).ok
        failed = condition.evaluate(self.beliefs(sofa__price=101, user__budget=100))
        assert not failed.ok and "user.budget" in failed.explanation and "101" in failed.explanation

    def test_unknown_reference_fails(self):
        condition = Condition(subject="sofa", predicate="price", operator="lte",
                              ref_subject="user", ref_predicate="budget")
        result = condition.evaluate(self.beliefs(sofa__price=100))
        assert not result.ok and "user.budget is unknown" in result.explanation

    def test_incomparable_types_fail_instead_of_raising(self):
        result = Condition(subject="s", predicate="p", operator="lt", value=5).evaluate(self.beliefs(s__p="text"))
        assert not result.ok and "cannot be compared" in result.explanation

    def test_booleans_are_not_ordered(self):
        assert not Condition(subject="s", predicate="p", operator="gt", value=0).evaluate(self.beliefs(s__p=True)).ok

    def test_nan_never_satisfies_ordering(self):
        assert not Condition(subject="s", predicate="p", operator="lte", value=5).evaluate(
            self.beliefs(s__p=math.nan)).ok

    def test_references_lists_every_belief_read(self):
        condition = Condition(subject="a", predicate="b", operator="lte", ref_subject="c", ref_predicate="d")
        assert condition.references() == [("a", "b"), ("c", "d")]

    def test_condition_serialises(self):
        condition = Condition(subject="s", predicate="p", operator="in", value=["x", "y"], description="d")
        assert Condition.model_validate_json(condition.model_dump_json()) == condition
