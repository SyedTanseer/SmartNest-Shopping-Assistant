"""Tests for the desire and intention managers."""

from __future__ import annotations

import json

import pytest

from smartnest.bdi import (
    Action,
    BeliefBase,
    Condition,
    Desire,
    DesireManager,
    DesireStatus,
    Intention,
    IntentionManager,
    IntentionStatus,
    select_sofa_desire,
    stay_within_budget_desire,
)
from smartnest.bdi.intentions import ActionStatus


def desire(desire_id="d1", priority=50, **kw) -> Desire:
    return Desire(desire_id=desire_id, description=f"goal {desire_id}", priority=priority, **kw)


# ══ Desires ═════════════════════════════════════════════════════════════════

class TestDesireBasics:
    def test_fields(self):
        d = Desire(description="x", priority=70, conditions=[Condition(subject="u", predicate="p", operator="exists")])
        assert d.desire_id.startswith("des_") and d.status == DesireStatus.ACTIVE
        assert d.priority == 70 and len(d.conditions) == 1

    @pytest.mark.parametrize("priority", [-1, 101, True])
    def test_priority_range_enforced(self, priority):
        with pytest.raises(ValueError):
            Desire(description="x", priority=priority)

    def test_description_required(self):
        with pytest.raises(ValueError):
            Desire(description="   ")

    def test_duplicate_id_rejected(self):
        manager = DesireManager()
        manager.add(desire("a"))
        with pytest.raises(ValueError):
            manager.add(desire("a"))

    def test_standard_desires(self):
        assert select_sofa_desire().goal_type == "select_product"
        budget = stay_within_budget_desire()
        assert budget.kind == "maintenance" and budget.priority > select_sofa_desire().priority


class TestPrioritisation:
    def test_all_sorted_by_priority_then_age(self):
        manager = DesireManager()
        for did, prio in (("low", 10), ("high", 90), ("mid_first", 50), ("mid_second", 50)):
            manager.add(desire(did, prio))
        assert [d.desire_id for d in manager.all()] == ["high", "mid_first", "mid_second", "low"]

    def test_several_desires_pursuable_simultaneously(self):
        manager = DesireManager()
        manager.add(desire("a", 20))
        manager.add(desire("b", 80))
        manager.add(desire("c", 50))
        assert [d.desire_id for d in manager.pursuable(BeliefBase())] == ["b", "c", "a"]

    def test_changing_priority_reorders(self):
        manager = DesireManager()
        manager.add(desire("a", 20))
        manager.add(desire("b", 80))
        assert manager.set_priority("a", 95) is True
        assert manager.set_priority("a", 95) is False
        assert [d.desire_id for d in manager.pursuable(BeliefBase())] == ["a", "b"]
        with pytest.raises(ValueError):
            manager.set_priority("a", 200)

    def test_conditions_gate_pursuit(self):
        manager = DesireManager()
        manager.add(desire("needs_width", 80, conditions=[Condition(subject="user", predicate="w", operator="exists")]))
        manager.add(desire("free", 10))
        beliefs = BeliefBase()
        assert [d.desire_id for d in manager.pursuable(beliefs)] == ["free"]
        waiting = manager.waiting_for_conditions(beliefs)
        assert waiting[0][0].desire_id == "needs_width" and "user.w" in waiting[0][1][0].explanation
        beliefs.update("user", "w", 210)
        assert [d.desire_id for d in manager.pursuable(beliefs)] == ["needs_width", "free"]

    def test_only_active_and_blocked_are_pursuable(self):
        manager = DesireManager()
        for did in ("active", "blocked", "suspended", "achieved", "dropped"):
            manager.add(desire(did))
        manager.set_status("blocked", DesireStatus.BLOCKED, "nothing feasible")
        manager.suspend("suspended", "later")
        manager.achieve("achieved")
        manager.drop("dropped")
        assert {d.desire_id for d in manager.pursuable(BeliefBase())} == {"active", "blocked"}


class TestDesireStatus:
    def test_transitions_and_reasons(self):
        manager = DesireManager()
        manager.add(desire("a"))
        assert manager.set_status("a", DesireStatus.BLOCKED, "no stock") is True
        assert manager.get("a").status_reason == "no stock"
        assert manager.set_status("a", DesireStatus.BLOCKED, "no stock") is False
        assert manager.set_status("a", DesireStatus.ACTIVE, "restocked") is True

    def test_terminal_statuses_stick(self):
        manager = DesireManager()
        manager.add(desire("a"))
        manager.achieve("a")
        assert manager.set_status("a", DesireStatus.ACTIVE) is False
        assert manager.get("a").status == DesireStatus.ACHIEVED

    def test_reactivate_only_from_suspended(self):
        manager = DesireManager()
        manager.add(desire("a"))
        assert manager.reactivate("a") is False
        manager.suspend("a", "pause")
        assert manager.reactivate("a") is True

    def test_unknown_desire(self):
        with pytest.raises(KeyError):
            DesireManager().achieve("nope")


class TestConflicts:
    def manager(self):
        m = DesireManager()
        m.add(desire("premium", 80, conflicts_with=["thrifty"]))
        m.add(desire("thrifty", 60))
        m.add(desire("independent", 10))
        return m

    def test_lower_priority_conflicting_desire_is_suspended(self):
        m = self.manager()
        changes = m.resolve_conflicts()
        assert [(d.desire_id) for d, _ in changes] == ["thrifty"]
        thrifty = m.get("thrifty")
        assert thrifty.status == DesireStatus.SUSPENDED and thrifty.suspended_by == "premium"
        assert m.get("premium").status == DesireStatus.ACTIVE
        assert m.get("independent").status == DesireStatus.ACTIVE

    def test_resolution_is_idempotent(self):
        m = self.manager()
        m.resolve_conflicts()
        assert m.resolve_conflicts() == []

    def test_loser_is_reactivated_when_winner_goes(self):
        m = self.manager()
        m.resolve_conflicts()
        m.drop("premium")
        changes = m.resolve_conflicts()
        assert [d.desire_id for d, _ in changes] == ["thrifty"]
        assert m.get("thrifty").status == DesireStatus.ACTIVE

    def test_priority_change_flips_the_winner(self):
        m = self.manager()
        m.resolve_conflicts()
        m.set_priority("thrifty", 95)
        m.resolve_conflicts()
        assert m.get("thrifty").status == DesireStatus.ACTIVE
        assert m.get("premium").status == DesireStatus.SUSPENDED and m.get("premium").suspended_by == "thrifty"

    def test_user_suspension_is_not_overridden(self):
        m = self.manager()
        m.suspend("independent", "user paused it")
        m.resolve_conflicts()
        assert m.get("independent").status == DesireStatus.SUSPENDED

    def test_conflict_declared_on_either_side(self):
        m = DesireManager()
        m.add(desire("a", 90))
        m.add(desire("b", 10, conflicts_with=["a"]))
        m.resolve_conflicts()
        assert m.get("b").status == DesireStatus.SUSPENDED


class TestDesirePersistence:
    def test_round_trip(self):
        m = DesireManager()
        m.add(select_sofa_desire())
        m.add(stay_within_budget_desire())
        m.set_status("des_select_sofa", DesireStatus.BLOCKED, "nothing fits")
        data = json.loads(json.dumps(m.to_list()))
        restored = DesireManager()
        restored.load(data)
        assert [d.desire_id for d in restored.all()] == [d.desire_id for d in m.all()]
        assert restored.get("des_select_sofa").status == DesireStatus.BLOCKED


# ══ Intentions ══════════════════════════════════════════════════════════════

def stock_beliefs(available=True, price=100, budget=150) -> BeliefBase:
    beliefs = BeliefBase()
    beliefs.update("P", "availability", "in_stock" if available else "out_of_stock")
    beliefs.update("P", "price", price)
    beliefs.update("user", "budget", budget)
    return beliefs


AVAILABLE = Condition(subject="P", predicate="availability", operator="in", value=["in_stock", "low_stock"])
AFFORDABLE = Condition(subject="P", predicate="price", operator="lte", ref_subject="user", ref_predicate="budget")


def make_intention(desire_id="d1", product="P", manual_last=True, **kw) -> Intention:
    return Intention(
        desire_id=desire_id,
        description=f"obtain {product}",
        preconditions=[AVAILABLE, AFFORDABLE],
        params={"product_id": product, "price": 100},
        actions=[
            Action(kind="check", description="check it"),
            Action(kind="recommend", params={"product_id": product}),
            Action(kind="reserve", manual=manual_last, preconditions=[AVAILABLE]),
        ],
        **kw,
    )


class TestIntentionLifecycle:
    def test_created_pending_then_activated(self):
        manager = IntentionManager()
        intention = manager.add(make_intention())
        assert intention.status == IntentionStatus.PENDING and intention.is_live
        assert manager.activate(intention.intention_id) is True
        assert intention.status == IntentionStatus.ACTIVE
        assert manager.activate(intention.intention_id) is False  # already active

    def test_duplicate_and_unknown(self):
        manager = IntentionManager()
        i = manager.add(make_intention())
        with pytest.raises(ValueError):
            manager.add(i)
        with pytest.raises(KeyError):
            manager.cancel("nope")

    def test_suspend_and_resume(self):
        manager, beliefs = IntentionManager(), stock_beliefs()
        i = manager.add(make_intention())
        manager.activate(i.intention_id)
        assert manager.suspend(i.intention_id, "paused") is True
        assert i.status == IntentionStatus.SUSPENDED and i.status_reason == "paused"
        assert manager.resume(i.intention_id, beliefs) is True
        assert i.status == IntentionStatus.ACTIVE

    def test_cannot_resume_while_invalid(self):
        manager = IntentionManager()
        i = manager.add(make_intention())
        manager.activate(i.intention_id)
        manager.suspend(i.intention_id)
        assert manager.resume(i.intention_id, stock_beliefs(available=False)) is False
        assert i.status == IntentionStatus.SUSPENDED

    def test_cancel(self):
        manager = IntentionManager()
        i = manager.add(make_intention())
        manager.activate(i.intention_id)
        assert manager.cancel(i.intention_id, "user changed their mind") is True
        assert i.status == IntentionStatus.CANCELLED and not i.is_live
        assert manager.cancel(i.intention_id) is False
        assert manager.activate(i.intention_id) is False

    def test_replace_retires_old_and_activates_new(self):
        manager = IntentionManager()
        old = manager.add(make_intention(product="P"))
        manager.activate(old.intention_id)
        new = manager.replace(old.intention_id, make_intention(product="Q"), "out of stock")
        assert old.status == IntentionStatus.REPLACED and old.replaced_by == new.intention_id
        assert new.status == IntentionStatus.ACTIVE
        assert manager.live_for("d1") is new

    def test_replace_suspended_intention(self):
        manager = IntentionManager()
        old = manager.add(make_intention())
        manager.activate(old.intention_id)
        manager.suspend(old.intention_id)
        manager.replace(old.intention_id, make_intention(product="Q"))
        assert old.status == IntentionStatus.REPLACED

    def test_cannot_replace_terminal(self):
        manager = IntentionManager()
        old = manager.add(make_intention())
        manager.cancel(old.intention_id)
        with pytest.raises(ValueError):
            manager.replace(old.intention_id, make_intention(product="Q"))

    def test_live_and_history(self):
        manager = IntentionManager()
        a = manager.add(make_intention(desire_id="d1"))
        manager.activate(a.intention_id)
        manager.replace(a.intention_id, make_intention(desire_id="d1", product="Q"))
        manager.add(make_intention(desire_id="d2"))
        assert len(manager.history_for("d1")) == 2
        assert [i.desire_id for i in manager.live()] == ["d1", "d2"]


class TestValidation:
    def test_valid_when_preconditions_hold(self):
        manager = IntentionManager()
        result = manager.validate(make_intention(), stock_beliefs())
        assert result.valid and result.failures == []

    def test_out_of_stock_is_recognised(self):
        result = IntentionManager.validate(make_intention(), stock_beliefs(available=False))
        assert not result.valid
        assert "P.availability = 'out_of_stock'" in result.explain()

    def test_over_budget_is_recognised(self):
        result = IntentionManager.validate(make_intention(), stock_beliefs(price=200, budget=150))
        assert not result.valid and "user.budget" in result.explain()

    def test_action_preconditions_count_until_the_action_is_done(self):
        intention = make_intention()
        intention.preconditions = []  # only the actions' own preconditions remain
        assert not IntentionManager.validate(intention, stock_beliefs(available=False)).valid
        for action in intention.actions:
            action.status = ActionStatus.DONE
        assert IntentionManager.validate(intention, stock_beliefs(available=False)).valid

    def test_duplicate_failures_reported_once(self):
        result = IntentionManager.validate(make_intention(), stock_beliefs(available=False))
        assert len([f for f in result.failures if "availability" in f.explanation]) == 1


class TestExecution:
    def setup_method(self):
        self.manager = IntentionManager()
        self.beliefs = stock_beliefs()
        self.intention = self.manager.add(make_intention())
        self.manager.activate(self.intention.intention_id)
        self.executed: list[str] = []

        def executor(action, intention):
            self.executed.append(action.kind)
            return {"did": action.kind}

        self.executor = executor

    def test_run_executes_automatic_steps_and_stops_at_manual(self):
        outcomes = self.manager.run(self.intention.intention_id, self.beliefs, self.executor)
        assert self.executed == ["check", "recommend"]
        assert [o.status for o in outcomes] == ["done", "done", "awaiting_confirmation"]
        assert self.intention.status == IntentionStatus.ACTIVE
        assert [a.status for a in self.intention.actions] == [ActionStatus.DONE, ActionStatus.DONE, ActionStatus.PENDING]
        assert self.intention.actions[0].result == {"did": "check"}
        assert self.intention.actions[0].executed_at is not None

    def test_confirm_runs_manual_step_and_completes(self):
        self.manager.run(self.intention.intention_id, self.beliefs, self.executor)
        outcome = self.manager.confirm(self.intention.intention_id, self.beliefs, self.executor)
        assert outcome.status == "done" and outcome.kind == "reserve"
        assert self.intention.status == IntentionStatus.COMPLETED
        assert self.executed == ["check", "recommend", "reserve"]

    def test_invalid_intention_is_suspended_and_nothing_runs(self):
        outcome = self.manager.execute_next(self.intention.intention_id, stock_beliefs(available=False), self.executor)
        assert outcome.status == "precondition_failed" and "availability" in outcome.detail
        assert self.executed == []
        assert self.intention.status == IntentionStatus.SUSPENDED

    def test_precondition_failure_midway_stops_execution(self):
        self.manager.execute_next(self.intention.intention_id, self.beliefs, self.executor)
        outcome = self.manager.execute_next(self.intention.intention_id, stock_beliefs(available=False), self.executor)
        assert outcome.status == "precondition_failed"
        assert self.executed == ["check"]
        assert self.intention.pending_actions()[0].kind == "recommend"

    def test_executor_failure_suspends_without_crashing(self):
        def failing(action, intention):
            raise RuntimeError("payment service down")

        outcome = self.manager.execute_next(self.intention.intention_id, self.beliefs, failing)
        assert outcome.status == "failed" and "payment service down" in outcome.detail
        assert self.intention.actions[0].status == ActionStatus.FAILED
        assert self.intention.status == IntentionStatus.SUSPENDED

    def test_nothing_runs_unless_active(self):
        self.manager.suspend(self.intention.intention_id)
        assert self.manager.execute_next(self.intention.intention_id, self.beliefs).status == "nothing_to_do"
        assert self.manager.run(self.intention.intention_id, self.beliefs) == []

    def test_all_automatic_plan_completes_by_itself(self):
        manager = IntentionManager()
        i = manager.add(make_intention(manual_last=False))
        manager.activate(i.intention_id)
        manager.run(i.intention_id, self.beliefs, self.executor)
        assert i.status == IntentionStatus.COMPLETED


class TestIntentionPersistence:
    def test_round_trip_preserves_progress(self):
        manager, beliefs = IntentionManager(), stock_beliefs()
        i = manager.add(make_intention())
        manager.activate(i.intention_id)
        manager.run(i.intention_id, beliefs)
        data = json.loads(json.dumps(manager.to_list()))
        restored = IntentionManager()
        restored.load(data)
        back = restored.get(i.intention_id)
        assert back.status == IntentionStatus.ACTIVE
        assert [a.status for a in back.actions] == [ActionStatus.DONE, ActionStatus.DONE, ActionStatus.PENDING]
        assert back.preconditions == i.preconditions
        assert restored.validate(back, beliefs).valid
