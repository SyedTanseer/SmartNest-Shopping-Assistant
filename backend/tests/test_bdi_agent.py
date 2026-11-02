"""Tests for the BDIAgent: the SOFA_A2 scenario, dynamic replanning, irrelevant events."""

from __future__ import annotations

import math

import pytest

from smartnest.agents.base_agent import BaseAgent
from smartnest.bdi import (
    BDIAgent,
    CatalogCandidateSource,
    DesireStatus,
    IntentionStatus,
    select_sofa_desire,
)
from smartnest.bdi.desires import Desire
from smartnest.core.events import EventStream
from smartnest.core.models import ActionLog, ActionResultType, EnvironmentalEvent, EventType
from smartnest.optimization import ConstraintValidator, UserRequirements, UtilityOptimizer
from tests.bdi_support import (
    BUDGET,
    REQUIREMENTS,
    SELECT,
    candidate_records,
    chosen,
    event,
    in_stock,
    kinds_since,
    make_agent,
    make_planner,
    out_of_stock,
    reference_choice,
)

ET = EventType


# ══ The assignment scenario ═════════════════════════════════════════════════

class TestSofaA2Scenario:
    def test_initially_selects_sofa_a2(self):
        agent = make_agent()
        assert chosen(agent) == "SOFA_A2" == reference_choice()
        intention = agent.current_intention(SELECT)
        assert intention.status == IntentionStatus.ACTIVE
        assert intention.params["source_agent_id"] == "research_agent_a"

    def test_initial_intention_executes_automatic_steps_and_awaits_confirmation(self):
        agent = make_agent()
        actions = agent.current_intention(SELECT).actions
        assert [a.status.value for a in actions] == ["done", "done", "pending"]
        assert actions[2].manual

    def test_out_of_stock_event_changes_the_intention(self):
        agent = make_agent()
        original = agent.current_intention(SELECT)

        log = agent.handle_event(out_of_stock("SOFA_A2"))

        # 1. availability belief updated
        belief = agent.beliefs.get("SOFA_A2", "availability")
        assert belief.value == "out_of_stock" and belief.source == "product_monitor"
        # 2. the old intention was recognised as invalid and retired
        assert original.status == IntentionStatus.REPLACED
        assert "SOFA_A2.availability = 'out_of_stock'" in original.status_reason
        # 3-5. desires reconsidered, optimizer re-run, best feasible alternative chosen
        replacement = agent.current_intention(SELECT)
        assert replacement.intention_id != original.intention_id
        assert replacement.params["product_id"] != "SOFA_A2"
        assert replacement.params["product_id"] == reference_choice(availability={"SOFA_A2": "out_of_stock"})
        assert original.replaced_by == replacement.intention_id
        # 6. a revised, active, executing intention
        assert replacement.status == IntentionStatus.ACTIVE
        assert [a.status.value for a in replacement.actions] == ["done", "done", "pending"]
        assert log.rule_id == "BDI_REVISED" and log.result_type == ActionResultType.INFO.value

    def test_the_replacement_is_feasible_by_the_constraint_validator(self):
        agent = make_agent()
        agent.handle_event(out_of_stock("SOFA_A2"))
        pid = chosen(agent)
        record = next(r for r in candidate_records() if r["product_id"] == pid)
        verdict = ConstraintValidator().evaluate(record, UserRequirements(
            budget=30000, max_width_cm=210, max_depth_cm=110, min_seating=3,
            preferred_style="minimalist", preferred_colour="neutral"))
        assert verdict.eligible

    def test_replacement_has_the_highest_utility_of_all_remaining_options(self):
        agent = make_agent()
        agent.handle_event(out_of_stock("SOFA_A2"))
        transition = next(t for t in agent.transitions if t.kind == "intention_replaced")
        ranking = transition.details["ranking"]
        assert ranking[0]["product_id"] == chosen(agent)
        assert all(ranking[0]["utility"] >= r["utility"] for r in ranking)
        assert all(r["product_id"] != "SOFA_A2" for r in ranking)
        assert "SOFA_A2" in [c["product_id"] for c in transition.details["rejected_candidates"]]

    def test_full_transition_is_logged_with_before_after_and_explanation(self):
        agent = make_agent()
        before = agent.current_intention(SELECT)
        start = len(agent.transitions)
        agent.handle_event(out_of_stock("SOFA_A2"))
        new = agent.transitions[start:]
        assert kinds_since(agent, start) == [
            "belief_updated", "intention_invalidated", "desires_reconsidered",
            "intention_replaced", "intention_progress",
        ]

        belief = new[0]
        assert belief.before == {"value": "in_stock"} and belief.after == {"value": "out_of_stock"}

        invalidated = new[1]
        assert invalidated.before["status"] == "active" and invalidated.after["status"] == "suspended"
        assert invalidated.before["product_id"] == "SOFA_A2"
        assert "requires in ['in_stock', 'low_stock']" in invalidated.explanation

        replaced = new[3]
        assert replaced.before["intention_id"] == before.intention_id
        assert replaced.before["product_id"] == "SOFA_A2"
        assert replaced.after["product_id"] == chosen(agent) and replaced.after["status"] == "active"
        assert replaced.before["utility"] > replaced.after["utility"]  # the loss is explicit
        for fragment in ("SOFA_A2 was dropped because", "out_of_stock", "UtilityOptimizer",
                         chosen(agent), "highest utility"):
            assert fragment in replaced.explanation, fragment
        assert replaced.trigger.startswith("event ") and "product_availability" in replaced.trigger

    def test_the_replacement_is_data_driven_not_hardcoded(self):
        """Change the catalogue and the replacement changes accordingly."""
        records = candidate_records()
        for r in records:
            if r["product_id"] == "SOFA_B3":
                r["rating"], r["review_count"], r["price"] = 5.0, 500, 15000  # make B3 very attractive
        class Source:
            def get_candidates(self):
                return [dict(r) for r in records]

        agent = make_agent(planner=make_planner(source=Source()))
        assert chosen(agent) == reference_choice(records=records) == "SOFA_B3"
        agent.handle_event(out_of_stock("SOFA_B3"))
        assert chosen(agent) == reference_choice(records=records, availability={"SOFA_B3": "out_of_stock"})
        assert chosen(agent) == "SOFA_A2"  # now A2 is the best remaining option

    def test_a_better_option_returning_in_stock_wins_the_intention_back(self):
        agent = make_agent()
        agent.handle_event(out_of_stock("SOFA_A2"))
        log = agent.handle_event(in_stock("SOFA_A2"))
        assert chosen(agent) == "SOFA_A2" and log.rule_id == "BDI_REVISED"

    def test_restock_event_for_a_product_already_believed_in_stock_changes_nothing(self):
        agent = make_agent()
        before = len(agent.transitions)
        assert agent.handle_event(in_stock("SOFA_A2")) is None
        assert len(agent.transitions) == before and chosen(agent) == "SOFA_A2"

    def test_agent_never_recommends_an_infeasible_product(self):
        agent = make_agent()
        for pid in ("SOFA_A2", "SOFA_B3", "SOFA_A3", "SOFA_B2", "SOFA_A6", "SOFA_B4", "SOFA_A7"):
            log = agent.handle_event(out_of_stock(pid))
        assert chosen(agent) is None
        assert agent.desires.get(SELECT).status == DesireStatus.BLOCKED
        assert log.rule_id == "BDI_BLOCKED" and log.result_type == ActionResultType.WARNING.value
        assert agent.current_intention(SELECT) is None

    def test_blocked_agent_recovers_when_stock_returns(self):
        agent = make_agent()
        for pid in ("SOFA_A2", "SOFA_B3", "SOFA_A3", "SOFA_B2", "SOFA_A6", "SOFA_B4", "SOFA_A7"):
            agent.handle_event(out_of_stock(pid))
        log = agent.handle_event(in_stock("SOFA_B2"))
        assert chosen(agent) == "SOFA_B2" and log.rule_id == "BDI_ADOPTED"
        assert agent.desires.get(SELECT).status == DesireStatus.ACTIVE
        assert "desire_unblocked" in [t.kind for t in agent.transitions]


# ══ Unrelated updates ═══════════════════════════════════════════════════════

class SpyOptimizer(UtilityOptimizer):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def optimize(self, candidates, requirements):
        self.calls += 1
        return super().optimize(candidates, requirements)


class TestIrrelevantUpdates:
    def agent(self):
        self.spy = SpyOptimizer()
        agent = make_agent(planner=make_planner(optimizer=self.spy))
        self.calls_after_setup = self.spy.calls
        return agent

    @pytest.mark.parametrize("ev", [
        out_of_stock("SOFA_A7"),                                                       # not the chosen one
        out_of_stock("SOFA_B1"),
        event(ET.PRODUCT_AVAILABILITY, product_name="Washing machine", availability="out_of_stock"),  # not a candidate
        event(ET.PRICE_CHANGE, product_id="SOFA_A7", new_price=29000),                 # rise on another product
        event(ET.PRICE_CHANGE, product_name="Refrigerator", new_price=34500),
        event(ET.FURNITURE_COMPATIBILITY, product_name="x", room_width=1),             # event type BDI does not track
        event(ET.MISSING_INFORMATION, missing_field="room_dimensions"),
    ])
    def test_no_replanning(self, ev):
        agent = self.agent()
        intention = agent.current_intention(SELECT)
        before = len(agent.transitions)
        log = agent.handle_event(ev)
        assert log is None
        assert agent.current_intention(SELECT) is intention and intention.status == IntentionStatus.ACTIVE
        assert self.spy.calls == self.calls_after_setup, "the optimizer must not run"
        assert "update_ignored" in kinds_since(agent, before)
        assert not {"intention_replaced", "intention_retained", "intention_adopted"} & set(kinds_since(agent, before))

    def test_ignored_updates_explain_why(self):
        agent = self.agent()
        agent.handle_event(out_of_stock("SOFA_A7"))
        ignored = agent.last_transition
        assert ignored.kind == "update_ignored" and "cannot improve the current choice" in ignored.explanation

    def test_identical_update_changes_nothing_at_all(self):
        agent = self.agent()
        before = len(agent.transitions)
        assert agent.handle_event(event(ET.BUDGET_UPDATE, sofa_budget=30000)) is None
        assert len(agent.transitions) == before  # not even a belief_updated entry
        assert self.spy.calls == self.calls_after_setup

    def test_total_budget_rise_does_not_replan_the_sofa(self):
        agent = self.agent()
        intention = agent.current_intention(SELECT)
        before = len(agent.transitions)
        assert agent.handle_event(event(ET.BUDGET_UPDATE, total_budget=250000)) is None
        assert self.spy.calls == self.calls_after_setup
        assert agent.current_intention(SELECT) is intention
        new = kinds_since(agent, before)
        assert "intention_retained" not in new and "intention_replaced" not in new
        reconsidered = next(t for t in agent.transitions[before:] if t.kind == "desires_reconsidered")
        assert "not affected, left as is: des_select_sofa" in reconsidered.explanation
        assert reconsidered.details["order"] == [BUDGET]

    def test_expenditure_updates_only_touch_the_budget_goal(self):
        agent = self.agent()
        agent.handle_event(event(ET.BUDGET_UPDATE, estimated_expenditure=50000))
        assert self.spy.calls == self.calls_after_setup

    def test_other_product_becoming_available_is_worth_a_look(self):
        agent = self.agent()
        agent.handle_event(out_of_stock("SOFA_A7"))
        calls = self.spy.calls
        agent.handle_event(in_stock("SOFA_A7"))
        assert self.spy.calls == calls + 1  # re-evaluated, because it could beat the current choice

    def test_irrelevant_updates_still_update_beliefs(self):
        agent = self.agent()
        agent.handle_event(out_of_stock("SOFA_A7"))
        assert agent.beliefs.value("SOFA_A7", "availability") == "out_of_stock"


# ══ Relevant changes: replanning ════════════════════════════════════════════

class TestDynamicReplanning:
    def test_sofa_budget_cut_replaces_the_intention(self):
        agent = make_agent()
        log = agent.handle_event(event(ET.BUDGET_UPDATE, sofa_budget=27000))
        expected = reference_choice({"sofa_budget": 27000})
        assert chosen(agent) == expected != "SOFA_A2"
        assert log.rule_id == "BDI_REVISED"
        invalidated = next(t for t in agent.transitions if t.kind == "intention_invalidated")
        assert "user.sofa_budget" in invalidated.explanation

    def test_sofa_budget_cut_below_everything_blocks_the_goal(self):
        agent = make_agent()
        log = agent.handle_event(event(ET.BUDGET_UPDATE, sofa_budget=10000))
        assert chosen(agent) is None and log.rule_id == "BDI_BLOCKED"
        assert reference_choice({"sofa_budget": 10000}) is None

    def test_total_budget_below_sofa_budget_becomes_the_cap(self):
        agent = make_agent()
        agent.handle_event(event(ET.BUDGET_UPDATE, total_budget=27000))
        assert chosen(agent) == reference_choice({"sofa_budget": 27000})

    @pytest.mark.parametrize("width", [205, 200, 195, 190, 150])
    def test_room_width_change(self, width):
        agent = make_agent()
        agent.handle_event(event(ET.ROOM_UPDATE, "user", max_width_cm=width))
        assert chosen(agent) == reference_choice({"max_width_cm": width})

    @pytest.mark.parametrize("depth", [100, 95, 90, 80])
    def test_room_depth_change(self, depth):
        agent = make_agent()
        agent.handle_event(event(ET.ROOM_UPDATE, "user", max_depth_cm=depth))
        assert chosen(agent) == reference_choice({"max_depth_cm": depth})

    def test_room_dimension_aliases(self):
        agent = make_agent()
        agent.handle_event(event(ET.ROOM_UPDATE, room_width=190, room_depth=100))
        assert agent.beliefs.value("user", "max_width_cm") == 190
        assert agent.beliefs.value("user", "max_depth_cm") == 100

    @pytest.mark.parametrize("style", ["contemporary", "scandinavian", "traditional"])
    def test_style_preference_change(self, style):
        agent = make_agent()
        agent.handle_event(event(ET.PREFERENCE_UPDATE, "user", preferred_style=style))
        assert chosen(agent) == reference_choice({"preferred_style": style})

    def test_colour_preference_change(self):
        agent = make_agent()
        agent.handle_event(event(ET.PREFERENCE_UPDATE, "user", preferred_colour="grey"))
        assert chosen(agent) == reference_choice({"preferred_colour": "grey"})

    def test_seating_requirement_change_can_block_everything(self):
        agent = make_agent()
        agent.handle_event(event(ET.PREFERENCE_UPDATE, "user", min_seating=4))
        assert chosen(agent) is None and agent.desires.get(SELECT).status == DesireStatus.BLOCKED

    def test_preference_that_does_not_change_the_winner_keeps_the_intention(self):
        agent = make_agent()
        intention = agent.current_intention(SELECT)
        log = agent.handle_event(event(ET.PREFERENCE_UPDATE, "user", preferred_style="scandinavian"))
        assert chosen(agent) == reference_choice({"preferred_style": "scandinavian"}) == "SOFA_A2"
        assert agent.current_intention(SELECT) is intention
        assert log.rule_id == "BDI_RETAINED"
        retained = next(t for t in agent.transitions if t.kind == "intention_retained")
        assert "same product" in retained.explanation and "Utility changed" in retained.explanation

    def test_cheaper_competitor_takes_over(self):
        agent = make_agent()
        agent.handle_event(event(ET.PRICE_CHANGE, "price_monitor", product_id="SOFA_B3", new_price=15000))
        assert chosen(agent) == reference_choice(prices={"SOFA_B3": 15000}) == "SOFA_B3"

    def test_chosen_products_price_change_is_retained_with_new_figures(self):
        agent = make_agent()
        intention = agent.current_intention(SELECT)
        agent.handle_event(event(ET.PRICE_CHANGE, "price_monitor", product_id="SOFA_A2", new_price=25000))
        assert agent.current_intention(SELECT) is intention
        assert intention.params["price"] == 25000
        assert intention.params["utility"] > 67.55  # cheaper -> better

    def test_chosen_products_price_over_budget_forces_a_change(self):
        agent = make_agent()
        agent.handle_event(event(ET.PRICE_CHANGE, "price_monitor", product_id="SOFA_A2", new_price=31000))
        assert chosen(agent) == reference_choice(prices={"SOFA_A2": 31000}) != "SOFA_A2"

    def test_several_changes_in_one_event_cause_one_reconsideration(self):
        agent = make_agent()
        before = len(agent.transitions)
        agent.handle_event(event(ET.ROOM_UPDATE, max_width_cm=190, max_depth_cm=100))
        assert kinds_since(agent, before).count("desires_reconsidered") == 1

    def test_sequence_of_changes_always_matches_the_optimizer(self):
        agent = make_agent()
        state = dict(REQUIREMENTS)
        stock: dict[str, str] = {}
        steps = [
            (out_of_stock("SOFA_A2"), lambda: stock.update(SOFA_A2="out_of_stock")),
            (event(ET.PREFERENCE_UPDATE, preferred_style="contemporary"), lambda: state.update(preferred_style="contemporary")),
            (event(ET.ROOM_UPDATE, max_width_cm=200), lambda: state.update(max_width_cm=200)),
            (event(ET.BUDGET_UPDATE, sofa_budget=28000), lambda: state.update(sofa_budget=28000)),
            (in_stock("SOFA_A2"), lambda: stock.update(SOFA_A2="in_stock")),
            (event(ET.PREFERENCE_UPDATE, preferred_style="minimalist"), lambda: state.update(preferred_style="minimalist")),
            (event(ET.BUDGET_UPDATE, sofa_budget=5000), lambda: state.update(sofa_budget=5000)),
            (event(ET.BUDGET_UPDATE, sofa_budget=30000), lambda: state.update(sofa_budget=30000)),
        ]
        for ev, apply in steps:
            agent.handle_event(ev)
            apply()
            assert chosen(agent) == reference_choice(state, stock), ev.payload

    def test_waiting_for_information_then_adopting(self):
        agent = make_agent(requirements={"total_budget": 200000, "sofa_budget": 30000})
        assert chosen(agent) is None
        waiting = [t for t in agent.transitions if t.kind == "desire_waiting"]
        assert waiting and "user.min_seating" in waiting[0].explanation  # room limits are optional
        agent.set_requirements({"max_width_cm": 210, "max_depth_cm": 110, "min_seating": 3,
                                "preferred_style": "minimalist", "preferred_colour": "neutral"})
        assert chosen(agent) == "SOFA_A2"

    def test_waiting_is_logged_only_once_per_missing_set(self):
        agent = make_agent(requirements={"total_budget": 200000})
        agent.deliberate("again")
        agent.deliberate("and again")
        assert len([t for t in agent.transitions if t.kind == "desire_waiting"]) == 1

    def test_invalidating_a_requirement_suspends_the_intention_until_it_is_known_again(self):
        agent = make_agent()
        intention = agent.current_intention(SELECT)
        agent.invalidate_belief("user", "max_width_cm", "room being remeasured")
        # Unknown width: the agent cannot confirm the choice, but nothing better can be planned either.
        assert intention.status == IntentionStatus.SUSPENDED
        assert "user.max_width_cm is unknown" in intention.status_reason
        assert agent.current_intention(SELECT) is intention
        assert any(t.kind == "desire_waiting" and "user.max_width_cm" in t.explanation for t in agent.transitions)
        before = len(agent.transitions)
        agent.set_requirement("max_width_cm", 210)
        assert agent.current_intention(SELECT) is intention  # the same commitment is resumed, not rebuilt
        assert intention.status == IntentionStatus.ACTIVE
        assert "intention_retained" in kinds_since(agent, before)


# ══ Multiple desires ════════════════════════════════════════════════════════

class TestMultipleDesires:
    def test_budget_goal_is_violated_and_recovers(self):
        agent = make_agent()
        log = agent.handle_event(event(ET.BUDGET_UPDATE, estimated_expenditure=260000))
        assert log.rule_id == "BDI_BLOCKED"
        assert agent.desires.get(BUDGET).status == DesireStatus.BLOCKED
        assert chosen(agent) == "SOFA_A2"  # the sofa goal is left alone
        agent.handle_event(event(ET.BUDGET_UPDATE, estimated_expenditure=100000))
        assert agent.desires.get(BUDGET).status == DesireStatus.ACTIVE

    def test_total_budget_below_every_price_leaves_nothing_affordable(self):
        agent = make_agent()
        agent.handle_event(event(ET.BUDGET_UPDATE, total_budget=20000))
        assert reference_choice({"total_budget": 20000}) is None
        assert chosen(agent) is None and agent.desires.get(SELECT).status == DesireStatus.BLOCKED
        # nothing is committed any more, so the budget goal has nothing to object to
        assert agent.desires.get(BUDGET).status == DesireStatus.ACTIVE

    def test_goals_are_reconsidered_in_priority_order(self):
        agent = make_agent(start=False)
        agent.add_desire(Desire(desire_id="des_second", description="another goal", goal_type="select_product",
                                priority=95, params={"category": "sofa"}))
        agent.deliberate("order")
        details = next(t for t in agent.transitions if t.kind == "desires_reconsidered").details
        assert details["order"][:2] == ["des_second", SELECT]

    def test_changing_a_priority_changes_the_order(self):
        agent = make_agent(start=False)
        agent.add_desire(Desire(desire_id="des_other", description="other", goal_type="select_product", priority=10))
        agent.set_desire_priority("des_other", 99)
        details = [t for t in agent.transitions if t.kind == "desires_reconsidered"][-1].details
        assert details["order"][0] == "des_other"

    def test_conflicting_goals_the_higher_priority_wins(self):
        agent = make_agent(start=False)
        agent.desires.get(SELECT).conflicts_with = ["des_budget_sofa"]
        agent.add_desire(Desire(desire_id="des_budget_sofa", description="cheapest sofa", goal_type="select_product",
                                priority=70, params={"category": "sofa"}))
        agent.deliberate("conflict")
        assert agent.desires.get("des_budget_sofa").status == DesireStatus.SUSPENDED
        assert chosen(agent) == "SOFA_A2" and chosen(agent, "des_budget_sofa") is None
        agent.drop_desire(SELECT)
        agent.deliberate("after drop")
        assert agent.desires.get("des_budget_sofa").status == DesireStatus.ACTIVE
        assert chosen(agent, "des_budget_sofa") == "SOFA_A2"

    def test_dropping_a_desire_cancels_its_intention(self):
        agent = make_agent()
        intention = agent.current_intention(SELECT)
        assert agent.drop_desire(SELECT, "no longer needed") is True
        assert intention.status == IntentionStatus.CANCELLED and chosen(agent) is None
        assert agent.drop_desire("nope") is False

    def test_unsupported_goal_type_is_skipped_not_fatal(self):
        agent = make_agent(start=False)
        agent.add_desire(Desire(desire_id="des_weird", description="teleport", goal_type="teleport"))
        agent.deliberate("x")
        assert chosen(agent) == "SOFA_A2"
        assert any("teleport" in t.explanation for t in agent.transitions)


# ══ Completing an intention ═════════════════════════════════════════════════

class TestConfirmation:
    def test_confirm_reserves_and_achieves_the_desire(self):
        agent = make_agent()
        intention = agent.current_intention(SELECT)
        assert agent.confirm(SELECT) is True
        assert intention.status == IntentionStatus.COMPLETED
        assert agent.desires.get(SELECT).status == DesireStatus.ACHIEVED
        assert intention.actions[-1].result["performed"] == "reserve_product"
        assert "desire_achieved" in [t.kind for t in agent.transitions]

    def test_an_achieved_goal_is_not_replanned(self):
        agent = make_agent()
        agent.confirm(SELECT)
        before = len(agent.transitions)
        assert agent.handle_event(out_of_stock("SOFA_A2")) is None
        assert "intention_replaced" not in kinds_since(agent, before)

    def test_confirm_with_nothing_to_confirm(self):
        agent = make_agent(requirements=None)
        assert agent.confirm() is False

    def test_confirmation_after_a_change_reserves_the_new_choice_never_the_old_one(self):
        agent = make_agent()
        old = agent.current_intention(SELECT)
        agent.handle_event(out_of_stock("SOFA_A2"))
        new = agent.current_intention(SELECT)
        assert agent.confirm(SELECT) is True
        assert old.actions[-1].status.value == "pending"  # SOFA_A2 was never reserved
        assert new.status == IntentionStatus.COMPLETED
        assert new.actions[-1].result["product_id"] == new.params["product_id"] != "SOFA_A2"

    def test_cancel_intention(self):
        agent = make_agent()
        intention = agent.current_intention(SELECT)
        assert agent.cancel_intention(intention.intention_id, "changed my mind") is True
        assert intention.status == IntentionStatus.CANCELLED
        assert agent.cancel_intention(intention.intention_id) is False
        assert agent.cancel_intention("missing") is False


# ══ Input robustness ════════════════════════════════════════════════════════

class TestMalformedInput:
    @pytest.mark.parametrize("ev", [
        event(ET.PRODUCT_AVAILABILITY),
        event(ET.PRODUCT_AVAILABILITY, product_id="SOFA_A2"),
        event(ET.PRODUCT_AVAILABILITY, availability="out_of_stock"),
        event(ET.PRODUCT_AVAILABILITY, product_id="SOFA_A2", availability=5),
        event(ET.PRICE_CHANGE, product_id="SOFA_A2", new_price=-1),
        event(ET.PRICE_CHANGE, product_id="SOFA_A2", new_price=math.nan),
        event(ET.PRICE_CHANGE, product_id="SOFA_A2", new_price="cheap"),
        event(ET.BUDGET_UPDATE, sofa_budget="lots"),
        event(ET.BUDGET_UPDATE, total_budget=0),
        event(ET.BUDGET_UPDATE),
        event(ET.ROOM_UPDATE),
        event(ET.ROOM_UPDATE, max_width_cm=-3),
        event(ET.PREFERENCE_UPDATE, preferred_style=""),
        event(ET.PREFERENCE_UPDATE, min_seating=2.5),
        event(ET.PREFERENCE_UPDATE),
    ])
    def test_rejected_without_changing_anything(self, ev):
        agent = make_agent()
        intention = agent.current_intention(SELECT)
        beliefs_before = agent.beliefs.as_dict()
        before = len(agent.transitions)
        assert agent.handle_event(ev) is None
        assert agent.beliefs.as_dict() == beliefs_before
        assert agent.current_intention(SELECT) is intention
        assert "update_rejected" in kinds_since(agent, before)

    def test_none_event(self):
        assert make_agent().handle_event(None) is None

    def test_setters_reject_invalid_values(self):
        agent = make_agent()
        with pytest.raises(ValueError):
            agent.set_requirement("sofa_budget", -1)
        with pytest.raises(ValueError):
            agent.set_requirements({"max_width_cm": 200, "min_seating": 0})
        assert agent.beliefs.value("user", "max_width_cm") == 210  # nothing partially applied
        with pytest.raises(ValueError):
            agent.set_belief("user", "sofa_budget", "x")

    def test_generic_beliefs_are_accepted(self):
        agent = make_agent()
        assert agent.set_belief("weather", "outlook", "rain") is None
        assert agent.beliefs.value("weather", "outlook") == "rain"

    def test_availability_values_are_normalised(self):
        agent = make_agent()
        agent.handle_event(event(ET.PRODUCT_AVAILABILITY, product_id="SOFA_A7", availability="Out of Stock"))
        assert agent.beliefs.value("SOFA_A7", "availability") == "out_of_stock"
        agent.handle_event(event(ET.PRODUCT_AVAILABILITY, product_id="SOFA_A7", availability="available"))
        assert agent.beliefs.value("SOFA_A7", "availability") == "in_stock"

    def test_a_failing_planner_does_not_crash_the_agent(self):
        class Boom:
            def get_candidates(self):
                raise RuntimeError("catalogue offline")

        agent = make_agent(planner=make_planner(source=Boom()))
        assert chosen(agent) is None
        assert any("catalogue offline" in t.explanation for t in agent.transitions)


# ══ Integration with the existing framework ═════════════════════════════════

class TestFrameworkIntegration:
    def test_is_a_base_agent(self):
        agent = make_agent()
        assert isinstance(agent, BaseAgent) and agent.agent_id == "bdi_test" and agent.name == "bdi_test"

    def test_handle_event_returns_action_logs_recorded_by_the_base_agent(self):
        agent = make_agent()
        log = agent.handle_event(out_of_stock("SOFA_A2"))
        assert isinstance(log, ActionLog) and log in agent.action_log
        assert log.event_id.startswith("evt_") and log.result_details["before"]["product_id"] == "SOFA_A2"
        assert log.result_details["after"]["product_id"] == chosen(agent)

    def test_initial_adoption_is_logged_as_an_action(self):
        agent = make_agent(start=False)
        log = agent.deliberate("start")
        assert log.rule_id == "BDI_ADOPTED" and log.event_id == "manual"

    def test_base_agent_state_mirrors_the_bdi_state(self):
        agent = make_agent()
        assert agent.state["beliefs"]["user.sofa_budget"] == 30000
        assert agent.state["intentions"][0]["product_id"] == "SOFA_A2"
        assert {d["desire_id"] for d in agent.state["desires"]} == {SELECT, BUDGET}
        agent.handle_event(out_of_stock("SOFA_A2"))
        assert agent.state["intentions"][0]["product_id"] == chosen(agent)

    def test_consumes_the_existing_event_stream(self):
        agent = make_agent()
        stream = EventStream()
        stream.emit(out_of_stock("SOFA_A7"))     # irrelevant
        stream.emit(out_of_stock("SOFA_A2"))     # relevant
        stream.emit(event(ET.FURNITURE_COMPATIBILITY, product_name="x"))
        logs = agent.process_events(stream)
        assert [log.rule_id for log in logs] == ["BDI_REVISED"]
        assert stream.size == 0 and chosen(agent) != "SOFA_A2"

    def test_processes_events_emitted_by_the_existing_worker_format(self):
        """ProductMonitorWorker emits product_name/availability payloads."""
        agent = make_agent(requirements=REQUIREMENTS)
        worker_event = EnvironmentalEvent(
            event_type=ET.PRODUCT_AVAILABILITY, source="product_monitor",
            payload={"product_name": "SOFA_A2", "availability": "out_of_stock", "previous_availability": "in_stock"})
        assert agent.handle_event(worker_event).rule_id == "BDI_REVISED"

    def test_reset_clears_the_bdi_state(self):
        agent = make_agent()
        agent.reset()
        assert len(agent.beliefs) == len(agent.desires) == len(agent.intentions) == 0
        assert agent.transitions == [] and agent.state == {} and agent.action_log == []

    def test_history_is_capped(self):
        agent = make_agent(max_history=10)
        for i in range(30):
            agent.set_belief("noise", "n", i)
        assert len(agent.transitions) == 10
        with pytest.raises(ValueError):
            BDIAgent("x", make_planner(), max_history=0)

    def test_every_decision_transition_carries_an_explanation(self):
        agent = make_agent()
        agent.handle_event(out_of_stock("SOFA_A2"))
        agent.handle_event(event(ET.BUDGET_UPDATE, sofa_budget=10000))
        for t in agent.transitions:
            if t.kind in ("intention_adopted", "intention_replaced", "intention_invalidated",
                          "desire_blocked", "update_ignored"):
                assert t.explanation, t.kind
        assert all(t.timestamp.tzinfo is not None for t in agent.transitions)
        assert t.format().startswith("[")
