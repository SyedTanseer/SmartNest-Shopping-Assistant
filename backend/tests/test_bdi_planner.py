"""Tests for the BDI planner: planning with the UtilityOptimizer, preconditions, relevance."""

from __future__ import annotations

import math

import pytest

from smartnest.bdi import (
    BDIPlanner,
    BeliefBase,
    BeliefChange,
    CatalogCandidateSource,
    IntentionManager,
    MaintainBudgetHandler,
    SelectProductHandler,
    effective_budget,
    normalise_requirement,
    select_sofa_desire,
    stay_within_budget_desire,
)
from smartnest.bdi.desires import Desire
from smartnest.cnp.product_catalog import ProductCatalog, catalog_agent_a, catalog_agent_b
from smartnest.optimization import UtilityOptimizer
from tests.bdi_support import REQUIREMENTS, make_planner, reference_choice

DESIRE = select_sofa_desire()


def beliefs_with(**overrides) -> BeliefBase:
    beliefs = BeliefBase()
    for predicate, value in {**REQUIREMENTS, **overrides}.items():
        if value is not None:
            beliefs.update("user", predicate, value, "user")
    return beliefs


def change(subject, predicate, old, new, kind="updated") -> BeliefChange:
    return BeliefChange(subject=subject, predicate=predicate, kind=kind, old_value=old, new_value=new)


def handler() -> SelectProductHandler:
    return make_planner()._handlers["select_product"]


# ── value helpers ───────────────────────────────────────────────────────────

class TestRequirementValues:
    @pytest.mark.parametrize("predicate,value,expected", [
        ("sofa_budget", 30000, 30000), ("max_width_cm", 210.5, 210.5), ("min_seating", 3, 3),
        ("min_seating", 3.0, 3), ("preferred_style", "  Minimalist ", "minimalist"),
        ("preferred_colour", "NEUTRAL", "neutral"), ("estimated_expenditure", 0, 0),
        ("something_else", {"any": "thing"}, {"any": "thing"}),
    ])
    def test_valid_values(self, predicate, value, expected):
        assert normalise_requirement(predicate, value) == (expected, None)

    @pytest.mark.parametrize("predicate,value", [
        ("sofa_budget", 0), ("sofa_budget", -5), ("sofa_budget", math.nan), ("sofa_budget", math.inf),
        ("sofa_budget", "30000"), ("sofa_budget", True), ("max_width_cm", -1), ("min_seating", 0),
        ("min_seating", 2.5), ("min_seating", True), ("preferred_style", ""), ("preferred_style", 5),
        ("estimated_expenditure", -1), ("total_budget", None),
    ])
    def test_invalid_values_explain_why(self, predicate, value):
        normalised, error = normalise_requirement(predicate, value)
        assert normalised is None and predicate in error

    def test_effective_budget_is_the_smaller_known_budget(self):
        assert effective_budget(30000, 200000) == 30000
        assert effective_budget(300000, 200000) == 200000
        assert effective_budget(None, 200000) == 200000
        assert effective_budget(30000, None) == 30000
        assert effective_budget(None, None) is None
        assert effective_budget(-5, "x") is None


# ── requirements from beliefs ───────────────────────────────────────────────

class TestRequirements:
    def test_built_from_beliefs(self):
        requirements, missing = handler().requirements(beliefs_with())
        assert missing == []
        assert (requirements.budget, requirements.max_width_cm, requirements.max_depth_cm) == (30000, 210, 110)
        assert requirements.preferred_style == "minimalist" and requirements.min_seating == 3

    def test_total_budget_alone_is_enough(self):
        requirements, _ = handler().requirements(beliefs_with(sofa_budget=None))
        assert requirements.budget == 200000

    def test_sofa_budget_is_capped_by_total_budget(self):
        requirements, _ = handler().requirements(beliefs_with(sofa_budget=250000, total_budget=200000))
        assert requirements.budget == 200000

    def test_missing_information_is_listed(self):
        requirements, missing = handler().requirements(BeliefBase())
        assert requirements is None
        assert "user.sofa_budget or user.total_budget" in missing
        assert "user.min_seating" in missing
        assert not {"user.max_width_cm", "user.max_depth_cm"} & set(missing)  # room limits are optional

    def test_invalidated_belief_counts_as_missing(self):
        beliefs = beliefs_with()
        beliefs.invalidate("user", "max_width_cm")
        _, missing = handler().requirements(beliefs)
        assert missing == ["user.max_width_cm"]

    def test_style_and_colour_are_optional(self):
        requirements, missing = handler().requirements(beliefs_with(preferred_style=None, preferred_colour=None))
        assert missing == [] and requirements.preferred_style is None

    def test_garbage_belief_is_reported_not_raised(self):
        beliefs = beliefs_with()
        beliefs.update("user", "preferred_style", 42)  # bypassed normalisation
        requirements, missing = handler().requirements(beliefs)
        assert requirements is None and "valid requirements" in missing[0]


# ── planning ────────────────────────────────────────────────────────────────

class TestPlanning:
    def test_selects_the_optimizers_highest_utility_feasible_product(self):
        outcome = make_planner().plan(DESIRE, beliefs_with(), [])
        assert outcome.status == "planned"
        assert outcome.intention.params["product_id"] == reference_choice()
        result = outcome.optimization
        assert result.verified_optimal and result.total_inspected == 14
        assert outcome.intention.params["utility"] == result.selected.utility

    def test_intention_preconditions_cover_every_hard_constraint(self):
        beliefs = beliefs_with()
        outcome = make_planner().plan(DESIRE, beliefs, [])
        intention = outcome.intention
        pid = intention.params["product_id"]
        actual = {(c.subject, c.predicate, c.operator, c.ref_predicate) for c in intention.preconditions}
        assert actual == {
            (pid, "availability", "in", None),
            (pid, "price", "lte", "sofa_budget"),
            (pid, "price", "lte", "total_budget"),
            (pid, "width_cm", "lte", "max_width_cm"),
            (pid, "depth_cm", "lte", "max_depth_cm"),
            (pid, "seating_capacity", "gte", "min_seating"),
        }
        assert IntentionManager.validate(intention, beliefs).valid

    def test_plan_has_ordered_actions_with_a_manual_final_step(self):
        actions = make_planner().plan(DESIRE, beliefs_with(), []).intention.actions
        assert [a.kind for a in actions] == ["check_constraints", "recommend_product", "reserve_product"]
        assert [a.manual for a in actions] == [False, False, True]
        assert all(a.preconditions for a in actions)

    def test_chosen_products_attributes_become_catalogue_beliefs(self):
        beliefs = beliefs_with()
        pid = make_planner().plan(DESIRE, beliefs, []).intention.params["product_id"]
        for predicate in ("availability", "price", "width_cm", "depth_cm", "seating_capacity"):
            belief = beliefs.get(pid, predicate)
            assert belief is not None and belief.source == "catalogue"

    def test_events_override_the_catalogue(self):
        beliefs = beliefs_with()
        beliefs.update("SOFA_A2", "availability", "out_of_stock", "product_monitor")
        outcome = make_planner().plan(DESIRE, beliefs, [])
        assert outcome.intention.params["product_id"] == reference_choice(availability={"SOFA_A2": "out_of_stock"})
        assert outcome.intention.params["product_id"] != "SOFA_A2"

    def test_catalogue_sourced_beliefs_never_override_the_catalogue(self):
        """Background knowledge must not freeze stale values in place."""
        beliefs = beliefs_with()
        beliefs.update("SOFA_A2", "availability", "out_of_stock", "catalogue")
        assert make_planner().plan(DESIRE, beliefs, []).intention.params["product_id"] == "SOFA_A2"

    def test_price_events_override_catalogue_price(self):
        beliefs = beliefs_with()
        beliefs.update("SOFA_B3", "price", 15000, "price_monitor")
        outcome = make_planner().plan(DESIRE, beliefs, [])
        assert outcome.intention.params["product_id"] == reference_choice(prices={"SOFA_B3": 15000})

    def test_blocked_when_nothing_is_feasible(self):
        outcome = make_planner().plan(DESIRE, beliefs_with(sofa_budget=1000), [])
        assert outcome.status == "blocked" and outcome.intention is None
        assert "none of the 14 candidates is feasible" in outcome.explanation
        assert "exceeds budget" in outcome.explanation

    def test_waiting_when_information_is_missing(self):
        outcome = make_planner().plan(DESIRE, BeliefBase(), [])
        assert outcome.status == "waiting" and outcome.intention is None
        assert "user.min_seating" in outcome.missing and "user.max_width_cm" not in outcome.missing

    @pytest.mark.parametrize("overrides", [
        {"sofa_budget": 28000}, {"max_width_cm": 200}, {"max_width_cm": 195}, {"max_depth_cm": 100},
        {"preferred_style": "contemporary"}, {"preferred_colour": "grey"}, {"min_seating": 3},
        {"sofa_budget": 27000, "max_width_cm": 192},
    ])
    def test_always_agrees_with_running_the_optimizer_directly(self, overrides):
        outcome = make_planner().plan(DESIRE, beliefs_with(**overrides), [])
        expected = reference_choice(overrides)
        assert (outcome.intention.params["product_id"] if outcome.intention else None) == expected

    def test_works_with_any_candidate_source(self):
        """4.2: only the CandidateSource schema matters, not where products come from."""
        class OneProduct:
            def get_candidates(self):
                return [{"product_id": "ONLY", "name": "Only", "price": 10000, "width_cm": 100, "depth_cm": 80,
                         "seating_capacity": 3, "material": "wood", "color": "neutral", "style": "minimalist",
                         "availability": "in_stock", "rating": 4.0, "review_count": 20,
                         "source_agent_id": "api"}]

        outcome = make_planner(source=OneProduct()).plan(DESIRE, beliefs_with(), [])
        assert outcome.intention.params["product_id"] == "ONLY"

    def test_malformed_candidates_are_rejected_not_chosen(self):
        class Dirty:
            def get_candidates(self):
                return [{"product_id": "BAD", "price": -5}]

        assert make_planner(source=Dirty()).plan(DESIRE, beliefs_with(), []).status == "blocked"


class TestPreconditionsDetectInvalidChoices:
    """The intention's preconditions let the agent notice trouble from beliefs alone."""

    def planned(self):
        beliefs = beliefs_with()
        return beliefs, make_planner().plan(DESIRE, beliefs, []).intention

    def test_valid_right_after_planning(self):
        beliefs, intention = self.planned()
        assert IntentionManager.validate(intention, beliefs).valid

    @pytest.mark.parametrize("subject_is_product,predicate,value,fragment", [
        (True, "availability", "out_of_stock", "availability"),
        (True, "price", 999999, "price"),
        (False, "sofa_budget", 100, "sofa_budget"),
        (False, "total_budget", 100, "total_budget"),
        (False, "max_width_cm", 50, "max_width_cm"),
        (False, "max_depth_cm", 50, "max_depth_cm"),
        (False, "min_seating", 9, "min_seating"),
    ])
    def test_each_hard_constraint_is_monitored(self, subject_is_product, predicate, value, fragment):
        beliefs, intention = self.planned()
        subject = intention.params["product_id"] if subject_is_product else "user"
        beliefs.update(subject, predicate, value, "event")
        result = IntentionManager.validate(intention, beliefs)
        assert not result.valid and fragment in result.explain()

    def test_style_change_does_not_invalidate(self):
        beliefs, intention = self.planned()
        beliefs.update("user", "preferred_style", "industrial", "user")
        assert IntentionManager.validate(intention, beliefs).valid


# ── relevance ───────────────────────────────────────────────────────────────

class TestRelevance:
    def ask(self, chg, beliefs=None, held=frozenset(), planner=None):
        planner = planner or make_planner()
        return planner.is_affected_by(DESIRE, chg, beliefs or beliefs_with(), set(held))

    def test_requirement_changes_matter(self):
        for predicate, old, new in (("max_width_cm", 210, 200), ("max_depth_cm", 110, 100),
                                    ("min_seating", 3, 4), ("preferred_style", "a", "b"),
                                    ("preferred_colour", "a", "b")):
            assert self.ask(change("user", predicate, old, new))[0], predicate

    def test_chosen_product_changes_matter(self):
        assert self.ask(change("SOFA_A2", "availability", "in_stock", "out_of_stock"), held={"SOFA_A2"})[0]
        assert self.ask(change("SOFA_A2", "price", 27499, 28000), held={"SOFA_A2"})[0]

    def test_other_product_becoming_unavailable_cannot_improve_the_choice(self):
        affected, reason = self.ask(change("SOFA_A7", "availability", "in_stock", "out_of_stock"), held={"SOFA_A2"})
        assert not affected and "cannot improve" in reason

    def test_other_product_becoming_available_might_beat_it(self):
        assert self.ask(change("SOFA_A7", "availability", "out_of_stock", "in_stock"), held={"SOFA_A2"})[0]

    def test_other_product_price_drop_matters_but_a_rise_does_not(self):
        assert self.ask(change("SOFA_A7", "price", 26499, 20000), held={"SOFA_A2"})[0]
        assert not self.ask(change("SOFA_A7", "price", 26499, 30000), held={"SOFA_A2"})[0]

    def test_with_no_earlier_belief_the_catalogue_is_the_baseline(self):
        """SOFA_A7 is listed at 26,499 and in stock; a first-ever event must be compared to that."""
        held = {"SOFA_A2"}
        assert not self.ask(change("SOFA_A7", "price", None, 29000, "added"), held=held)[0]    # a rise
        assert self.ask(change("SOFA_A7", "price", None, 20000, "added"), held=held)[0]        # a real drop
        assert not self.ask(change("SOFA_A7", "availability", None, "in_stock", "added"), held=held)[0]  # no news
        assert not self.ask(change("SOFA_A7", "availability", None, "out_of_stock", "added"), held=held)[0]
        assert self.ask(change("SOFA_B6", "availability", None, "in_stock", "added"), held=held)[0]  # B6 is listed out of stock

    def test_unknown_products_and_predicates_are_ignored(self):
        assert not self.ask(change("Washing machine", "availability", "in_stock", "out_of_stock"))[0]
        assert not self.ask(change("user", "favourite_film", None, "Alien", "added"))[0]
        assert not self.ask(change("SOFA_A2", "reviews_text", None, "nice", "added"), held={"SOFA_A2"})[0]

    def test_total_budget_change_is_ignored_while_sofa_budget_caps_the_purchase(self):
        beliefs = beliefs_with(total_budget=250000)
        affected, reason = self.ask(change("user", "total_budget", 200000, 250000), beliefs)
        assert not affected and "effective sofa budget is still 30000" in reason

    def test_total_budget_change_matters_when_it_becomes_the_cap(self):
        beliefs = beliefs_with(total_budget=20000)  # now below sofa_budget
        assert self.ask(change("user", "total_budget", 200000, 20000), beliefs)[0]

    def test_sofa_budget_change_matters_only_if_it_moves_the_cap(self):
        beliefs = beliefs_with(sofa_budget=250000, total_budget=200000)  # capped at 200000 either way
        assert not self.ask(change("user", "sofa_budget", 300000, 250000), beliefs)[0]
        assert self.ask(change("user", "sofa_budget", 30000, 25000), beliefs_with(sofa_budget=25000))[0]

    def test_budget_goal_watches_total_budget_and_expenditure(self):
        planner = make_planner()
        budget = stay_within_budget_desire()
        for predicate in ("total_budget", "estimated_expenditure"):
            assert planner.is_affected_by(budget, change("user", predicate, 1, 2), beliefs_with(), set())[0]
        assert not planner.is_affected_by(budget, change("user", "max_width_cm", 1, 2), beliefs_with(), set())[0]


# ── maintenance goal ────────────────────────────────────────────────────────

class TestMaintainBudget:
    def plan(self, beliefs, price=None):
        from smartnest.bdi import Intention
        live = [Intention(desire_id="d", description="x", params={"price": price})] if price is not None else []
        return MaintainBudgetHandler().plan(stay_within_budget_desire(), beliefs, live)

    def test_waiting_without_a_total_budget(self):
        assert self.plan(BeliefBase()).status == "waiting"

    def test_satisfied(self):
        beliefs = beliefs_with()
        beliefs.update("user", "estimated_expenditure", 50000)
        outcome = self.plan(beliefs, price=27499)
        assert outcome.status == "satisfied" and "27,499" in outcome.explanation

    def test_violated_by_the_committed_price(self):
        assert self.plan(beliefs_with(total_budget=20000), price=27499).status == "violated"

    def test_violated_by_recorded_expenditure(self):
        beliefs = beliefs_with()
        beliefs.update("user", "estimated_expenditure", 250000)
        outcome = self.plan(beliefs)
        assert outcome.status == "violated" and "recorded expenditure" in outcome.explanation


# ── planner dispatch ────────────────────────────────────────────────────────

class TestPlannerDispatch:
    def test_supports_known_goal_types(self):
        planner = make_planner()
        assert planner.supports("select_product") and planner.supports("maintain_budget")
        assert not planner.supports("teleport")

    def test_unknown_goal_type_raises(self):
        with pytest.raises(ValueError, match="teleport"):
            make_planner().plan(Desire(description="x", goal_type="teleport"), BeliefBase(), [])


class TestCatalogCandidateSource:
    def test_combines_catalogues_and_sets_source(self):
        records = CatalogCandidateSource([catalog_agent_a(True), catalog_agent_b(True)]).get_candidates()
        assert len(records) == 14
        assert {r["source_agent_id"] for r in records} == {"research_agent_a", "research_agent_b"}

    def test_fills_missing_source_from_catalogue_agent_id(self):
        catalog = ProductCatalog("shop_x", [{"product_id": "P", "price": 1}])
        assert CatalogCandidateSource([catalog]).get_candidates()[0]["source_agent_id"] == "shop_x"

    def test_records_are_copies(self):
        catalog = catalog_agent_a(True)
        CatalogCandidateSource([catalog]).get_candidates()[0]["price"] = 1
        assert catalog.get_product("SOFA_A1")["price"] != 1
