"""When the budget is tight: fill everything if it is possible at all, otherwise say exactly why and what would fix it."""

from __future__ import annotations

import pytest

from smartnest.furnishing import (
    FurnishingCoordinator,
    FurnishingRequest,
    LayeredCatalog,
    SimulatedCatalog,
    SnapshotCatalog,
)
from smartnest.furnishing.assistant import ChatAssistant
from smartnest.optimization import UtilityOptimizer

KIDS = ("kids_bed", "mattress", "wardrobe", "study_table", "study_chair", "bookshelf", "wall_decor", "table_lamp",
        "rug", "bean_bag", "toy_storage")
HAVE_REAL = all(SnapshotCatalog().listings(c).records for c in KIDS)


def sim():
    return LayeredCatalog(SnapshotCatalog("/nonexistent"), SimulatedCatalog(), allow_simulated=True)


def kids_request(budget, **kw):
    return FurnishingRequest(budget, None, scope=("kids_bedroom",), audience="child", age=10,
                             themes=("monster", "cartoon"), **kw)


def minimum_total(c: FurnishingCoordinator) -> float:
    """Independent recomputation: the sum over items of the cheapest buyable listing."""
    total = 0.0
    for slot in c._slots:
        r = UtilityOptimizer().optimize(c._research[slot.profile.id]["records"],
                                        {"budget": 1e9, "min_seating": slot.min_seating if slot.profile.needs_seating else 1})
        prices = [e.price for e in (*r.ranking, *r.unverified)]
        total += min(prices) if prices else 0.0
    return round(total, 2)


class TestFeasibility:
    @pytest.mark.parametrize("budget", range(12000, 90001, 6000))
    def test_every_item_is_filled_whenever_the_cheapest_choices_fit_the_budget(self, budget):
        c = FurnishingCoordinator(sim())
        plan = c.create_plan(kids_request(budget))
        floor = minimum_total(c)
        assert plan.summary["spent"] <= budget + 1e-6
        if floor <= budget:
            assert plan.summary["unfilled_count"] == 0, f"floor {floor} fits {budget} but an item is unfilled"
            assert plan.summary["shortfall"] == 0 and plan.summary["budget_needed"] is None
        else:
            assert plan.summary["unfilled_count"] >= 1
            assert plan.summary["shortfall"] == pytest.approx(floor - budget, abs=0.01)
            assert plan.summary["budget_needed"] >= floor

    @pytest.mark.parametrize("budget", range(40000, 200001, 20000))
    def test_the_whole_home_is_filled_whenever_it_fits(self, budget):
        c = FurnishingCoordinator(sim())
        plan = c.create_plan(FurnishingRequest(budget, 2))
        if minimum_total(c) <= budget:
            assert plan.summary["unfilled_count"] == 0
        assert plan.summary["spent"] <= budget + 1e-6

    def test_the_hard_ceilings_still_hold_after_rebalancing(self):
        c = FurnishingCoordinator(sim())
        floor_budget = None
        c.create_plan(kids_request(100000))
        floor_budget = minimum_total(c) + 150            # just enough: forces the rebalanced split
        plan = FurnishingCoordinator(sim()).create_plan(kids_request(floor_budget))
        assert plan.summary["unfilled_count"] == 0 and plan.summary["spent"] <= floor_budget
        for ln in plan.lines.values():
            assert ln["product"]["price"] <= ln["ceiling"] + 1e-6
        assert sum(ln["ceiling"] for ln in plan.lines.values()) <= floor_budget + 1e-6

    def test_exactly_the_minimum_budget_fills_every_item_at_its_cheapest(self):
        c = FurnishingCoordinator(sim())
        c.create_plan(kids_request(100000))
        floor = minimum_total(c)
        plan = FurnishingCoordinator(sim()).create_plan(kids_request(floor))
        assert plan.summary["unfilled_count"] == 0 and plan.summary["spent"] == pytest.approx(floor)

    def test_one_rupee_below_the_minimum_leaves_something_out_and_says_so(self):
        c = FurnishingCoordinator(sim())
        c.create_plan(kids_request(100000))
        floor = minimum_total(c)
        plan = FurnishingCoordinator(sim()).create_plan(kids_request(floor - 1))
        assert plan.summary["unfilled_count"] >= 1 and plan.summary["shortfall"] == pytest.approx(1, abs=0.01)
        assert plan.summary["spent"] <= floor - 1

    def test_raising_the_budget_as_suggested_fills_the_plan(self):
        c = FurnishingCoordinator(sim())
        c.create_plan(kids_request(100000))
        small = FurnishingCoordinator(sim())
        plan = small.create_plan(kids_request(minimum_total(c) - 3000))
        assert plan.summary["budget_needed"]
        plan = small.handle_event({"type": "budget_change", "total_budget": plan.summary["budget_needed"]})
        assert plan.summary["unfilled_count"] == 0 and plan.summary["within_budget"]

    def test_an_out_of_stock_event_raises_the_minimum(self):
        c = FurnishingCoordinator(sim())
        plan = c.create_plan(kids_request(100000))
        before = minimum_total(c)
        cheapest_wardrobe = min(c._research["wardrobe"]["records"], key=lambda r: r["price"])
        c.handle_event({"type": "out_of_stock", "product_id": cheapest_wardrobe["product_id"]})
        c._research["wardrobe"]["records"] = [dict(r, **(c._overrides.get(r["product_id"]) or {}))
                                              for r in c._research["wardrobe"]["records"]]
        assert minimum_total(c) >= before


class TestExplanations:
    def tight(self):
        c = FurnishingCoordinator(sim())
        c.create_plan(kids_request(100000))
        return FurnishingCoordinator(sim()).create_plan(kids_request(minimum_total(c) - 2000))

    def test_the_unfilled_card_states_the_cheapest_price_and_what_is_left(self):
        plan = self.tight()
        line = next(ln for ln in plan.lines.values() if ln["status"] == "unfilled")
        assert "cheapest" in line["why"] and "is left" in line["why"] and line["cheapest_price"] > 0
        assert "nothing was relaxed" in line["why"].lower()

    def test_the_chat_summary_gives_the_numbers_and_the_fix(self):
        plan = self.tight()
        text = ChatAssistant.describe_plan(plan)
        assert "too small for everything" in text and "Raise the budget to about" in text
        assert "₹" in text and "drop an item" in text

    def test_a_plan_that_fits_has_no_shortfall_message(self):
        c = FurnishingCoordinator(sim())
        plan = c.create_plan(kids_request(100000))
        assert "too small" not in ChatAssistant.describe_plan(plan)


@pytest.mark.skipif(not HAVE_REAL, reason="no real kid's-room snapshots")
class TestRealKidsRoomAtTightBudgets:
    def plan(self, budget):
        return FurnishingCoordinator(LayeredCatalog(SnapshotCatalog())).create_plan(kids_request(budget))

    def test_25000_cannot_fit_a_wardrobe_and_says_exactly_why(self):
        plan = self.plan(25000)
        un = [ln for ln in plan.lines.values() if ln["status"] == "unfilled"]
        assert [ln["category"] for ln in un] == ["wardrobe"]
        assert plan.summary["minimum_total"] == 29581 and plan.summary["shortfall"] == 4581
        assert plan.summary["budget_needed"] == 30000
        assert "12,319" in un[0]["why"] and "7,249" in un[0]["why"]

    def test_30000_fills_all_eleven_items(self):
        plan = self.plan(30000)
        assert plan.summary["unfilled_count"] == 0 and plan.summary["spent"] <= 30000 and len(plan.lines) == 11

    def test_the_suggested_budget_really_works(self):
        c = FurnishingCoordinator(LayeredCatalog(SnapshotCatalog()))
        plan = c.create_plan(kids_request(25000))
        plan = c.handle_event({"type": "budget_change", "total_budget": plan.summary["budget_needed"]})
        assert plan.summary["unfilled_count"] == 0
