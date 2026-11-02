"""Furnishing layer: categories, rooms, budget, catalogs, signed agent bus, coordinator and replanning."""

from __future__ import annotations

import json

import pytest

from smartnest.core.models import MessageEnvelope, MessageType
from smartnest.furnishing import (
    CATEGORIES,
    ActivityLog,
    AgentBus,
    AgentCallError,
    FurnishingCoordinator,
    FurnishingRequest,
    LayeredCatalog,
    PlanError,
    SimulatedCatalog,
    SnapshotCatalog,
    parse_home_type,
    rooms_for,
    slots_for,
)
from smartnest.furnishing import budget as budget_mod
from smartnest.furnishing.catalog import ORIGIN_REAL, ORIGIN_SIMULATED, CategoryListings
from smartnest.security import sign_envelope


def coordinator(**kw) -> FurnishingCoordinator:
    return FurnishingCoordinator(LayeredCatalog(SnapshotCatalog("/nonexistent-dir"), SimulatedCatalog(),
                                                allow_simulated=True), **kw)


def planned(budget=200000, bedrooms=2, **kw):
    c = coordinator()
    return c, c.create_plan(FurnishingRequest(budget, bedrooms, **kw))


# ── Categories ───────────────────────────────────────────────────────────────

class TestCategories:
    def test_weights_are_sane(self):
        assert all(0 < c.default_weight < 1 for c in CATEGORIES.values())

    @pytest.mark.parametrize("cat,name,price", [
        ("refrigerator", "Fridge magnet set", 499), ("refrigerator", "Door Gasket for fridge", 9000),
        ("sofa", "Sofa cover 3 seater", 2500), ("mattress", "Mattress protector queen", 1200),
        ("bed", "Bed", 800)])
    def test_accessories_are_flagged(self, cat, name, price):
        assert CATEGORIES[cat].accessory_reason(name, price)

    def test_real_product_is_not_flagged(self):
        assert CATEGORIES["refrigerator"].accessory_reason("Brand 250 L Double Door Refrigerator", 24990) is None

    def test_implausibly_expensive_is_flagged(self):
        assert "above" in CATEGORIES["mattress"].accessory_reason("Mattress", 900000)


# ── Rooms ────────────────────────────────────────────────────────────────────

class TestRooms:
    @pytest.mark.parametrize("text,n", [("I moved into a 2BHK flat", 2), ("3 bhk", 3), ("a 1-BHK", 1),
                                        ("two bedroom", None), ("2 bedroom apartment", 2), ("studio", None),
                                        ("9BHK", None)])
    def test_parse_home_type(self, text, n):
        assert parse_home_type(text) == n

    def test_two_bhk_rooms(self):
        rooms = rooms_for(2)
        assert [r.label for r in rooms] == ["Living room", "Bedroom 1", "Bedroom 2", "Kitchen"]
        assert len(slots_for(rooms)) == 10

    @pytest.mark.parametrize("n", [0, 6, -1])
    def test_bad_bedroom_count(self, n):
        with pytest.raises(ValueError):
            rooms_for(n)

    def test_dimensions_are_optional_and_carried(self):
        slots = slots_for(rooms_for(1, {"living_room": {"width_cm": 300, "depth_cm": 200}}))
        sofa = next(s for s in slots if s.profile.id == "sofa")
        assert (sofa.max_width_cm, sofa.max_depth_cm) == (300, 200)
        assert next(s for s in slots if s.profile.id == "bed").max_width_cm is None

    def test_slot_round_trip(self):
        for s in slots_for(rooms_for(2)):
            assert type(s).from_dict(s.to_dict()) == s


# ── Budget ───────────────────────────────────────────────────────────────────

class TestBudget:
    @pytest.mark.parametrize("budget", [1, 99999.99, 200000, 1_234_567.89])
    @pytest.mark.parametrize("beds", [1, 2, 4])
    def test_ceilings_sum_exactly_to_budget(self, budget, beds):
        slots = slots_for(rooms_for(beds))
        assert round(sum(budget_mod.allocate(budget, slots).values()), 2) == round(budget, 2)

    @pytest.mark.parametrize("bad", [0, -5])
    def test_bad_budget(self, bad):
        with pytest.raises(ValueError):
            budget_mod.allocate(bad, slots_for(rooms_for(1)))

    def test_reallocate_never_exceeds_total(self):
        slots = slots_for(rooms_for(2))
        settled = {slots[0].slot_id: 10000.0, slots[1].slot_id: 5000.0}
        open_ids = {s.slot_id for s in slots[2:]}
        out = budget_mod.reallocate(100000, slots, settled, open_ids)
        assert set(out) == open_ids
        assert round(sum(out.values()) + 15000, 2) == 100000

    def test_reallocate_with_nothing_left(self):
        slots = slots_for(rooms_for(1))
        out = budget_mod.reallocate(1000, slots, {slots[0].slot_id: 1000.0}, {slots[1].slot_id})
        assert out == {slots[1].slot_id: 0.0}

    def test_summary(self):
        slots = slots_for(rooms_for(1))
        s = budget_mod.summarise(100000, slots, {slots[0].slot_id: 30000.0, slots[1].slot_id: 5000.0})
        assert s.spent == 35000 and s.remaining == 65000 and s.within_budget
        assert s.by_room == {"Living room": 35000.0} and len(s.unfilled) == len(slots) - 2

    def test_summary_flags_overspend(self):
        slots = slots_for(rooms_for(1))
        assert not budget_mod.summarise(100, slots, {slots[0].slot_id: 500.0}).within_budget


# ── Catalogs ─────────────────────────────────────────────────────────────────

class TestCatalogs:
    def test_simulated_is_deterministic_and_labelled(self):
        a, b = SimulatedCatalog(seed=3).listings("bed"), SimulatedCatalog(seed=3).listings("bed")
        assert a.records == b.records and a.simulated
        assert all(r["data_origin"] == ORIGIN_SIMULATED and r["source_agent_id"] == "simulated_sample" for r in a.records)

    def test_different_seeds_differ(self):
        assert SimulatedCatalog(1).listings("sofa").records != SimulatedCatalog(2).listings("sofa").records

    def test_simulated_data_is_as_sparse_as_the_real_thing(self):
        records = [r for c in CATEGORIES for r in SimulatedCatalog(per_category=40).listings(c).records]
        rated = sum(r["rating"] is not None for r in records) / len(records)
        assert 0.05 < rated < 0.30
        assert all(r["width_cm"] is None and r["style"] is None and r["colour"] is None for r in records)

    def test_unknown_category(self):
        with pytest.raises(KeyError):
            SimulatedCatalog().listings("spaceship")

    def test_snapshot_round_trip(self, tmp_path):
        snap = SnapshotCatalog(tmp_path)
        original = SimulatedCatalog().listings("wardrobe")
        snap.save(CategoryListings("wardrobe", original.records, ORIGIN_REAL, "2026-11-01T00:00:00+00:00", "q"))
        loaded = snap.listings("wardrobe")
        assert loaded.records == original.records and loaded.origin == ORIGIN_REAL and loaded.query == "q"

    def test_snapshot_missing_and_corrupt(self, tmp_path):
        snap = SnapshotCatalog(tmp_path)
        assert snap.listings("sofa").origin == "none"
        (tmp_path / "sofa.json").write_text("{not json", encoding="utf-8")
        got = snap.listings("sofa")
        assert got.origin == "none" and "unreadable" in got.notes[0]
        (tmp_path / "sofa.json").write_text(json.dumps({"records": "x"}), encoding="utf-8")
        assert snap.listings("sofa").origin == "none"

    def test_category_id_cannot_escape_the_directory(self, tmp_path):
        with pytest.raises(KeyError):
            SnapshotCatalog(tmp_path).path_for("../../etc/passwd")

    def test_layered_prefers_real_and_only_falls_back_when_allowed(self, tmp_path):
        snap = SnapshotCatalog(tmp_path)
        real = SimulatedCatalog().listings("bed")
        snap.save(CategoryListings("bed", real.records[:3], ORIGIN_REAL, "t"))
        allowed = LayeredCatalog(snap, SimulatedCatalog(), allow_simulated=True)
        assert allowed.listings("bed").origin == ORIGIN_REAL and len(allowed.listings("bed").records) == 3
        assert allowed.listings("sofa").simulated
        strict = LayeredCatalog(snap, SimulatedCatalog(), allow_simulated=False)
        assert strict.listings("sofa").records == [] and strict.listings("bed").records


# ── Signed agent bus ─────────────────────────────────────────────────────────

class TestBus:
    def make(self):
        log = ActivityLog()
        bus = AgentBus(log)
        bus.register("echo", lambda task, data: {"task": task, "data": data})
        return bus, log

    def test_call_round_trip_and_logs_verified_message(self):
        bus, log = self.make()
        assert bus.call("echo", "ping", {"x": 1}) == {"task": "ping", "data": {"x": 1}}
        msg = [e for e in log.events() if e.kind == "message"][0]
        assert msg.verified is True and msg.peer == "echo" and msg.duration_ms is not None

    def test_failing_agent_becomes_agent_call_error(self):
        bus, log = self.make()
        bus.register("bad", lambda t, d: 1 / 0)
        with pytest.raises(AgentCallError) as err:
            bus.call("bad", "go", {})
        assert "ZeroDivisionError" in err.value.reason
        assert any(e.kind == "warning" for e in log.events())

    def test_unsigned_and_forged_messages_never_reach_the_handler(self):
        bus, _ = self.make()
        seen = []
        bus.register("victim", lambda t, d: seen.append(t) or {})
        router = bus._routers["victim"]
        unsigned = MessageEnvelope(sender="coordinator", receiver="victim", message_type=MessageType.TASK_REQUEST,
                                   payload={"task_type": "x", "data": {}})
        assert router.route(unsigned).payload["reason"] == "authentication_failed"
        from smartnest.security import AgentKeyPair
        forged = sign_envelope(unsigned, AgentKeyPair.generate("coordinator"))      # right name, wrong key
        assert router.route(forged).payload["reason"] == "authentication_failed"
        assert seen == []

    def test_a_worker_cannot_command_another_worker(self):
        bus, _ = self.make()
        bus.register("a", lambda t, d: {})
        bus.register("b", lambda t, d: {})
        with pytest.raises(AgentCallError):
            bus.call("b", "do", {}, sender="a")           # workers may not send task_request commands

    def test_replayed_message_is_rejected(self):
        bus, _ = self.make()
        env = sign_envelope(MessageEnvelope(sender="coordinator", receiver="echo", message_type=MessageType.TASK_REQUEST,
                                            payload={"task_type": "x", "data": {}}), bus._keys["coordinator"])
        router = bus._routers["echo"]
        assert router.route(env).message_type == MessageType.INFORM_RESULT.value
        assert router.route(env).payload["reason"] == "message_rejected"


# ── Coordinator ──────────────────────────────────────────────────────────────

class TestCoordinator:
    def test_plan_covers_every_room_within_budget(self):
        _, plan = planned()
        assert len(plan.lines) == 10 and plan.summary["within_budget"]
        assert plan.summary["spent"] <= 200000
        assert {ln["room"] for ln in plan.lines.values()} == {"Living room", "Bedroom 1", "Bedroom 2", "Kitchen"}
        assert plan.summary["spent"] == pytest.approx(sum(ln["product"]["price"] for ln in plan.lines.values()
                                                          if ln["product"]))

    def test_every_pick_respects_its_item_ceiling(self):
        _, plan = planned()
        for ln in plan.lines.values():
            if ln["product"]:
                assert ln["product"]["price"] <= ln["ceiling"]

    @pytest.mark.parametrize("budget", [20000, 50000, 90000, 150000, 500000])
    @pytest.mark.parametrize("beds", [1, 2, 3])
    def test_budget_is_never_exceeded(self, budget, beds):
        _, plan = planned(budget, beds)
        assert plan.summary["spent"] <= budget + 1e-6
        assert plan.summary["within_budget"]

    def test_tiny_budget_leaves_items_unfilled_instead_of_breaking_the_limit(self):
        _, plan = planned(15000, 2)
        assert plan.summary["unfilled_count"] > 0 and plan.summary["spent"] <= 15000
        unfilled = [ln for ln in plan.lines.values() if ln["status"] == "unfilled"]
        assert all("nothing was relaxed" in ln["why"].lower() for ln in unfilled)

    def test_unconfirmed_stock_is_never_called_confirmed(self):
        _, plan = planned()
        for ln in plan.lines.values():
            if ln["product"] and ln["product"]["availability"] != "in_stock":
                assert ln["status"] == "unverified"
            if ln["status"] == "confirmed":
                assert ln["product"]["availability"] == "in_stock"

    def test_simulated_products_have_no_buy_link_or_image(self):
        _, plan = planned()
        for ln in plan.lines.values():
            p = ln["product"]
            assert p["simulated"] and p["buy_url"] is None and p["image_url"] is None
            assert "not a real product" in p["link_label"]

    def test_plan_is_deterministic(self):
        a = planned()[1].to_dict()
        b = planned()[1].to_dict()
        for d in (a, b):
            d.pop("updated_at")
        assert a == b

    def test_plan_is_json_serialisable(self):
        json.dumps(planned()[1].to_dict())

    @pytest.mark.parametrize("bad", [0, -1, float("nan"), "100", None, True, 1e12])
    def test_bad_budget_rejected(self, bad):
        with pytest.raises(PlanError):
            coordinator().create_plan(FurnishingRequest(bad, 2))

    @pytest.mark.parametrize("bad", [0, 6, "2", True, None])
    def test_bad_bedrooms_rejected(self, bad):
        with pytest.raises(PlanError):
            coordinator().create_plan(FurnishingRequest(100000, bad))

    def test_no_snapshot_and_no_simulation_gives_an_honest_empty_plan(self):
        c = FurnishingCoordinator(LayeredCatalog(SnapshotCatalog("/nonexistent-dir")))
        plan = c.create_plan(FurnishingRequest(200000, 2))
        assert plan.summary["unfilled_count"] == 10 and plan.summary["spent"] == 0

    def test_accessories_are_dropped_before_ranking(self):
        class Cat:
            def listings(self, category_id):
                base = SimulatedCatalog().listings(category_id)
                junk = dict(base.records[0], product_id="JUNK", name="Fridge magnet set", price=499.0)
                return CategoryListings(category_id, [junk, *base.records], base.origin, base.retrieved_at)
        plan = FurnishingCoordinator(Cat()).create_plan(FurnishingRequest(200000, 2))
        fridge = plan.lines["kitchen.refrigerator"]
        assert fridge["product"]["product_id"] != "JUNK" and fridge["dropped_accessories"] >= 1

    def test_activity_log_shows_every_agent_and_only_verified_messages(self):
        c, _ = planned()
        events = c.log.events()
        agents = {e.agent for e in events}
        assert {"coordinator", "space_planning", "product_research", "comparison", "budget"} <= agents
        assert all(e.verified for e in events if e.kind == "message")
        assert [e.seq for e in events] == sorted(e.seq for e in events)


class TestReplanning:
    def test_out_of_stock_replaces_the_choice_and_stays_in_budget(self):
        c, plan = planned()
        sofa = plan.lines["living_room.sofa"]["product"]["product_id"]
        new = c.handle_event({"type": "out_of_stock", "product_id": sofa})
        assert new.lines["living_room.sofa"]["product"]["product_id"] != sofa
        assert new.version == plan.version + 1 and new.summary["within_budget"]
        assert any(t["kind"] == "replaced" and t["slot_id"] == "living_room.sofa" for t in c.transitions)

    def test_unrelated_slots_are_retained(self):
        c, plan = planned()
        before = {k: v["product"]["product_id"] for k, v in plan.lines.items()}
        sofa = before["living_room.sofa"]
        new = c.handle_event({"type": "out_of_stock", "product_id": sofa})
        for k, pid in before.items():
            if k != "living_room.sofa" and pid != sofa:
                assert new.lines[k]["product"]["product_id"] == pid

    def test_out_of_stock_affects_every_slot_holding_that_product(self):
        c, plan = planned()
        bed = plan.lines["bedroom_1.bed"]["product"]["product_id"]
        assert plan.lines["bedroom_2.bed"]["product"]["product_id"] == bed       # same cheapest bed in both rooms
        new = c.handle_event({"type": "out_of_stock", "product_id": bed})
        for room in ("bedroom_1", "bedroom_2"):
            assert new.lines[f"{room}.bed"]["product"]["product_id"] != bed

    def test_budget_cut_reduces_spend(self):
        c, plan = planned(200000)
        new = c.handle_event({"type": "budget_change", "total_budget": 80000})
        assert new.summary["spent"] <= 80000 < plan.summary["spent"]

    def test_budget_increase_never_overspends(self):
        c, _ = planned(60000)
        new = c.handle_event({"type": "budget_change", "total_budget": 400000})
        assert new.summary["spent"] <= 400000 and new.summary["unfilled_count"] == 0

    def test_price_change_is_applied(self):
        c, plan = planned()
        pid = plan.lines["living_room.sofa"]["product"]["product_id"]
        new = c.handle_event({"type": "price_change", "product_id": pid, "price": 999999})
        assert new.lines["living_room.sofa"]["product"]["product_id"] != pid       # now over its ceiling

    def test_remove_item(self):
        c, plan = planned()
        new = c.handle_event({"type": "remove_item", "slot_id": "living_room.tv_unit"})
        assert "living_room.tv_unit" not in new.lines and len(new.lines) == 9
        assert new.summary["spent"] <= 200000

    def test_preference_change_is_recorded(self):
        c, _ = planned()
        c.handle_event({"type": "preference_change", "style": "classic", "colour": "grey"})
        assert c.beliefs.value("user", "style") == "classic"

    @pytest.mark.parametrize("event", [
        {"type": "nonsense"}, {"type": "out_of_stock", "product_id": "NOPE"}, {"type": "out_of_stock"},
        {"type": "budget_change", "total_budget": -5}, {"type": "budget_change", "total_budget": "lots"},
        {"type": "price_change", "product_id": "NOPE", "price": 5}, {"type": "remove_item", "slot_id": "x"}, {}])
    def test_bad_events_are_rejected_without_changing_the_plan(self, event):
        c, plan = planned()
        with pytest.raises(PlanError):
            c.handle_event(event)
        assert c.plan is plan and c.plan.version == 1

    def test_invalid_price_value(self):
        c, plan = planned()
        pid = plan.lines["living_room.sofa"]["product"]["product_id"]
        for bad in (0, -1, "x", float("nan"), True):
            with pytest.raises(PlanError):
                c.handle_event({"type": "price_change", "product_id": pid, "price": bad})

    def test_event_before_plan(self):
        with pytest.raises(PlanError):
            coordinator().handle_event({"type": "budget_change", "total_budget": 5})

    def test_shortlist_toggle(self):
        c, plan = planned()
        pid = plan.lines["living_room.sofa"]["product"]["product_id"]
        assert c.toggle_shortlist(pid) == [pid]
        assert c.toggle_shortlist(pid) == []
        with pytest.raises(PlanError):
            c.toggle_shortlist("nope")


class TestUserChoices:
    def pricey(self, c, category):
        return max(c._research[category]["records"], key=lambda r: r["price"])

    def test_user_choice_replaces_the_automatic_pick(self):
        c, plan = planned()
        cheapest = plan.lines["living_room.sofa"]["product"]["product_id"]
        alt = plan.lines["living_room.sofa"]["alternatives"][0]
        new = c.handle_event({"type": "choose_product", "slot_id": "living_room.sofa", "product_id": alt["product_id"]})
        line = new.lines["living_room.sofa"]
        assert line["product"]["product_id"] == alt["product_id"] != cheapest
        assert line["user_choice"] and line["why"].startswith("Chosen by you")
        assert new.summary["within_budget"]

    def test_choice_that_would_break_the_total_budget_is_refused_and_the_plan_unchanged(self):
        c, plan = planned(100000, 1)
        sofa = plan.lines["living_room.sofa"]
        assert sofa["product"] is not None
        record = self.pricey(c, "sofa")
        room = 100000 - (plan.summary["spent"] - sofa["product"]["price"])
        assert record["price"] > room                     # precondition: this pick cannot fit
        before = plan.summary["spent"]
        with pytest.raises(PlanError):
            c.handle_event({"type": "choose_product", "slot_id": "living_room.sofa",
                            "product_id": record["product_id"]})
        assert c.plan is plan and c.plan.summary["spent"] == before and c._choices == {}

    def test_choice_cannot_use_a_product_from_another_category(self):
        c, _ = planned()
        wardrobe = c._research["wardrobe"]["records"][0]["product_id"]
        with pytest.raises(PlanError):
            c.handle_event({"type": "choose_product", "slot_id": "living_room.sofa", "product_id": wardrobe})

    def test_choice_of_out_of_stock_product_is_refused(self):
        c, _ = planned()
        pid = c._research["sofa"]["records"][3]["product_id"]
        c.handle_event({"type": "out_of_stock", "product_id": pid})
        with pytest.raises(PlanError):
            c.handle_event({"type": "choose_product", "slot_id": "living_room.sofa", "product_id": pid})

    def test_clear_choice_goes_back_to_the_automatic_pick(self):
        c, plan = planned()
        auto = plan.lines["living_room.sofa"]["product"]["product_id"]
        alt = plan.lines["living_room.sofa"]["alternatives"][0]["product_id"]
        c.handle_event({"type": "choose_product", "slot_id": "living_room.sofa", "product_id": alt})
        back = c.handle_event({"type": "clear_choice", "slot_id": "living_room.sofa"})
        assert back.lines["living_room.sofa"]["product"]["product_id"] == auto
        assert not back.lines["living_room.sofa"]["user_choice"]

    def test_many_choices_never_exceed_the_budget(self):
        c, plan = planned(120000, 2)
        for slot_id, line in plan.lines.items():
            if line["alternatives"]:
                best = max(line["alternatives"], key=lambda a: a["price"])
                try:
                    c.handle_event({"type": "choose_product", "slot_id": slot_id, "product_id": best["product_id"]})
                except PlanError:
                    pass
                assert c.plan.summary["spent"] <= 120000 + 1e-6

    def test_unknown_slot(self):
        c, _ = planned()
        with pytest.raises(PlanError):
            c.handle_event({"type": "choose_product", "slot_id": "nowhere.sofa", "product_id": "x"})


class TestLeftoverIsNotWasted:
    @pytest.mark.parametrize("budget", [40000, 60000, 80000, 100000, 120000, 160000])
    def test_an_item_stays_unfilled_only_if_the_remaining_money_cannot_buy_it(self, budget):
        c, plan = planned(budget, 2)
        remaining = plan.summary["remaining"]
        for line in plan.lines.values():
            if line["status"] != "unfilled":
                continue
            slot = next(s for s in c._slots if s.slot_id == line["slot_id"])
            buyable = [r for r in c._research[line["category"]]["records"]
                       if r["price"] <= remaining and r["availability"] != "out_of_stock"
                       and (not slot.profile.needs_seating or r["seating_capacity"] is None
                            or r["seating_capacity"] >= slot.min_seating)]
            assert not buyable, f"{line['label']} is unfilled although {buyable[0]['name']} costs {buyable[0]['price']}"

    def test_the_greedy_fill_never_breaks_the_budget(self):
        for budget in range(30000, 200001, 10000):
            _, plan = planned(budget, 2)
            assert plan.summary["spent"] <= budget + 1e-6
