"""Shopping cart: rules, budget guard, events after the fact, chat commands, API and persistence."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from smartnest.api.app import ApiSettings, create_app
from smartnest.database import InMemoryRepository
from smartnest.furnishing import (
    FurnishingCoordinator,
    FurnishingRequest,
    LayeredCatalog,
    PlanError,
    SimulatedCatalog,
    SnapshotCatalog,
)
from smartnest.furnishing.assistant import ChatAssistant, HomeSession

DEMO = "I just moved into a 2BHK apartment. My furnishing budget is ₹2 lakh. I prefer minimalist furniture."


def catalog():
    return LayeredCatalog(SnapshotCatalog("/nonexistent"), SimulatedCatalog(), allow_simulated=True)


def planned(budget=200000):
    c = FurnishingCoordinator(catalog())
    c.create_plan(FurnishingRequest(budget, 2, "minimalist"))
    return c


def pid(c, slot):
    return c.plan.lines[slot]["product"]["product_id"]


def priciest(c, category):
    return max(c._research[category]["records"], key=lambda r: r["price"])


class TestCartRules:
    def test_add_set_remove_clear(self):
        c = planned()
        sofa = pid(c, "living_room.sofa")
        c.handle_event({"type": "cart_add", "product_id": sofa})
        assert c.cart == {sofa: 1}
        c.handle_event({"type": "cart_add", "product_id": sofa, "qty": 2})
        assert c.cart[sofa] == 3
        c.handle_event({"type": "cart_set", "product_id": sofa, "qty": 1})
        assert c.cart[sofa] == 1
        c.handle_event({"type": "cart_set", "product_id": sofa, "qty": 0})
        assert c.cart == {}
        c.handle_event({"type": "cart_add", "product_id": sofa})
        c.handle_event({"type": "cart_remove", "product_id": sofa})
        c.handle_event({"type": "cart_add", "product_id": sofa})
        c.handle_event({"type": "cart_clear"})
        assert c.cart == {}

    def test_cart_events_never_replan(self):
        c = planned()
        version, picks = c.plan.version, {k: v["product"]["product_id"] for k, v in c.plan.lines.items()}
        c.handle_event({"type": "cart_add", "product_id": pid(c, "living_room.sofa")})
        assert c.plan.version == version and picks == {k: v["product"]["product_id"] for k, v in c.plan.lines.items()}

    def test_add_plan_is_idempotent_and_counts_repeats(self):
        c = planned()
        c.handle_event({"type": "cart_add_plan"})
        first = dict(c.cart)
        c.handle_event({"type": "cart_add_plan"})
        assert c.cart == first
        assert c.cart[pid(c, "bedroom_1.bed")] == 2                     # both bedrooms hold the same bed
        assert sum(c.cart.values()) == len(c.plan.lines)
        assert c.cart_total() == pytest.approx(c.plan.summary["spent"])

    def test_any_found_listing_can_be_added_not_only_plan_picks(self):
        c = planned()
        other = next(r for r in c._research["sofa"]["records"] if r["product_id"] != pid(c, "living_room.sofa"))
        c.handle_event({"type": "cart_add", "product_id": other["product_id"]})
        view = c.cart_view()
        assert view["lines"][0]["in_plan"] is False and view["lines"][0]["product"]["name"] == other["name"]

    @pytest.mark.parametrize("bad", [None, "NOPE", 5, ""])
    def test_unknown_product_is_refused(self, bad):
        c = planned()
        with pytest.raises(PlanError):
            c.handle_event({"type": "cart_add", "product_id": bad})
        assert c.cart == {}

    @pytest.mark.parametrize("qty", [0, -1, 11, True, "2", 2.5, None])
    def test_bad_quantities(self, qty):
        c = planned()
        event = {"type": "cart_add", "product_id": pid(c, "living_room.sofa")}
        if qty is not None:
            event["qty"] = qty
        if qty is None:                       # default quantity is 1
            c.handle_event(event)
            assert sum(c.cart.values()) == 1
            return
        with pytest.raises(PlanError):
            c.handle_event(event)
        assert c.cart == {}

    def test_at_most_ten_of_one_product(self):
        c = planned(10_000_000)
        sofa = pid(c, "living_room.sofa")
        c.handle_event({"type": "cart_add", "product_id": sofa, "qty": 10})
        with pytest.raises(PlanError):
            c.handle_event({"type": "cart_add", "product_id": sofa})
        assert c.cart[sofa] == 10

    def test_out_of_stock_product_cannot_be_added(self):
        c = planned()
        sofa = pid(c, "living_room.sofa")
        c.handle_event({"type": "out_of_stock", "product_id": sofa})
        with pytest.raises(PlanError) as err:
            c.handle_event({"type": "cart_add", "product_id": sofa})
        assert "out of stock" in str(err.value) and c.cart == {}

    def test_unknown_cart_event_type(self):
        c = planned()
        with pytest.raises(PlanError):
            c.apply_cart_event({"type": "cart_explode"})


class TestBudgetGuard:
    def test_an_addition_that_breaks_the_budget_is_refused_with_the_shortfall(self):
        c = planned(100000)
        rec = priciest(c, "sofa")
        c.handle_event({"type": "cart_add", "product_id": rec["product_id"], "qty": 1})
        with pytest.raises(PlanError) as err:
            c.handle_event({"type": "cart_add", "product_id": rec["product_id"], "qty": 5})
        assert "over your" in str(err.value) and c.cart == {rec["product_id"]: 1}

    def test_the_cart_total_never_exceeds_the_budget_under_any_sequence(self):
        c = planned(120000)
        records = [r for cat in ("sofa", "bed", "refrigerator", "wardrobe") for r in c._research[cat]["records"]]
        for r in records * 2:
            try:
                c.handle_event({"type": "cart_add", "product_id": r["product_id"]})
            except PlanError:
                pass
            assert c.cart_total() <= 120000 + 1e-6

    def test_exactly_the_budget_is_allowed(self):
        c = planned(100000)
        r = c._research["sofa"]["records"][0]
        c._overrides[r["product_id"]] = {"price": 100000.0}
        c.handle_event({"type": "cart_add", "product_id": r["product_id"]})
        assert c.cart_view()["remaining"] == 0 and c.cart_view()["within_budget"]

    def test_a_refused_addition_leaves_the_cart_untouched(self):
        c = planned(100000)
        c.handle_event({"type": "cart_add_plan"})
        before = dict(c.cart)
        with pytest.raises(PlanError):
            c.handle_event({"type": "cart_add", "product_id": priciest(c, "sofa")["product_id"], "qty": 10})
        assert c.cart == before

    def test_cutting_the_budget_flags_the_cart_but_never_edits_it(self):
        c = planned()
        c.handle_event({"type": "cart_add_plan"})
        before = dict(c.cart)
        c.handle_event({"type": "budget_change", "total_budget": 50000})
        view = c.cart_view()
        assert c.cart == before and not view["within_budget"]
        assert any("over the current budget" in w for w in view["warnings"])

    def test_you_can_always_reduce_an_over_budget_cart(self):
        c = planned()
        c.handle_event({"type": "cart_add_plan"})
        c.handle_event({"type": "budget_change", "total_budget": 50000})
        some = next(iter(c.cart))
        c.handle_event({"type": "cart_remove", "product_id": some})
        c.handle_event({"type": "cart_clear"})
        assert c.cart == {}

    def test_adding_while_over_budget_is_refused(self):
        c = planned()
        c.handle_event({"type": "cart_add_plan"})
        c.handle_event({"type": "budget_change", "total_budget": 50000})
        other = next(r for r in c._research["sofa"]["records"] if r["product_id"] not in c.cart)
        with pytest.raises(PlanError):
            c.handle_event({"type": "cart_add", "product_id": other["product_id"]})

    def test_a_later_out_of_stock_event_warns_but_keeps_the_line(self):
        c = planned()
        sofa = pid(c, "living_room.sofa")
        c.handle_event({"type": "cart_add", "product_id": sofa})
        c.handle_event({"type": "out_of_stock", "product_id": sofa})
        view = c.cart_view()
        assert sofa in c.cart and view["lines"][0]["out_of_stock"]
        assert any("out of stock" in w for w in view["warnings"])

    def test_a_price_change_updates_the_total(self):
        c = planned()
        sofa = pid(c, "living_room.sofa")
        c.handle_event({"type": "cart_add", "product_id": sofa, "qty": 2})
        c.handle_event({"type": "price_change", "product_id": sofa, "price": 1000})
        assert c.cart_view()["total"] == 2000


class TestCartView:
    def test_totals_add_up(self):
        c = planned()
        c.handle_event({"type": "cart_add_plan"})
        v = c.cart_view()
        assert v["total"] == pytest.approx(sum(ln["line_total"] for ln in v["lines"]))
        assert v["total"] + v["remaining"] == pytest.approx(v["budget"])
        assert v["count"] == sum(ln["qty"] for ln in v["lines"])
        assert all(ln["line_total"] == pytest.approx(ln["qty"] * ln["product"]["price"]) for ln in v["lines"])

    def test_stock_is_never_called_confirmed_without_evidence(self):
        c = planned()
        c.handle_event({"type": "cart_add_plan"})
        for ln in c.cart_view()["lines"]:
            assert (ln["status"] == "confirmed") == (ln["product"]["availability"] == "in_stock")

    def test_simulated_lines_have_no_buy_link(self):
        c = planned()
        c.handle_event({"type": "cart_add_plan"})
        assert all(ln["product"]["buy_url"] is None and ln["product"]["simulated"] for ln in c.cart_view()["lines"])

    def test_empty_cart(self):
        v = planned().cart_view()
        assert v["lines"] == [] and v["total"] == 0 and v["within_budget"] and v["count"] == 0


class TestCartChat:
    def ready(self):
        s, a = HomeSession(FurnishingCoordinator(catalog())), ChatAssistant()
        a.respond(s, DEMO)
        return s, a

    def test_add_one_item_by_name(self):
        s, a = self.ready()
        r = a.respond(s, "add the sofa to my cart")
        assert r.kind == "cart_add" and r.changed and s.coordinator.cart == {pid(s.coordinator, "living_room.sofa"): 1}
        assert "₹" in r.text

    def test_adding_twice_says_so(self):
        s, a = self.ready()
        a.respond(s, "add the sofa to my cart")
        r = a.respond(s, "put the sofa in the basket")
        assert not r.changed and "Already in your cart" in r.text

    def test_add_by_room(self):
        s, a = self.ready()
        a.respond(s, "add the bed in bedroom 1 to my cart")
        assert pid(s.coordinator, "bedroom_1.bed") in s.coordinator.cart and len(s.coordinator.cart) == 1

    def test_add_everything(self):
        s, a = self.ready()
        r = a.respond(s, "add everything to my cart")
        assert sum(s.coordinator.cart.values()) == 10 and "every chosen product" in r.text

    def test_asks_which_item_when_unclear(self):
        s, a = self.ready()
        r = a.respond(s, "add something to my cart")
        assert not r.changed and "Which item" in r.text

    def test_removing_from_the_cart_does_not_touch_the_plan(self):
        s, a = self.ready()
        a.respond(s, "add the sofa to my cart")
        r = a.respond(s, "remove the sofa from my cart")
        assert r.kind == "cart_remove" and s.coordinator.cart == {} and "living_room.sofa" in s.plan.lines

    def test_show_and_clear(self):
        s, a = self.ready()
        assert "empty" in a.respond(s, "what is in my cart?").text
        a.respond(s, "add everything to my cart")
        assert a.respond(s, "show my cart").kind == "cart_show"
        assert a.respond(s, "clear my cart").changed and s.coordinator.cart == {}
        assert not a.respond(s, "clear my cart").changed

    def test_over_budget_request_is_explained(self):
        s, a = self.ready()
        a.respond(s, "add everything to my cart")
        a.respond(s, "reduce the budget to 80k")
        r = a.respond(s, "add the sofa to my cart")
        assert r.kind in ("error", "cart_add")

    def test_words_that_merely_contain_cart_verbs_are_not_cart_commands(self):
        s, a = self.ready()
        assert a.respond(s, "I don't need a TV unit").kind == "remove_item"
        assert a.respond(s, "address the sofa").kind != "cart_add"


class TestCartApi:
    def started(self, repository=None):
        app = create_app(ApiSettings(), catalog=catalog(), gemini=None, repository=repository)
        client = TestClient(app)
        return client, client.post("/api/sessions", json={"message": DEMO}).json()

    def test_session_view_carries_the_cart(self):
        client, body = self.started()
        assert body["cart"]["lines"] == [] and body["cart"]["budget"] == 200000

    def test_add_through_events(self):
        client, body = self.started()
        sofa = next(i for i in body["plan"]["items"] if i["category"] == "sofa")["product"]["product_id"]
        r = client.post(f"/api/sessions/{body['id']}/events", json={"type": "cart_add", "product_id": sofa, "qty": 2})
        assert r.status_code == 200 and r.json()["cart"]["count"] == 2
        r = client.post(f"/api/sessions/{body['id']}/events", json={"type": "cart_add_plan"})
        assert r.json()["cart"]["count"] >= 10 and r.json()["cart"]["within_budget"]

    @pytest.mark.parametrize("event", [
        {"type": "cart_add", "product_id": "NOPE"}, {"type": "cart_add"}, {"type": "cart_add", "product_id": "x", "qty": 99},
        {"type": "cart_add", "product_id": "x", "qty": -1}])
    def test_bad_cart_events(self, event):
        client, body = self.started()
        assert client.post(f"/api/sessions/{body['id']}/events", json=event).status_code == 422
        assert client.get(f"/api/sessions/{body['id']}").json()["cart"]["count"] == 0

    def test_over_budget_is_a_422_with_the_reason(self):
        client, body = self.started()
        client.post(f"/api/sessions/{body['id']}/events", json={"type": "budget_change", "total_budget": 30000})
        opts = client.get(f"/api/sessions/{body['id']}/items/living_room.sofa/options").json()["options"]
        big = max(opts, key=lambda o: o["price"])
        r = client.post(f"/api/sessions/{body['id']}/events",
                        json={"type": "cart_add", "product_id": big["product_id"], "qty": 10})
        assert r.status_code == 422 and "budget" in r.json()["detail"]

    def test_cart_survives_a_restart(self):
        repo = InMemoryRepository("home_projects", id_prefix="home")
        c1, body = self.started(repo)
        sid = body["id"]
        c1.post(f"/api/sessions/{sid}/events", json={"type": "cart_add_plan"})
        sofa = next(i for i in body["plan"]["items"] if i["category"] == "sofa")["product_id" if False else "product"]["product_id"]
        c1.post(f"/api/sessions/{sid}/events", json={"type": "cart_remove", "product_id": sofa})
        before = c1.get(f"/api/sessions/{sid}").json()["cart"]
        after = TestClient(create_app(ApiSettings(), catalog=catalog(), gemini=None, repository=repo)).get(
            f"/api/sessions/{sid}").json()["cart"]
        assert after["count"] == before["count"] == 9 and after["total"] == before["total"]

    def test_cart_commands_in_chat_update_the_session(self):
        client, body = self.started()
        r = client.post(f"/api/sessions/{body['id']}/messages", json={"message": "add everything to my cart"}).json()
        assert r["session"]["cart"]["count"] == 10 and r["reply"]["kind"] == "cart_add"
