"""Chat layer: intent parsing, money parsing, the assistant's actions and the Gemini guard rails."""

from __future__ import annotations

import pytest

from smartnest.furnishing import FurnishingCoordinator, LayeredCatalog, SimulatedCatalog, SnapshotCatalog
from smartnest.furnishing.assistant import MAX_MESSAGE_CHARS, ChatAssistant, HomeSession
from smartnest.furnishing.intent import categories_in, parse_colour, parse_message, parse_money, parse_style, room_in
from smartnest.furnishing.money import inr

DEMO = ("I just moved into a 2BHK apartment. My furnishing budget is ₹2 lakh. I prefer minimalist furniture. "
        "Help me furnish the entire apartment.")


def session() -> HomeSession:
    return HomeSession(FurnishingCoordinator(LayeredCatalog(SnapshotCatalog("/nonexistent"), SimulatedCatalog(),
                                                           allow_simulated=True)))


def ready(assistant=None):
    s, a = session(), assistant or ChatAssistant()
    a.respond(s, DEMO)
    return s, a


class TestMoney:
    @pytest.mark.parametrize("text,expected", [
        ("₹2 lakh", 200000), ("2 lakhs", 200000), ("1.5 lakh", 150000), ("Rs 1,50,000", 150000),
        ("rs. 80000", 80000), ("INR 45,000", 45000), ("80k", 80000), ("2 crore", 20000000),
        ("my budget is 250000", 250000), ("budget: 2,00,000", 200000), ("₹ 99999.50", 99999.5)])
    def test_amounts(self, text, expected):
        assert parse_money(text) == expected

    @pytest.mark.parametrize("text", ["I live in a 2BHK", "3 seater sofa", "I need 4 chairs", "no numbers here", "",
                                      "flat 12 on floor 3"])
    def test_numbers_that_are_not_money_are_ignored(self, text):
        assert parse_money(text) is None

    def test_inr_grouping(self):
        assert [inr(x) for x in (0, 999, 1000, 12345, 200000, 1234567, 12345678, -5000)] == [
            "₹0", "₹999", "₹1,000", "₹12,345", "₹2,00,000", "₹12,34,567", "₹1,23,45,678", "-₹5,000"]


class TestParsing:
    def test_demo_message(self):
        p = parse_message(DEMO, has_plan=False)
        assert (p.action, p.bedrooms, p.budget, p.style) == ("new_request", 2, 200000.0, "minimalist")

    @pytest.mark.parametrize("text,words", [("minimal look", "minimalist"), ("something Modern", "modern"),
                                            ("nothing", None)])
    def test_style(self, text, words):
        assert parse_style(text) == words

    def test_colour(self):
        assert parse_colour("a gray sofa") == "grey" and parse_colour("no colour") is None

    @pytest.mark.parametrize("text,cats", [("the tv unit", ["tv_unit"]), ("my fridge and sofa", ["sofa", "refrigerator"]),
                                           ("bedroom", []), ("dining table", ["dining_table"]), ("a couch", ["sofa"])])
    def test_categories(self, text, cats):
        assert categories_in(text) == cats

    @pytest.mark.parametrize("text,room", [("bedroom 2", "bedroom_2"), ("the Living Room", "living_room"),
                                           ("kitchen", "kitchen"), ("master bedroom", "bedroom_1"), ("garage", None)])
    def test_rooms(self, text, room):
        assert room_in(text) == room

    @pytest.mark.parametrize("text,action", [
        ("increase the budget to 3 lakh", "set_budget"), ("I don't need a TV unit", "remove_item"),
        ("the sofa is out of stock", "out_of_stock"), ("why this sofa?", "why"), ("how much is left?", "budget_status"),
        ("make it modern", "set_preference"), ("show bedroom 2", "overview"),
        ("compare options for the wardrobe", "compare"), ("blah blah", "unknown")])
    def test_actions_with_a_plan(self, text, action):
        assert parse_message(text, has_plan=True).action == action

    def test_without_a_plan_only_a_request_is_understood(self):
        assert parse_message("why this sofa?", has_plan=False).action == "help"
        assert parse_message("", has_plan=True).action == "help"


class TestAssistant:
    def test_demo_scenario_builds_the_plan(self):
        s, a = session(), ChatAssistant()
        r = a.respond(s, DEMO)
        assert r.kind == "new_request" and r.changed and s.plan is not None
        assert "2BHK" in r.text and "₹2,00,000" in r.text and s.plan.summary["within_budget"]

    def test_missing_information_is_asked_for_not_guessed(self):
        s, a = session(), ChatAssistant()
        r = a.respond(s, "I have a 2BHK")
        assert r.needs == ["budget"] and s.plan is None
        r = a.respond(s, "around 1.5 lakh")
        assert r.changed and s.plan.request.total_budget == 150000 and s.plan.request.bedrooms == 2

    def test_budget_only_asks_which_room_or_home(self):
        s, a = session(), ChatAssistant()
        assert a.respond(s, "my budget is 3 lakh").needs == ["rooms"]

    def test_budget_change(self):
        s, a = ready()
        r = a.respond(s, "reduce the budget to 80k")
        assert r.changed and s.plan.summary["total_budget"] == 80000 and s.plan.summary["spent"] <= 80000

    def test_bigger_budget_that_changes_nothing_says_so(self):
        s, a = ready()
        spent = s.plan.summary["spent"]
        r = a.respond(s, "increase the budget to 2.001 lakh")
        assert s.plan.summary["spent"] == spent and "unspent" in r.text

    def test_bigger_budget_that_changes_the_plan_does_not_claim_otherwise(self):
        s, a = ready()
        spent = s.plan.summary["spent"]
        r = a.respond(s, "increase the budget to 10 lakh")
        assert (s.plan.summary["spent"] != spent) == ("unspent" not in r.text)

    def test_remove_item_by_category_and_by_room(self):
        s, a = ready()
        a.respond(s, "I don't need a TV unit")
        assert "living_room.tv_unit" not in s.plan.lines
        a.respond(s, "skip the wardrobe in bedroom 2")
        assert "bedroom_2.wardrobe" not in s.plan.lines and "bedroom_1.wardrobe" in s.plan.lines
        a.respond(s, "no need for mattress")
        assert not [k for k in s.plan.lines if k.endswith("mattress")]

    def test_out_of_stock_replans(self):
        s, a = ready()
        before = s.plan.lines["living_room.sofa"]["product"]["product_id"]
        r = a.respond(s, "the sofa is out of stock")
        assert r.changed and s.plan.lines["living_room.sofa"]["product"]["product_id"] != before

    def test_questions_do_not_change_the_plan(self):
        s, a = ready()
        version = s.plan.version
        for q in ("why this sofa?", "how much is left?", "show bedroom 1", "compare wardrobes"):
            assert not a.respond(s, q).changed
        assert s.plan.version == version

    def test_events_are_recorded_so_the_plan_can_be_rebuilt(self):
        s, a = ready()
        a.respond(s, "I don't need a TV unit")
        a.respond(s, "reduce the budget to 150000")
        rebuilt = session()
        rebuilt.start(s.request)
        for ev in s.events:
            rebuilt.apply(ev)

        def picks(sess):
            return {k: v["product"]["product_id"] if v["product"] else None for k, v in sess.plan.lines.items()}
        assert picks(rebuilt) == picks(s)

    @pytest.mark.parametrize("bad", ["", "   ", "x" * (MAX_MESSAGE_CHARS + 1)])
    def test_empty_and_oversized_messages(self, bad):
        s, a = session(), ChatAssistant()
        assert a.respond(s, bad).kind == "error" and s.plan is None

    def test_nonsense_before_a_plan_shows_help(self):
        s, a = session(), ChatAssistant()
        assert a.respond(s, "tell me a joke").kind == "help"

    def test_history_is_kept(self):
        s, a = ready()
        assert [h["role"] for h in s.history[:2]] == ["user", "assistant"]


class FakeGemini:
    """Stands in for GeminiService."""

    def __init__(self, text=None, extraction=None, fail=False, brief=None):
        self.text, self.extraction, self.fail, self.brief = text, extraction, fail, brief

    def extract_requirements(self, message):
        if self.fail:
            raise RuntimeError("down")
        return self.extraction

    def _structured(self, prompt, schema, system, what):
        if self.fail:
            raise RuntimeError("down")
        if what == "home brief":
            return schema(**(self.brief or {}))
        return schema(text=self.text)


class TestGeminiGuards:
    def test_rewording_is_used_when_it_keeps_the_numbers(self):
        reply = ChatAssistant().respond(session(), DEMO)
        rewrite = reply.text.replace("I planned", "Here is what I planned")
        s2, a = session(), ChatAssistant(FakeGemini(text=rewrite))
        assert a.respond(s2, DEMO).wording == "gemini"

    def test_rewording_with_an_invented_number_is_rejected(self):
        s, a = session(), ChatAssistant(FakeGemini(text="Great news, your plan costs ₹12,345 only and is "
                                                        "unverified and wonderful for everyone."))
        r = a.respond(s, DEMO)
        assert r.wording == "rules" and "12,345" not in r.text

    def test_rewording_that_drops_a_number_is_rejected(self):
        plain = ChatAssistant().respond(session(), DEMO).text
        without = plain.replace("₹2,00,000", "your budget")                  # same facts minus one figure
        s, a = session(), ChatAssistant(FakeGemini(text=without + " unverified"))
        r = a.respond(s, DEMO)
        assert r.wording == "rules" and "₹2,00,000" in r.text

    def test_rewording_that_drops_the_unverified_caution_is_rejected(self):
        s, a = session(), ChatAssistant(FakeGemini(text="Your plan is lovely and everything is ready to buy now."))
        r = a.respond(s, DEMO)
        # simulated data here includes unverified items, so a rewrite without the caution must not be used
        assert r.wording == "rules"

    def test_gemini_failure_falls_back_to_rules(self):
        s, a = session(), ChatAssistant(FakeGemini(fail=True))
        r = a.respond(s, DEMO)
        assert r.wording == "rules" and s.plan is not None

    def test_gemini_fills_in_what_the_rules_missed(self):
        brief = {"budget": 250000, "rooms": ["living_room", "kitchen"], "themes": ["japandi"]}
        s, a = session(), ChatAssistant(FakeGemini(brief=brief))
        r = a.respond(s, "the hall and the kitchen, spend around two and a half hundred thousand rupees")
        assert s.plan is not None and s.plan.request.total_budget == 250000 and r.kind == "new_request"
        assert s.plan.request.scope == ("living_room", "kitchen")
