"""Versatile requests: any room, who it is for, themes, budget phrasings, missing data and quota-gated live search."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from smartnest.api.app import ApiSettings, create_app
from smartnest.database import InMemoryRepository
from smartnest.furnishing import (
    CATEGORIES,
    CategoryListings,
    FurnishingCoordinator,
    FurnishingRequest,
    LayeredCatalog,
    PlanError,
    SimulatedCatalog,
    SnapshotCatalog,
    rooms_for_scope,
    slots_for,
)
from smartnest.furnishing.assistant import ChatAssistant, HomeSession
from smartnest.furnishing.brief import (
    parse_age,
    parse_audience,
    parse_limit_phrase,
    parse_scope,
    parse_themes,
    theme_matches,
)
from smartnest.furnishing.catalog import ORIGIN_REAL
from smartnest.furnishing.categories import kid_signal, mature_reason
from smartnest.furnishing.intent import parse_message, parse_money
from smartnest.furnishing.live import LiveSearcher, LiveSearchError
from smartnest.integrations.products.serpapi import SerpApiSettings

USER_PROMPT = ("i want to decorate my 10 year old son's bedroom  under 50000, please find things that would be good for "
               "his bedroom, he is very much interested in cartoons and monsters as such so that would be a nice addition")
FIXTURE = Path(__file__).parent / "fixtures" / "serpapi_real_sample.json"


def empty_catalog(tmp_path):
    return LayeredCatalog(SnapshotCatalog(tmp_path), SimulatedCatalog(), allow_simulated=False)


def sim_catalog():
    return LayeredCatalog(SnapshotCatalog("/nonexistent"), SimulatedCatalog(), allow_simulated=True)


# ── Parsing ──────────────────────────────────────────────────────────────────

class TestParsing:
    @pytest.mark.parametrize("text,value", [
        ("under 50000", 50000), ("below Rs 45,000", 45000), ("within ₹30000", 30000), ("up to 80,000", 80000),
        ("budget of 1.5 lakh", None), ("max 2500", 2500), ("around 60000", 60000), ("under 3 seats", None),
        ("less than 500", None), ("nothing", None)])
    def test_limit_phrases(self, text, value):
        assert parse_limit_phrase(text) == value

    def test_money_still_prefers_units_and_signs(self):
        assert parse_money("under 2 lakh") == 200000 and parse_money("under ₹45k") == 45000
        assert parse_money("under 50000") == 50000

    @pytest.mark.parametrize("text,age", [("my 10 year old", 10), ("a 7-year-old boy", 7), ("aged 15", 15),
                                          ("he is 12 yrs old", 12), ("no age here", None), ("2BHK", None)])
    def test_age(self, text, age):
        assert parse_age(text) == age

    @pytest.mark.parametrize("text,expected", [
        ("my 10 year old son", ("child", 10)), ("my daughter", ("child", None)), ("a 16 year old", ("teen", 16)),
        ("my teenager", ("teen", None)), ("a 30 year old", ("adult", 30)), ("my bedroom", (None, None))])
    def test_audience(self, text, expected):
        assert parse_audience(text) == expected

    @pytest.mark.parametrize("text,scope", [
        ("my son's bedroom", ["kids_bedroom"]), ("my 8 year old's room", ["kids_bedroom"]), ("my bedroom", ["bedroom"]),
        ("my living room and kitchen", ["living_room", "kitchen"]), ("the study room", ["study"]),
        ("nursery for the baby", ["kids_bedroom"]), ("my room", ["bedroom"]), ("2BHK flat", []),
        ("a study table", [])])
    def test_scope(self, text, scope):
        assert parse_scope(text) == scope

    def test_my_living_room_is_not_also_a_bedroom(self):
        assert parse_scope("furnish my living room") == ["living_room"]

    def test_the_users_prompt(self):
        assert parse_themes(USER_PROMPT) == ["cartoon", "monster"]
        assert parse_audience(USER_PROMPT) == ("child", 10) and parse_scope(USER_PROMPT) == ["kids_bedroom"]
        assert parse_limit_phrase(USER_PROMPT) == 50000

    @pytest.mark.parametrize("text,themes", [
        ("she loves unicorns and rainbows", ["unicorn", "rainbow"]), ("he is into dinosaurs", ["dinosaur"]),
        ("space themed room", ["space"]), ("a football fan", ["football"]), ("make it nice", []),
        ("he likes it very much", [])])
    def test_themes(self, text, themes):
        assert parse_themes(text) == themes

    def test_themes_are_capped_and_clean(self):
        out = parse_themes("loves cartoon, monster, dinosaur, space, robot, pirate, dragon, unicorn")
        assert len(out) <= 5 and all(t.isalpha() for t in out)

    def test_theme_matching_uses_the_title_only(self):
        assert theme_matches("Monster Cartoon Poster Set", ["monster", "dinosaur"]) == ["monster"]
        assert theme_matches("Dinosaurs Wall Art", ["dinosaur"]) == ["dinosaur"]
        assert theme_matches("Plain Poster", ["monster"]) == []
        assert theme_matches("Monstera plant poster", ["monster"]) == []          # a different word

    def test_the_old_flow_still_works(self):
        p = parse_message("2BHK budget 2 lakh minimalist", has_plan=False)
        assert (p.action, p.bedrooms, p.budget, p.scope) == ("new_request", 2, 200000.0, [])

    def test_a_lone_number_or_filler_is_not_a_request(self):
        assert parse_message("1", has_plan=False).action == "help"
        assert parse_message("hello", has_plan=False).action == "help"


# ── The conversation that failed ─────────────────────────────────────────────

class TestTheUsersConversation:
    def ready(self, catalog=None):
        s = HomeSession(FurnishingCoordinator(catalog or sim_catalog()))
        return s, ChatAssistant()

    def test_one_message_plans_just_the_childs_bedroom(self):
        s, a = self.ready()
        r = a.respond(s, USER_PROMPT)
        req = s.plan.request
        assert r.kind == "new_request" and req.bedrooms is None and req.scope == ("kids_bedroom",)
        assert (req.total_budget, req.audience, req.age, req.themes) == (50000, "child", 10, ("cartoon", "monster"))
        assert {i["room"] for i in s.plan.items} == {"Kid's bedroom"} if hasattr(s.plan, "items") else True
        assert {ln["room_id"] for ln in s.plan.lines.values()} == {"kids_bedroom"}
        assert "living_room.sofa" not in s.plan.lines and not any("kitchen" in k for k in s.plan.lines)
        assert "10-year-old" in r.text

    def test_it_never_assumes_a_home_when_only_a_room_was_described(self):
        s, a = self.ready()
        a.respond(s, "decorate my daughter's room")
        assert s.plan is None and s.draft["scope"] == ["kids_bedroom"]
        r = a.respond(s, "1")                                    # the reply that used to build a 1BHK
        assert s.plan is None and r.needs == ["budget"]

    def test_a_bare_number_answers_the_budget_question(self):
        s, a = self.ready()
        a.respond(s, "decorate my daughter's room, she loves unicorns")
        r = a.respond(s, "40000")
        assert r.changed and s.plan.request.total_budget == 40000 and s.plan.request.themes == ("unicorn",)

    def test_a_bare_number_is_ignored_when_nothing_was_asked(self):
        s, a = self.ready()
        r = a.respond(s, "40000")
        assert s.plan is None and r.kind == "help"

    def test_the_budget_can_be_given_first(self):
        s, a = self.ready()
        a.respond(s, "I have 60000 to spend")
        assert s.plan is None
        a.respond(s, "it's for my bedroom")
        assert s.plan.request.scope == ("bedroom",) and s.plan.request.total_budget == 60000

    def test_what_was_understood_is_echoed_back(self):
        s, a = self.ready()
        r = a.respond(s, "decorate my 10 year old son's bedroom, he loves dinosaurs")
        assert "10-year-old" in r.text and "dinosaur" in r.text and "budget" in r.text

    @pytest.mark.parametrize("message,rooms", [
        ("furnish my living room and kitchen, budget of 1.5 lakh", {"living_room", "kitchen"}),
        ("I need a bedroom set under ₹80k", {"bedroom_1"}), ("set up my study, 25000", {"study"})])
    def test_other_requests(self, message, rooms):
        s, a = self.ready()
        a.respond(s, message)
        assert s.plan is not None and {ln["room_id"] for ln in s.plan.lines.values()} == rooms

    def test_a_whole_home_request_still_works_and_wins_over_a_room_word(self):
        s, a = self.ready()
        a.respond(s, "2BHK, bedroom and kitchen, budget 2 lakh")
        assert s.plan.request.bedrooms == 2 and len(s.plan.lines) == 10

    def test_changes_work_after_a_room_plan(self):
        s, a = self.ready()
        a.respond(s, USER_PROMPT)
        assert a.respond(s, "reduce the budget to 30000").changed
        assert s.plan.request.scope == ("kids_bedroom",) and s.plan.request.themes == ("cartoon", "monster")
        assert s.plan.summary["spent"] <= 30000
        r = a.respond(s, "I don't need a bookshelf")
        assert "kids_bedroom.bookshelf" not in s.plan.lines and r.changed


class TestGeminiBriefIsValidated:
    class Fake:
        def __init__(self, brief):
            self.brief = brief

        def _structured(self, prompt, schema, system, what):
            return schema(**self.brief)

        def extract_requirements(self, message):
            raise RuntimeError("not used")

    def plan_from(self, brief, message="my little one needs a cosy corner, whatever you suggest"):
        s = HomeSession(FurnishingCoordinator(sim_catalog()))
        ChatAssistant(self.Fake(brief)).respond(s, message)
        return s

    def test_a_good_brief_is_used(self):
        s = self.plan_from({"rooms": ["kids_bedroom"], "budget": 45000, "audience": "child", "age": 6, "themes": ["space"]})
        assert s.plan.request.scope == ("kids_bedroom",) and s.plan.request.age == 6 and s.plan.request.themes == ("space",)

    @pytest.mark.parametrize("bad", [
        {"rooms": ["garage"], "budget": 45000}, {"rooms": ["kids_bedroom"], "budget": -5},
        {"rooms": ["kids_bedroom"], "budget": 1e15}, {"rooms": ["kids_bedroom"], "budget": 5000, "age": 400},
        {"rooms": ["kids_bedroom"], "budget": 5000, "audience": "alien"}])
    def test_invalid_values_are_dropped_not_trusted(self, bad):
        s = self.plan_from(bad)
        if s.plan is not None:
            req = s.plan.request
            assert req.total_budget <= 1e9 and (req.age is None or req.age <= 99) and req.audience in (None, "child", "teen", "adult")
            assert all(k in ("kids_bedroom",) for k in req.scope)

    def test_bad_theme_words_are_dropped(self):
        s = self.plan_from({"rooms": ["bedroom"], "budget": 30000, "themes": ["ok", "<script>", "x" * 50, "dragon"]})
        assert s.plan.request.themes in (("ok", "dragon"), ("dragon",))

    def test_rules_win_over_gemini(self):
        s = self.plan_from({"rooms": ["bedroom"], "budget": 99999}, "decorate my son's bedroom under 50000 please sir")
        assert s.plan.request.total_budget == 50000 and s.plan.request.scope == ("kids_bedroom",)

    def test_gemini_failure_falls_back_to_asking(self):
        class Down:
            def _structured(self, *a, **k):
                raise RuntimeError("down")
        s = HomeSession(FurnishingCoordinator(sim_catalog()))
        r = ChatAssistant(Down()).respond(s, "something cosy for my little one please")
        assert s.plan is None and r.kind == "help"


# ── Planning a room ──────────────────────────────────────────────────────────

class TestRoomPlans:
    def test_kids_room_slots(self):
        assert [s.profile.id for s in slots_for(rooms_for_scope(["kids_bedroom"]))] == [
            "kids_bed", "mattress", "wardrobe", "study_table", "study_chair", "bookshelf", "wall_decor", "table_lamp",
            "rug", "bean_bag", "toy_storage"]

    def test_unknown_room_and_empty_scope_are_refused(self):
        with pytest.raises(ValueError):
            rooms_for_scope(["garage"])
        with pytest.raises(ValueError):
            rooms_for_scope([])

    @pytest.mark.parametrize("kw", [
        {"bedrooms": None, "scope": ()}, {"bedrooms": None, "scope": ("garage",)}, {"bedrooms": 2, "audience": "alien"},
        {"bedrooms": 2, "age": -1}, {"bedrooms": 2, "age": True}, {"bedrooms": 2, "themes": ("x" * 40,)},
        {"bedrooms": 2, "themes": tuple("abcdefghi")}])
    def test_bad_requests_are_refused(self, kw):
        with pytest.raises(PlanError):
            FurnishingCoordinator(sim_catalog()).create_plan(FurnishingRequest(50000, **kw))

    def test_a_room_plan_stays_inside_the_budget(self):
        for budget in (8000, 20000, 50000, 200000):
            c = FurnishingCoordinator(sim_catalog())
            plan = c.create_plan(FurnishingRequest(budget, None, scope=("kids_bedroom",), audience="child"))
            assert plan.summary["spent"] <= budget and plan.summary["within_budget"]

    def test_the_request_is_in_the_plan_json(self):
        c = FurnishingCoordinator(sim_catalog())
        d = c.create_plan(FurnishingRequest(50000, None, scope=("kids_bedroom",), age=10, themes=("monster",))).to_dict()
        assert d["request"]["scope"] == ["kids_bedroom"] and d["request"]["label"] == "kid's bedroom"
        json.dumps(d)

    def test_assumptions_explain_the_childs_room(self):
        c = FurnishingCoordinator(sim_catalog())
        plan = c.create_plan(FurnishingRequest(50000, None, scope=("kids_bedroom",), age=10))
        assert any("child's room (age 10)" in a for a in plan.assumptions)


class TestMissingData:
    def test_categories_without_listings_are_flagged_and_get_no_budget_share(self, tmp_path):
        c = FurnishingCoordinator(empty_catalog(tmp_path))
        plan = c.create_plan(FurnishingRequest(50000, None, scope=("kids_bedroom",)))
        assert plan.summary["no_data_count"] == 11 and plan.summary["spent"] == 0
        for ln in plan.lines.values():
            assert ln["no_data"] and ln["status"] == "unfilled" and "no" in ln["why"].lower() and ln["search_query"]

    def test_the_money_goes_to_items_that_can_be_bought(self, tmp_path):
        snaps = SnapshotCatalog(tmp_path)
        for cat in ("mattress", "wardrobe"):
            snaps.save(CategoryListings(cat, SimulatedCatalog().listings(cat).records, ORIGIN_REAL, "t"))
        c = FurnishingCoordinator(LayeredCatalog(snaps))
        plan = c.create_plan(FurnishingRequest(50000, None, scope=("kids_bedroom",)))
        filled = [ln for ln in plan.lines.values() if ln["product"]]
        assert {ln["category"] for ln in filled} == {"mattress", "wardrobe"}
        assert round(sum(ln["ceiling"] for ln in plan.lines.values()), 2) == 50000       # the whole budget is shared by them
        assert all(ln["ceiling"] == 0 for ln in plan.lines.values() if ln["no_data"])

    def test_the_wall_decor_search_query_carries_the_theme_only_where_it_makes_sense(self):
        assert CATEGORIES["wall_decor"].search_query(("monster", "cartoon")) == "monster cartoon wall poster"
        assert CATEGORIES["study_table"].search_query(("monster",)) == "study table"

    def test_the_chat_explains_what_is_missing(self, tmp_path):
        s = HomeSession(FurnishingCoordinator(empty_catalog(tmp_path)))
        r = ChatAssistant().respond(s, USER_PROMPT)
        assert "no product listings yet" in r.text and "Search live" in r.text


class TestThemes:
    def test_a_title_that_mentions_the_theme_wins_over_a_cheaper_unthemed_one(self):
        c = FurnishingCoordinator(sim_catalog())
        plan = c.create_plan(FurnishingRequest(50000, None, scope=("kids_bedroom",), themes=("monster",)))
        decor = plan.lines["kids_bedroom.wall_decor"]
        records = c._research["wall_decor"]["records"]
        themed = [r for r in records if "monster" in r["name"].lower()]
        assert themed, "the simulated posters include a monster set"
        assert decor["product"]["theme_matches"] == ["monster"] and "monster" in decor["product"]["name"].lower()
        assert "mentions monster" in decor["why"]
        assert plan.summary["themed_items"] >= 1

    def test_with_no_matching_title_the_optimizer_order_stands_and_it_says_so(self):
        c = FurnishingCoordinator(sim_catalog())
        with_theme = c.create_plan(FurnishingRequest(50000, None, scope=("kids_bedroom",), themes=("submarine",)))
        without = FurnishingCoordinator(sim_catalog()).create_plan(FurnishingRequest(50000, None, scope=("kids_bedroom",)))
        pick = lambda plan: {k: (v["product"] or {}).get("product_id") for k, v in plan.lines.items()}
        assert pick(with_theme) == pick(without)
        assert with_theme.summary["themed_items"] == 0
        assert "cannot say any pick matches" in ChatAssistant.describe_plan(with_theme)

    def test_themes_never_break_the_budget_or_stock_rules(self):
        c = FurnishingCoordinator(sim_catalog())
        plan = c.create_plan(FurnishingRequest(9000, None, scope=("kids_bedroom",), themes=("monster", "cartoon")))
        assert plan.summary["spent"] <= 9000
        for ln in plan.lines.values():
            if ln["product"]:
                assert ln["product"]["price"] <= ln["ceiling"]
                assert (ln["status"] == "confirmed") == (ln["product"]["availability"] == "in_stock")

    def test_changing_the_theme_replans(self):
        c = FurnishingCoordinator(sim_catalog())
        c.create_plan(FurnishingRequest(50000, None, scope=("kids_bedroom",)))
        plan = c.handle_event({"type": "preference_change", "themes": ["monster"]})
        assert plan.request.themes == ("monster",) and plan.summary["themed_items"] >= 1


# ── Live search ──────────────────────────────────────────────────────────────

def settings(tmp_path, limit=3):
    return SerpApiSettings(api_key="test-key-1234567890", usage_file=tmp_path / "usage.json", lifetime_limit=limit)


def transport(seen=None, payload=None):
    body = payload if payload is not None else json.loads(FIXTURE.read_text(encoding="utf-8"))["search"]

    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(str(request.url))
        return httpx.Response(200, json=body)
    return httpx.MockTransport(handler)


class TestLiveSearcher:
    def make(self, tmp_path, limit=3, **kw):
        return LiveSearcher(settings(tmp_path, limit), SnapshotCatalog(tmp_path / "catalog"), transport=transport(**kw))

    def test_one_search_costs_one_request_and_saves_a_snapshot(self, tmp_path):
        seen = []
        searcher = self.make(tmp_path, seen=seen)
        found = searcher.search("study_table")
        assert len(seen) == 1 and searcher.used == 1 and searcher.remaining == 2
        assert found.records and SnapshotCatalog(tmp_path / "catalog").listings("study_table").records
        assert all(r["category"] == "study_table" and r["data_origin"] == ORIGIN_REAL for r in found.records)

    def test_the_query_is_the_category_term_plus_theme_where_it_applies(self, tmp_path):
        seen = []
        self.make(tmp_path, seen=seen).search("wall_decor", ("monster", "cartoon"))
        assert "monster+cartoon+wall+poster" in seen[0] or "monster%20cartoon%20wall%20poster" in seen[0]

    def test_the_lifetime_cap_is_enforced(self, tmp_path):
        searcher = self.make(tmp_path, limit=1)
        searcher.search("bookshelf")
        with pytest.raises(LiveSearchError) as err:
            searcher.search("study_table")
        assert err.value.code == "quota" and "used up" in str(err.value) and searcher.used == 1

    def test_an_already_exhausted_allowance_sends_nothing(self, tmp_path):
        seen = []
        searcher = self.make(tmp_path, limit=0, seen=seen)
        with pytest.raises(LiveSearchError):
            searcher.search("bookshelf")
        assert seen == [] and searcher.used == 0

    def test_an_empty_result_does_not_overwrite_a_good_snapshot(self, tmp_path):
        catalog = SnapshotCatalog(tmp_path / "catalog")
        good = SimulatedCatalog().listings("bookshelf")
        catalog.save(CategoryListings("bookshelf", good.records, ORIGIN_REAL, "t"))
        searcher = LiveSearcher(settings(tmp_path), catalog, transport=transport(payload={"shopping_results": []}))
        assert searcher.search("bookshelf").records == []
        assert catalog.listings("bookshelf").records == good.records

    def test_a_failing_api_becomes_a_clean_error(self, tmp_path):
        def boom(request):
            return httpx.Response(500, text="boom")
        searcher = LiveSearcher(settings(tmp_path), SnapshotCatalog(tmp_path / "c"), transport=httpx.MockTransport(boom))
        with pytest.raises(LiveSearchError) as err:
            searcher.search("bookshelf")
        assert err.value.code == "failed" and "test-key" not in str(err.value)

    def test_unknown_category(self, tmp_path):
        with pytest.raises(KeyError):
            self.make(tmp_path).search("spaceship")

    def test_the_api_key_is_never_saved(self, tmp_path):
        self.make(tmp_path).search("study_table")
        for f in (tmp_path / "catalog").glob("*.json"):
            assert "test-key" not in f.read_text(encoding="utf-8")


class FakeSearcher:
    """Writes listings the way the real one does, without any network."""

    def __init__(self, snapshots, remaining=5, error=None, empty=False):
        self.snapshots, self._remaining, self.error, self.empty, self.calls = snapshots, remaining, error, empty, []

    def status(self):
        return {"enabled": True, "remaining": self._remaining, "limit": 10, "used": 10 - self._remaining}

    def search(self, category, themes=()):
        self.calls.append((category, tuple(themes)))
        if self.error:
            raise self.error
        if self.empty:
            return CategoryListings(category, [], ORIGIN_REAL, "t", "q")
        records = [dict(r, data_origin=ORIGIN_REAL) for r in SimulatedCatalog().listings(category).records]
        found = CategoryListings(category, records, ORIGIN_REAL, "2026-11-01T00:00:00+00:00",
                                 CATEGORIES[category].search_query(tuple(themes)))
        self.snapshots.save(found)
        return found


class TestLiveSearchApi:
    def started(self, tmp_path, searcher="fake", repository=None, **fake):
        snaps = SnapshotCatalog(tmp_path)
        fake_searcher = FakeSearcher(snaps, **fake) if searcher == "fake" else searcher
        app = create_app(ApiSettings(), catalog=LayeredCatalog(snaps), gemini=None, repository=repository,
                         searcher=fake_searcher)
        client = TestClient(app)
        body = client.post("/api/sessions", json={"message": USER_PROMPT}).json()
        return client, body, fake_searcher

    def test_health_reports_the_allowance(self, tmp_path):
        client, _, _ = self.started(tmp_path, remaining=4)
        assert client.get("/api/health").json()["live_search"] == {"enabled": True, "remaining": 4, "limit": 10, "used": 6}

    def test_health_when_it_is_off(self, tmp_path):
        client, _, _ = self.started(tmp_path, searcher=None)
        assert client.get("/api/health").json()["live_search"] == {"enabled": False}

    def test_a_search_fills_the_item_and_replans(self, tmp_path):
        client, body, fake = self.started(tmp_path)
        before = body["plan"]["summary"]
        assert before["no_data_count"] == 11
        r = client.post(f"/api/sessions/{body['id']}/search", json={"category": "study_table"})
        assert r.status_code == 200 and r.json()["found"] > 0
        plan = r.json()["session"]["plan"]
        table = next(i for i in plan["items"] if i["category"] == "study_table")
        assert table["product"] is not None and not table["no_data"] and plan["summary"]["no_data_count"] == 10
        assert plan["summary"]["spent"] <= 50000 and plan["summary"]["within_budget"]
        assert fake.calls == [("study_table", ("cartoon", "monster"))]

    def test_searching_everything_fills_the_room_inside_the_budget(self, tmp_path):
        client, body, _ = self.started(tmp_path)
        for cat in ("kids_bed", "mattress", "wardrobe", "study_table", "study_chair", "bookshelf", "wall_decor",
                    "table_lamp", "rug", "bean_bag", "toy_storage"):
            r = client.post(f"/api/sessions/{body['id']}/search", json={"category": cat})
            assert r.status_code == 200
        plan = client.get(f"/api/sessions/{body['id']}").json()["plan"]
        assert plan["summary"]["no_data_count"] == 0 and plan["summary"]["spent"] <= 50000
        decor = next(i for i in plan["items"] if i["category"] == "wall_decor")
        assert decor["product"]["theme_matches"]                    # the simulated posters include monster / cartoon sets

    def test_quota_exhausted_is_a_409_with_the_reason(self, tmp_path):
        err = LiveSearchError("quota", "The SerpApi allowance is used up (10 of 10 requests).")
        client, body, _ = self.started(tmp_path, error=err, remaining=0)
        r = client.post(f"/api/sessions/{body['id']}/search", json={"category": "study_table"})
        assert r.status_code == 409 and "used up" in r.json()["detail"] and r.json()["code"] == "quota"
        assert client.get(f"/api/sessions/{body['id']}").json()["plan"]["version"] == body["plan"]["version"]

    def test_switched_off_is_a_409(self, tmp_path):
        client, body, _ = self.started(tmp_path, searcher=None)
        r = client.post(f"/api/sessions/{body['id']}/search", json={"category": "study_table"})
        assert r.status_code == 409 and "switched off" in r.json()["detail"]

    def test_a_failed_search_is_a_502_and_changes_nothing(self, tmp_path):
        client, body, _ = self.started(tmp_path, error=LiveSearchError("failed", "The search did not work"))
        r = client.post(f"/api/sessions/{body['id']}/search", json={"category": "study_table"})
        assert r.status_code == 502 and client.get(f"/api/sessions/{body['id']}").json()["plan"]["version"] == 1

    def test_an_empty_search_is_a_404(self, tmp_path):
        client, body, _ = self.started(tmp_path, empty=True)
        assert client.post(f"/api/sessions/{body['id']}/search", json={"category": "study_table"}).status_code == 404

    @pytest.mark.parametrize("category", ["sofa", "refrigerator", "spaceship", "", "x" * 60])
    def test_only_categories_in_the_plan_can_be_searched(self, tmp_path, category):
        client, body, fake = self.started(tmp_path)
        assert client.post(f"/api/sessions/{body['id']}/search", json={"category": category}).status_code == 422
        assert fake.calls == []

    def test_search_needs_a_plan_and_a_session(self, tmp_path):
        client, _, fake = self.started(tmp_path)
        sid = client.post("/api/sessions", json={}).json()["id"]
        assert client.post(f"/api/sessions/{sid}/search", json={"category": "study_table"}).status_code == 422
        assert client.post("/api/sessions/nope/search", json={"category": "study_table"}).status_code == 404
        assert fake.calls == []

    def test_a_search_survives_a_restart(self, tmp_path):
        repo = InMemoryRepository("home_projects", id_prefix="home")
        client, body, _ = self.started(tmp_path, repository=repo)
        client.post(f"/api/sessions/{body['id']}/search", json={"category": "study_table"})
        before = client.get(f"/api/sessions/{body['id']}").json()
        snaps = SnapshotCatalog(tmp_path)                               # the saved snapshot is on disk
        again = TestClient(create_app(ApiSettings(), catalog=LayeredCatalog(snaps), gemini=None, repository=repo,
                                      searcher=None)).get(f"/api/sessions/{body['id']}").json()
        pick = lambda v: {i["slot_id"]: (i["product"] or {}).get("product_id") for i in v["plan"]["items"]}
        assert pick(again) == pick(before) and again["plan"]["request"]["scope"] == ["kids_bedroom"]
        assert again["plan"]["request"]["themes"] == ["cartoon", "monster"]

    def test_create_session_with_a_room_scope(self, tmp_path):
        client, _, _ = self.started(tmp_path)
        r = client.post("/api/sessions", json={"total_budget": 30000, "scope": ["bedroom"], "audience": "teen", "age": 15,
                                               "themes": ["Gaming"]})
        assert r.status_code == 201
        req = r.json()["plan"]["request"]
        assert req["scope"] == ["bedroom"] and req["themes"] == ["gaming"] and req["age"] == 15

    @pytest.mark.parametrize("payload", [{"scope": ["bedroom"], "total_budget": 100, "audience": "alien"},
                                         {"scope": ["bedroom"], "total_budget": 100, "age": 200},
                                         {"scope": ["a"] * 9, "total_budget": 100}])
    def test_bad_room_bodies(self, tmp_path, payload):
        client, _, _ = self.started(tmp_path)
        assert client.post("/api/sessions", json=payload).status_code == 422

    def test_an_unknown_room_is_a_422_not_a_crash(self, tmp_path):
        client, _, _ = self.started(tmp_path)
        assert client.post("/api/sessions", json={"total_budget": 100, "scope": ["garage"]}).status_code == 422


# ── Suitability, ranking and plan reset ──────────────────────────────────────

class TestChildSuitability:
    @pytest.mark.parametrize("title", ["Dark Thriller Poster", "Psychological Anime Wall Art", "Horror Movie Poster",
                                       "Beer Mug Wall Decor", "18+ Adult Poster"])
    def test_obviously_mature_titles_are_flagged(self, title):
        assert mature_reason(title)

    @pytest.mark.parametrize("title", ["Cute Cartoon Monster Poster for Kids", "Dinosaur Wall Sticker", "Space Poster"])
    def test_ordinary_titles_are_not(self, title):
        assert mature_reason(title) is None

    def test_kid_signal_counts_child_words_in_the_title_only(self):
        assert kid_signal("Cute Cartoon Monster Poster, Kids Room Decor") >= 3
        assert kid_signal("Monster Anime Poster Naoki Urasawa") == 0

    def listings(self, names):
        records = SimulatedCatalog().listings("wall_decor").records[:len(names)]
        return [dict(r, name=n, price=300.0 + i, availability="in_stock") for i, (r, n) in enumerate(zip(records, names))]

    class Cat:
        def __init__(self, records):
            self.records = records

        def listings(self, category_id):
            return CategoryListings(category_id, self.records if category_id == "wall_decor" else [], ORIGIN_REAL, "t")

    def plan_for(self, names, audience):
        c = FurnishingCoordinator(self.Cat(self.listings(names)))
        plan = c.create_plan(FurnishingRequest(20000, None, scope=("kids_bedroom",), audience=audience,
                                               themes=("cartoon", "monster")))
        return c, plan.lines["kids_bedroom.wall_decor"]

    def test_mature_titles_are_dropped_for_a_child_with_the_reason_recorded(self):
        c, line = self.plan_for(["Dark Thriller Monster Poster", "Cute Cartoon Monster Poster for Kids"], "child")
        assert line["product"]["name"].startswith("Cute") and line["dropped_accessories"] == 1
        assert "not for a child" in c._research["wall_decor"]["dropped"][0]["reason"]

    def test_the_same_titles_are_not_dropped_for_an_adult(self):
        c, line = self.plan_for(["Dark Thriller Monster Poster"], "adult")
        assert line["product"] is not None and line["dropped_accessories"] == 0

    def test_a_title_matching_more_themes_beats_one_matching_fewer(self):
        _, line = self.plan_for(["Monster Anime Poster", "Cartoon Monster Poster"], None)
        assert line["product"]["theme_matches"] == ["cartoon", "monster"]

    def test_for_a_child_the_child_oriented_title_wins_among_equal_theme_matches(self):
        _, line = self.plan_for(["Monster Poster Glossy A4", "Monster Poster for Kids Room"], "child")
        assert "Kids" in line["product"]["name"]

    def test_theme_matches_carry_a_suitability_caution(self):
        _, line = self.plan_for(["Cartoon Monster Poster for Kids"], "child")
        assert "matched by title only" in line["why"]


class TestNewRequestResetsPlanState:
    def test_a_removed_item_does_not_leak_into_the_next_plan(self):
        c = FurnishingCoordinator(sim_catalog())
        c.create_plan(FurnishingRequest(200000, 2))
        c.handle_event({"type": "remove_item", "slot_id": "living_room.sofa"})
        plan = c.create_plan(FurnishingRequest(200000, 2))
        assert "living_room.sofa" in plan.lines

    def test_a_choice_does_not_leak_into_the_next_plan(self):
        c = FurnishingCoordinator(sim_catalog())
        plan = c.create_plan(FurnishingRequest(200000, 2))
        alt = plan.lines["living_room.sofa"]["alternatives"][0]["product_id"]
        c.handle_event({"type": "choose_product", "slot_id": "living_room.sofa", "product_id": alt})
        again = c.create_plan(FurnishingRequest(200000, 2))
        assert not again.lines["living_room.sofa"]["user_choice"]

    def test_the_audience_applies_to_the_new_plan_not_the_old_one(self):
        class Spy:
            def listings(self, category_id):
                base = SimulatedCatalog().listings(category_id)
                if category_id == "wall_decor":
                    base.records[0] = dict(base.records[0], name="Dark Thriller Poster")
                return base
        c = FurnishingCoordinator(Spy())
        c.create_plan(FurnishingRequest(50000, None, scope=("kids_bedroom",), audience=None))
        assert c._research["wall_decor"]["dropped"] == []
        c.create_plan(FurnishingRequest(50000, None, scope=("kids_bedroom",), audience="child"))
        assert len(c._research["wall_decor"]["dropped"]) == 1          # re-researched under the new audience

    def test_the_cart_survives_a_new_request(self):
        c = FurnishingCoordinator(sim_catalog())
        plan = c.create_plan(FurnishingRequest(200000, 2))
        sofa = plan.lines["living_room.sofa"]["product"]["product_id"]
        c.handle_event({"type": "cart_add", "product_id": sofa})
        c.create_plan(FurnishingRequest(200000, 2))
        assert c.cart == {sofa: 1}


# ── The real captured data (skipped on a fresh clone) ────────────────────────

KIDS = ("kids_bed", "mattress", "wardrobe", "study_table", "study_chair", "bookshelf", "wall_decor", "table_lamp",
        "rug", "bean_bag", "toy_storage")
HAVE_KIDS = all(SnapshotCatalog().listings(c).records for c in KIDS)


@pytest.mark.skipif(not HAVE_KIDS, reason="no real kid's-room snapshots (run src/capture_catalog.py)")
class TestRealKidsRoom:
    @pytest.fixture(scope="class")
    def plan(self):
        c = FurnishingCoordinator(LayeredCatalog(SnapshotCatalog()))
        return c, c.create_plan(FurnishingRequest(50000, None, scope=("kids_bedroom",), audience="child", age=10,
                                                  themes=("cartoon", "monster")))

    def test_every_item_is_filled_from_real_listings_inside_the_budget(self, plan):
        _, p = plan
        assert p.summary["no_data_count"] == 0 and p.summary["unfilled_count"] == 0
        assert p.summary["spent"] <= 50000 and len(p.lines) == 11
        index = {r["product_id"]: r for c in KIDS for r in SnapshotCatalog().listings(c).records}
        for ln in p.lines.values():
            real = index[ln["product"]["product_id"]]                 # KeyError would mean an invented product
            assert ln["product"]["name"] == real["name"] and ln["product"]["price"] == real["price"]
            assert ln["product"]["data_origin"] == "serpapi" and ln["product"]["buy_url"] == real["listing_url"]
            assert ln["product"]["price"] <= ln["ceiling"]

    def test_the_themed_picks_really_mention_the_theme_and_are_for_children(self, plan):
        _, p = plan
        decor = p.lines["kids_bedroom.wall_decor"]["product"]
        assert decor["theme_matches"] and "kids" in decor["name"].lower()
        assert not CATEGORIES["wall_decor"].accessory_reason(decor["name"], decor["price"])
        from smartnest.furnishing.categories import mature_reason
        assert all(mature_reason(ln["product"]["name"]) is None for ln in p.lines.values())

    def test_the_search_queries_that_produced_the_data_are_recorded(self):
        assert SnapshotCatalog().listings("wall_decor").query == "monster cartoon wall poster"
        assert SnapshotCatalog().listings("study_table").query == "study table"

    def test_no_stock_is_claimed_without_evidence(self, plan):
        _, p = plan
        index = {r["product_id"]: r for c in KIDS for r in SnapshotCatalog().listings(c).records}
        for ln in p.lines.values():
            assert (ln["status"] == "confirmed") == (index[ln["product"]["product_id"]]["availability"] == "in_stock")
