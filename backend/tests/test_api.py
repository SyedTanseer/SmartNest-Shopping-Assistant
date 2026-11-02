"""HTTP API: validation, sessions, chat, events, analytics, persistence, rate limiting and information hygiene."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from smartnest.api.app import ApiSettings, create_app
from smartnest.database import InMemoryRepository
from smartnest.furnishing import LayeredCatalog, SimulatedCatalog, SnapshotCatalog

DEMO = ("I just moved into a 2BHK apartment. My furnishing budget is ₹2 lakh. I prefer minimalist furniture. "
        "Help me furnish the entire apartment.")


def make_app(repository=None, **settings):
    catalog = LayeredCatalog(SnapshotCatalog("/nonexistent"), SimulatedCatalog(), allow_simulated=True)
    return create_app(ApiSettings(**settings), catalog=catalog, gemini=None, repository=repository)


@pytest.fixture
def client():
    return TestClient(make_app())


@pytest.fixture
def started(client):
    r = client.post("/api/sessions", json={"message": DEMO})
    assert r.status_code == 201
    return client, r.json()


class TestStatus:
    def test_health(self, client):
        body = client.get("/api/health").json()
        assert body["status"] == "ok" and body["gemini"] is False and body["persistence"] == "memory"

    def test_catalog_reports_origin_per_category(self, client):
        cats = client.get("/api/catalog").json()["categories"]
        assert len(cats) == 16 and all(c["origin"] == "simulated" and c["listings"] > 0 for c in cats)


class TestSessions:
    def test_create_from_a_message(self, started):
        _, body = started
        assert body["has_plan"] and body["plan"]["summary"]["within_budget"] and len(body["plan"]["items"]) == 10
        assert [m["role"] for m in body["history"][:2]] == ["user", "assistant"]

    def test_create_from_fields(self, client):
        r = client.post("/api/sessions", json={"total_budget": 150000, "bedrooms": 1, "style": "Modern"})
        assert r.status_code == 201 and r.json()["plan"]["request"]["style"] == "modern"
        assert len(r.json()["plan"]["items"]) == 7

    def test_create_empty_then_chat(self, client):
        sid = client.post("/api/sessions", json={}).json()["id"]
        assert client.get(f"/api/sessions/{sid}").json()["has_plan"] is False
        r = client.post(f"/api/sessions/{sid}/messages", json={"message": DEMO}).json()
        assert r["reply"]["kind"] == "new_request" and r["session"]["has_plan"]

    def test_unknown_session(self, client):
        assert client.get("/api/sessions/nope").status_code == 404
        assert client.post("/api/sessions/nope/messages", json={"message": "hi"}).status_code == 404

    def test_session_ids_are_unguessable(self, client):
        ids = {client.post("/api/sessions", json={}).json()["id"] for _ in range(20)}
        assert len(ids) == 20 and all(len(i) >= 16 for i in ids)

    @pytest.mark.parametrize("payload", [
        {"total_budget": -5, "bedrooms": 2}, {"total_budget": 0, "bedrooms": 2}, {"total_budget": 1e12, "bedrooms": 2},
        {"total_budget": "lots", "bedrooms": 2}, {"total_budget": 1000, "bedrooms": 0}, {"total_budget": 1000, "bedrooms": 9},
        {"bedrooms": 2.5, "total_budget": 1000}, {"message": "x" * 1001}, {"surprise": 1}])
    def test_bad_creation_bodies(self, client, payload):
        assert client.post("/api/sessions", json=payload).status_code == 422


class TestChat:
    def test_questions_and_changes(self, started):
        client, body = started
        sid = body["id"]
        r = client.post(f"/api/sessions/{sid}/messages", json={"message": "I don't need a TV unit"}).json()
        assert r["reply"]["changed"] and len(r["session"]["plan"]["items"]) == 9
        r = client.post(f"/api/sessions/{sid}/messages", json={"message": "why this sofa?"}).json()
        assert not r["reply"]["changed"] and "sofa" in r["reply"]["text"].lower()

    @pytest.mark.parametrize("bad", [{"message": ""}, {}, {"message": "x" * 1001}, {"message": 5}, {"msg": "hi"}])
    def test_bad_chat_bodies(self, started, bad):
        client, body = started
        assert client.post(f"/api/sessions/{body['id']}/messages", json=bad).status_code == 422

    def test_not_json(self, started):
        client, body = started
        assert client.post(f"/api/sessions/{body['id']}/messages", content=b"not json",
                           headers={"content-type": "application/json"}).status_code == 422


class TestEvents:
    def test_budget_change_and_stock(self, started):
        client, body = started
        sid = body["id"]
        r = client.post(f"/api/sessions/{sid}/events", json={"type": "budget_change", "total_budget": 150000})
        assert r.status_code == 200 and r.json()["plan"]["summary"]["spent"] <= 150000
        pid = next(i["product"]["product_id"] for i in r.json()["plan"]["items"] if i["category"] == "sofa" and i["product"])
        r2 = client.post(f"/api/sessions/{sid}/events", json={"type": "out_of_stock", "product_id": pid})
        assert r2.status_code == 200 and r2.json()["plan"]["version"] > r.json()["plan"]["version"]

    @pytest.mark.parametrize("event", [
        {"type": "explode"}, {"type": "out_of_stock", "product_id": "NOPE"}, {"type": "budget_change"},
        {"type": "budget_change", "total_budget": -1}, {"type": "remove_item", "slot_id": "x.y"},
        {"type": "choose_product", "slot_id": "living_room.sofa", "product_id": "NOPE"}, {}])
    def test_bad_events_leave_the_plan_untouched(self, started, event):
        client, body = started
        sid = body["id"]
        assert client.post(f"/api/sessions/{sid}/events", json=event).status_code == 422
        assert client.get(f"/api/sessions/{sid}").json()["plan"]["version"] == body["plan"]["version"]

    def test_event_without_a_plan(self, client):
        sid = client.post("/api/sessions", json={}).json()["id"]
        r = client.post(f"/api/sessions/{sid}/events", json={"type": "budget_change", "total_budget": 5})
        assert r.status_code == 422 and "no plan" in r.json()["detail"]

    def test_choose_product_from_options(self, started):
        client, body = started
        sid = body["id"]
        opts = client.get(f"/api/sessions/{sid}/items/living_room.sofa/options").json()
        target = next(o for o in opts["options"] if o["fits_budget"] and not o["is_pick"])
        r = client.post(f"/api/sessions/{sid}/events", json={"type": "choose_product", "slot_id": "living_room.sofa",
                                                              "product_id": target["product_id"]})
        if r.status_code == 200:
            sofa = next(i for i in r.json()["plan"]["items"] if i["slot_id"] == "living_room.sofa")
            assert sofa["product"]["product_id"] == target["product_id"] and sofa["user_choice"]
            assert r.json()["plan"]["summary"]["within_budget"]
        else:                                                  # the optimizer may refuse (e.g. stock) - never a 500
            assert r.status_code == 422


class TestViews:
    def test_rooms(self, started):
        client, body = started
        rooms = client.get(f"/api/sessions/{body['id']}/rooms").json()["rooms"]
        assert [r["label"] for r in rooms] == ["Living room", "Bedroom 1", "Bedroom 2", "Kitchen"]
        assert sum(r["spent"] for r in rooms) == pytest.approx(body["plan"]["summary"]["spent"])

    def test_budget_analytics_add_up(self, started):
        client, body = started
        b = client.get(f"/api/sessions/{body['id']}/budget").json()
        assert b["total"] == 200000 and b["spent"] + b["remaining"] == pytest.approx(b["total"])
        assert sum(x["value"] for x in b["by_room"]) == pytest.approx(b["spent"])
        assert sum(x["value"] for x in b["by_category"]) == pytest.approx(b["spent"])
        assert sum(x["allocated"] for x in b["allocation_vs_spend"]) == pytest.approx(b["total"])

    def test_budget_before_a_plan(self, client):
        sid = client.post("/api/sessions", json={}).json()["id"]
        assert client.get(f"/api/sessions/{sid}/budget").status_code == 422

    def test_options_flag_what_fits(self, started):
        client, body = started
        opts = client.get(f"/api/sessions/{body['id']}/items/kitchen.refrigerator/options").json()
        assert opts["options"] == sorted(opts["options"], key=lambda o: o["price"])
        assert all(o["fits_budget"] == (o["price"] <= opts["budget_available"]) for o in opts["options"])
        assert client.get(f"/api/sessions/{body['id']}/items/nowhere.sofa/options").status_code == 422

    def test_shortlist_and_compare(self, started):
        client, body = started
        sid = body["id"]
        sofa = next(i for i in body["plan"]["items"] if i["category"] == "sofa")
        ids = [sofa["product"]["product_id"], sofa["alternatives"][0]["product_id"]]
        for pid in ids:
            assert client.post(f"/api/sessions/{sid}/shortlist", json={"product_id": pid}).status_code == 200
        assert client.get(f"/api/sessions/{sid}").json()["shortlist"] == ids
        cmp = client.post(f"/api/sessions/{sid}/compare", json={"product_ids": ids}).json()["products"]
        assert [p["product_id"] for p in cmp] == ids and cmp[0]["is_pick"] and not cmp[1]["is_pick"]
        assert client.post(f"/api/sessions/{sid}/shortlist", json={"product_id": "NOPE"}).status_code == 422
        assert client.post(f"/api/sessions/{sid}/compare", json={"product_ids": ["a"]}).status_code == 422
        assert client.post(f"/api/sessions/{sid}/compare", json={"product_ids": ["a", "b"]}).status_code == 422

    def test_activity_log_and_roster(self, started):
        client, body = started
        a = client.get(f"/api/sessions/{body['id']}/activity").json()
        assert {x["id"] for x in a["agents"]} == {"coordinator", "space_planning", "product_research", "comparison", "budget"}
        assert a["messages"] > 0 and a["unverified_messages"] == 0 and a["security"]["accepted"] >= a["messages"]
        last = a["events"][-1]["seq"]
        assert client.get(f"/api/sessions/{body['id']}/activity?since={last}").json()["events"] == []
        assert all(e["verified"] for e in a["events"] if e["kind"] == "message")


class TestPersistence:
    def test_session_survives_a_restart_and_rebuilds_identically(self):
        repo = InMemoryRepository("home_projects", id_prefix="home")
        c1 = TestClient(make_app(repository=repo))
        sid = c1.post("/api/sessions", json={"message": DEMO}).json()["id"]
        c1.post(f"/api/sessions/{sid}/messages", json={"message": "I don't need a TV unit"})
        c1.post(f"/api/sessions/{sid}/messages", json={"message": "reduce the budget to 150000"})
        before = c1.get(f"/api/sessions/{sid}").json()
        c2 = TestClient(make_app(repository=repo))                       # a new process with the same database
        after = c2.get(f"/api/sessions/{sid}").json()

        def picks(view):
            return {i["slot_id"]: (i["product"] or {}).get("product_id") for i in view["plan"]["items"]}
        assert picks(after) == picks(before) and after["plan"]["summary"]["spent"] == before["plan"]["summary"]["spent"]
        assert len(after["history"]) == len(before["history"])

    def test_corrupt_stored_session_is_treated_as_missing(self):
        repo = InMemoryRepository("home_projects", id_prefix="home")
        repo.upsert("bad", {"request": {"total_budget": -1, "bedrooms": 2}, "events": []})
        assert TestClient(make_app(repository=repo)).get("/api/sessions/bad").status_code == 404

    def test_a_failing_database_does_not_break_requests(self):
        class Broken:
            def upsert(self, *a, **k): raise RuntimeError("down")
            def get(self, *a, **k): raise RuntimeError("down")
        c = TestClient(make_app(repository=Broken()))
        assert c.post("/api/sessions", json={"message": DEMO}).status_code == 201


class TestHardening:
    def test_rate_limit(self):
        c = TestClient(make_app(requests_per_minute=5))
        codes = [c.post("/api/sessions", json={}).status_code for _ in range(8)]
        assert codes[:5] == [201] * 5 and set(codes[5:]) == {429}
        assert c.get("/api/health").status_code == 200            # reads are not throttled

    def test_rate_limit_is_per_visitor_behind_the_vercel_proxy(self, monkeypatch):
        monkeypatch.setenv("VERCEL", "1")
        c = TestClient(make_app(requests_per_minute=2))
        a, b = {"x-forwarded-for": "203.0.113.5, 10.0.0.1"}, {"x-forwarded-for": "198.51.100.7"}
        assert [c.post("/api/sessions", json={}, headers=a).status_code for _ in range(3)] == [201, 201, 429]
        assert c.post("/api/sessions", json={}, headers=b).status_code == 201        # another visitor is unaffected

    def test_the_forwarded_header_is_ignored_off_vercel(self, monkeypatch):
        monkeypatch.delenv("VERCEL", raising=False)
        c = TestClient(make_app(requests_per_minute=2))
        codes = [c.post("/api/sessions", json={}, headers={"x-forwarded-for": f"198.51.100.{i}"}).status_code
                 for i in range(3)]
        assert codes == [201, 201, 429]                       # a visitor cannot dodge the limit by faking the header

    def test_cors_allows_only_the_dev_origins(self, client):
        ok = client.options("/api/sessions", headers={"Origin": "http://localhost:5173",
                                                       "Access-Control-Request-Method": "POST"})
        assert ok.headers.get("access-control-allow-origin") == "http://localhost:5173"
        bad = client.options("/api/sessions", headers={"Origin": "http://evil.example",
                                                        "Access-Control-Request-Method": "POST"})
        assert "access-control-allow-origin" not in bad.headers

    def test_internal_errors_do_not_leak_details(self):
        app = make_app()
        app.state.assistant.respond = lambda *a, **k: 1 / 0
        c = TestClient(app, raise_server_exceptions=False)
        sid = c.post("/api/sessions", json={}).json()["id"]
        r = c.post(f"/api/sessions/{sid}/messages", json={"message": "hello"})
        assert r.status_code == 500 and r.json() == {"detail": "internal error"}

    def test_responses_contain_no_secrets(self, started):
        client, body = started
        from dotenv import dotenv_values
        from pathlib import Path
        secrets = [v for k, v in dotenv_values(Path(__file__).resolve().parents[1] / ".env").items()
                   if v and len(v) > 12 and any(t in k for t in ("KEY", "URI", "PASSWORD"))]
        blob = json.dumps(body) + client.get("/api/health").text + client.get(f"/api/sessions/{body['id']}/activity").text
        for secret in secrets:
            assert secret not in blob

    def test_simulated_products_never_expose_a_buy_link(self, started):
        _, body = started
        for item in body["plan"]["items"]:
            assert item["product"]["simulated"] and item["product"]["buy_url"] is None
            assert all(a["buy_url"] is None for a in item["alternatives"])


class TestSettingsFromEnvFile:
    def test_the_env_file_is_read_and_the_real_environment_wins(self, tmp_path, monkeypatch):
        f = tmp_path / ".env"
        f.write_text("SMARTNEST_ALLOW_LIVE_SEARCH=1\nSMARTNEST_USE_GEMINI=0\nSMARTNEST_CORS_ORIGINS=http://a.example\n",
                     encoding="utf-8")
        monkeypatch.delenv("SMARTNEST_ALLOW_LIVE_SEARCH", raising=False)
        monkeypatch.setenv("SMARTNEST_USE_GEMINI", "1")
        s = ApiSettings.from_env(env_file=f)
        assert s.live_search is True and s.use_gemini is True and s.cors_origins == ["http://a.example"]

    def test_an_explicit_environ_ignores_any_file(self, tmp_path):
        f = tmp_path / ".env"
        f.write_text("SMARTNEST_ALLOW_LIVE_SEARCH=1\n", encoding="utf-8")
        assert ApiSettings.from_env({}, env_file=f).live_search is False

    def test_a_missing_file_is_fine_and_live_search_defaults_off(self, tmp_path, monkeypatch):
        monkeypatch.delenv("SMARTNEST_ALLOW_LIVE_SEARCH", raising=False)
        assert ApiSettings.from_env(env_file=tmp_path / "nope.env").live_search is False
