"""End-to-end on the REAL saved SerpApi snapshots: the final demo scenario through the HTTP API.

Skipped when ``backend/data/catalog`` has no snapshots (a fresh clone); run ``python src/capture_catalog.py`` to create
them.  These tests make no network request and spend no SerpApi quota.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from smartnest.api.app import ApiSettings, create_app
from smartnest.furnishing import CATEGORIES, SnapshotCatalog

SNAP = SnapshotCatalog()
CAPTURED = ("sofa", "tv_unit", "bed", "mattress", "wardrobe", "dining_table", "refrigerator")   # captured with SerpApi
HAVE = all(SNAP.listings(c).records for c in CAPTURED)
pytestmark = pytest.mark.skipif(not HAVE, reason="no real catalog snapshots (run src/capture_catalog.py)")

DEMO = ("I just moved into a 2BHK apartment. My furnishing budget is ₹2 lakh. I prefer minimalist furniture. "
        "Help me furnish the entire apartment.")


@pytest.fixture(scope="module")
def started():
    client = TestClient(create_app(ApiSettings(use_gemini=False), gemini=None, repository=None))
    body = client.post("/api/sessions", json={"message": DEMO}).json()
    return client, body


def real_index():
    index = {}
    for cat in CAPTURED:
        for r in SNAP.listings(cat).records:
            index[r["product_id"]] = r
    return index


def test_the_demo_scenario_furnishes_the_whole_home_within_budget(started):
    _, body = started
    plan = body["plan"]
    assert plan["request"]["bedrooms"] == 2 and plan["request"]["total_budget"] == 200000
    assert plan["request"]["style"] == "minimalist"
    assert len(plan["items"]) == 10 and plan["summary"]["unfilled_count"] == 0
    assert plan["summary"]["spent"] <= 200000 and plan["summary"]["within_budget"]
    assert {i["room"] for i in plan["items"]} == {"Living room", "Bedroom 1", "Bedroom 2", "Kitchen"}


def test_every_recommended_product_is_a_real_snapshot_listing_with_unaltered_facts(started):
    _, body = started
    index = real_index()
    for item in body["plan"]["items"]:
        p = item["product"]
        real = index[p["product_id"]]                       # raises KeyError if the product was invented
        assert p["name"] == real["name"] and p["price"] == real["price"] and p["retailer"] == real["retailer"]
        assert p["data_origin"] == "serpapi" and not p["simulated"]
        assert p["rating"] == real["rating"]                # a missing rating stays missing, never filled in
        assert p["buy_url"] == real["listing_url"]


def test_links_are_never_mislabelled(started):
    _, body = started
    for item in body["plan"]["items"]:
        p = item["product"]
        assert p["buy_url"].startswith("https://")
        if p["url_type"] == "retailer":
            assert "google." not in p["buy_url"] and p["link_label"] == "Retailer page"
        else:
            assert p["link_label"] == "Google Shopping page" and "google." in p["buy_url"]


def test_stock_is_only_confirmed_where_the_listing_says_so(started):
    _, body = started
    index = real_index()
    for item in body["plan"]["items"]:
        real = index[item["product"]["product_id"]]
        assert (item["status"] == "confirmed") == (real["availability"] == "in_stock")


def test_chosen_prices_respect_their_ceilings_and_sum_to_the_summary(started):
    _, body = started
    items = body["plan"]["items"]
    assert all(i["product"]["price"] <= i["ceiling"] for i in items)
    assert sum(i["ceiling"] for i in items) == pytest.approx(200000, abs=0.01)
    assert sum(i["product"]["price"] for i in items) == pytest.approx(body["plan"]["summary"]["spent"])


def test_no_accessory_is_recommended(started):
    _, body = started
    for item in body["plan"]["items"]:
        assert CATEGORIES[item["category"]].accessory_reason(item["product"]["name"], item["product"]["price"]) is None


def test_a_full_conversation_stays_consistent():
    client = TestClient(create_app(ApiSettings(use_gemini=False), gemini=None, repository=None))
    sid = client.post("/api/sessions", json={"message": DEMO}).json()["id"]

    def say(text):
        return client.post(f"/api/sessions/{sid}/messages", json={"message": text}).json()

    r = say("I don't need a TV unit")
    assert len(r["session"]["plan"]["items"]) == 9
    r = say("the fridge is out of stock")
    fridge = next(i for i in r["session"]["plan"]["items"] if i["category"] == "refrigerator")
    assert fridge["product"] is not None and fridge["product"]["product_id"] in real_index()
    r = say("reduce the budget to 1 lakh")
    plan = r["session"]["plan"]
    assert plan["summary"]["spent"] <= 100000 and plan["summary"]["within_budget"]
    r = say("increase the budget to 3 lakh")
    assert r["session"]["plan"]["summary"]["spent"] <= 300000
    transitions = client.get(f"/api/sessions/{sid}").json()["transitions"]
    assert {t["kind"] for t in transitions} >= {"adopted", "replaced"}
    activity = client.get(f"/api/sessions/{sid}/activity").json()
    assert activity["unverified_messages"] == 0 and activity["messages"] > 30


def test_the_api_makes_no_serpapi_requests():
    from smartnest.integrations.products.serpapi.client import UsageLedger
    from smartnest.integrations.products.serpapi.config import SerpApiSettings
    ledger = UsageLedger(SerpApiSettings.__dataclass_fields__["usage_file"].default, 10 ** 9)
    before = ledger.used
    client = TestClient(create_app(ApiSettings(use_gemini=False), gemini=None, repository=None))
    client.post("/api/sessions", json={"message": DEMO})
    assert UsageLedger(SerpApiSettings.__dataclass_fields__["usage_file"].default, 10 ** 9).used == before


def test_snapshot_responses_contain_no_credentials():
    blob = json.dumps({c: SNAP.listings(c).records for c in CAPTURED})
    assert "api_key" not in blob.lower() and "serpapi.com" not in blob
