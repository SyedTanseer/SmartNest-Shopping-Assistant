"""Tests for ProductCatalog — eligibility filtering, data integrity.

Verifies:
    - Catalog construction and retrieval.
    - get_in_stock_products filters correctly.
    - find_eligible_products with budget constraint.
    - find_eligible_products with width constraint.
    - find_eligible_products with style constraint.
    - find_eligible_products with seating capacity.
    - find_eligible_products with max_width override.
    - Default catalogues (Agent A, Agent B) are independent.
    - Products are clearly simulated (no URLs).
"""

from __future__ import annotations

import pytest

from smartnest.cnp.product_catalog import (
    ProductCatalog,
    catalog_agent_a,
    catalog_agent_b,
)


class TestCatalogConstruction:
    def test_get_all_products(self):
        cat = ProductCatalog("test", [
            {"product_id": "P1", "name": "Sofa 1", "price": 10000, "width_cm": 200},
            {"product_id": "P2", "name": "Sofa 2", "price": 20000, "width_cm": 180},
        ])
        assert len(cat.get_all_products()) == 2

    def test_get_product_by_id(self):
        cat = ProductCatalog("test", [
            {"product_id": "P1", "name": "Sofa 1", "price": 10000, "width_cm": 200},
        ])
        p = cat.get_product("P1")
        assert p is not None
        assert p["product_id"] == "P1"
        assert p["price"] == 10000

    def test_get_product_not_found(self):
        cat = ProductCatalog("test", [])
        assert cat.get_product("NONEXISTENT") is None

    def test_agent_id(self):
        cat = ProductCatalog("my_agent", [])
        assert cat.agent_id == "my_agent"


class TestInStockFilter:
    def test_in_stock_only(self):
        cat = ProductCatalog("test", [
            {"product_id": "P1", "price": 100, "width_cm": 100, "availability": "in_stock"},
            {"product_id": "P2", "price": 100, "width_cm": 100, "availability": "out_of_stock"},
        ])
        in_stock = cat.get_in_stock_products()
        assert len(in_stock) == 1
        assert in_stock[0]["product_id"] == "P1"


class TestEligibilityFiltering:
    @pytest.fixture()
    def catalog(self):
        return ProductCatalog("test", [
            {"product_id": "A", "price": 20000, "width_cm": 200, "style": "minimalist",
             "seating_capacity": 3, "availability": "in_stock"},
            {"product_id": "B", "price": 28000, "width_cm": 230, "style": "minimalist",
             "seating_capacity": 3, "availability": "in_stock"},
            {"product_id": "C", "price": 25000, "width_cm": 195, "style": "modern",
             "seating_capacity": 2, "availability": "in_stock"},
            {"product_id": "D", "price": 15000, "width_cm": 180, "style": "minimalist",
             "seating_capacity": 3, "availability": "out_of_stock"},
        ])

    def test_budget_filter(self, catalog):
        eligible = catalog.find_eligible_products(max_budget=25000, room_width=250)
        ids = {p["product_id"] for p in eligible}
        assert "A" in ids
        assert "C" in ids
        assert "B" not in ids  # 28000 > 25000

    def test_width_filter(self, catalog):
        eligible = catalog.find_eligible_products(max_budget=100000, room_width=210)
        ids = {p["product_id"] for p in eligible}
        assert "A" in ids
        assert "C" in ids
        assert "B" not in ids  # 230 > 210

    def test_style_filter(self, catalog):
        eligible = catalog.find_eligible_products(
            max_budget=100000, room_width=250, preferred_style="minimalist"
        )
        ids = {p["product_id"] for p in eligible}
        assert "A" in ids
        assert "C" not in ids  # style is "modern"

    def test_seating_capacity_filter(self, catalog):
        eligible = catalog.find_eligible_products(
            max_budget=100000, room_width=250, seating_capacity=3
        )
        ids = {p["product_id"] for p in eligible}
        assert "A" in ids
        assert "C" not in ids  # seating_capacity=2

    def test_out_of_stock_excluded(self, catalog):
        eligible = catalog.find_eligible_products(max_budget=100000, room_width=250)
        ids = {p["product_id"] for p in eligible}
        assert "D" not in ids  # out_of_stock

    def test_max_width_override(self, catalog):
        """max_width takes precedence over room_width if provided."""
        eligible = catalog.find_eligible_products(
            max_budget=100000, room_width=250, max_width=200
        )
        ids = {p["product_id"] for p in eligible}
        assert "A" in ids  # 200 <= 200
        assert "B" not in ids  # 230 > 200

    def test_no_eligible_products(self, catalog):
        eligible = catalog.find_eligible_products(max_budget=1000, room_width=100)
        assert len(eligible) == 0

    def test_combined_constraints(self, catalog):
        eligible = catalog.find_eligible_products(
            max_budget=22000, room_width=210, preferred_style="minimalist",
            seating_capacity=3
        )
        assert len(eligible) == 1
        assert eligible[0]["product_id"] == "A"


class TestDefaultCatalogues:
    def test_agent_a_has_three_products(self):
        cat = catalog_agent_a()
        assert len(cat.get_all_products()) == 3
        assert cat.agent_id == "research_agent_a"

    def test_agent_b_has_three_products(self):
        cat = catalog_agent_b()
        assert len(cat.get_all_products()) == 3
        assert cat.agent_id == "research_agent_b"

    def test_catalogues_are_independent(self):
        cat_a = catalog_agent_a()
        cat_b = catalog_agent_b()
        a_ids = {p["product_id"] for p in cat_a.get_all_products()}
        b_ids = {p["product_id"] for p in cat_b.get_all_products()}
        assert a_ids.isdisjoint(b_ids)

    def test_all_products_simulated_no_urls(self):
        """Products must not contain real retailer URLs."""
        for cat in [catalog_agent_a(), catalog_agent_b()]:
            for p in cat.get_all_products():
                for key, val in p.items():
                    if isinstance(val, str):
                        assert "http" not in val.lower(), f"URL found in {p['product_id']}.{key}"


# ── Fixes: unknown width, availability states ───────────────────────────────

class TestEligibilityFixes:
    def test_unknown_room_width_applies_no_width_filter(self):
        catalog = catalog_agent_a()
        eligible = catalog.find_eligible_products(max_budget=30000, room_width=None)
        assert {p["product_id"] for p in eligible} == {"SOFA_A1", "SOFA_A2", "SOFA_A3"}

    def test_low_stock_is_eligible_but_out_of_stock_is_not(self):
        catalog = ProductCatalog("x", [
            {"product_id": "L", "price": 100, "width_cm": 100, "availability": "low_stock"},
            {"product_id": "O", "price": 100, "width_cm": 100, "availability": "out_of_stock"},
            {"product_id": "D", "price": 100, "width_cm": 100, "availability": "discontinued"},
            {"product_id": "N", "price": 100, "width_cm": 100},
        ])
        eligible = catalog.find_eligible_products(max_budget=1000, room_width=200)
        assert [p["product_id"] for p in eligible] == ["L"]

    def test_eligible_results_are_copies(self):
        catalog = catalog_agent_a()
        eligible = catalog.find_eligible_products(max_budget=30000, room_width=None)
        eligible[0]["price"] = 1
        assert catalog.get_product(eligible[0]["product_id"])["price"] != 1
