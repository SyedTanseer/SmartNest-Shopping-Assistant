"""Tests for each SmartNest domain rule.

Each rule is tested for:
    - Positive case: condition matches and action produces correct result type.
    - Negative case: condition does not match.
    - Malformed payload: condition handles missing/None values safely.
"""

from __future__ import annotations

import pytest

from smartnest.core.models import (
    ActionResultType,
    ActionLog,
    EnvironmentalEvent,
    EventType,
)
from smartnest.reasoning.rules import (
    budget_exceeded_rule,
    furniture_incompatibility_rule,
    missing_information_rule,
    product_unavailable_rule,
    valid_product_rule,
)


# ── Helpers ────────────────────────────────────────────────────────────────

def _make_event(event_type: EventType, payload: dict) -> EnvironmentalEvent:
    return EnvironmentalEvent(
        event_id="test_evt",
        event_type=event_type,
        source="test",
        payload=payload,
    )


# ── Rule 1: Budget Exceeded ──────────────────────────────────────────────────

class TestBudgetExceededRule:
    def test_positive_budget_exceeded(self):
        rule = budget_exceeded_rule()
        event = _make_event(EventType.BUDGET_UPDATE, {
            "total_budget": 200_000,
            "estimated_expenditure": 218_000,
        })
        assert rule.evaluate(event) is True
        log = rule.execute(event)
        assert log.result_type == ActionResultType.WARNING
        assert "Budget exceeded" in log.result_message
        assert log.result_details["overrun"] == 18_000

    def test_negative_budget_within_limit(self):
        rule = budget_exceeded_rule()
        event = _make_event(EventType.BUDGET_UPDATE, {
            "total_budget": 200_000,
            "estimated_expenditure": 150_000,
        })
        assert rule.evaluate(event) is False

    def test_malformed_missing_budget_fields(self):
        rule = budget_exceeded_rule()
        event = _make_event(EventType.BUDGET_UPDATE, {})
        assert rule.evaluate(event) is False

    def test_wrong_event_type(self):
        rule = budget_exceeded_rule()
        event = _make_event(EventType.PRODUCT_AVAILABILITY, {
            "total_budget": 200_000,
            "estimated_expenditure": 218_000,
        })
        assert rule.evaluate(event) is False


# ── Rule 2: Furniture Incompatibility ───────────────────────────────────────

class TestFurnitureIncompatibilityRule:
    def test_positive_product_too_wide(self):
        rule = furniture_incompatibility_rule()
        event = _make_event(EventType.FURNITURE_COMPATIBILITY, {
            "product_name": "Three-seater sofa",
            "product_width": 230,
            "room_width": 210,
        })
        assert rule.evaluate(event) is True
        log = rule.execute(event)
        assert log.result_type == ActionResultType.REJECTION
        assert "Three-seater sofa" in log.result_message
        assert log.result_details["excess"] == 20

    def test_negative_product_fits(self):
        rule = furniture_incompatibility_rule()
        event = _make_event(EventType.FURNITURE_COMPATIBILITY, {
            "product_name": "Single chair",
            "product_width": 60,
            "room_width": 210,
        })
        assert rule.evaluate(event) is False

    def test_malformed_missing_widths(self):
        rule = furniture_incompatibility_rule()
        event = _make_event(EventType.FURNITURE_COMPATIBILITY, {
            "product_name": "Sofa",
        })
        assert rule.evaluate(event) is False


# ── Rule 3: Product Unavailable ─────────────────────────────────────────────

class TestProductUnavailableRule:
    def test_positive_out_of_stock(self):
        rule = product_unavailable_rule()
        event = _make_event(EventType.PRODUCT_AVAILABILITY, {
            "product_name": "Refrigerator REF_001",
            "availability": "out_of_stock",
        })
        assert rule.evaluate(event) is True
        log = rule.execute(event)
        assert log.result_type == ActionResultType.FLAG
        assert "alternative" in log.result_message.lower()
        assert log.result_details["alternative_required"] is True

    def test_negative_in_stock(self):
        rule = product_unavailable_rule()
        event = _make_event(EventType.PRODUCT_AVAILABILITY, {
            "product_name": "Refrigerator REF_001",
            "availability": "in_stock",
        })
        assert rule.evaluate(event) is False

    def test_malformed_missing_availability(self):
        rule = product_unavailable_rule()
        event = _make_event(EventType.PRODUCT_AVAILABILITY, {
            "product_name": "Unknown",
        })
        assert rule.evaluate(event) is False


# ── Rule 4: Missing Information ──────────────────────────────────────────────

class TestMissingInformationRule:
    def test_positive_missing_dimensions(self):
        rule = missing_information_rule()
        event = _make_event(EventType.MISSING_INFORMATION, {
            "missing_field": "room_dimensions",
            "context": "Cannot evaluate furniture fit without room measurements.",
        })
        assert rule.evaluate(event) is True
        log = rule.execute(event)
        assert log.result_type == ActionResultType.REQUEST
        assert "room_dimensions" in log.result_message
        assert log.result_details["missing_field"] == "room_dimensions"

    def test_negative_no_missing_field(self):
        rule = missing_information_rule()
        event = _make_event(EventType.MISSING_INFORMATION, {})
        assert rule.evaluate(event) is False

    def test_wrong_event_type(self):
        rule = missing_information_rule()
        event = _make_event(EventType.BUDGET_UPDATE, {
            "missing_field": "room_dimensions",
        })
        assert rule.evaluate(event) is False


# ── Rule 5: Valid Product ────────────────────────────────────────────────────

class TestValidProductRule:
    def test_positive_valid_product(self):
        rule = valid_product_rule()
        event = _make_event(EventType.PRODUCT_VALIDATION, {
            "product_name": "Dining table",
            "product_price": 24_999,
            "available_budget": 30_000,
            "fits_room": True,
        })
        assert rule.evaluate(event) is True
        log = rule.execute(event)
        assert log.result_type == ActionResultType.APPROVAL
        assert "Dining table" in log.result_message
        assert log.result_details["remaining_budget"] == 5_001

    def test_negative_price_exceeds_budget(self):
        rule = valid_product_rule()
        event = _make_event(EventType.PRODUCT_VALIDATION, {
            "product_name": "Expensive table",
            "product_price": 35_000,
            "available_budget": 30_000,
            "fits_room": True,
        })
        assert rule.evaluate(event) is False

    def test_negative_does_not_fit_room(self):
        rule = valid_product_rule()
        event = _make_event(EventType.PRODUCT_VALIDATION, {
            "product_name": "Wide table",
            "product_price": 20_000,
            "available_budget": 30_000,
            "fits_room": False,
        })
        assert rule.evaluate(event) is False

    def test_malformed_missing_fields(self):
        rule = valid_product_rule()
        event = _make_event(EventType.PRODUCT_VALIDATION, {
            "product_name": "Table",
        })
        assert rule.evaluate(event) is False
