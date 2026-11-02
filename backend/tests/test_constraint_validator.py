"""Tests for record validation and hard constraints (Assignment 3.2).

Covers every boundary listed in the assignment: limits are inclusive,
one unit over is rejected, and malformed records fail in a controlled
way instead of receiving defaults.
"""

from __future__ import annotations

import math

import pytest

from smartnest.optimization import (
    ConstraintValidator,
    InvalidRequirementsError,
    ScoringConfig,
    UserRequirements,
    validate_record,
)

REQ = UserRequirements(budget=30000, max_width_cm=210, max_depth_cm=110, min_seating=3,
                       preferred_style="minimalist", preferred_colour="neutral")


def _record(**overrides):
    data = dict(product_id="P1", name="Test sofa", price=25000, width_cm=200, depth_cm=100,
                seating_capacity=3, material="fabric", color="neutral", style="minimalist",
                availability="in_stock", rating=4.2, review_count=40, source_agent_id="agent_x")
    data.update(overrides)
    return data


def _status(**overrides):
    return ConstraintValidator().evaluate(_record(**overrides), REQ)


# ── Requirements ─────────────────────────────────────────────────────────────

class TestRequirements:
    def test_valid_requirements_normalise_text(self):
        r = UserRequirements(budget=1, max_width_cm=1, max_depth_cm=1,
                             preferred_style=" Minimalist ", preferred_colour="NEUTRAL")
        assert r.preferred_style == "minimalist"
        assert r.preferred_colour == "neutral"

    @pytest.mark.parametrize("field,value", [
        ("budget", 0), ("budget", -1), ("budget", math.nan), ("budget", math.inf),
        ("max_width_cm", 0), ("max_depth_cm", -10), ("min_seating", 0), ("min_seating", 2.5),
        ("budget", None), ("budget", "30000"),
    ])
    def test_invalid_requirements_raise(self, field, value):
        data = dict(budget=30000, max_width_cm=210, max_depth_cm=110, min_seating=3)
        data[field] = value
        with pytest.raises(InvalidRequirementsError, match=field):
            UserRequirements(**data)

    def test_zero_budget_is_a_controlled_error(self):
        with pytest.raises(InvalidRequirementsError, match="budget must be greater than 0"):
            UserRequirements(budget=0, max_width_cm=210, max_depth_cm=110)

    def test_from_cnp_maps_3_1_keys(self):
        r = UserRequirements.from_cnp({
            "max_budget": 30000, "room_width": 210, "room_depth": 110,
            "seating_capacity": 3, "preferred_style": "minimalist", "preferred_color": "neutral",
        })
        assert (r.budget, r.max_width_cm, r.max_depth_cm, r.min_seating) == (30000, 210, 110, 3)
        assert r.preferred_colour == "neutral"

    def test_from_cnp_missing_depth_raises(self):
        with pytest.raises(InvalidRequirementsError, match="max_depth_cm"):
            UserRequirements.from_cnp({"max_budget": 30000, "room_width": 210})


# ── Hard-constraint boundaries ───────────────────────────────────────────────

class TestBoundaries:
    def test_price_exactly_equal_to_budget_is_eligible(self):
        assert _status(price=30000).status == "eligible"

    def test_price_one_rupee_over_budget_is_rejected(self):
        res = _status(price=30001)
        assert res.status == "rejected"
        assert res.reasons == ["price ₹30,001 exceeds budget ₹30,000 by ₹1"]

    def test_width_exactly_at_limit_is_eligible(self):
        assert _status(width_cm=210).eligible

    def test_width_one_cm_over_is_rejected(self):
        res = _status(width_cm=211)
        assert not res.eligible
        assert "width 211 cm exceeds maximum width 210 cm by 1 cm" in res.reasons

    def test_depth_exactly_at_limit_is_eligible(self):
        assert _status(depth_cm=110).eligible

    def test_depth_one_cm_over_is_rejected(self):
        assert "depth" in _status(depth_cm=111).reasons[0]

    def test_seating_exactly_minimum_is_eligible(self):
        assert _status(seating_capacity=3).eligible

    def test_seating_below_minimum_rejected(self):
        assert _status(seating_capacity=2).reasons == ["seating capacity 2 is below the minimum 3"]

    @pytest.mark.parametrize("availability", ["out_of_stock", "discontinued"])
    def test_confirmed_out_of_stock_is_rejected(self, availability):
        res = _status(availability=availability)
        assert res.status == "rejected"
        assert "confirmed out of stock" in res.reasons[0]

    @pytest.mark.parametrize("availability", ["backorder", "unknown"])
    def test_unconfirmed_availability_is_unverified_not_rejected_and_not_in_stock(self, availability):
        # (previously rejected; unknown availability is now a separate, clearly-labelled category)
        res = _status(availability=availability)
        assert res.status == "unverified" and not res.eligible
        assert res.availability == "unknown" and res.gaps == ["availability"]
        assert "check stock on the retailer" in res.reasons[0]

    def test_low_stock_counts_as_confirmed(self):
        assert _status(availability="low_stock").eligible

    def test_available_states_are_configurable(self):
        strict = ConstraintValidator(ScoringConfig(available_states=["in_stock"]))
        assert strict.evaluate(_record(availability="low_stock"), REQ).status == "unverified"  # no longer confirmed

    def test_every_violation_is_reported(self):
        res = _status(price=40000, width_cm=250, depth_cm=150, seating_capacity=1,
                      availability="out_of_stock")
        assert len(res.reasons) == 5


# ── Malformed records ────────────────────────────────────────────────────────

class TestInvalidRecords:
    # width_cm / depth_cm are no longer required: a missing size is "unknown", not invalid
    # (see tests/test_optional_dimensions.py).  Malformed sizes are still invalid, below.
    # Rating, review count and seating are optional too now (see tests/test_optional_attributes.py).
    @pytest.mark.parametrize("field", ["price"])
    def test_missing_numeric_field_is_invalid_not_defaulted(self, field):
        record = _record()
        del record[field]
        res = ConstraintValidator().evaluate(record, REQ)
        assert res.status == "invalid"
        assert res.candidate is None
        assert f"{field} is missing" in res.reasons[0]

    @pytest.mark.parametrize("field,value,fragment", [
        ("price", -1, "greater than 0"),
        ("price", 0, "greater than 0"),
        ("width_cm", -200, "greater than 0"),
        ("depth_cm", 0, "greater than 0"),
        ("rating", 5.1, "between 0 and 5"),
        ("rating", -0.5, "must not be negative"),
        ("price", math.nan, "finite"),
        ("width_cm", math.inf, "finite"),
        ("rating", -math.inf, "finite"),
        ("review_count", 2.5, "whole number"),
        ("seating_capacity", "three", "must be a number"),
        ("price", True, "must be a number"),
    ])
    def test_invalid_numeric_values(self, field, value, fragment):
        res = _status(**{field: value})
        assert res.status == "invalid"
        assert any(fragment in r for r in res.reasons)

    @pytest.mark.parametrize("field,value", [("style", ""), ("style", "   "), ("style", 7),
                                             ("name", ""), ("availability", 1)])  # a missing colour is now allowed
    def test_empty_or_invalid_text_fields(self, field, value):
        res = _status(**{field: value})
        assert res.status == "invalid"

    def test_colour_key_spelling_accepted(self):
        record = _record()
        record["colour"] = record.pop("color")
        assert ConstraintValidator().evaluate(record, REQ).eligible

    def test_missing_product_id(self):
        res = _status(product_id="")
        assert res.status == "invalid"
        assert res.product_id == "<unknown>"

    @pytest.mark.parametrize("record", [None, 42, "sofa", ["a"]])
    def test_non_mapping_record(self, record):
        res = ConstraintValidator().evaluate(record, REQ)
        assert res.status == "invalid"

    def test_all_errors_listed(self):
        candidate, errors = validate_record({"product_id": "X"}, require_complete=True)  # the strict CNP mode
        assert candidate is None
        assert len(errors) >= 10
        lenient, few = validate_record({"product_id": "X"})   # only name, price and source are mandatory now
        assert lenient is None and len(few) == 3

    def test_valid_record_is_normalised(self):
        candidate, errors = validate_record(_record(style=" Minimalist ", color="Neutral"))
        assert errors == []
        assert candidate.style == "minimalist" and candidate.colour == "neutral"
