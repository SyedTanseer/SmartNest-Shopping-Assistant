"""SmartNest domain-specific rules for Assignment 1.2.

Each rule is a condition-action pair that operates on an
``EnvironmentalEvent``.  Rules are registered into the ``RuleEngine``
which evaluates them in priority order.

The condition functions inspect ``event.payload`` and return ``True`` when
the rule should fire.  The action functions build and return an
``ActionLog`` describing the action taken.

Rules are defined as factory functions that return ``Rule`` objects so they
can be independently registered, tested and extended without modifying the
``RuleEngine``.
"""

from __future__ import annotations

from smartnest.core.models import (
    ActionResultType,
    ActionLog,
    EnvironmentalEvent,
    EventType,
)
from smartnest.reasoning.rule_engine import Rule


# ── Helper ──────────────────────────────────────────────────────────────────

def _safe_get(event: EnvironmentalEvent, *keys, default=None):
    """Safely retrieve nested keys from an event payload."""
    obj = event.payload
    for key in keys:
        if not isinstance(obj, dict):
            return default
        obj = obj.get(key)
        if obj is None:
            return default
    return obj


# ── Rule: Budget Exceeded ────────────────────────────────────────────────────

def budget_exceeded_rule() -> Rule:
    """Trigger a warning when estimated expenditure exceeds total budget."""
    def condition(event: EnvironmentalEvent) -> bool:
        if event.event_type != EventType.BUDGET_UPDATE:
            return False
        total = _safe_get(event, "total_budget")
        estimated = _safe_get(event, "estimated_expenditure")
        if total is None or estimated is None:
            return False
        return estimated > total

    def action(event: EnvironmentalEvent) -> ActionLog:
        total = _safe_get(event, "total_budget", default=0)
        estimated = _safe_get(event, "estimated_expenditure", default=0)
        overrun = estimated - total
        return ActionLog(
            event_id=event.event_id,
            rule_id="R001_BUDGET_EXCEEDED",
            rule_name="Budget Exceeded Warning",
            action_description="Notify user that estimated expenditure exceeds the total budget.",
            result_type=ActionResultType.WARNING,
            result_message=(
                f"Budget exceeded: estimated ₹{estimated:,} vs budget ₹{total:,} "
                f"(overrun ₹{overrun:,})"
            ),
            result_details={
                "total_budget": total,
                "estimated_expenditure": estimated,
                "overrun": overrun,
            },
        )

    return Rule(
        rule_id="R001_BUDGET_EXCEEDED",
        name="Budget Exceeded Warning",
        priority=10,
        condition=condition,
        action=action,
        explanation="Fires when estimated expenditure exceeds the total budget.",
    )


# ── Rule: Furniture Compatibility (reject oversized) ───────────────────────

def furniture_incompatibility_rule() -> Rule:
    """Reject a product whose dimensions exceed the available room space."""
    def condition(event: EnvironmentalEvent) -> bool:
        if event.event_type != EventType.FURNITURE_COMPATIBILITY:
            return False
        product_width = _safe_get(event, "product_width")
        room_width = _safe_get(event, "room_width")
        if product_width is None or room_width is None:
            return False
        return product_width > room_width

    def action(event: EnvironmentalEvent) -> ActionLog:
        product_name = _safe_get(event, "product_name", default="Unknown product")
        product_width = _safe_get(event, "product_width", default=0)
        room_width = _safe_get(event, "room_width", default=0)
        excess = product_width - room_width
        return ActionLog(
            event_id=event.event_id,
            rule_id="R002_FURNITURE_INCOMPATIBLE",
            rule_name="Furniture Incompatibility Rejection",
            action_description="Reject the product because it does not fit the room.",
            result_type=ActionResultType.REJECTION,
            result_message=(
                f"Reject '{product_name}': width {product_width} cm exceeds "
                f"room width {room_width} cm by {excess} cm."
            ),
            result_details={
                "product_name": product_name,
                "product_width": product_width,
                "room_width": room_width,
                "excess": excess,
            },
        )

    return Rule(
        rule_id="R002_FURNITURE_INCOMPATIBLE",
        name="Furniture Incompatibility Rejection",
        priority=20,
        condition=condition,
        action=action,
        explanation="Fires when a product's dimensions exceed available room space.",
    )


# ── Rule: Product Unavailable ──────────────────────────────────────────────

def product_unavailable_rule() -> Rule:
    """Flag an out-of-stock product and indicate an alternative is needed."""
    def condition(event: EnvironmentalEvent) -> bool:
        if event.event_type != EventType.PRODUCT_AVAILABILITY:
            return False
        availability = _safe_get(event, "availability", default="")
        return str(availability).lower() in ("out_of_stock", "out of stock", "unavailable")

    def action(event: EnvironmentalEvent) -> ActionLog:
        product_name = _safe_get(event, "product_name", default="Unknown product")
        availability = _safe_get(event, "availability", default="unknown")
        return ActionLog(
            event_id=event.event_id,
            rule_id="R003_PRODUCT_UNAVAILABLE",
            rule_name="Product Unavailable Flag",
            action_description="Flag the product as unavailable and request an alternative.",
            result_type=ActionResultType.FLAG,
            result_message=(
                f"Product '{product_name}' is {availability}. "
                f"An alternative product is required."
            ),
            result_details={
                "product_name": product_name,
                "availability": availability,
                "alternative_required": True,
            },
        )

    return Rule(
        rule_id="R003_PRODUCT_UNAVAILABLE",
        name="Product Unavailable Flag",
        priority=30,
        condition=condition,
        action=action,
        explanation="Fires when a product is out of stock or unavailable.",
    )


# ── Rule: Missing Information ──────────────────────────────────────────────

def missing_information_rule() -> Rule:
    """Request missing room measurements when dimensions are not provided."""
    def condition(event: EnvironmentalEvent) -> bool:
        if event.event_type != EventType.MISSING_INFORMATION:
            return False
        missing_field = _safe_get(event, "missing_field", default="")
        return bool(missing_field)

    def action(event: EnvironmentalEvent) -> ActionLog:
        missing_field = _safe_get(event, "missing_field", default="information")
        context = _safe_get(event, "context", default="")
        return ActionLog(
            event_id=event.event_id,
            rule_id="R004_MISSING_INFO",
            rule_name="Missing Information Request",
            action_description=f"Request the missing {missing_field}.",
            result_type=ActionResultType.REQUEST,
            result_message=(
                f"Missing information: '{missing_field}' is not provided."
                + (f" Context: {context}" if context else "")
            ),
            result_details={
                "missing_field": missing_field,
                "context": context,
            },
        )

    return Rule(
        rule_id="R004_MISSING_INFO",
        name="Missing Information Request",
        priority=40,
        condition=condition,
        action=action,
        explanation="Fires when required information (e.g. room dimensions) is missing.",
    )


# ── Rule: Valid Product Candidate ──────────────────────────────────────────

def valid_product_rule() -> Rule:
    """Mark a product as a suitable candidate when price and dimensions fit."""
    def condition(event: EnvironmentalEvent) -> bool:
        if event.event_type != EventType.PRODUCT_VALIDATION:
            return False
        price = _safe_get(event, "product_price")
        budget = _safe_get(event, "available_budget")
        fits = _safe_get(event, "fits_room")
        if price is None or budget is None or fits is None:
            return False
        return price <= budget and fits is True

    def action(event: EnvironmentalEvent) -> ActionLog:
        product_name = _safe_get(event, "product_name", default="Unknown product")
        price = _safe_get(event, "product_price", default=0)
        budget = _safe_get(event, "available_budget", default=0)
        remaining = budget - price
        return ActionLog(
            event_id=event.event_id,
            rule_id="R005_VALID_PRODUCT",
            rule_name="Valid Product Candidate",
            action_description="Mark the product as a suitable candidate.",
            result_type=ActionResultType.APPROVAL,
            result_message=(
                f"Product '{product_name}' is a suitable candidate: "
                f"price ₹{price:,} within budget ₹{budget:,} "
                f"(remaining ₹{remaining:,}), dimensions fit the room."
            ),
            result_details={
                "product_name": product_name,
                "product_price": price,
                "available_budget": budget,
                "remaining_budget": remaining,
                "fits_room": True,
            },
        )

    return Rule(
        rule_id="R005_VALID_PRODUCT",
        name="Valid Product Candidate",
        priority=50,
        condition=condition,
        action=action,
        explanation="Fires when a product's price is within budget and dimensions fit the room.",
    )


# ── Rule: Price Decrease (Assignment 2.1) ────────────────────────────────────

def price_decrease_rule() -> Rule:
    """Notify the user when a monitored product's price drops."""
    def condition(event: EnvironmentalEvent) -> bool:
        if event.event_type != EventType.PRICE_CHANGE:
            return False
        old_price = _safe_get(event, "old_price")
        new_price = _safe_get(event, "new_price")
        if old_price is None or new_price is None:
            return False
        return new_price < old_price

    def action(event: EnvironmentalEvent) -> ActionLog:
        product_name = _safe_get(event, "product_name", default="Unknown product")
        old_price = _safe_get(event, "old_price", default=0)
        new_price = _safe_get(event, "new_price", default=0)
        savings = old_price - new_price
        return ActionLog(
            event_id=event.event_id,
            rule_id="R006_PRICE_DECREASE",
            rule_name="Price Decrease Notification",
            action_description="Notify user that a monitored product's price has dropped.",
            result_type=ActionResultType.INFO,
            result_message=(
                f"Price drop for '{product_name}': ₹{old_price:,} → ₹{new_price:,} "
                f"(save ₹{savings:,})"
            ),
            result_details={
                "product_name": product_name,
                "old_price": old_price,
                "new_price": new_price,
                "savings": savings,
            },
        )

    return Rule(
        rule_id="R006_PRICE_DECREASE",
        name="Price Decrease Notification",
        priority=60,
        condition=condition,
        action=action,
        explanation="Fires when a monitored product's price decreases.",
    )


# ── Rule: Price Increase (Assignment 2.1) ────────────────────────────────────

def price_increase_rule() -> Rule:
    """Warn the user when a monitored product's price rises."""
    def condition(event: EnvironmentalEvent) -> bool:
        if event.event_type != EventType.PRICE_CHANGE:
            return False
        old_price = _safe_get(event, "old_price")
        new_price = _safe_get(event, "new_price")
        if old_price is None or new_price is None:
            return False
        return new_price > old_price

    def action(event: EnvironmentalEvent) -> ActionLog:
        product_name = _safe_get(event, "product_name", default="Unknown product")
        old_price = _safe_get(event, "old_price", default=0)
        new_price = _safe_get(event, "new_price", default=0)
        increase = new_price - old_price
        return ActionLog(
            event_id=event.event_id,
            rule_id="R007_PRICE_INCREASE",
            rule_name="Price Increase Warning",
            action_description="Warn user that a monitored product's price has risen.",
            result_type=ActionResultType.WARNING,
            result_message=(
                f"Price increase for '{product_name}': ₹{old_price:,} → ₹{new_price:,} "
                f"(+₹{increase:,})"
            ),
            result_details={
                "product_name": product_name,
                "old_price": old_price,
                "new_price": new_price,
                "increase": increase,
            },
        )

    return Rule(
        rule_id="R007_PRICE_INCREASE",
        name="Price Increase Warning",
        priority=55,
        condition=condition,
        action=action,
        explanation="Fires when a monitored product's price increases.",
    )


# ── Rule: Budget Threshold Warning (Assignment 2.1) ──────────────────────────

def budget_threshold_rule() -> Rule:
    """Warn when expenditure crosses 90% of the total budget but is not exceeded."""
    def condition(event: EnvironmentalEvent) -> bool:
        if event.event_type != EventType.BUDGET_UPDATE:
            return False
        total = _safe_get(event, "total_budget")
        estimated = _safe_get(event, "estimated_expenditure")
        if total is None or estimated is None:
            return False
        # Fire when between 90% and 100% (budget exceeded rule handles >100%)
        return 0.9 * total <= estimated <= total

    def action(event: EnvironmentalEvent) -> ActionLog:
        total = _safe_get(event, "total_budget", default=0)
        estimated = _safe_get(event, "estimated_expenditure", default=0)
        pct = (estimated / total * 100) if total else 0
        return ActionLog(
            event_id=event.event_id,
            rule_id="R008_BUDGET_THRESHOLD",
            rule_name="Budget Threshold Warning",
            action_description="Warn that expenditure is approaching the budget limit.",
            result_type=ActionResultType.WARNING,
            result_message=(
                f"Budget threshold: expenditure ₹{estimated:,} is {pct:.1f}% "
                f"of budget ₹{total:,}. Approaching limit."
            ),
            result_details={
                "total_budget": total,
                "estimated_expenditure": estimated,
                "percentage": round(pct, 1),
            },
        )

    return Rule(
        rule_id="R008_BUDGET_THRESHOLD",
        name="Budget Threshold Warning",
        priority=12,
        condition=condition,
        action=action,
        explanation="Fires when expenditure is between 90% and 100% of the budget.",
    )


# ── Rule: Product Back In Stock (Assignment 2.1) ─────────────────────────────

def product_available_rule() -> Rule:
    """Notify when a previously unavailable product is back in stock."""
    def condition(event: EnvironmentalEvent) -> bool:
        if event.event_type != EventType.PRODUCT_AVAILABILITY:
            return False
        availability = _safe_get(event, "availability", default="")
        return str(availability).lower() in ("in_stock", "in stock", "available")

    def action(event: EnvironmentalEvent) -> ActionLog:
        product_name = _safe_get(event, "product_name", default="Unknown product")
        availability = _safe_get(event, "availability", default="available")
        return ActionLog(
            event_id=event.event_id,
            rule_id="R009_PRODUCT_AVAILABLE",
            rule_name="Product Available Notification",
            action_description="Notify that the product is back in stock.",
            result_type=ActionResultType.INFO,
            result_message=(
                f"Product '{product_name}' is now {availability}. "
                f"Available for purchase."
            ),
            result_details={
                "product_name": product_name,
                "availability": availability,
            },
        )

    return Rule(
        rule_id="R009_PRODUCT_AVAILABLE",
        name="Product Available Notification",
        priority=35,
        condition=condition,
        action=action,
        explanation="Fires when a product becomes available again.",
    )


# ── Registration helper ─────────────────────────────────────────────────────

def all_smartnest_rules() -> list[Rule]:
    """Return all default SmartNest rules for Assignments 1.2 and 2.1."""
    return [
        budget_exceeded_rule(),
        budget_threshold_rule(),
        furniture_incompatibility_rule(),
        product_unavailable_rule(),
        product_available_rule(),
        missing_information_rule(),
        valid_product_rule(),
        price_increase_rule(),
        price_decrease_rule(),
    ]
