"""Budget allocation and analytics (the Budget agent's deterministic core).

``allocate`` splits the total budget over the slots in proportion to their category weights.  The weights of the
slots actually being bought are renormalised, so a 1BHK and a 3BHK both sum to exactly the budget.  A ceiling is a
HARD limit handed to the optimizer, so the sum of picks can never exceed the budget.

``reallocate`` moves the money of slots that could not be filled (nothing eligible under the ceiling) to the slots
that are still open, in proportion to their weight, without ever exceeding the total.
"""

from __future__ import annotations

from dataclasses import dataclass

from smartnest.furnishing.rooms import Slot


def _split(amount: float, slots: list[Slot]) -> dict[str, float]:
    weight = sum(s.profile.default_weight for s in slots)
    out = {s.slot_id: round(amount * s.profile.default_weight / weight, 2) for s in slots}
    drift = round(amount - sum(out.values()), 2)
    out[slots[0].slot_id] = round(out[slots[0].slot_id] + drift, 2)
    return out


def allocate(budget: float, slots: list[Slot]) -> dict[str, float]:
    """Ceiling per slot id; the ceilings sum to ``budget`` (to the paisa)."""
    if budget <= 0:
        raise ValueError("budget must be positive")
    return _split(budget, slots) if slots else {}


def reallocate(budget: float, slots: list[Slot], spent: dict[str, float], open_ids: set[str]) -> dict[str, float]:
    """New ceilings for ``open_ids`` after the other slots are settled at ``spent`` (slot id -> price)."""
    remaining = round(budget - sum(spent.values()), 2)
    open_slots = [s for s in slots if s.slot_id in open_ids]
    if not open_slots:
        return {}
    if remaining <= 0:
        return {s.slot_id: 0.0 for s in open_slots}
    return _split(remaining, open_slots)


@dataclass(frozen=True)
class BudgetSummary:
    total_budget: float
    spent: float
    remaining: float
    utilisation: float                  # spent / total, 0-1
    by_room: dict[str, float]
    by_category: dict[str, float]
    unfilled: list[str]                 # slot ids with no purchase
    within_budget: bool


def summarise(budget: float, slots: list[Slot], prices: dict[str, float]) -> BudgetSummary:
    """Analytics for a set of chosen prices (``slot_id -> price``)."""
    by_room: dict[str, float] = {}
    by_cat: dict[str, float] = {}
    for slot in slots:
        price = prices.get(slot.slot_id)
        if price is None:
            continue
        by_room[slot.room_label] = round(by_room.get(slot.room_label, 0.0) + price, 2)
        by_cat[slot.profile.label] = round(by_cat.get(slot.profile.label, 0.0) + price, 2)
    spent = round(sum(prices.get(s.slot_id, 0.0) for s in slots), 2)
    return BudgetSummary(total_budget=budget, spent=spent, remaining=round(budget - spent, 2),
                         utilisation=round(spent / budget, 4) if budget else 0.0, by_room=by_room, by_category=by_cat,
                         unfilled=[s.slot_id for s in slots if s.slot_id not in prices],
                         within_budget=spent <= budget + 1e-9)
