"""Turning a refreshed product snapshot into environmental events.

When live discovery is run again, the difference between the old and new
snapshots is *new information about the world*: a price moved, a product
went out of stock or disappeared.  :func:`detect_changes` expresses each
difference as an ordinary :class:`EnvironmentalEvent` — the same type the
workers and the BDI agent already use — so the existing belief-update and
replanning machinery handles it unchanged.
"""

from __future__ import annotations

from typing import Sequence

from smartnest.core.models import EnvironmentalEvent, EventType

_PRICE_EPSILON = 0.005


def detect_changes(previous: Sequence[dict], current: Sequence[dict],
                   source: str = "live_refresh") -> list[EnvironmentalEvent]:
    """Events describing how ``current`` differs from ``previous``.

    - price differs               -> ``PRICE_CHANGE`` (``product_id``, ``old_price``, ``new_price``)
    - availability differs        -> ``PRODUCT_AVAILABILITY``
    - listed before, now missing  -> ``PRODUCT_AVAILABILITY`` with ``unavailable`` (``removed_from_catalog``)

    New products produce no event (they are simply new candidates).
    """
    before = {r["product_id"]: r for r in previous if r.get("product_id")}
    after = {r["product_id"]: r for r in current if r.get("product_id")}
    events: list[EnvironmentalEvent] = []

    for pid in sorted(before):
        old = before[pid]
        new = after.get(pid)
        name = old.get("name")
        if new is None:
            events.append(EnvironmentalEvent(
                event_type=EventType.PRODUCT_AVAILABILITY, source=source,
                payload={"product_id": pid, "product_name": name, "availability": "unavailable",
                         "previous_availability": old.get("availability"), "removed_from_catalog": True}))
            continue
        old_price, new_price = old.get("price"), new.get("price")
        if isinstance(old_price, (int, float)) and isinstance(new_price, (int, float)) \
                and abs(new_price - old_price) > _PRICE_EPSILON:
            events.append(EnvironmentalEvent(
                event_type=EventType.PRICE_CHANGE, source=source,
                payload={"product_id": pid, "product_name": name, "old_price": old_price,
                         "new_price": new_price, "change": new_price - old_price}))
        if old.get("availability") != new.get("availability") and new.get("availability"):
            events.append(EnvironmentalEvent(
                event_type=EventType.PRODUCT_AVAILABILITY, source=source,
                payload={"product_id": pid, "product_name": name, "availability": new["availability"],
                         "previous_availability": old.get("availability")}))
    return events
