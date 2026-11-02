"""JSON views of a session for the interface: plan, room-wise breakdown, budget analytics, comparison."""

from __future__ import annotations

from typing import Any

from smartnest.furnishing.assistant import HomeSession
from smartnest.furnishing.coordinator import PlanError


def room_view(session: HomeSession) -> list[dict[str, Any]]:
    plan = session.plan
    if plan is None:
        return []
    rooms = []
    for room in plan.rooms:
        items = [ln for ln in plan.lines.values() if ln["room_id"] == room["room_id"]]
        spent = sum(ln["product"]["price"] for ln in items if ln.get("product"))
        rooms.append({
            "room_id": room["room_id"], "label": room["label"], "spent": round(spent, 2),
            "allocated": round(sum(ln["ceiling"] for ln in items), 2), "items": items,
            "filled": sum(1 for ln in items if ln.get("product")), "count": len(items),
        })
    return rooms


def budget_view(session: HomeSession) -> dict[str, Any] | None:
    plan = session.plan
    if plan is None:
        return None
    s = plan.summary
    return {
        "total": s["total_budget"], "spent": s["spent"], "remaining": s["remaining"], "utilisation": s["utilisation"],
        "within_budget": s["within_budget"],
        "by_room": [{"name": k, "value": v} for k, v in s["by_room"].items()],
        "by_category": [{"name": k, "value": v} for k, v in s["by_category"].items()],
        "items": [{"label": ln["label"], "room": ln["room"], "category": ln["category_label"],
                   "ceiling": ln["ceiling"], "price": (ln.get("product") or {}).get("price"), "status": ln["status"]}
                  for ln in plan.lines.values()],
        "allocation_vs_spend": [{"room": r["label"], "allocated": r["allocated"], "spent": r["spent"]}
                                for r in room_view(session)],
        "counts": {"confirmed": s["confirmed"], "unverified": s["unverified"], "unfilled": s["unfilled_count"]},
    }


def session_view(session: HomeSession) -> dict[str, Any]:
    plan = session.plan
    co = session.coordinator
    return {
        "id": session.id, "created_at": session.created_at, "has_plan": plan is not None,
        "plan": plan.to_dict() if plan else None, "rooms": room_view(session), "budget": budget_view(session),
        "shortlist": list(co.shortlist), "cart": co.cart_view() if plan else None, "transitions": co.transitions[-60:], "history": session.history[-60:],
        "draft": session.draft,
    }


def compare_view(session: HomeSession, product_ids: list[str]) -> list[dict[str, Any]]:
    """Side-by-side data for products that appear in the plan (as a pick or an alternative)."""
    plan = session.plan
    if plan is None:
        raise PlanError("there is no plan yet")
    index: dict[str, tuple[dict, dict]] = {}
    for line in plan.lines.values():
        for item in [line.get("product"), *(line.get("alternatives") or [])]:
            if item:
                index.setdefault(item["product_id"], (line, item))
    out = []
    for pid in product_ids:
        if pid not in index:
            raise PlanError(f"unknown product {pid!r}")
        line, item = index[pid]
        picked = (line.get("product") or {}).get("product_id") == pid
        out.append({**item, "item": line["label"], "slot_id": line["slot_id"], "is_pick": picked,
                    "utility": line.get("utility") if picked else item.get("utility"),
                    "completeness": line.get("completeness") if picked else item.get("completeness"),
                    "contributions": line.get("contributions") if picked else None})
    return out


def options_view(session: HomeSession, slot_id: str) -> dict[str, Any]:
    """Every listing found for an item, and whether it still fits the budget left once this item is excluded."""
    co, plan = session.coordinator, session.plan
    if plan is None or slot_id not in plan.lines:
        raise PlanError(f"unknown item {slot_id!r}")
    line = plan.lines[slot_id]
    others = plan.summary["spent"] - ((line.get("product") or {}).get("price") or 0)
    room = plan.summary["total_budget"] - others
    records = co._research[line["category"]]["records"]
    rows = []
    for r in records:
        stock = (co._overrides.get(r["product_id"]) or {}).get("availability") or r.get("availability")
        price = (co._overrides.get(r["product_id"]) or {}).get("price") or r["price"]
        rows.append({"product_id": r["product_id"], "name": r["name"], "price": price, "retailer": r.get("retailer"),
                     "rating": r.get("rating"), "availability": stock, "fits_budget": price <= room,
                     "is_pick": (line.get("product") or {}).get("product_id") == r["product_id"]})
    rows.sort(key=lambda x: x["price"])
    return {"slot_id": slot_id, "label": line["label"], "budget_available": round(room, 2), "options": rows}


def agents_view(session: HomeSession) -> dict[str, Any]:
    co = session.coordinator
    events = co.log.events()
    roster = []
    for agent_id in co.bus.agent_ids:
        mine = [e for e in events if e.agent == agent_id]
        received = [e for e in events if e.peer == agent_id and e.kind == "message"]
        roster.append({"id": agent_id, "role": "coordinator" if agent_id == "coordinator" else "worker",
                       "actions": len(mine), "requests_received": len(received),
                       "avg_ms": round(sum(e.duration_ms or 0 for e in received) / len(received), 2) if received else None,
                       "fingerprint": (co.bus.registry.get(agent_id).fingerprint if co.bus.registry.get(agent_id) else None)})
    return {"agents": roster, "security": {"accepted": len(co.bus.audit.accepted()), "rejected": len(co.bus.audit.rejected()),
                     "by_code": co.bus.audit.summary()},
            "messages": sum(1 for e in events if e.kind == "message"),
            "unverified_messages": sum(1 for e in events if e.kind == "message" and not e.verified),
            "total_events": len(events)}
