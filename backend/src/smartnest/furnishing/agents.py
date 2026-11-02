"""The four worker agents.  Each one is a small, single-purpose component reachable only through the signed bus.

    SpacePlanningAgent    "2BHK" -> rooms -> the items to buy (and the assumptions made)        task: plan
    ProductResearchAgent  fetches a category's listings, removes accessories and says why        task: research
    ComparisonAgent       runs the UtilityOptimizer for one item and returns ranked options     task: compare
    BudgetAgent           splits the budget, re-splits it, and produces the analytics           task: allocate | reallocate | summarise

None of them calls Gemini or the network.  The ComparisonAgent is the only place that ranks products, and it does so
with the existing :class:`UtilityOptimizer`, so every hard constraint (price ceiling, stock, size, seats) still holds.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from smartnest.furnishing import budget as budget_mod
from smartnest.furnishing.brief import theme_matches
from smartnest.furnishing.catalog import ProductCatalog
from smartnest.furnishing.categories import category, kid_signal, mature_reason
from smartnest.furnishing.rooms import Slot, parse_home_type, rooms_for, rooms_for_items, rooms_for_scope, slots_for
from smartnest.optimization import UtilityOptimizer
from smartnest.optimization.utility_optimizer import CandidateEvaluation


class SpacePlanningAgent:
    agent_id = "space_planning"

    def handle(self, task: str, data: dict[str, Any]) -> dict[str, Any]:
        if task != "plan":
            raise ValueError(f"unknown task {task!r}")
        bedrooms = data.get("bedrooms")
        scope = list(data.get("scope") or [])
        items = list(data.get("items") or [])
        assumptions: list[str] = []
        if bedrooms is None and not scope and not items:
            bedrooms = parse_home_type(str(data.get("home_text") or ""))
        rooms: list = []
        if bedrooms is not None:
            rooms = rooms_for(int(bedrooms), data.get("room_dimensions"))
            assumptions.append(f"A {bedrooms}BHK has a living room, {bedrooms} bedroom(s) and a kitchen.")
            assumptions.append("Living room: sofa (3 seats or more) and TV unit. Each bedroom: bed, mattress, wardrobe. "
                               "Kitchen: refrigerator and a dining table that seats 4 or more.")
        elif scope:
            rooms = rooms_for_scope(scope, data.get("room_dimensions"))
            for room in rooms:
                assumptions.append(f"{room.label}: " + ", ".join(category(c).label.lower() for c in room.categories) + ".")
            if "kids_bedroom" in scope:
                age = data.get("age")
                assumptions.append("This is a child's room" + (f" (age {age})" if age else "") +
                                   ": SmartNest looks for a single/kid's bed and a study table. Sizes are only checked "
                                   "where a listing states them.")
        if items:
            have = {c for r in rooms for c in r.categories}
            extra = [i for i in items if i not in have]
            if extra:
                rooms += rooms_for_items(extra, "Extras" if rooms else "Shopping list")
                assumptions.append("Things you asked for: " + ", ".join(category(c).label.lower() for c in extra) + ".")
                custom = [category(c).label.lower() for c in extra if category(c).custom]
                if custom:
                    assumptions.append("No curated knowledge for " + ", ".join(custom) + ": SmartNest searches for them by "
                                       "name and judges prices only against the other results it finds.")
        if not rooms:
            raise ValueError("neither a home type (e.g. 2BHK), a room nor anything to buy was given")
        slots = slots_for(rooms)
        if not any(r.width_cm or r.depth_cm for r in rooms):
            assumptions.append("No room dimensions were given, so no size limits were applied; products whose size "
                               "is stated are not checked against a room.")
        return {"rooms": [asdict(r) for r in rooms], "slots": [s.to_dict() for s in slots],
                "assumptions": assumptions}


class ProductResearchAgent:
    agent_id = "product_research"

    def __init__(self, catalog: ProductCatalog) -> None:
        self._catalog = catalog

    def handle(self, task: str, data: dict[str, Any]) -> dict[str, Any]:
        if task != "research":
            raise ValueError(f"unknown task {task!r}")
        profile = category(data["category"])
        fetch = getattr(self._catalog, "listings_for", None)
        listings = (fetch(profile.id, tuple(data.get("themes") or ())) if (fetch and profile.theme_sensitive)
                    else self._catalog.listings(profile.id))
        child = data.get("audience") in ("child", "teen")
        kept, dropped = [], []
        for record in listings.records:
            name = str(record.get("name") or "")
            reason = profile.accessory_reason(name, float(record.get("price") or 0))
            if reason is None and data.get("audience") == "child":
                reason = mature_reason(name)
            if reason:
                dropped.append({"name": record.get("name"), "price": record.get("price"), "reason": reason})
            else:
                kept.append(record)
        if profile.derived_floor and len(kept) >= 5:
            # No curated price range for a custom category, so take it from the results: a listing far below the typical
            # price of the batch is almost always an accessory, a part or a sample.
            prices = sorted(float(r["price"]) for r in kept)
            median = prices[len(prices) // 2]
            floor, still = 0.25 * median, []
            for record in kept:
                if float(record["price"]) < floor:
                    dropped.append({"name": record.get("name"), "price": record.get("price"),
                                    "reason": f"price Rs {float(record['price']):,.0f} is far below the typical "
                                              f"{profile.label.lower()} in these results (median Rs {median:,.0f}); "
                                              "likely an accessory"})
                else:
                    still.append(record)
            kept = still
        return {"category": profile.id, "records": kept, "dropped": dropped, "origin": listings.origin,
                "retrieved_at": listings.retrieved_at, "query": listings.query,
                "rejected_at_parse": listings.rejected, "notes": listings.notes,
                "found": len(listings.records)}


def _option(evaluation: CandidateEvaluation, record: dict[str, Any], state: str,
            themes: list[str] | None = None) -> dict[str, Any]:
    b = evaluation.breakdown
    return {
        "state": state, "rank": evaluation.rank, "product_id": evaluation.product_id, "name": evaluation.name,
        "price": evaluation.price, "utility": b.total_utility if b else None,
        "completeness": b.completeness if b else None, "unavailable": evaluation.unavailable,
        "components": b.components if b else {}, "weights": b.weights if b else {},
        "contributions": b.contributions if b else {}, "excluded": b.excluded if b else {},
        "availability": evaluation.availability, "gaps": list(evaluation.gaps), "fit": evaluation.fit_label,
        "reasons": list(evaluation.reasons), "record": record,
        "theme_matches": theme_matches(evaluation.name or "", themes or []),
    }


class ComparisonAgent:
    agent_id = "comparison"

    def __init__(self) -> None:
        self._optimizer = UtilityOptimizer()

    def handle(self, task: str, data: dict[str, Any]) -> dict[str, Any]:
        if task == "cheapest":
            return self._cheapest(data)
        if task != "compare":
            raise ValueError(f"unknown task {task!r}")
        slot = Slot.from_dict(data["slot"])
        ceiling = float(data["ceiling"])
        records = [dict(r) for r in data["records"]]
        overrides: dict[str, dict[str, Any]] = data.get("overrides") or {}
        for record in records:                       # events (out of stock, price change) override the snapshot
            record.update(overrides.get(record["product_id"], {}))
        if ceiling <= 0:
            return {"slot_id": slot.slot_id, "selected": None, "options": [], "inspected": len(records),
                    "rejected": len(records), "note": "no budget is left for this item"}
        requirements = {"budget": ceiling, "min_seating": slot.min_seating if slot.profile.needs_seating else 1,
                        "max_width_cm": slot.max_width_cm, "max_depth_cm": slot.max_depth_cm,
                        "preferred_style": data.get("style"), "preferred_colour": data.get("colour")}
        result = self._optimizer.optimize(records, requirements)
        by_id = {r["product_id"]: r for r in records}
        themes = [str(t) for t in (data.get("themes") or [])]
        options = [_option(e, by_id.get(e.product_id, {}), "confirmed", themes) for e in result.ranking[:5]]
        options += [_option(e, by_id.get(e.product_id, {}), "unverified", themes) for e in result.unverified[:5]]
        # A stated theme is a preference that outranks the utility order: options whose TITLE mentions it come first
        # (a themed listing with unconfirmed stock is still labelled unverified).  Nothing is inferred; if no title
        # mentions the theme the optimizer's order stands.
        if themes and any(o["theme_matches"] for o in options):
            child = data.get("audience") in ("child", "teen")
            # themed first; then the title that matches MORE of the themes; for a child, then the one worded for
            # children; only after that stock state and the optimizer's rank.
            options.sort(key=lambda o: (0 if o["theme_matches"] else 1, -len(o["theme_matches"]),
                                        -(kid_signal(o["name"] or "") if child else 0),
                                        0 if o["state"] == "confirmed" else 1, o["rank"] or 0))
        selected = options[0] if options else None
        rejected = [{"name": e.name, "price": e.price, "reasons": e.reasons[:2]}
                    for e in result.evaluations if e.status in ("rejected", "invalid")][:8]
        return {"slot_id": slot.slot_id, "selected": selected, "options": options, "inspected": result.total_inspected,
                "eligible": result.eligible_count, "unverified": result.unverified_count,
                "rejected": result.rejected_count, "rejected_examples": rejected, "explanation": result.explanation,
                "ceiling": ceiling, "themed": bool(themes and any(o["theme_matches"] for o in options))}


    def _cheapest(self, data: dict[str, Any]) -> dict[str, Any]:
        """The lowest price among options that could be bought for this item (confirmed or unverified), ignoring any
        budget ceiling: what the item costs at minimum.  ``None`` if no listing is valid for it at all."""
        slot = Slot.from_dict(data["slot"])
        records = [dict(r) for r in data["records"]]
        overrides: dict[str, dict[str, Any]] = data.get("overrides") or {}
        for record in records:
            record.update(overrides.get(record["product_id"], {}))
        if not records:
            return {"slot_id": slot.slot_id, "cheapest": None}
        requirements = {"budget": 1e9, "min_seating": slot.min_seating if slot.profile.needs_seating else 1,
                        "max_width_cm": slot.max_width_cm, "max_depth_cm": slot.max_depth_cm}
        result = self._optimizer.optimize(records, requirements)
        prices = [e.price for e in (*result.ranking, *result.unverified) if e.price is not None]
        return {"slot_id": slot.slot_id, "cheapest": min(prices) if prices else None}


class BudgetAgent:
    agent_id = "budget"

    def handle(self, task: str, data: dict[str, Any]) -> dict[str, Any]:
        slots = [Slot.from_dict(s) for s in data.get("slots", [])]
        if task == "allocate":
            return {"ceilings": budget_mod.allocate(float(data["budget"]), slots)}
        if task == "reallocate":
            return {"ceilings": budget_mod.reallocate(float(data["budget"]), slots, data.get("settled") or {},
                                                      set(data.get("open") or []))}
        if task == "summarise":
            return asdict(budget_mod.summarise(float(data["budget"]), slots, data.get("prices") or {}))
        raise ValueError(f"unknown task {task!r}")
