"""The shopping cart: a buy list the user fills from any listing SmartNest has found.

SmartNest does not sell anything and has no checkout; the cart is the list of what you intend to buy, with each line
linking out to the retailer (or the labelled Google Shopping page) and a running total.

Rules (all enforced here, none left to the interface):
    * only listings SmartNest actually found can be added (an invented product id is refused);
    * a product known to be out of stock cannot be added;
    * the cart total can never exceed the total budget: an addition that would is refused with the exact shortfall;
    * stock stays "unverified" unless the listing states it;
    * a later event (budget cut, price change, stock change) never silently edits the cart; the cart view reports
      what no longer holds (over budget, now out of stock) so the user decides.
"""

from __future__ import annotations

from typing import Any

MAX_QTY = 10
CART_EVENTS = ("cart_add", "cart_set", "cart_remove", "cart_clear", "cart_add_plan")


class CartMixin:
    """Cart behaviour for :class:`FurnishingCoordinator` (uses its research, overrides, request and plan)."""

    cart: dict[str, int]

    def _init_cart(self) -> None:
        self.cart = {}

    # ── Lookups ──────────────────────────────────────────────────────────

    def _record_of(self, product_id: Any) -> dict[str, Any]:
        from smartnest.furnishing.coordinator import PlanError
        for research in self._research.values():                      # type: ignore[attr-defined]
            for record in research["records"]:
                if record["product_id"] == product_id:
                    return record
        raise PlanError(f"{product_id!r} is not a product SmartNest has found")

    def _live_price(self, record: dict[str, Any]) -> float:
        return float((self._overrides.get(record["product_id"]) or {}).get("price") or record["price"])  # type: ignore[attr-defined]

    def _live_stock(self, record: dict[str, Any]) -> str:
        return (self._overrides.get(record["product_id"]) or {}).get("availability") or record.get("availability") or "unknown"  # type: ignore[attr-defined]

    def cart_total(self, cart: dict[str, int] | None = None) -> float:
        cart = self.cart if cart is None else cart
        return round(sum(self._live_price(self._record_of(pid)) * q for pid, q in cart.items()), 2)

    # ── Changes ──────────────────────────────────────────────────────────

    def apply_cart_event(self, event: dict[str, Any]) -> None:
        from smartnest.furnishing.coordinator import PlanError
        kind = event.get("type")
        proposed = dict(self.cart)
        if kind == "cart_clear":
            proposed = {}
        elif kind == "cart_add_plan":
            if self.plan is None:                                       # type: ignore[attr-defined]
                raise PlanError("there is no plan yet")
            wanted: dict[str, int] = {}
            for line in self.plan.lines.values():                       # type: ignore[attr-defined]
                if line.get("product"):
                    wanted[line["product"]["product_id"]] = wanted.get(line["product"]["product_id"], 0) + 1
            for pid, count in wanted.items():
                proposed[pid] = max(proposed.get(pid, 0), count)        # idempotent: never doubles a quantity
        else:
            pid = event.get("product_id")
            record = self._record_of(pid)                               # refuses an unknown product
            current = proposed.get(pid, 0)
            if kind == "cart_remove":
                proposed.pop(pid, None)
            elif kind in ("cart_add", "cart_set"):
                qty = event.get("qty", 1)
                low = 0 if kind == "cart_set" else 1
                if isinstance(qty, bool) or not isinstance(qty, int) or not low <= qty <= MAX_QTY:
                    raise PlanError(f"quantity must be a whole number from {low} to {MAX_QTY}")
                new = current + qty if kind == "cart_add" else qty
                if new > MAX_QTY:
                    raise PlanError(f"at most {MAX_QTY} of one product")
                if new > current and self._live_stock(record) == "out_of_stock":
                    raise PlanError(f"{record['name']} is out of stock")
                if new:
                    proposed[pid] = new
                else:
                    proposed.pop(pid, None)
            else:
                raise PlanError(f"unknown cart event {kind!r}")
        total = self.cart_total(proposed)
        budget = float(self._request.total_budget)                      # type: ignore[attr-defined]
        if total > budget + 1e-9 and total > self.cart_total():
            from smartnest.furnishing.money import inr
            raise PlanError(f"that would bring the cart to {inr(total)}, {inr(total - budget)} over your "
                            f"{inr(budget)} budget")
        self.cart = proposed
        self.log.record("coordinator", "info", f"cart: {kind} -> {sum(proposed.values())} item(s), "   # type: ignore[attr-defined]
                        f"₹{total:,.0f}", details={k: v for k, v in event.items() if k != "type"})

    # ── View ─────────────────────────────────────────────────────────────

    def cart_view(self) -> dict[str, Any]:
        from smartnest.furnishing.coordinator import _product_view
        picks = {ln["product"]["product_id"] for ln in (self.plan.lines.values() if self.plan else ())  # type: ignore[attr-defined]
                 if ln.get("product")}
        lines, warnings = [], []
        for pid, qty in self.cart.items():
            record = self._record_of(pid)
            price, stock = self._live_price(record), self._live_stock(record)
            view = _product_view({"product_id": pid, "name": record["name"], "price": price, "record": record,
                                  "availability": stock})
            if stock == "out_of_stock":
                warnings.append(f"{record['name']} is now out of stock - remove it or pick another.")
            lines.append({"product": view, "qty": qty, "line_total": round(price * qty, 2),
                          "category": record.get("category"), "status": "confirmed" if stock == "in_stock" else "unverified",
                          "in_plan": pid in picks, "out_of_stock": stock == "out_of_stock"})
        total = round(sum(ln["line_total"] for ln in lines), 2)
        budget = float(self._request.total_budget) if self._request else 0.0                    # type: ignore[attr-defined]
        if budget and total > budget + 1e-9:
            warnings.append("The cart is over the current budget. Remove something or raise the budget.")
        return {"lines": lines, "count": sum(ln["qty"] for ln in lines), "total": total, "budget": budget,
                "remaining": round(budget - total, 2), "within_budget": total <= budget + 1e-9,
                "unverified": sum(1 for ln in lines if ln["status"] == "unverified"), "warnings": warnings}
