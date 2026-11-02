#!/usr/bin/env python3
"""SmartNest - whole-home furnishing demo (multi-agent, real saved listings, no network).

    python src/furnishing_run.py                # the final demo scenario
    python src/furnishing_run.py --simulated    # labelled sample listings instead of real snapshots

Shows the four agents talking over signed messages, the plan room by room, the budget analytics, and BDI-style
replanning after an out-of-stock event and a budget cut.  Exit code 0 when every check passes.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from smartnest.furnishing import (FurnishingCoordinator, FurnishingRequest, LayeredCatalog, SimulatedCatalog,  # noqa: E402
                                  SnapshotCatalog)
from smartnest.furnishing.money import inr  # noqa: E402

DEMO = "2BHK, furnishing budget Rs 2 lakh, minimalist"
checks = []


def check(name, ok, detail=""):
    checks.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{' - ' + detail if detail else ''}")


def show(plan):
    for room in plan.rooms:
        print(f"\n  {room['label']}")
        for ln in (l for l in plan.lines.values() if l["room_id"] == room["room_id"]):
            p = ln["product"]
            if p:
                print(f"    {ln['category_label']:13s} {inr(p['price']):>10s}  {ln['status']:10s} u={ln['utility']:5.1f}  "
                      f"{p['name'][:46]} [{p['retailer']}]")
            else:
                print(f"    {ln['category_label']:13s} {'-':>10s}  unfilled")
    s = plan.summary
    print(f"\n  Spent {inr(s['spent'])} of {inr(s['total_budget'])} ({s['utilisation']:.0%}); remaining {inr(s['remaining'])}; "
          f"{s['confirmed']} confirmed, {s['unverified']} unverified, {s['unfilled_count']} unfilled")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulated", action="store_true")
    args = ap.parse_args()
    catalog = LayeredCatalog(SnapshotCatalog(), SimulatedCatalog(), allow_simulated=args.simulated)
    co = FurnishingCoordinator(catalog)
    print("=" * 78 + f"\nSmartNest furnishing demo: {DEMO}\n" + "=" * 78)
    plan = co.create_plan(FurnishingRequest(200000, 2, "minimalist"))
    show(plan)
    print("\nAgent traffic (signed Ed25519 envelopes):")
    msgs = [e for e in co.log.events() if e.kind == "message"]
    for e in msgs[:6]:
        print(f"    {e.agent:12s} -> {e.peer:16s} {e.summary}  verified={e.verified}  {e.duration_ms} ms")
    print(f"    ... {len(msgs)} messages in total")

    print("\nChecks:")
    s = plan.summary
    check("all 10 items planned", len(plan.lines) == 10 and s["unfilled_count"] == 0)
    check("total within budget", s["within_budget"] and s["spent"] <= 200000)
    check("every message signature verified", all(e.verified for e in msgs))
    check("no unverified item is called confirmed", all(l["status"] != "confirmed" or
          l["product"]["availability"] == "in_stock" for l in plan.lines.values()))

    sofa = plan.lines["living_room.sofa"]["product"]
    print(f"\nEvent: {sofa['name'][:40]} goes out of stock")
    plan2 = co.handle_event({"type": "out_of_stock", "product_id": sofa["product_id"]})
    new = plan2.lines["living_room.sofa"]["product"]
    print(f"  replaced by {new['name'][:46]} at {inr(new['price'])}" if new else "  no replacement fits")
    check("out-of-stock product replaced", new is None or new["product_id"] != sofa["product_id"])
    check("other items kept their choice", all(
        plan2.lines[k]["product"]["product_id"] == v["product"]["product_id"]
        for k, v in plan.lines.items() if k != "living_room.sofa" and v["product"]["product_id"] != sofa["product_id"]))

    print("\nEvent: budget cut to Rs 1 lakh")
    plan3 = co.handle_event({"type": "budget_change", "total_budget": 100000})
    show(plan3)
    check("replanned within the new budget", plan3.summary["spent"] <= 100000 and plan3.summary["within_budget"])
    print("\nBDI transitions:")
    for t in co.transitions[-6:]:
        print(f"    {t['kind']:11s} {t['label']:24s} {t['reason']}")
    check("transitions explained", all(t["reason"] for t in co.transitions))
    print(f"\n{sum(checks)}/{len(checks)} checks passed")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
