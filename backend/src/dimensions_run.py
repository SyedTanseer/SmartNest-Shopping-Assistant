#!/usr/bin/env python3
"""SmartNest - optional dimensions: conditional spatial validation and dynamic utility weights.

Run from the ``backend`` directory:

    python src/dimensions_run.py

Four sofas, one of them with no stated size at all, evaluated under three situations:
  1. the user gave NO room limits  -> fit is not a question, SpaceFit is excluded, weights are redistributed
  2. the user gave a width and depth -> verified / not verified / incompatible are kept apart
  3. the user gave only a width limit -> only that axis is checked and scored

Everything is computed by the deterministic UtilityOptimizer; nothing here is invented or simulated data
beyond the four example records.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from smartnest.optimization import ScoringConfig, UserRequirements, UtilityOptimizer  # noqa: E402

BASE = dict(material="fabric", colour="neutral", style="minimalist", availability="in_stock", source_agent_id="demo")
CATALOGUE = [
    {**BASE, "product_id": "COMPACT",    "name": "Compact 3-seater (200x95 cm)",         "price": 26000, "width_cm": 200, "depth_cm": 95,
     "seating_capacity": 3, "rating": 4.2, "review_count": 120},
    {**BASE, "product_id": "GRAND",      "name": "Grand 3-seater (235x105 cm)",          "price": 24000, "width_cm": 235, "depth_cm": 105,
     "seating_capacity": 3, "rating": 4.5, "review_count": 300},
    {**BASE, "product_id": "ONLINE_ONLY", "name": "Online listing (size not stated)",    "price": 18000,
     "seating_capacity": 3, "rating": 4.8, "review_count": 450},
    {**BASE, "product_id": "NO_DEPTH",   "name": "Wide listing (width 205 cm, no depth)", "price": 22000, "width_cm": 205,
     "seating_capacity": 3, "rating": 4.0, "review_count": 80},
]


def sep(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def show(result) -> None:
    print(f"  weights used for a fit-verified product: {fmt(ScoringConfig().effective_weights(True))}")
    print(f"  weights used when SpaceFit is excluded : {fmt(ScoringConfig().effective_weights(False))}\n")
    if result.selected:
        s = result.selected
        print(f"  RECOMMENDED: {s.name}  U = {s.utility:.2f}   [{s.fit_label}]")
    else:
        print("  RECOMMENDED: nothing - no candidate has a verified fit")
    for title, rows in (("Ranked (eligible)", result.ranking), ("Fit not verified (listed separately, never recommended as a fit)", result.unverified)):
        print(f"\n  {title}:")
        for e in rows or []:
            used = "SpaceFit counted" if e.breakdown.spatial_included else "SpaceFit excluded"
            print(f"    #{e.rank} {e.name:40} U = {e.utility:6.2f}   [{e.fit_label}; {used}; weights sum {sum(e.breakdown.weights.values()):.4f}]")
            for reason in e.reasons:
                print(f"         why: {reason}")
        if not rows:
            print("    (none)")
    print("\n  Dimensionally incompatible / rejected:")
    shown = False
    for e in result.rejected:
        shown = True
        print(f"    {e.name:40} [{e.fit_label or 'invalid record'}]  {'; '.join(e.reasons)}")
    if not shown:
        print("    (none)")
    print(f"\n  {result.explanation}")


def fmt(weights: dict) -> str:
    return ", ".join(f"{k} {v:.4f}" for k, v in weights.items()) + f"  (sum {sum(weights.values()):.4f})"


def main() -> int:
    optimizer = UtilityOptimizer()
    common = dict(budget=30000, min_seating=3, preferred_style="minimalist", preferred_colour="neutral")
    failures: list[str] = []

    def check(condition: bool, text: str) -> None:
        print(f"    [{'OK' if condition else 'FAIL'}] {text}")
        if not condition:
            failures.append(text)

    sep("SmartNest - optional dimensions, conditional spatial validation, dynamic weights")

    sep("1. No room limits given: fit is not a question")
    r1 = optimizer.optimize(CATALOGUE, UserRequirements(**common))
    show(r1)
    check(r1.selected is not None and r1.selected.product_id == "ONLINE_ONLY", "the best product is recommended even though its size is not stated")
    check(all(not e.breakdown.spatial_included for e in r1.ranking), "SpaceFit is excluded for every product")
    check(all(abs(sum(e.breakdown.weights.values()) - 1) < 1e-9 for e in r1.ranking), "effective weights sum to 1")
    check(r1.unverified_count == 0 and not r1.incompatible, "nothing is called unverified or incompatible when no limits were given")

    sep("2. Width 210 cm and depth 110 cm given: three kinds of outcome")
    r2 = optimizer.optimize(CATALOGUE, UserRequirements(**common, max_width_cm=210, max_depth_cm=110))
    show(r2)
    check(r2.selected is not None and r2.selected.product_id == "COMPACT", "only the fit-verified product is recommended")
    check({e.product_id for e in r2.unverified} == {"ONLINE_ONLY", "NO_DEPTH"}, "products with a missing size are 'Fit not verified'")
    check([e.product_id for e in r2.incompatible] == ["GRAND"], "the product wider than the limit is 'Dimensionally incompatible'")
    check(max(e.utility for e in r2.unverified) > r2.selected.utility, "an unverified product may score higher, but it is still not recommended as a fit")
    check(r2.selected.breakdown.spatial_included and abs(r2.selected.breakdown.weights["spatial"] - 0.20) < 1e-12, "the verified product keeps the 20% SpaceFit weight")

    sep("3. Only a width limit (210 cm) given: only that axis is checked and scored")
    r3 = optimizer.optimize(CATALOGUE, UserRequirements(**common, max_width_cm=210))
    show(r3)
    check({e.product_id for e in r3.ranking} == {"COMPACT", "NO_DEPTH"}, "a missing depth does not matter when no depth limit was given")
    check({e.product_id for e in r3.unverified} == {"ONLINE_ONLY"}, "only the product with no stated width is unverified")

    sep("Checks")
    print(f"  {'ALL CHECKS PASSED' if not failures else str(len(failures)) + ' CHECK(S) FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
