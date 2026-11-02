#!/usr/bin/env python3
"""SmartNest — live Gemini verification.

Run from the ``backend`` directory (needs ``GEMINI_API_KEY`` in backend/.env):

    python src/gemini_check.py            # 1. a real structured-extraction request
    python src/gemini_check.py --search   # 1 + 2. a real Google-Search-grounded product discovery

What it proves, when it succeeds:
  1. Your key and model work: a real request is sent, the reply parses into the validated
     ``RequirementsExtraction`` schema.
  2. (--search) Google Search grounding returns real source URLs, and the product pipeline turns
     them into validated listings (or rejects them with reasons).

The key is never printed.  Exit status: 0 = verified, 1 = the API call failed, 2 = not configured.
"""

from __future__ import annotations

import sys

from smartnest.integrations.gemini import (
    GeminiError,
    GeminiService,
    GenaiClient,
    load_gemini_settings,
)
from smartnest.integrations.gemini.errors import GeminiConfigError
from smartnest.integrations.products import (
    GeminiSearchProvider,
    HttpImageVerifier,
    ProductDiscoveryError,
    ProductQuery,
    RedirectResolver,
)

SAMPLE = "I want a minimalist three-seater sofa under 30k rupees, it must fit a 2.1 metre wall."


def main(search: bool = False) -> int:
    try:
        settings = load_gemini_settings()
    except GeminiConfigError as exc:
        print(f"NOT CONFIGURED: {exc}")
        print("Add GEMINI_API_KEY=<your key> to backend/.env (it is git-ignored), then run this again.")
        return 2
    print(f"Configured: model={settings.model}, key=set (hidden)")

    client = GenaiClient(settings)
    service = GeminiService(client)

    print(f'\n[1] Real structured extraction request: "{SAMPLE}"')
    try:
        extraction = service.extract_requirements(SAMPLE)
    except GeminiError as exc:
        print(f"FAIL: {exc}")
        return 1
    print("    Parsed and validated:", extraction.model_dump(exclude_none=True, exclude_defaults=True))
    if extraction.sofa_budget != 30000 or extraction.min_seating != 3:
        print("    WARNING: the values differ from what was asked (expected sofa_budget 30000, min_seating 3).")
    else:
        print("    OK: sofa_budget = 30000, min_seating = 3, as stated.")

    if not search:
        print("\nVerified: a live Gemini request succeeded. Add --search to test live product discovery.")
        return 0

    print("\n[2] Real Google-Search-grounded product discovery (this can take a minute)...")
    provider = GeminiSearchProvider(service, resolver=RedirectResolver(), image_verifier=HttpImageVerifier())
    try:
        result = provider.discover(ProductQuery(category="sofa", max_price=30000, min_seating=3, style="minimalist"))
    except ProductDiscoveryError as exc:
        print(f"FAIL: {exc}")
        return 1
    print(f"    {result.summary()}")
    for q in result.search_queries:
        print(f"    search query: {q}")
    for url in result.sources:
        print(f"    source: {url}")
    for listing in result.listings:
        print(f"    ACCEPTED {listing['name'][:50]} | ₹{listing['price']:,.0f} | {listing['width_cm']:g}x{listing['depth_cm']:g} cm "
              f"| {listing['availability']} | {listing['url']}")
    for rejected in result.rejected:
        print(f"    REJECTED {(rejected.name or '?')[:50]} | {'; '.join(rejected.reasons)[:140]}")
    for note in result.notes:
        print(f"    note: {note}")
    if not result.sources:
        print("\nPartial: the search ran but reported no sources.")
        return 1
    print("\nVerified: live grounded search returned real source URLs.")
    return 0


if __name__ == "__main__":
    sys.exit(main(search="--search" in sys.argv[1:]))
