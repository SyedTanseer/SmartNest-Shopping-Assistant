#!/usr/bin/env python3
"""SmartNest — live SerpApi verification, from a real Google Shopping search to the recommendation stage.

Run from the ``backend`` directory (needs ``SERPAPI_API_KEY`` in ``backend/.env``):

    python src/serpapi_check.py                     # search (cached 6 h), score, recommend - all from real listings
    python src/serpapi_check.py --atlas             # cache in MongoDB Atlas: a repeat run within 6 h costs 0 requests
    python src/serpapi_check.py --dump raw.json     # also save the raw responses (they contain no API key)
    python src/serpapi_check.py --explain           # also ask Gemini to explain the recommendation (needs GEMINI_API_KEY)

What it shows, for the request "a minimalist three-seater sofa in India under ₹30,000" (no room limits given):

  * how many SerpApi requests were made and how many answers came from the cache (and how old they are),
  * every product as parsed: name, price, merchant, URL and URL type, stock state, and which optional attributes
    SerpApi did not state (they stay null),
  * how the UtilityOptimizer treats them: confirmed recommendations, separately-labelled UNVERIFIED candidates
    (stock / seating not confirmed), and rejections with reasons,
  * the information-completeness note and the explanation for the top recommendation.

A fresh search costs 1 SerpApi request plus up to ``SERPAPI_MAX_DETAIL_CALLS`` detail lookups; the usage ledger
(``SERPAPI_LIFETIME_LIMIT``) refuses requests beyond the cap.

Exit codes: 0 = real listings were retrieved, 1 = the API call failed, 2 = not configured.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from smartnest.integrations.products import ProductDiscoveryError, ProductQuery  # noqa: E402
from smartnest.integrations.products.serpapi import (  # noqa: E402
    SEARCH_CACHE_COLLECTION,
    SearchCache,
    SerpApiClient,
    SerpApiConfigError,
    SerpApiProvider,
    load_serpapi_settings,
)
from smartnest.optimization import UserRequirements, UtilityOptimizer  # noqa: E402

#: The request: no room limits, a style preference, three seats, a ₹30,000 budget.
REQUIREMENTS = UserRequirements(budget=30000, min_seating=3, preferred_style="minimalist")
URL_LABELS = {"retailer": "DIRECT RETAILER LISTING", "google_shopping": "GOOGLE SHOPPING PAGE (not a retailer link)"}


class RecordingClient(SerpApiClient):
    """Keeps every raw response so it can be inspected or dumped."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.raw: list[dict] = []

    def search(self, query):
        out = super().search(query)
        self.raw.append({"call": "search", "q": query, "response": out})
        return out

    def product_details(self, page_token):
        out = super().product_details(page_token)
        self.raw.append({"call": "product_details", "response": out})
        return out


def describe(listing: dict) -> str:
    link = listing["listing_url"] or "NO USABLE URL"
    return (f"{listing['name'][:90]}\n      ₹{listing['price']:,.0f} at {listing['merchant']}  |  "
            f"{URL_LABELS.get(listing['listing_url_type'], listing['listing_url_type'])}\n      {link}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dump", metavar="FILE", help="save the raw SerpApi responses to this JSON file")
    parser.add_argument("--atlas", action="store_true", help="use MongoDB Atlas for the search cache")
    parser.add_argument("--explain", action="store_true", help="ask Gemini to explain the recommendation")
    args = parser.parse_args(argv)

    try:
        settings = load_serpapi_settings()
    except SerpApiConfigError as exc:
        print(f"Not configured: {exc}")
        print("Get a key at https://serpapi.com/manage-api-key and add SERPAPI_API_KEY=... to backend/.env")
        return 2

    connection = None
    if args.atlas:
        from smartnest.database import BaseRepository, MongoConnection, load_settings
        connection = MongoConnection(load_settings())
        connection.connect()
        repository = BaseRepository(connection.collection(SEARCH_CACHE_COLLECTION), id_prefix="sc")
        where = "MongoDB Atlas"
    else:
        from smartnest.database import InMemoryRepository
        repository = InMemoryRepository(SEARCH_CACHE_COLLECTION, id_prefix="sc")
        where = "in-memory (this run only)"

    client = RecordingClient(settings)
    ttl = timedelta(hours=settings.cache_ttl_hours)
    provider = SerpApiProvider(client, settings, cache=SearchCache(repository, ttl=ttl))
    query = ProductQuery(category="sofa", max_price=REQUIREMENTS.budget, min_seating=3, style="minimalist")

    print("SmartNest - live SerpApi Google Shopping -> optimizer check")
    print(f"  request: minimalist three-seater sofa in India, budget ₹{REQUIREMENTS.budget:,.0f}, no room limits given")
    print(f"  country={settings.country} language={settings.language} cache={where} ttl={settings.cache_ttl_hours:g} h")
    if client.ledger is not None:
        print(f"  usage cap: {client.ledger.used} of {client.ledger.limit} lifetime requests used before this run")
    try:
        result = provider.discover(query)
    except ProductDiscoveryError as exc:
        print(f"\nFAILED: {exc}")
        return 1

    print(f"\nSearch text: {result.search_queries[0]!r}")
    print(f"SerpApi requests made this run: {result.api_calls} | answers served from cache: {result.cache_hits} | "
          f"stale: {result.stale}")
    print(f"Data retrieved from SerpApi at: {result.retrieved_at.isoformat()}")
    if client.ledger is not None:
        print(f"Lifetime usage after this run: {client.ledger.used} of {client.ledger.limit}")
    print(f"Parsed: {len(result.listings)} usable listings, {len(result.rejected)} rejected "
          f"(of {result.extracted_count} shopping results considered)")

    search_raw = next((r["response"] for r in client.raw if r["call"] == "search"), None)
    if search_raw is not None:
        items = search_raw.get("shopping_results") or []
        counts = Counter(k for item in items if isinstance(item, dict) for k in item)
        print(f"\nFields present on the {len(items)} shopping_results items: "
              + ", ".join(f"{k}({n})" for k, n in sorted(counts.items())))

    print("\nProducts as parsed (missing attributes stay null - nothing is invented):")
    for l in result.listings:
        unstated = [f for f in ("rating", "review_count", "style", "colour", "material", "seating_capacity",
                                "width_cm", "depth_cm") if l.get(f) is None]
        print(f"  - {describe(l)}\n      stock: {l['availability']} | not stated by SerpApi: {', '.join(unstated) or 'none'}")
    for r in result.rejected:
        print(f"  x REJECTED {r.name}: {'; '.join(r.reasons)}")
    for note in result.notes:
        print(f"  note: {note}")

    optimized = UtilityOptimizer().optimize(result.listings, REQUIREMENTS)
    by_id = {l["product_id"]: l for l in result.listings}
    print("\nUtilityOptimizer")
    print(f"  weights per component are redistributed over the criteria that have data; completeness shows how much did.")
    print(f"  {len(optimized.ranking)} confirmed | {optimized.unverified_count} unverified | {optimized.rejected_count} rejected")
    print("\nCONFIRMED recommendations (in stock, every checkable constraint met):")
    for e in optimized.ranking:
        l = by_id[e.product_id]
        mark = "  <- RECOMMENDED" if e is optimized.ranking[0] else ""
        print(f"  #{e.rank} U = {e.utility:.2f}  completeness {e.completeness:.1%}{mark}\n      {describe(l)}")
        print(f"      {e.breakdown.completeness_note}")
    if not optimized.ranking:
        print("  (none)")
    print("\nUNVERIFIED candidates (not recommended as confirmed; see what is unconfirmed):")
    for e in optimized.unverified:
        l = by_id[e.product_id]
        print(f"  #{e.rank} U = {e.utility:.2f}  completeness {e.completeness:.1%}  unconfirmed: {', '.join(e.gaps)}"
              f"\n      {describe(l)}\n      {e.availability_label}\n      why: {'; '.join(e.reasons)}")
    if not optimized.unverified:
        print("  (none)")
    print("\nRejected by the optimizer:")
    for e in optimized.rejected:
        print(f"  - {e.name[:60]} ₹{e.price:,.0f}: {'; '.join(e.reasons)}")
    if not optimized.rejected:
        print("  (none)")

    print(f"\nOptimizer explanation:\n  {optimized.explanation}")

    if optimized.selected is not None:
        from smartnest.integrations.gemini import GeminiService
        from smartnest.integrations.gemini.schemas import RecommendationFacts
        sel = optimized.selected
        l = by_id[sel.product_id]
        b = sel.breakdown
        facts = RecommendationFacts(
            product_id=sel.product_id, product_name=sel.name, price=sel.price, utility=sel.utility,
            component_scores=dict(b.components), retailer=l["merchant"], url=l["url"], url_type=l["listing_url_type"],
            feasible_count=optimized.eligible_count, inspected_count=optimized.total_inspected, budget=REQUIREMENTS.budget,
            runner_up_name=optimized.ranking[1].name if len(optimized.ranking) > 1 else None,
            runner_up_utility=optimized.ranking[1].utility if len(optimized.ranking) > 1 else None,
            fit_status=sel.fit_label, availability_status=sel.availability_label,
            completeness_percent=round(b.completeness * 100, 1), unavailable_attributes=list(b.unavailable),
            requirements={"min_seating": 3, "preferred_style": "minimalist"})
        explanation = GeminiService.template_explanation(facts)
        if args.explain:
            from smartnest.integrations.gemini import GenaiClient, load_gemini_settings
            try:
                explanation = GeminiService(GenaiClient(load_gemini_settings())).explain_recommendation(facts)
            except Exception as exc:  # the template is the honest fallback
                print(f"  (Gemini explanation unavailable: {type(exc).__name__}; showing the template)")
        print(f"\nRecommendation: {sel.name}\n  ₹{sel.price:,.0f} at {l['merchant']}  |  {URL_LABELS[l['listing_url_type']]}"
              f"\n  {l['listing_url']}\n  Explanation ({explanation.source}): {explanation.text}")

    if args.dump:
        Path(args.dump).write_text(json.dumps(client.raw, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nRaw responses saved to {args.dump}")
    print("\nVerified: real SerpApi data reached the optimizer." if result.listings else "\nNo usable listings were retrieved.")
    return 0 if result.listings else 1


if __name__ == "__main__":
    sys.exit(main())
