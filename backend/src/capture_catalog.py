#!/usr/bin/env python3
"""SmartNest - capture REAL SerpApi listings for each furnishing category and save them as snapshots.

    python src/capture_catalog.py --plan                          # show what would be spent; no request is made
    python src/capture_catalog.py --atlas --spend 5               # spend at most 5 requests (1 search per category)
    python src/capture_catalog.py --from-cache sofa --atlas       # sofa from the old Atlas cache, 0 requests

Each category costs exactly ONE SerpApi request (a Google Shopping search, no product-details lookups).  Details
cost extra requests, so listings captured here link to the Google Shopping product page, which the interface labels
as such - it never calls a Google page a retailer link.

Safety:
  * the lifetime cap (``SERPAPI_LIFETIME_LIMIT`` in backend/.env) must be loaded, or the script refuses to run
    (``unlimited`` removes the cap on purpose; ``--spend`` still limits each run);
  * ``--spend N`` is an additional per-run cap;
  * raw responses are cached in MongoDB Atlas (``--atlas``) and parsed listings are saved to ``backend/data/catalog``
    (git-ignored), so nothing is ever searched twice;
  * categories that already have a snapshot are skipped unless ``--force``.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from smartnest.furnishing import CATEGORIES, SnapshotCatalog  # noqa: E402
from smartnest.furnishing.catalog import ORIGIN_REAL, CategoryListings  # noqa: E402
from smartnest.integrations.products import ProductDiscoveryError, ProductQuery  # noqa: E402
from smartnest.integrations.products.serpapi import (  # noqa: E402
    SEARCH_CACHE_COLLECTION,
    SearchCache,
    SerpApiClient,
    SerpApiConfigError,
    SerpApiProvider,
    load_serpapi_settings,
)
from smartnest.integrations.products.serpapi.client import UsageLedger  # noqa: E402

#: Searches used for capture, one per category (plain wording; style is not searched on because Google Shopping
#: listings do not state a style, so a style word would only bias the search).
QUERIES = {
    "bed": "queen size bed", "mattress": "queen size mattress", "wardrobe": "wardrobe 3 door",
    "dining_table": "4 seater dining table set", "refrigerator": "double door refrigerator 250 L",
    "tv_unit": "tv unit",
    # a child's room (captured for the "decorate my 10 year old son's bedroom ... cartoons and monsters" request)
    "kids_bed": "kids single bed", "study_table": "study table", "bookshelf": "bookshelf",
    "wall_decor": "monster cartoon wall poster", "study_chair": "study chair", "table_lamp": "table lamp",
    "rug": "kids rug", "bean_bag": "bean bag", "toy_storage": "toy storage organizer",
}


def make_cache(atlas: bool, ttl: timedelta):
    if atlas:
        from smartnest.database import BaseRepository, MongoConnection, load_settings
        connection = MongoConnection(load_settings())
        connection.connect()
        return SearchCache(BaseRepository(connection.collection(SEARCH_CACHE_COLLECTION), id_prefix="sc"), ttl=ttl), connection
    from smartnest.database import InMemoryRepository
    return SearchCache(InMemoryRepository(SEARCH_CACHE_COLLECTION, id_prefix="sc"), ttl=ttl), None


def to_listings(category_id: str, result, query_text: str) -> CategoryListings:
    records = []
    for r in result.listings:
        r = dict(r)
        r["category"] = category_id
        r["data_origin"] = ORIGIN_REAL
        records.append(r)
    rejected = [{"name": x.name, "reasons": x.reasons} for x in result.rejected]
    return CategoryListings(category_id, records, ORIGIN_REAL, result.retrieved_at.isoformat(), query_text, rejected,
                            list(result.notes))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", action="store_true", help="only show what would be done")
    ap.add_argument("--atlas", action="store_true", help="cache raw responses in MongoDB Atlas")
    ap.add_argument("--spend", type=int, default=0, metavar="N", help="max SerpApi requests this run may make")
    ap.add_argument("--only", nargs="*", choices=sorted(QUERIES), help="capture only these categories")
    ap.add_argument("--force", action="store_true", help="re-capture categories that already have a snapshot")
    ap.add_argument("--from-cache", nargs="*", default=[], metavar="CATEGORY",
                    help="rebuild a snapshot from the cached responses of an earlier search (0 requests)")
    args = ap.parse_args()

    try:
        settings = load_serpapi_settings()
    except SerpApiConfigError as exc:
        print(f"SerpApi is not configured: {exc}")
        return 2
    if settings.lifetime_limit is None and not settings.uncapped:
        print("REFUSING TO RUN: SERPAPI_LIFETIME_LIMIT is not loaded, so the quota cap would not be enforced.\n"
              "(To remove the cap on purpose, set SERPAPI_LIFETIME_LIMIT=unlimited in backend/.env.)")
        return 2
    ledger = UsageLedger(settings.usage_file, 10 ** 9)
    remaining = None if settings.lifetime_limit is None else settings.lifetime_limit - ledger.used
    snaps = SnapshotCatalog()
    wanted = [c for c in (args.only or QUERIES) if args.force or not snaps.listings(c).records]
    if remaining is None:
        print(f"Lifetime cap: none (removed on purpose), used {ledger.used}. This run is still limited by --spend.")
    else:
        print(f"Lifetime cap {settings.lifetime_limit}, used {ledger.used}, remaining {remaining}")
    print(f"Snapshot directory: {snaps.directory}")
    for c in QUERIES:
        state = "has a snapshot" if snaps.listings(c).records else "needs capture"
        print(f"  {c:13s} query={QUERIES[c]!r:36s} {state}{'  <- will search' if c in wanted else ''}")
    budget = args.spend if remaining is None else min(args.spend, remaining)
    print(f"Planned spend: {len(wanted)} request(s); this run may spend {budget}.")
    if args.plan:
        return 0

    settings = dataclasses.replace(settings, max_detail_calls=0)
    cache, connection = make_cache(args.atlas, timedelta(hours=settings.cache_ttl_hours))
    code = 0
    try:
        # 1. Rebuild from earlier cached responses (no request).
        for cat in args.from_cache:
            long_ttl = SearchCache(cache._repo, ttl=timedelta(days=3650))
            client = SerpApiClient(settings)
            client.ledger = UsageLedger(settings.usage_file, ledger.used)         # zero allowance: any call raises
            provider = SerpApiProvider(client, dataclasses.replace(settings, max_detail_calls=3), cache=long_ttl)
            text = {"sofa": "3 seater minimalist sofa"}.get(cat) or QUERIES[cat]
            result = provider.discover(ProductQuery(category=text.replace("3 seater minimalist ", ""),
                                                    min_seating=3 if cat == "sofa" else None,
                                                    max_price=30000 if cat == "sofa" else None,
                                                    style="minimalist" if cat == "sofa" else None))
            if result.api_calls:
                print(f"  {cat}: unexpectedly needed {result.api_calls} API call(s)")
            snaps.save(to_listings(cat, result, text))
            print(f"  {cat}: {len(result.listings)} listings from cache (0 requests)")

        # 2. Fresh searches within the cap.
        client = SerpApiClient(settings)
        client.ledger = UsageLedger(settings.usage_file, ledger.used + budget)
        provider = SerpApiProvider(client, settings, cache=cache)
        for cat in [c for c in wanted if c not in args.from_cache]:
            if client.ledger.remaining < 1:
                print(f"  {cat}: SKIPPED - this run's allowance is spent")
                continue
            try:
                result = provider.discover(ProductQuery(category=QUERIES[cat]))
            except ProductDiscoveryError as exc:
                print(f"  {cat}: FAILED - {exc}")
                code = 1
                continue
            snaps.save(to_listings(cat, result, QUERIES[cat]))
            prices = sorted(r["price"] for r in result.listings)
            print(f"  {cat}: {len(result.listings)} listings (+{len(result.rejected)} rejected), "
                  f"{result.api_calls} request(s), {result.cache_hits} cache hit(s)"
                  + (f", price Rs {prices[0]:,.0f}-{prices[-1]:,.0f}" if prices else ""))
    finally:
        if connection is not None:
            connection.close()
    print(f"Lifetime used now: {UsageLedger(settings.usage_file, 10 ** 9).used} of "
          f"{settings.lifetime_limit if settings.lifetime_limit is not None else 'unlimited'}")
    return code


if __name__ == "__main__":
    sys.exit(main())
