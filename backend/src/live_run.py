#!/usr/bin/env python3
"""SmartNest — complete live workflow: Gemini -> SerpApi -> UtilityOptimizer -> BDI agent -> MongoDB.

Run from the ``backend`` directory (needs ``GEMINI_API_KEY`` and ``SERPAPI_API_KEY`` in ``backend/.env``):

    python src/live_run.py --atlas                      # uses the 6-hour search cache only: 0 SerpApi requests
    python src/live_run.py --atlas --allow-requests 4   # permit a fresh search (1 search + up to 3 detail lookups)

Every language task is a REAL Gemini call and every product comes from REAL SerpApi Google Shopping data (served
from the MongoDB cache when it is fresh; each answer states when it was retrieved).  By default the run may make
**no** new SerpApi requests: if the cache is cold it stops and says so instead of spending quota.

The scenario is built for real listings, which rarely state sizes, ratings, style or colour:

  1. A request in the user's words, with no room limits.  Gemini extracts the requirements; real listings are
     discovered; the optimizer recommends from what the data states (and says what it could not score).
  2. New information (SIMULATED event, labelled as such): the recommended listing goes out of stock.  The BDI agent
     updates its beliefs, recognises its intention is invalid and recommends the next confirmed option.
  3. The user adds room limits ("210 cm wide, 110 cm deep").  Real listings do not state their size, so no product
     can be verified to fit: SmartNest withdraws the confirmed recommendation and lists them as "fit not verified".
     Constraints are NOT weakened to keep a recommendation.
  4. Persistence: knowledge records, agent state saved to MongoDB, the agent restarted and restored.

Exit codes: 0 all checks passed, 1 a check failed, 2 not configured or the cache was cold and no requests allowed.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import adaptive_run as demo  # noqa: E402  (reuses its output helpers and world builder)
from smartnest.adaptive import AdaptiveSession  # noqa: E402
from smartnest.bdi import BDIAgent, BDIPlanner, RepositoryAgentStateStore  # noqa: E402
from smartnest.core.models import EnvironmentalEvent, EventType  # noqa: E402
from smartnest.integrations.gemini import GeminiService  # noqa: E402
from smartnest.integrations.products import ProductDiscoveryError  # noqa: E402
from smartnest.integrations.products.serpapi import (  # noqa: E402
    SEARCH_CACHE_COLLECTION,
    SearchCache,
    SerpApiCandidateSource,
    SerpApiClient,
    SerpApiConfigError,
    SerpApiProvider,
    load_serpapi_settings,
)
from smartnest.integrations.products.serpapi.client import UsageLedger  # noqa: E402
from smartnest.knowledge import KnowledgeKind, KnowledgeStore, Provenance  # noqa: E402
from smartnest.optimization import UserRequirements, UtilityOptimizer  # noqa: E402

USER_ID = "live_user"
PROJECT_ID = "project_live_living_room"
AGENT_ID = "bdi_live"
MESSAGE = "I'd like a minimalist three-seater sofa under ₹30,000 for my living room."
ROOM_MESSAGE = "My sofa wall is only 210 cm wide and the space is 110 cm deep."
SIMPLE = UserRequirements(budget=30000, min_seating=3, preferred_style="minimalist")


def main(atlas: bool, allow_requests: int, detail_calls: int) -> int:
    failures: list[str] = []

    def check(condition: bool, text: str) -> None:
        print(f"    [{'OK' if condition else 'FAIL'}] {text}")
        if not condition:
            failures.append(text)

    try:
        settings = dataclasses.replace(load_serpapi_settings(), max_detail_calls=detail_calls)
    except SerpApiConfigError as exc:
        print(f"Not configured: {exc}")
        return 2

    client_llm, _, knowledge_repo, state_repo, connection, created = demo.build_world(True, atlas)
    try:
        # SerpApi: the lifetime ledger already caps the project; this run additionally may spend only `allow_requests`.
        serp = SerpApiClient(settings)
        used_before = UsageLedger(settings.usage_file, 10 ** 9).used
        serp.ledger = UsageLedger(settings.usage_file, used_before + allow_requests)
        if connection is not None:
            from smartnest.database import BaseRepository
            cache_repo = BaseRepository(connection.collection(SEARCH_CACHE_COLLECTION), id_prefix="sc")
        else:
            from smartnest.database import InMemoryRepository
            cache_repo = InMemoryRepository(SEARCH_CACHE_COLLECTION, id_prefix="sc")
        ttl = timedelta(hours=settings.cache_ttl_hours)
        source = SerpApiCandidateSource(SerpApiProvider(serp, settings, cache=SearchCache(cache_repo, ttl=ttl)), max_age=ttl)

        knowledge = KnowledgeStore(knowledge_repo)
        gemini = GeminiService(client_llm)
        store = RepositoryAgentStateStore(state_repo)

        def new_agent() -> BDIAgent:
            return BDIAgent(AGENT_ID, BDIPlanner.default(UtilityOptimizer(), source), state_store=store)

        agent = new_agent()
        session = AdaptiveSession(agent=agent, gemini=gemini, source=source, knowledge=knowledge,
                                  user_id=USER_ID, project_id=PROJECT_ID)

        demo.sep("SmartNest - live workflow: Gemini -> SerpApi -> UtilityOptimizer -> BDI -> MongoDB")
        print(f"  Gemini model     : {demo.load_gemini_settings().model if hasattr(demo, 'load_gemini_settings') else 'see .env'} (real calls)")
        print(f"  Product source   : SerpApi Google Shopping, gl={settings.country} (real data; cache TTL {settings.cache_ttl_hours:g} h)")
        print(f"  Persistence      : {'MongoDB Atlas' if atlas else 'in-memory repositories'}")
        print(f"  SerpApi requests : this run may make {allow_requests}; lifetime used so far {used_before} "
              f"of {settings.lifetime_limit if settings.lifetime_limit is not None else 'unlimited'}")

        # ── 1. The request ──────────────────────────────────────────────────────
        demo.sep("1. The request (no room limits given)")
        said = Provenance(source="user", source_type="user", method="project setup")
        knowledge.create(KnowledgeKind.HOME_PROJECT, f"project:{PROJECT_ID}", {"name": "Living room", "location": "India"}, said)
        print(f'  User: "{MESSAGE}"\n')
        turn = session.request(MESSAGE)
        if turn.discovery is None:
            print("  Product discovery did not run:", "; ".join(turn.notes))
            print("  The search cache is cold and this run may not make SerpApi requests. "
                  "Re-run with --allow-requests 4 to spend quota on a fresh search.")
            return 2
        d = turn.discovery
        print(f"  SerpApi: {d.api_calls} request(s) made this run, {d.cache_hits} answer(s) from the cache; "
              f"data retrieved at {d.retrieved_at.isoformat()}")
        demo.show_turn(session, turn)
        rec = turn.recommendation
        oracle = UtilityOptimizer().optimize(source.current_listings, SIMPLE)
        print()
        check(turn.extraction is not None and turn.extraction.sofa_budget == 30000 and turn.extraction.min_seating == 3
              and turn.extraction.preferred_style == "minimalist", "Gemini extracted ₹30,000 / 3 seats / minimalist")
        check(d.api_calls <= allow_requests, f"SerpApi requests this run ({d.api_calls}) stayed within the allowance ({allow_requests})")
        check(len(d.listings) > 0, f"real listings were retrieved ({len(d.listings)})")
        check(rec is not None, "the agent adopted a recommendation")
        if rec is not None:
            listing = next(l for l in source.current_listings if l["product_id"] == rec.product_id)
            check(oracle.selected is not None and rec.product_id == oracle.selected.product_id,
                  "it is the optimizer's top confirmed product (independent re-run)")
            check(rec.availability == "in_stock", "its stock is confirmed in the retrieved data")
            check(rec.url_type == "retailer" and (rec.url or "").startswith("https://"),
                  "it has a genuine retailer purchase link (not a Google Shopping page)")
            check(rec.price <= 30000, f"it respects the budget (₹{rec.price:,.0f})")
            check(rec.facts.unavailable_attributes and (rec.completeness or 1) < 1,
                  f"its limited evidence is stated (completeness {rec.completeness:.0%}; unavailable: "
                  f"{', '.join(rec.facts.unavailable_attributes)})")
            check(all(listing.get(k) is None for k in rec.facts.unavailable_attributes
                      if k in ("rating", "style", "colour")), "attributes SerpApi did not state are null, not invented")
        check(all(o["product_id"] != (rec.product_id if rec else None) for o in turn.unverified_options),
              "unverified candidates are listed separately from the recommendation")

        # ── 2. New information ───────────────────────────────────────────────────
        demo.sep("2. New information: the recommended listing goes out of stock")
        first = rec
        mark = len(agent.transitions)
        print("  (SIMULATED event on the real recommended listing - live stock rarely changes on demand)\n")
        turn2 = session.ingest_event(EnvironmentalEvent(
            event_type=EventType.PRODUCT_AVAILABILITY, source="simulated_event",
            payload={"product_id": first.product_id, "product_name": first.name, "availability": "out_of_stock"}))
        demo.show_turn(session, turn2)
        print("\n  Transition log for this change:")
        for t in agent.transitions[mark:]:
            if t.kind in ("belief_updated", "intention_invalidated", "intention_replaced", "intention_adopted"):
                print("    " + t.format().replace("\n", "\n    "))
        second = turn2.recommendation
        print()
        check(agent.beliefs.value(first.product_id, "availability") in ("out_of_stock", "unavailable"), "the availability belief was updated")
        check(any(t.kind == "intention_invalidated" for t in agent.transitions), "the agent recognised its intention was invalid")
        rest = [l for l in source.current_listings if l["product_id"] != first.product_id]
        expected = UtilityOptimizer().optimize(rest, SIMPLE).selected
        check(second is not None and expected is not None and second.product_id == expected.product_id and second.product_id != first.product_id,
              "it replanned to the optimizer's best remaining confirmed product")
        if second is not None:
            check(second.url_type == "retailer" and second.availability == "in_stock", "the new choice is a confirmed in-stock retailer listing")

        # ── 3. Room limits ───────────────────────────────────────────────────────
        demo.sep("3. The user adds room limits - real listings cannot be verified against them")
        print(f'  User: "{ROOM_MESSAGE}"\n')
        turn3 = session.chat(ROOM_MESSAGE)
        demo.show_turn(session, turn3)
        print()
        width, depth = agent.beliefs.value("user", "max_width_cm"), agent.beliefs.value("user", "max_depth_cm")
        check(width == 210 and depth == 110, f"Gemini extracted the limits and validation applied them ({width} x {depth} cm)")
        limited = UserRequirements(budget=30000, max_width_cm=210, max_depth_cm=110, min_seating=3, preferred_style="minimalist")
        result = UtilityOptimizer().optimize(rest, limited)
        check(result.selected is None, "independent re-run: no remaining listing can be verified to fit")
        check(turn3.recommendation is None, "SmartNest withdrew the confirmed recommendation instead of claiming a fit")
        check(len(turn3.unverified_options) > 0 and all("fit" in o["gaps"] and o["fit_label"] == "Fit not verified"
                                                          for o in turn3.unverified_options),
              "the candidates are shown as 'Fit not verified', with the reason")
        check(all(not (e.spatial and e.spatial.status == "verified") for e in result.evaluations),
              "no listing is marked 'Fit verified' (none states its dimensions)")

        # ── 4. Persistence ───────────────────────────────────────────────────────
        demo.sep("4. Persistence: knowledge, saved agent state, restart and restore")
        for kind in KnowledgeKind:
            n = knowledge.count(kind)
            if n:
                print(f"  {kind.value:16} {n} record(s)")
        example = knowledge.find(KnowledgeKind.PRODUCT_SPEC)[:1]
        if example:
            doc = example[0].to_document()
            print("\n  Example product record (provenance shown):")
            print("    " + json.dumps({k: doc.get(k) for k in ("knowledge_id", "entity", "attributes", "provenance")},
                                      ensure_ascii=False, default=str)[:700])
        check(knowledge.count(KnowledgeKind.PRODUCT_SPEC) >= 1, "listings were persisted as knowledge with SerpApi provenance")
        if example:
            check(example[0].provenance.method == "serpapi_google_shopping", "their provenance method is serpapi_google_shopping")
        agent.save_state()
        saved = state_repo.get(AGENT_ID)
        print(f"\n  Saved agent state in {'MongoDB Atlas' if atlas else 'memory'}: beliefs={len(saved['beliefs'])} "
              f"desires={len(saved['desires'])} intentions={len(saved['intentions'])} transitions={len(saved['transitions'])}")
        beliefs_before, transitions_before = len(agent.beliefs), len(agent.transitions)
        width_before = agent.beliefs.value("user", "max_width_cm")
        stock_before = agent.beliefs.value(first.product_id, "availability")
        del agent
        print("  ... agent process restarted ...")
        agent = new_agent()
        print(f"  state_restored = {agent.state_restored}")
        check(agent.state_restored, "state was restored from the store")
        check(len(agent.beliefs) == beliefs_before, f"all {beliefs_before} beliefs were restored")
        check(len(agent.transitions) >= transitions_before, "the decision history was restored")
        check(agent.beliefs.value("user", "max_width_cm") == width_before == 210, "the room limit survived the restart")
        check(agent.beliefs.value(first.product_id, "availability") == stock_before,
              "it still remembers that the first recommendation went out of stock")
        if atlas:
            check(connection is not None, "the run used MongoDB Atlas")

        demo.sep("Usage and result")
        used_after = UsageLedger(settings.usage_file, 10 ** 9).used
        print(f"  SerpApi requests made by this run: {used_after - used_before} (lifetime used: {used_after})")
        if failures:
            print(f"  {len(failures)} check(s) FAILED:")
            for text in failures:
                print(f"    - {text}")
            return 1
        print("  All checks passed. Nothing was invented: unstated attributes are null and unverified requirements are labelled.")
        return 0
    except ProductDiscoveryError as exc:
        print(f"  Product discovery failed: {exc}")
        return 2
    finally:
        for repo, document_id in created:
            try:
                repo.delete(document_id)
            except Exception:
                pass
        if connection is not None:
            connection.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SmartNest live end-to-end workflow", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--atlas", action="store_true", help="persist knowledge, agent state and the search cache in MongoDB Atlas")
    parser.add_argument("--allow-requests", type=int, default=0, metavar="N",
                        help="SerpApi requests this run may make (default 0: use the cache only)")
    parser.add_argument("--detail-calls", type=int, default=3, metavar="N", help="product-detail lookups per search (default 3)")
    args = parser.parse_args()
    sys.exit(main(atlas=args.atlas, allow_requests=args.allow_requests, detail_calls=args.detail_calls))
