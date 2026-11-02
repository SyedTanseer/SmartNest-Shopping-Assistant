#!/usr/bin/env python3
"""SmartNest — Assignment 4.2: knowledge representation and adaptive API integration.

Run from the ``backend`` directory:

    python src/adaptive_run.py                 # SIMULATED Gemini + market (no key, no network)
    python src/adaptive_run.py --live          # REAL Gemini (language) + REAL SerpApi Google Shopping (products);
                                               #   needs GEMINI_API_KEY and SERPAPI_API_KEY in backend/.env
    python src/adaptive_run.py --hybrid        # REAL Gemini for language, SIMULATED market (needs GEMINI_API_KEY only)
    python src/adaptive_run.py --atlas         # also persist knowledge and agent state in MongoDB Atlas
    python src/adaptive_run.py --live --atlas

The story (the same in both modes):
  1. The user asks for "a minimalist three-seater sofa under ₹30,000".
  2. Gemini extracts structured requirements; they are validated and recorded as knowledge.
  3. Product discovery retrieves listings; incomplete or unverifiable ones are rejected with reasons.
  4. Listings are recorded as knowledge (with provenance), the UtilityOptimizer ranks the feasible
     ones and the BDI agent adopts the best.
  5. New information arrives (a listing goes out of stock): the agent updates its beliefs, finds
     its intention invalid, and replans.
  6. The user changes their mind in words ("contemporary instead"): Gemini proposes the change,
     validation applies it, and the agent replans.

SIMULATED mode is for demonstration only: the "Gemini" is a scripted stand-in and the products are
invented, with ``.invalid`` links that cannot resolve.  Only ``--live`` shows real data.
Google Search grounding is NOT used for product discovery; SerpApi supplies the listings.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import timedelta
from typing import Any

from smartnest.adaptive import AdaptiveSession, SessionTurn
from smartnest.bdi import BDIAgent, BDIPlanner, RepositoryAgentStateStore
from smartnest.core.models import EnvironmentalEvent, EventType
from smartnest.database import BaseRepository, InMemoryRepository, MongoConnection, load_settings
from smartnest.integrations.gemini import (
    GeminiService,
    PlanChangeRequest,
    ProductListExtraction,
    RecommendationExplanation,
    RequirementsExtraction,
)
from smartnest.integrations.gemini.client import LLMResponse
from smartnest.integrations.gemini.testing import grounded_response, json_response
from smartnest.integrations.products import GeminiSearchProvider, LiveCandidateSource
from smartnest.knowledge import (
    KNOWLEDGE_COLLECTION,
    KnowledgeKind,
    KnowledgeStore,
    Provenance,
    Relationship,
)
from smartnest.optimization import UserRequirements, UtilityOptimizer

USER_ID = "demo_user"
PROJECT_ID = "project_living_room"
MESSAGE = "I'd like a minimalist three-seater sofa under ₹30,000 for my living room."
FOLLOW_UP = "Actually, I'd prefer a contemporary style instead of minimalist."


def sep(title: str) -> None:
    print(f"\n{'=' * 80}\n  {title}\n{'=' * 80}")


# ══ The simulated world (SIMULATED mode only) ════════════════════════════════

class SimulatedMarket:
    """Invented listings with unresolvable ``.invalid`` links.  Not real products."""

    def __init__(self) -> None:
        def item(slug, name, price, w, d, seats, material, colour, style, rating, reviews, stock="in_stock"):
            return {"name": name, "price": price, "width_cm": w, "depth_cm": d, "height_cm": 85,
                    "seating_capacity": seats, "material": material, "colour": colour, "style": style,
                    "rating": rating, "review_count": reviews, "availability": stock, "retailer": "SimFurnish",
                    "source_url": f"https://simfurnish.invalid/p/{slug}", "dimensions_unit": "cm", "currency": "INR"}

        self.listings = [
            item("nordic-lite", "SIMULATED Nordic Lite 3-Seater", 22999, 205, 88, 3, "fabric", "neutral", "minimalist", 4.2, 310),
            item("haven-compact", "SIMULATED Haven Compact 3-Seater", 27499, 198, 92, 3, "leather", "grey", "minimalist", 4.5, 88),
            item("urban-contemporary", "SIMULATED Urban Contemporary 3-Seater", 28999, 200, 95, 3, "fabric", "taupe", "contemporary", 4.6, 150),
            item("grand-lounge", "SIMULATED Grand Lounge 3-Seater", 31999, 207, 96, 3, "fabric", "beige", "minimalist", 4.7, 220),
            item("wide-sectional", "SIMULATED Wide Sectional 3-Seater", 25999, 225, 95, 3, "fabric", "neutral", "minimalist", 4.4, 95),
            item("budget-loveseat", "SIMULATED Budget Loveseat", 16999, 150, 80, 2, "fabric", "neutral", "minimalist", 4.0, 60),
            item("mystery-sofa", "SIMULATED Mystery Sofa 3-Seater", 21999, 200, 90, 3, "fabric", "grey", "minimalist", None, None),
        ]

    def find(self, slug: str) -> dict:
        return next(l for l in self.listings if l["source_url"].endswith("/" + slug))

    def set_out_of_stock(self, slug: str) -> None:
        self.find(slug)["availability"] = "out_of_stock"


class SimulatedGemini:
    """A scripted stand-in for Gemini.  It parses simple phrases and replays the simulated market.

    This is NOT a language model; it exists so the real pipeline can be demonstrated without
    credentials.  Every call is routed by the schema or tool the real service asked for.
    """

    STYLES = ("minimalist", "contemporary", "scandinavian", "modern", "traditional", "industrial")

    def __init__(self, market: SimulatedMarket) -> None:
        self.market = market
        self.calls = 0

    def generate(self, prompt: str, *, system=None, schema=None, grounding=False, url_context=False,
                 temperature=0.0, max_output_tokens=None) -> LLMResponse:
        self.calls += 1
        if grounding:
            return self._search()
        if schema is RequirementsExtraction:
            return json_response(self._requirements(self._message(prompt)))
        if schema is PlanChangeRequest:
            return json_response(self._change(self._message(prompt)))
        if schema is ProductListExtraction:
            return json_response({"products": [self._extraction(l) for l in self.market.listings]})
        if schema is RecommendationExplanation:
            return json_response(self._explanation(prompt))
        raise AssertionError(f"unexpected call: schema={schema}")

    @staticmethod
    def _message(prompt: str) -> str:
        match = re.search(r"<user_message>\n(.*?)\n</user_message>", prompt, re.DOTALL)
        return match.group(1) if match else prompt

    def _requirements(self, text: str) -> dict:
        out: dict[str, Any] = {"category": "sofa", "confidence": 0.95, "assumptions": [], "unclear": []}
        money = re.search(r"(?:under|below|up to|max)\s*(?:₹|rs\.?\s*)?([\d,]+)\s*(k)?", text, re.IGNORECASE)
        if money:
            amount = float(money.group(1).replace(",", "")) * (1000 if money.group(2) else 1)
            out["sofa_budget"] = amount
        for word, n in (("one", 1), ("two", 2), ("three", 3), ("four", 4)):
            if re.search(rf"\b{word}[- ]seater\b", text, re.IGNORECASE):
                out["min_seating"] = n
        if m := re.search(r"\b(\d)[- ]seater\b", text):
            out["min_seating"] = int(m.group(1))
        for style in self.STYLES:
            if style in text.lower():
                out["preferred_style"] = style
        out["unclear"] = ["room size was not stated"]
        return out

    def _change(self, text: str) -> dict:
        # "I'd prefer X instead of Y": the style the user wants comes before "instead".
        wanted = text.lower().split("instead")[0]
        if re.search(r"prefer|switch|want|change|like", wanted):
            for style in self.STYLES:
                if style in wanted:
                    return {"intent": "modify_requirements", "confidence": 0.9,
                            "changes": [{"field": "preferred_style", "action": "set", "value": style}]}
        return {"intent": "other", "changes": [], "confidence": 0.5}

    def _search(self) -> LLMResponse:
        lines = []
        for l in self.market.listings:
            rating = f"rated {l['rating']} ({l['review_count']} reviews)" if l["rating"] is not None else "no rating shown"
            stock = "in stock" if l["availability"] == "in_stock" else "out of stock"
            lines.append(f"{l['name']} ({l['retailer']}): Rs {l['price']:,}, {l['width_cm']} x {l['depth_cm']} x "
                         f"{l['height_cm']} cm, {l['seating_capacity']}-seater, {l['material']}, {l['colour']}, "
                         f"{l['style']} style, {rating}, {stock}. {l['source_url']}")
        return grounded_response("\n".join(lines), [l["source_url"] for l in self.market.listings],
                                 queries=["SIMULATED search: three-seater sofa India"])

    @staticmethod
    def _extraction(l: dict) -> dict:
        return {**l, "style_inferred": False}

    @staticmethod
    def _explanation(prompt: str) -> dict:
        facts = json.loads(prompt.split("FACTS:\n", 1)[1])
        text = (f"{facts['product_name']} is the best match for your requirements at ₹{facts['price']:,.0f}, with an "
                f"overall score of {facts['utility']:.2f} out of 100.")
        if facts.get("runner_up_name"):
            text += (f" It beat {facts['runner_up_name']} ({facts['runner_up_utility']:.2f}) among "
                     f"{facts['feasible_count']} options that meet your requirements.")
        return {"text": text, "highlights": []}


class HybridClient:
    """Real Gemini for every language task; the simulated market for the Google Search step only.

    For keys whose project has no Google Search grounding quota (the grounded call is rejected with
    429).  Requirement extraction, product extraction from the (simulated) search text, explanations
    and conversational changes are all REAL model calls; only the search results are invented.
    """

    def __init__(self, real, simulated: SimulatedGemini) -> None:
        self._real = real
        self._simulated = simulated

    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        if kwargs.get("grounding"):
            return self._simulated._search()
        return self._real.generate(prompt, **kwargs)


class IdentityResolver:
    def resolve(self, url: str) -> str:
        return url


# ══ Output helpers ═══════════════════════════════════════════════════════════

def show_discovery(turn: SessionTurn) -> None:
    d = turn.discovery
    if d is None:
        return
    print(f"  Discovery ({d.provider_id}): {d.summary()}")
    if d.api_calls or d.cache_hits:
        print(f"    SerpApi requests made: {d.api_calls}; served from cache: {d.cache_hits}"
              f"{'; STALE data in use' if d.stale else ''}; data fetched {d.retrieved_at.isoformat()}")
    for q in d.search_queries:
        print(f"    search query: {q}")
    for source in d.sources[:8]:
        print(f"    source: {source}")
    for l in d.listings:
        link = l["url"] or f"(no direct retailer link; {l.get('url_type', 'n/a')}: {l.get('product_page_url')})"
        size = f"{l['width_cm']:g}x{l['depth_cm']:g} cm" if l.get("width_cm") and l.get("depth_cm") else "size not stated"
        print(f"    ACCEPTED  {l['name'][:44]:44} ₹{l['price']:>9,.0f}  {size}  "
              f"{l['availability']:12}  {l.get('merchant') or ''}  {link}")
    for r in d.rejected:
        known = ""
        if r.details:
            price = r.details.get("price")
            known = f"[{r.details.get('merchant') or '?'}, {('₹%s' % format(price, ',.0f')) if price else 'no price'}] "
        print(f"    REJECTED  {(r.name or '?')[:44]:44} {known}{'; '.join(r.reasons)[:120]}")
    for note in d.notes:
        print(f"    note: {note}")


def show_ranking(session: AdaptiveSession) -> None:
    outcome = session.agent.last_outcomes.get("des_select_sofa")
    if outcome is None or outcome.optimization is None:
        return
    result = outcome.optimization
    print(f"  UtilityOptimizer: {result.eligible_count} feasible of {result.total_inspected} candidates")
    for e in result.ranking[:5]:
        mark = "★" if e.rank == 1 else " "
        print(f"    {mark} #{e.rank} {(e.name or e.product_id)[:44]:44} ₹{e.price:>9,.0f}  U = {e.utility:6.2f}")
    for e in result.rejected[:6]:
        print(f"      ✗ {(e.name or e.product_id)[:44]:44} {e.reasons[0][:90]}")


def show_turn(session: AdaptiveSession, turn: SessionTurn) -> None:
    if turn.extraction is not None:
        print("  Gemini's structured extraction (validated):")
        print("    " + json.dumps(turn.extraction.model_dump(exclude_none=True, exclude_defaults=False), ensure_ascii=False)[:420])
    if turn.change_request is not None:
        print("  Gemini's interpretation of the follow-up (validated):")
        print("    " + json.dumps(turn.change_request.model_dump(exclude_none=True), ensure_ascii=False))
    if turn.applied:
        print(f"  Requirements applied to the agent's beliefs after validation (known + newly stated): {turn.applied}")
    for name, reason in turn.rejected.items():
        print(f"  Rejected value {name}: {reason}")
    show_discovery(turn)
    show_ranking(session)
    for log in turn.action_logs:
        print(f"  BDI decision: {log.rule_name} - {log.action_description}")
    for note in turn.notes:
        print(f"  note: {note}")
    for need in turn.needs:
        print(f"  needs: {need}")
    rec = turn.recommendation
    for o in turn.unverified_options:
        print(f"  NOT VERIFIED (not a confirmed recommendation): {o['name']} ₹{o['price']:,.0f} "
              f"[{', '.join(o['gaps'])} unconfirmed; {o['availability_label']}]")
    if rec is None:
        print("  RECOMMENDATION: none")
        return
    print(f"  RECOMMENDATION: {rec.name}")
    print(f"    price ₹{rec.price:,.0f} | utility {rec.utility:.2f}/100 | retailer {rec.retailer}")
    kind = {"retailer": "direct retailer listing", "google_shopping": "Google Shopping page, NOT a retailer link"}.get(
        rec.url_type or ("retailer" if rec.url else ""), "link")
    print(f"    link: {rec.listing_url or rec.url} ({kind}; verified {rec.verified_at})")
    if rec.completeness is not None and rec.completeness < 1:
        print(f"    score rests on {rec.completeness:.0%} of the applicable criteria; not available: "
              f"{', '.join(rec.facts.unavailable_attributes) or 'none'}")
    print(f"    why ({rec.explanation.source}): {rec.explanation.text}")


# ══ Main ═════════════════════════════════════════════════════════════════════

def build_world(live: bool, atlas: bool, hybrid: bool = False):
    """Returns (llm_client, market_or_None, knowledge_repo, state_repo, connection_or_None, tracker)."""
    connection = None
    created_ids: list[tuple[Any, str]] = []

    class Tracking:
        """Wraps a repository and remembers what this run created, so it can clean up after itself."""

        def __init__(self, inner):
            self.inner = inner

        def upsert(self, document_id, document):
            if self.inner.get(document_id) is None:
                created_ids.append((self.inner, document_id))
            return self.inner.upsert(document_id, document)

        def __getattr__(self, name):
            return getattr(self.inner, name)

    if atlas:
        connection = MongoConnection(load_settings())
        connection.connect()
        knowledge_repo = Tracking(BaseRepository(connection.collection(KNOWLEDGE_COLLECTION), id_prefix="kn"))
        state_repo = Tracking(BaseRepository(connection.collection("agent_states"), id_prefix="agent"))
    else:
        knowledge_repo = Tracking(InMemoryRepository(KNOWLEDGE_COLLECTION, id_prefix="kn"))
        state_repo = Tracking(InMemoryRepository("agent_states", id_prefix="agent"))

    if live:
        from smartnest.integrations.gemini import GenaiClient, load_gemini_settings
        client = GenaiClient(load_gemini_settings())
        market = None
    elif hybrid:
        from smartnest.integrations.gemini import GenaiClient, load_gemini_settings
        market = SimulatedMarket()
        client = HybridClient(GenaiClient(load_gemini_settings()), SimulatedGemini(market))
    else:
        market = SimulatedMarket()
        client = SimulatedGemini(market)
    return client, market, knowledge_repo, state_repo, connection, created_ids


def main(live: bool = False, atlas: bool = False, hybrid: bool = False) -> int:
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    failures: list[str] = []

    def check(condition: bool, text: str) -> None:
        print(f"    [{'OK' if condition else 'FAIL'}] {text}")
        if not condition:
            failures.append(text)

    client, market, knowledge_repo, state_repo, connection, created = build_world(live, atlas, hybrid)
    try:
        sep("SmartNest - Assignment 4.2: Knowledge Representation and Adaptive API Integration")
        if live:
            from smartnest.integrations.gemini import load_gemini_settings
            settings = load_gemini_settings()
            print(f"  MODE: LIVE - real Gemini API (model {settings.model}) for language; "
                  "real SerpApi Google Shopping for products (no Google Search grounding)")
            print("  Products below come from SerpApi; prices are as retrieved (or as cached, see the cache lines).")
        elif hybrid:
            from smartnest.integrations.gemini import load_gemini_settings
            print(f"  MODE: HYBRID - REAL Gemini (model {load_gemini_settings().model}) for all language tasks; "
                  "the product search is SIMULATED (invented listings, links end in .invalid).")
            print("  Extraction, product parsing, explanations and conversation changes are live model calls.")
        else:
            print("  MODE: SIMULATED - scripted stand-in for Gemini and an invented market (links end in .invalid).")
            print("  Nothing below is real data. Run with --live (and GEMINI_API_KEY) for the real thing.")
        print(f"  Persistence: {'MongoDB Atlas' if atlas else 'in-memory repositories'}")

        knowledge = KnowledgeStore(knowledge_repo)
        gemini = GeminiService(client)
        provider = GeminiSearchProvider(gemini, resolver=IdentityResolver(), provider_id="simulated_search")
        serp_source = None
        if live:
            from smartnest.integrations.products.serpapi import (
                SEARCH_CACHE_COLLECTION, SerpApiCandidateSource, SerpApiConfigError, load_serpapi_settings)
            try:
                serp_settings = load_serpapi_settings()
            except SerpApiConfigError as exc:
                print(f"  CANNOT RUN --live: {exc}")
                return 2
            cache_repo = (BaseRepository(connection.collection(SEARCH_CACHE_COLLECTION), id_prefix="sc")
                          if atlas else InMemoryRepository(SEARCH_CACHE_COLLECTION, id_prefix="sc"))
            serp_source = SerpApiCandidateSource.from_settings(serp_settings, cache_repository=cache_repo)
        if hybrid:
            gemini = GeminiService(client, max_api_retries=1, backoff_seconds=2.0)
            provider = GeminiSearchProvider(gemini, resolver=IdentityResolver(), provider_id="simulated_search")
        source = serp_source or LiveCandidateSource(provider, max_age=timedelta(hours=6))
        planner = BDIPlanner.default(UtilityOptimizer(), source)
        agent = BDIAgent("bdi_adaptive_demo", planner, state_store=RepositoryAgentStateStore(state_repo))
        session = AdaptiveSession(agent=agent, gemini=gemini, source=source, knowledge=knowledge,
                                  user_id=USER_ID, project_id=PROJECT_ID)

        # ── 1. Knowledge about the home project ─────────────────────────────────
        sep("1. Home project knowledge (already known before the user speaks)")
        said_so = Provenance(source="user", source_type="user", method="project setup")
        project = knowledge.create(KnowledgeKind.HOME_PROJECT, f"project:{PROJECT_ID}",
                                   {"name": "Living room refresh", "location": "Bengaluru"}, said_so)
        room = knowledge.create(KnowledgeKind.ROOM, "room:living_room",
                                {"name": "living room", "width_cm": 210, "length_cm": 350, "purpose": "seating"}, said_so,
                                relationships=[Relationship(relation="part_of", target_id=project.knowledge_id,
                                                            target_kind="home_project")])
        knowledge.create(KnowledgeKind.REQUIREMENTS, f"user:{USER_ID}",
                         {"max_width_cm": 210, "max_depth_cm": 110, "preferred_colour": "neutral"}, said_so,
                         relationships=[Relationship(relation="part_of", target_id=project.knowledge_id,
                                                     target_kind="home_project")])
        print(f"  project {project.knowledge_id}: {project.attributes}")
        print(f"  room    {room.knowledge_id}: {room.attributes}")
        print("  requirements on file: sofa wall 210 cm, max depth 110 cm, neutral colour")

        # ── 2-4. The request ────────────────────────────────────────────────────
        sep("2. The request")
        print(f'  User: "{MESSAGE}"\n')
        turn = session.request(MESSAGE)
        show_turn(session, turn)

        accepted = source.current_listings
        requirements = UserRequirements(budget=30000, max_width_cm=210, max_depth_cm=110, min_seating=3,
                                        preferred_style="minimalist", preferred_colour="neutral")
        oracle = UtilityOptimizer().optimize(accepted, requirements)
        rec = turn.recommendation
        print()
        check(turn.extraction is not None and turn.extraction.sofa_budget == 30000
              and turn.extraction.min_seating == 3 and turn.extraction.preferred_style == "minimalist",
              "Gemini's extraction was parsed into validated requirements (₹30,000, 3 seats, minimalist)")
        check(turn.discovery is not None and len(turn.discovery.listings) > 0, "product discovery returned accepted listings")
        check(rec is not None, "the agent adopted a recommendation")
        if rec is not None and oracle.selected is not None:
            check(rec.product_id == oracle.selected.product_id,
                  f"it is the optimizer's highest-utility feasible product ({oracle.selected.product_id}) - "
                  "verified by running the UtilityOptimizer independently")
            check(rec.price <= 30000 and rec.url is not None, "it satisfies the budget and carries a source link")
        if not live and turn.discovery is not None:
            reasons = " ".join(" ".join(r.reasons) for r in turn.discovery.rejected)
            check("rating is missing" in reasons, "the listing with no rating was rejected, not defaulted")

        # ── 5. New information ─────────────────────────────────────────────────
        sep("3. New information: the recommended listing goes out of stock")
        first = rec
        mark = len(agent.transitions)
        if market is not None:
            slug = first.url.rsplit("/", 1)[-1]
            market.set_out_of_stock(slug)
            print(f"  (simulated market change: {first.name} is now out of stock; the market is searched again)\n")
            turn2 = session.refresh_products()
        else:
            print("  (live listings rarely change on demand, so a SIMULATED availability event is injected for the "
                  "real recommended listing)\n")
            turn2 = session.ingest_event(EnvironmentalEvent(
                event_type=EventType.PRODUCT_AVAILABILITY, source="simulated_event",
                payload={"product_id": first.product_id, "product_name": first.name, "availability": "out_of_stock"}))
        show_turn(session, turn2)
        print("\n  Transition log for this change:")
        for t in agent.transitions[mark:]:
            if t.kind in ("belief_updated", "intention_invalidated", "intention_replaced", "intention_adopted",
                          "desire_blocked"):
                print("    " + t.format().replace("\n", "\n    "))
        second = turn2.recommendation
        print()
        check(agent.beliefs.value(first.product_id, "availability") in ("out_of_stock", "unavailable"),
              "the availability belief was updated from the new information")
        check(any(t.kind == "intention_invalidated" for t in agent.transitions), "the agent recognised its intention was invalid")
        check(second is not None and second.product_id != first.product_id,
              "a different, feasible product was selected automatically")
        if second is not None:
            rest = [l for l in source.current_listings if l["product_id"] != first.product_id]
            expected = UtilityOptimizer().optimize(rest, requirements).selected
            check(expected is not None and second.product_id == expected.product_id,
                  f"the new choice ({second.product_id}) is the optimizer's best remaining option")

        # ── 6. Conversational change ───────────────────────────────────────────
        sep("4. Conversational preference change")
        print(f'  User: "{FOLLOW_UP}"\n')
        turn3 = session.chat(FOLLOW_UP)
        show_turn(session, turn3)
        print()
        check(agent.beliefs.value("user", "preferred_style") == "contemporary",
              "the style belief changed from minimalist to contemporary (via validated structured output)")
        third = turn3.recommendation
        if third is not None:
            expected = UtilityOptimizer().optimize(
                [l for l in source.current_listings if l["product_id"] != first.product_id],
                UserRequirements(budget=30000, max_width_cm=210, max_depth_cm=110, min_seating=3,
                                 preferred_style="contemporary", preferred_colour="neutral")).selected
            check(expected is not None and third.product_id == expected.product_id,
                  f"the agent now recommends the optimizer's best contemporary option ({third.product_id})")

        # ── 7. Knowledge ────────────────────────────────────────────────────────
        sep("5. Knowledge recorded (provenance, versions, relationships)")
        for kind in KnowledgeKind:
            n = knowledge.count(kind)
            if n:
                print(f"  {kind.value:16} {n} record(s)")
        if third is not None:
            record = knowledge.get(f"prd_{third.product_id}")
            if record is not None:
                print(f"\n  Example - the recommended product's record {record.knowledge_id}:")
                print("    " + json.dumps(record.to_document(), ensure_ascii=False, default=str)[:900])
        recs = knowledge.find(KnowledgeKind.AGENT_KNOWLEDGE)
        if recs:
            r = recs[0]
            print(f"\n  The agent's recommendation record is at version {r.version}; "
                  f"it replaced {len(r.history)} earlier recommendation(s):")
            for h in r.history:
                print(f"    v{h.version}: {h.previous.get('statement') or h.previous.get('data', {}).get('name')}")
        check(knowledge.count(KnowledgeKind.PRODUCT_SPEC) >= len(accepted) - 1, "every accepted listing was persisted as knowledge")
        check(knowledge.count(KnowledgeKind.REQUIREMENTS) == 1 and knowledge.latest(
            KnowledgeKind.REQUIREMENTS, f"user:{USER_ID}").version >= 2,
            "the user's requirements record was updated in place (versioned), not duplicated")
        check(knowledge.count(KnowledgeKind.OBSERVATION) >= 1, "the new information was recorded as an observation")

        sep("Result")
        if failures:
            print(f"  {len(failures)} check(s) FAILED:")
            for text in failures:
                print(f"    - {text}")
            return 1
        print("  All checks passed.")
        if hybrid:
            print("  (HYBRID mode: the language tasks were real Gemini calls; the products were simulated - "
                  "use --live, with Google Search grounding quota, for real products.)")
        elif not live:
            print("  (SIMULATED mode: this exercised the real pipeline against a scripted stand-in - "
                  "use --live for real Gemini and real products.)")
        return 0
    finally:
        for repo, document_id in created:
            try:
                repo.delete(document_id)
            except Exception:
                pass
        if connection is not None:
            connection.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SmartNest adaptive integration demonstration")
    parser.add_argument("--live", action="store_true", help="use the real Gemini API (needs GEMINI_API_KEY in backend/.env)")
    parser.add_argument("--hybrid", action="store_true",
                        help="real Gemini for language tasks, simulated market (no Google Search quota needed)")
    parser.add_argument("--atlas", action="store_true", help="persist knowledge and agent state in MongoDB Atlas")
    args = parser.parse_args()
    if args.live and args.hybrid:
        parser.error("choose --live or --hybrid, not both")
    sys.exit(main(live=args.live, atlas=args.atlas, hybrid=args.hybrid))
