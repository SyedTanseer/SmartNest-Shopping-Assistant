"""End-to-end tests of the adaptive session: Gemini -> validation -> discovery -> knowledge -> BDI -> optimizer.

Gemini is replaced by the demo's scripted stand-in and the market by its invented listings, so the
*real* service, provider, normalizer, knowledge store, BDI agent and optimizer all run for real.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

import adaptive_run as demo
from smartnest.adaptive import AdaptiveSession, loosens
from smartnest.bdi import BDIAgent, BDIPlanner
from smartnest.core.models import EnvironmentalEvent, EventType
from smartnest.database import InMemoryRepository
from smartnest.integrations.gemini import (
    GeminiAPIError,
    GeminiService,
    PlanChangeRequest,
    ProductListExtraction,
    RecommendationExplanation,
    RequirementsExtraction,
)
from smartnest.integrations.products import GeminiSearchProvider, LiveCandidateSource, ProductDiscoveryError
from smartnest.knowledge import KnowledgeKind, KnowledgeStore, Provenance, Relationship
from smartnest.optimization import UserRequirements, UtilityOptimizer
from tests.fake_mongo import BrokenCollection

USER = "u1"
SAID = Provenance(source="user", source_type="user")
MESSAGE = demo.MESSAGE

REQ_MIN = UserRequirements(budget=30000, max_width_cm=210, max_depth_cm=110, min_seating=3,
                           preferred_style="minimalist", preferred_colour="neutral")


class CountingGemini(demo.SimulatedGemini):
    """The scripted Gemini, counting calls per purpose."""

    def __init__(self, market):
        super().__init__(market)
        self.by_kind: dict[str, int] = {}

    def generate(self, prompt, *, schema=None, grounding=False, **kw):
        kind = "search" if grounding else getattr(schema, "__name__", "plain")
        self.by_kind[kind] = self.by_kind.get(kind, 0) + 1
        return super().generate(prompt, schema=schema, grounding=grounding, **kw)


class World:
    def __init__(self, llm_class=CountingGemini, knowledge=True, known_room=True, explain=True, repo=None):
        self.market = demo.SimulatedMarket()
        self.llm = llm_class(self.market)
        self.gemini = GeminiService(self.llm, sleep=lambda s: None)
        self.provider = GeminiSearchProvider(self.gemini, resolver=demo.IdentityResolver(), provider_id="simulated_search")
        self.source = LiveCandidateSource(self.provider, max_age=timedelta(hours=6))
        self.agent = BDIAgent("adaptive_test", BDIPlanner.default(UtilityOptimizer(), self.source))
        self.repo = repo or InMemoryRepository("knowledge", id_prefix="kn")
        self.knowledge = KnowledgeStore(self.repo) if knowledge else None
        if self.knowledge and known_room:
            self.knowledge.create(KnowledgeKind.REQUIREMENTS, f"user:{USER}",
                                  {"max_width_cm": 210, "max_depth_cm": 110, "preferred_colour": "neutral"}, SAID)
        self.session = AdaptiveSession(agent=self.agent, gemini=self.gemini, source=self.source,
                                       knowledge=self.knowledge, user_id=USER, project_id="proj", explain=explain)

    def oracle(self, requirements=REQ_MIN, exclude=()):
        listings = [l for l in self.source.current_listings if l["product_id"] not in exclude]
        return UtilityOptimizer().optimize(listings, requirements).selected


@pytest.fixture()
def world():
    return World()


# ══ The scenario ════════════════════════════════════════════════════════════

class TestRequest:
    def test_gemini_extraction_is_validated_and_applied_as_beliefs(self, world):
        turn = world.session.request(MESSAGE)
        assert turn.extraction.sofa_budget == 30000 and turn.extraction.min_seating == 3
        beliefs = world.agent.beliefs
        assert beliefs.value("user", "sofa_budget") == 30000 and beliefs.value("user", "min_seating") == 3
        assert beliefs.value("user", "preferred_style") == "minimalist"
        assert beliefs.value("user", "max_width_cm") == 210  # from the home-project knowledge, not invented
        assert turn.rejected == {}

    def test_discovery_funnel_and_rejections(self, world):
        turn = world.session.request(MESSAGE)
        d = turn.discovery
        assert (len(d.sources), d.extracted_count, len(d.listings), len(d.rejected)) == (7, 7, 6, 1)
        assert "rating is missing" in " ".join(d.rejected[0].reasons)
        assert all(l["url"].startswith("https://simfurnish.invalid/") for l in d.listings)

    def test_the_recommendation_is_the_optimizers_highest_utility_feasible_product(self, world):
        turn = world.session.request(MESSAGE)
        best = world.oracle()
        assert turn.recommendation.product_id == best.product_id
        assert turn.recommendation.utility == pytest.approx(best.utility)
        assert turn.recommendation.price <= 30000
        assert turn.recommendation.url and turn.recommendation.verified_at

    def test_infeasible_listings_never_win(self, world):
        world.session.request(MESSAGE)
        outcome = world.agent.last_outcomes["des_select_sofa"]
        rejected = {e.name: e.reasons[0] for e in outcome.optimization.rejected}
        assert any("exceeds budget" in r for r in rejected.values())      # ₹31,999 > ₹30,000
        assert any("exceeds maximum width" in r for r in rejected.values())  # 225 cm > 210 cm
        assert any("seating capacity" in r for r in rejected.values())      # 2 seats
        assert world.session.current.name not in rejected

    def test_the_bdi_agent_adopts_it(self, world):
        turn = world.session.request(MESSAGE)
        intention = world.agent.current_intention("des_select_sofa")
        assert intention.params["product_id"] == turn.recommendation.product_id and intention.status.value == "active"
        assert [l.rule_id for l in turn.action_logs] == ["BDI_ADOPTED"]

    def test_an_explanation_is_produced_from_the_optimizers_facts(self, world):
        rec = world.session.request(MESSAGE).recommendation
        assert rec.explanation.source == "gemini" and rec.name in rec.explanation.text
        assert f"{rec.utility:.2f}" in rec.explanation.text
        assert rec.facts.feasible_count == 3 and rec.facts.inspected_count == 6 and rec.facts.runner_up_name

    def test_what_was_learned_is_recorded_with_provenance(self, world):
        turn = world.session.request(MESSAGE)
        k = world.knowledge
        requirements = k.latest(KnowledgeKind.REQUIREMENTS, f"user:{USER}")
        assert requirements.version == 2 and requirements.attributes["sofa_budget"] == 30000
        assert requirements.provenance.source == "gemini" and requirements.provenance.source_type == "llm"
        assert requirements.history[0].previous["sofa_budget"] is None  # absent before the request
        assert k.count(KnowledgeKind.PRODUCT_SPEC) == 6
        product = k.get(f"prd_{turn.recommendation.product_id}")
        assert product.provenance.source_type == "web" and product.provenance.url == turn.recommendation.url
        assert product.provenance.method == "google_search_grounding" and product.confidence == 0.8
        assert product.related("derived_from")[0].target_id == requirements.knowledge_id
        agent_note = k.find(KnowledgeKind.AGENT_KNOWLEDGE)[0]
        assert agent_note.attributes["data"]["product_id"] == turn.recommendation.product_id
        assert agent_note.related("about")[0].target_id == f"prd_{turn.recommendation.product_id}"


class TestNewInformation:
    def test_an_availability_change_found_by_refreshing_triggers_replanning(self, world):
        first = world.session.request(MESSAGE).recommendation
        world.market.set_out_of_stock(first.url.rsplit("/", 1)[-1])
        turn = world.session.refresh_products()

        assert world.agent.beliefs.value(first.product_id, "availability") == "out_of_stock"       # 1. belief updated
        assert any(t.kind == "intention_invalidated" for t in world.agent.transitions)             # 2. recognised invalid
        assert [l.rule_id for l in turn.action_logs] == ["BDI_REVISED"]                             # 3. reconsidered
        assert turn.recommendation.product_id != first.product_id and turn.changed_recommendation  # 4. new recommendation
        assert turn.recommendation.product_id == world.oracle(exclude=(first.product_id,)).product_id

    def test_the_old_intention_is_replaced_and_the_change_is_explained(self, world):
        first = world.session.request(MESSAGE).recommendation
        old = world.agent.current_intention("des_select_sofa")
        world.market.set_out_of_stock(first.url.rsplit("/", 1)[-1])
        world.session.refresh_products()
        assert old.status.value == "replaced"
        replaced = next(t for t in world.agent.transitions if t.kind == "intention_replaced")
        assert replaced.before["product_id"] == first.product_id and replaced.after["product_id"] == world.session.current.product_id
        assert "out_of_stock" in replaced.explanation

    def test_the_change_is_recorded_as_an_observation_and_the_product_record_is_versioned(self, world):
        first = world.session.request(MESSAGE).recommendation
        world.market.set_out_of_stock(first.url.rsplit("/", 1)[-1])
        world.session.refresh_products()
        observation = world.knowledge.find(KnowledgeKind.OBSERVATION)[0]
        assert observation.attributes["observation_type"] == "product_availability"
        assert observation.attributes["details"]["availability"] == "out_of_stock" and observation.provenance.source_type == "event"
        record = world.knowledge.get(f"prd_{first.product_id}")
        assert record.version == 2 and record.attributes["availability"] == "out_of_stock"
        assert record.history[0].previous["availability"] == "in_stock"

    def test_the_recommendation_record_keeps_a_history(self, world):
        first = world.session.request(MESSAGE).recommendation
        world.market.set_out_of_stock(first.url.rsplit("/", 1)[-1])
        world.session.refresh_products()
        record = world.knowledge.find(KnowledgeKind.AGENT_KNOWLEDGE)[0]
        assert record.version == 2 and first.name in record.history[0].previous["statement"]

    def test_a_price_change_event_from_elsewhere_is_handled_the_same_way(self, world):
        first = world.session.request(MESSAGE).recommendation
        turn = world.session.ingest_event(EnvironmentalEvent(
            event_type=EventType.PRICE_CHANGE, source="price_monitor",
            payload={"product_id": first.product_id, "new_price": 31000}))   # now over budget
        assert world.agent.beliefs.value(first.product_id, "price") == 31000
        assert turn.recommendation.product_id != first.product_id
        assert turn.recommendation.product_id == world.oracle(exclude=(first.product_id,)).product_id

    def test_an_unrelated_event_changes_nothing(self, world):
        first = world.session.request(MESSAGE).recommendation
        explanations = world.llm.by_kind.get("RecommendationExplanation", 0)
        turn = world.session.ingest_event(EnvironmentalEvent(
            event_type=EventType.PRODUCT_AVAILABILITY, source="monitor",
            payload={"product_name": "Washing machine", "availability": "out_of_stock"}))
        assert turn.recommendation.product_id == first.product_id and not turn.changed_recommendation
        assert turn.action_logs == [] and world.llm.by_kind.get("RecommendationExplanation", 0) == explanations

    def test_new_products_found_on_refresh_are_considered(self, world):
        world.session.request(MESSAGE)
        world.market.listings.append(dict(world.market.find("nordic-lite"), name="SIMULATED New Arrival 3-Seater",
                                          source_url="https://simfurnish.invalid/p/new-arrival", price=12999, rating=4.9,
                                          review_count=900))
        turn = world.session.refresh_products()
        assert turn.recommendation.name == "SIMULATED New Arrival 3-Seater"
        assert turn.recommendation.product_id == world.oracle().product_id

    def test_the_explanation_is_only_requested_when_the_recommendation_changes(self, world):
        world.session.request(MESSAGE)
        asked = world.llm.by_kind["RecommendationExplanation"]
        world.session.refresh_products()          # nothing changed in the market
        world.session.refresh_products()
        assert world.llm.by_kind["RecommendationExplanation"] == asked


class TestConversation:
    FOLLOW_UP = demo.FOLLOW_UP

    CONTEMPORARY = UserRequirements(budget=30000, max_width_cm=210, max_depth_cm=110, min_seating=3,
                                    preferred_style="contemporary", preferred_colour="neutral")

    def test_changing_style_in_words_is_validated_applied_and_judged_by_the_optimizer(self, world):
        first = world.session.request(MESSAGE).recommendation
        turn = world.session.chat(self.FOLLOW_UP)
        assert turn.change_request.changes[0].value == "contemporary"
        assert world.agent.beliefs.value("user", "preferred_style") == "contemporary"
        expected = world.oracle(self.CONTEMPORARY)
        assert turn.recommendation.product_id == expected.product_id
        # "minimalist" is a compatible style, so the cheaper minimalist sofa may legitimately still win:
        assert turn.changed_recommendation == (turn.recommendation.product_id != first.product_id)

    def test_changing_style_in_words_can_flip_the_recommendation(self, world):
        first = world.session.request(MESSAGE).recommendation
        world.market.set_out_of_stock(first.url.rsplit("/", 1)[-1])
        middle = world.session.refresh_products().recommendation          # the second-best minimalist sofa
        turn = world.session.chat(self.FOLLOW_UP)
        assert turn.recommendation.product_id == world.oracle(self.CONTEMPORARY).product_id
        assert turn.recommendation.product_id != middle.product_id and turn.changed_recommendation
        assert "Contemporary" in turn.recommendation.name

    def test_the_market_is_searched_again_with_the_new_preference_before_deciding(self, world):
        world.session.request(MESSAGE)
        searches = world.llm.by_kind["search"]
        world.session.chat(self.FOLLOW_UP)
        assert world.llm.by_kind["search"] == searches + 1 and world.source.query.style == "contemporary"

    def test_the_change_is_recorded_in_the_versioned_requirements(self, world):
        world.session.request(MESSAGE)
        world.session.chat(self.FOLLOW_UP)
        record = world.knowledge.latest(KnowledgeKind.REQUIREMENTS, f"user:{USER}")
        assert record.attributes["preferred_style"] == "contemporary" and record.version == 3
        assert record.history[-1].previous["preferred_style"] == "minimalist"

    def test_an_unclear_request_asks_for_clarification_and_changes_nothing(self, world):
        class Vague(CountingGemini):
            def _change(self, text):
                return {"intent": "modify_requirements", "changes": [], "confidence": 0.3,
                        "clarification_needed": "Which feature should change?"}
        w = World(Vague)
        first = w.session.request(MESSAGE).recommendation
        turn = w.session.chat("make it better")
        assert any("Which feature should change?" in n for n in turn.notes)
        assert turn.recommendation.product_id == first.product_id and not turn.changed_recommendation

    def test_chat_that_is_not_a_change_leaves_the_plan_alone(self, world):
        first = world.session.request(MESSAGE).recommendation
        turn = world.session.chat("thanks, how long does delivery take?")
        assert turn.recommendation.product_id == first.product_id and any("No change" in n for n in turn.notes)

    def test_confirming_the_choice_reserves_it(self, world):
        class Confirm(CountingGemini):
            def _change(self, text):
                return {"intent": "confirm_choice", "changes": [], "confidence": 0.9}
        w = World(Confirm)
        w.session.request(MESSAGE)
        turn = w.session.chat("yes, that one")
        assert any("reserved" in n for n in turn.notes)
        assert w.agent.desires.get("des_select_sofa").status.value == "achieved"

    def test_invalid_values_proposed_by_the_model_are_rejected_by_validation(self, world):
        class BadValue(CountingGemini):
            def _change(self, text):
                return {"intent": "modify_requirements", "confidence": 0.9, "changes": [
                    {"field": "max_width_cm", "action": "set", "value": -50},
                    {"field": "min_seating", "action": "set", "value": 2.5},
                    {"field": "preferred_style", "action": "set", "value": ""},
                    {"field": "sofa_budget", "action": "set", "value": None}]}
        w = World(BadValue)
        first = w.session.request(MESSAGE).recommendation
        turn = w.session.chat("change things")
        assert set(turn.rejected) == {"max_width_cm", "min_seating", "preferred_style", "sofa_budget"}
        assert w.agent.beliefs.value("user", "max_width_cm") == 210 and turn.recommendation.product_id == first.product_id

    def test_clearing_a_preference(self, world):
        class Clear(CountingGemini):
            def _change(self, text):
                return {"intent": "modify_requirements", "confidence": 0.9,
                        "changes": [{"field": "preferred_style", "action": "clear", "value": None}]}
        w = World(Clear)
        w.session.request(MESSAGE)
        w.session.chat("any style is fine")
        assert w.agent.beliefs.value("user", "preferred_style") is None
        assert "preferred_style" not in w.knowledge.latest(KnowledgeKind.REQUIREMENTS, f"user:{USER}").attributes


class TestHybridMode:
    """Real language model for text tasks, simulated market for the search step."""

    class Real:
        def __init__(self):
            self.calls = []

        def generate(self, prompt, **kw):
            self.calls.append(kw)
            return demo.SimulatedGemini(demo.SimulatedMarket()).generate(prompt, **kw)

    def test_search_goes_to_the_simulation_and_everything_else_to_the_real_client(self):
        real = self.Real()
        market = demo.SimulatedMarket()
        hybrid = demo.HybridClient(real, demo.SimulatedGemini(market))
        search = hybrid.generate("find sofas", grounding=True, url_context=True)
        assert real.calls == [] and "SIMULATED" in search.text and search.sources[0].uri.endswith(".invalid/p/nordic-lite")
        hybrid.generate("extract", schema=RequirementsExtraction)
        assert len(real.calls) == 1 and real.calls[0]["schema"] is RequirementsExtraction

    def test_the_whole_session_runs_through_a_hybrid_client(self):
        market = demo.SimulatedMarket()
        real = self.Real()
        client = demo.HybridClient(real, demo.SimulatedGemini(market))
        gemini = GeminiService(client, sleep=lambda s: None)
        source = LiveCandidateSource(GeminiSearchProvider(gemini, resolver=demo.IdentityResolver(), provider_id="simulated_search"))
        agent = BDIAgent("hybrid", BDIPlanner.default(UtilityOptimizer(), source))
        knowledge = KnowledgeStore(InMemoryRepository("knowledge", id_prefix="kn"))
        knowledge.create(KnowledgeKind.REQUIREMENTS, f"user:{USER}",
                         {"max_width_cm": 210, "max_depth_cm": 110, "preferred_colour": "neutral"}, SAID)
        session = AdaptiveSession(agent=agent, gemini=gemini, source=source, knowledge=knowledge, user_id=USER)
        turn = session.request(MESSAGE)
        assert turn.recommendation is not None and turn.recommendation.url.startswith("https://simfurnish.invalid/")
        assert {c.get("schema") for c in real.calls if c.get("schema")} >= {RequirementsExtraction, ProductListExtraction}


# ══ Gemini cannot override hard constraints ═════════════════════════════════

class TestLoosening:
    @pytest.mark.parametrize("name,old,new,expected", [
        ("max_width_cm", 210, 300, True), ("max_width_cm", 210, 200, False), ("max_depth_cm", 110, 120, True),
        ("sofa_budget", 30000, 40000, True), ("sofa_budget", 30000, 20000, False), ("total_budget", 1, 2, True),
        ("min_seating", 3, 2, True), ("min_seating", 3, 4, False),
        ("preferred_style", "a", "b", False), ("max_width_cm", None, 300, False), ("max_width_cm", 210, 210, False),
    ])
    def test_which_changes_relax_a_limit(self, name, old, new, expected):
        assert loosens(name, old, new) is expected

    def test_a_hallucinated_wider_wall_is_held_not_applied(self):
        class Hallucinating(CountingGemini):
            def _requirements(self, text):
                return {**super()._requirements(text), "max_width_cm": 500.0, "sofa_budget": 90000.0}
        w = World(Hallucinating)
        turn = w.session.request(MESSAGE)
        assert w.agent.beliefs.value("user", "max_width_cm") == 210            # the limit on file stands
        assert w.agent.beliefs.value("user", "sofa_budget") == 90000           # first statement of a limit: applied
        assert turn.pending == {"max_width_cm": (210, 500.0)} and w.session.pending_changes == {"max_width_cm": 500.0}
        assert any("relaxes a limit" in n for n in turn.notes)
        listing = next(l for l in w.source.current_listings if l["product_id"] == turn.recommendation.product_id)
        assert listing["width_cm"] <= 210, "the recommendation must still fit the original 210 cm wall"
        rejected = {e.name: e.reasons[0] for e in w.agent.last_outcomes["des_select_sofa"].optimization.rejected}
        assert any("exceeds maximum width 210" in r for r in rejected.values())  # the 225 cm sofa is still out

    def test_a_held_change_is_applied_only_after_explicit_confirmation(self):
        class Wider(CountingGemini):
            def _requirements(self, text):
                return {**super()._requirements(text), "max_width_cm": 230.0}
        w = World(Wider)
        w.session.request(MESSAGE)
        wide = "SIMULATED Wide Sectional 3-Seater"  # 225 cm: too wide for 210, fine for 230
        optimization = lambda: w.agent.last_outcomes["des_select_sofa"].optimization
        assert wide in {e.name for e in optimization().rejected}
        assert w.agent.beliefs.value("user", "max_width_cm") == 210
        turn = w.session.confirm_pending()
        assert wide in {e.name for e in optimization().ranking}
        assert w.agent.beliefs.value("user", "max_width_cm") == 230 and w.session.pending_changes == {}
        assert turn.recommendation.product_id == w.oracle(UserRequirements(
            budget=30000, max_width_cm=230, max_depth_cm=110, min_seating=3, preferred_style="minimalist",
            preferred_colour="neutral")).product_id

    def test_nothing_to_confirm(self, world):
        world.session.request(MESSAGE)
        assert "nothing waiting" in world.session.confirm_pending().notes[0]

    def test_chat_cannot_loosen_a_limit_silently_either(self):
        class Raise(CountingGemini):
            def _change(self, text):
                return {"intent": "modify_requirements", "confidence": 0.9,
                        "changes": [{"field": "sofa_budget", "action": "set", "value": 60000}]}
        w = World(Raise)
        first = w.session.request(MESSAGE).recommendation
        turn = w.session.chat("raise my budget")
        assert w.agent.beliefs.value("user", "sofa_budget") == 30000 and turn.pending["sofa_budget"] == (30000, 60000.0)
        assert turn.recommendation.product_id == first.product_id
        w.session.confirm_pending()
        assert w.agent.beliefs.value("user", "sofa_budget") == 60000

    def test_tightening_a_limit_applies_immediately(self):
        class Tighter(CountingGemini):
            def _change(self, text):
                return {"intent": "modify_requirements", "confidence": 0.9,
                        "changes": [{"field": "sofa_budget", "action": "set", "value": 26000}]}
        w = World(Tighter)
        w.session.request(MESSAGE)
        turn = w.session.chat("keep it under 26k")
        assert w.agent.beliefs.value("user", "sofa_budget") == 26000 and turn.pending == {}
        assert turn.recommendation.price <= 26000

    def test_model_claims_cannot_make_an_infeasible_product_feasible(self):
        class Overclaiming(CountingGemini):
            @staticmethod
            def _extraction(l):
                return {**l, "style_inferred": False, "feasible": True, "fits_room": True, "utility": 100,
                        "utility_score": 100, "within_budget": True, "approved": True}
        w = World(Overclaiming)
        w.session.request(MESSAGE)
        outcome = w.agent.last_outcomes["des_select_sofa"]
        rejected_ids = {e.product_id for e in outcome.optimization.rejected}
        assert len(rejected_ids) == 3                       # over budget, too wide, too few seats: still rejected
        assert w.session.current.product_id == w.oracle().product_id

    def test_a_listing_whose_text_disagrees_with_the_extraction_is_rejected(self):
        class Lying(CountingGemini):
            @staticmethod
            def _extraction(l):
                return {**l, "price": 9999} if "Nordic" in l["name"] else {**l}
        w = World(Lying)
        turn = w.session.request(MESSAGE)
        assert any("does not appear in the retrieved text" in " ".join(r.reasons) for r in turn.discovery.rejected)
        assert "Nordic" not in turn.recommendation.name


# ══ Failure handling ════════════════════════════════════════════════════════

class TestFailures:
    def test_gemini_failure_at_extraction_is_reported_and_nothing_changes(self):
        class Down(CountingGemini):
            def generate(self, prompt, *, schema=None, **kw):
                if schema is RequirementsExtraction:
                    raise GeminiAPIError("quota exceeded", 429, False)
                return super().generate(prompt, schema=schema, **kw)
        w = World(Down)
        turn = w.session.request(MESSAGE)
        assert any("Could not understand the request" in n for n in turn.notes)
        assert turn.recommendation is None and w.agent.beliefs.value("user", "sofa_budget") is None

    def test_invalid_model_output_is_rejected_not_applied(self):
        class Garbage(CountingGemini):
            def _requirements(self, text):
                return {"sofa_budget": -30000, "confidence": 0.9}
        w = World(Garbage)
        turn = w.session.request(MESSAGE)
        assert any("Could not understand the request" in n for n in turn.notes)
        assert w.agent.beliefs.value("user", "sofa_budget") is None

    def test_discovery_failure_keeps_the_previous_listings_and_recommendation(self):
        w = World()
        first = w.session.request(MESSAGE).recommendation

        class Failing:
            provider_id = "x"
            def discover(self, query):
                raise ProductDiscoveryError("x", "search failed: timeout")
        w.source._provider = Failing()
        turn = w.session.refresh_products()
        assert any("Product discovery failed" in n for n in turn.notes)
        assert turn.recommendation.product_id == first.product_id and not turn.changed_recommendation
        assert "timeout" in w.source.last_error

    def test_no_search_results_leave_the_goal_waiting_not_blocked(self):
        class NoWeb(CountingGemini):
            def _search(self):
                from smartnest.integrations.gemini.testing import grounded_response
                return grounded_response("I could not find anything.", [])
        w = World(NoWeb)
        turn = w.session.request(MESSAGE)
        assert turn.recommendation is None and w.agent.desires.get("des_select_sofa").status.value == "active"
        assert "product candidates" in turn.needs

    def test_missing_room_information_is_optional_and_not_invented(self):
        w = World(known_room=False)
        turn = w.session.request(MESSAGE)
        assert not {"user.max_width_cm", "user.max_depth_cm"} & set(turn.needs)  # not required any more
        assert w.agent.beliefs.value("user", "max_width_cm") is None            # and never invented
        assert w.agent.beliefs.value("user", "max_depth_cm") is None
        assert turn.recommendation is not None                                   # planned without room limits

    def test_knowledge_store_failure_does_not_stop_the_session(self):
        from smartnest.database import BaseRepository
        w = World(knowledge=False)
        w.session.knowledge = KnowledgeStore(BaseRepository(BrokenCollection(RuntimeError("db down")), id_prefix="kn"))
        w.agent.set_requirements({"max_width_cm": 210, "max_depth_cm": 110, "preferred_colour": "neutral"})
        turn = w.session.request(MESSAGE)
        assert turn.recommendation is not None and any("Knowledge was not saved" in n for n in turn.notes)

    def test_explanation_failure_falls_back_to_the_template(self):
        class NoWriter(CountingGemini):
            def generate(self, prompt, *, schema=None, **kw):
                if schema is RecommendationExplanation:
                    raise GeminiAPIError("down", 503, False)
                return super().generate(prompt, schema=schema, **kw)
        rec = World(NoWriter).session.request(MESSAGE).recommendation
        assert rec.explanation.source == "template" and rec.name in rec.explanation.text

    def test_explanations_can_be_switched_off(self):
        w = World(explain=False)
        rec = w.session.request(MESSAGE).recommendation
        assert rec.explanation.source == "template" and "RecommendationExplanation" not in w.llm.by_kind

    def test_a_session_works_without_a_knowledge_store(self):
        w = World(knowledge=False)
        w.agent.set_requirements({"max_width_cm": 210, "max_depth_cm": 110})
        assert w.session.request(MESSAGE).recommendation is not None

    def test_stale_listings_are_not_offered(self):
        w = World()
        w.session.request(MESSAGE)
        w.source._max_age = timedelta(microseconds=1)
        assert w.source.get_candidates() == [] and len(w.source.stale_listings) == 6

    def test_desires_are_not_duplicated_when_a_session_is_created_twice(self):
        w = World()
        AdaptiveSession(agent=w.agent, gemini=w.gemini, source=w.source)
        assert len(w.agent.desires) == 2
