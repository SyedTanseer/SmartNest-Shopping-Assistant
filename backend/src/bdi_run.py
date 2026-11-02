#!/usr/bin/env python3
"""SmartNest — Assignment 4.1: BDI architecture demonstration.

Run from the ``backend`` directory:

    python src/bdi_run.py            # state kept in an in-memory repository
    python src/bdi_run.py --atlas    # state kept in MongoDB Atlas (agent_states collection)
    python src/bdi_run.py --atlas --keep   # ... and leave the saved record in Atlas

Story
-----
The user wants a sofa: total budget Rs 2,00,000, sofa budget Rs 30,000,
maximum width 210 cm, minimalist style (room depth 110 cm, 3 seats and a
neutral colour are taken from the Assignment 3.2 scenario).

 1. The agent adopts an intention using the existing UtilityOptimizer.
 2. An event reports that the chosen sofa (SOFA_A2) is out of stock.  The
    agent updates its belief, recognises that its intention is invalid,
    reconsiders its desires, re-runs the optimizer and commits to the
    highest-utility feasible alternative — nothing about the replacement
    is hardcoded.  The complete before/after transition is printed.
 3. Unrelated updates do not cause any replanning.
 4. Real changes (preferences, room size, budget) do cause replanning; if
    nothing is feasible the agent recommends nothing, and recovers when
    the situation improves.
 5. The agent's state is saved, the agent is "restarted", and the new
    instance restores its beliefs, desires, intentions and history and
    carries on.
"""

from __future__ import annotations

import argparse
import logging
import sys

from smartnest.bdi import (
    BDIAgent,
    BDIPlanner,
    CatalogCandidateSource,
    RepositoryAgentStateStore,
    select_sofa_desire,
    stay_within_budget_desire,
)
from smartnest.cnp.product_catalog import catalog_agent_a, catalog_agent_b
from smartnest.core.events import EventStream
from smartnest.core.models import EnvironmentalEvent, EventType
from smartnest.database import BaseRepository, Collections, InMemoryRepository, MongoConnection, load_settings
from smartnest.optimization import UserRequirements, UtilityOptimizer

AGENT_ID = "bdi_home_planner_demo"

REQUIREMENTS = {
    "total_budget": 200000,     # Rs 2,00,000
    "sofa_budget": 30000,
    "max_width_cm": 210,
    "max_depth_cm": 110,        # from the 3.2 scenario
    "min_seating": 3,           # from the 3.2 scenario
    "preferred_style": "minimalist",
    "preferred_colour": "neutral",  # from the 3.2 scenario
}


def sep(title: str) -> None:
    print(f"\n{'=' * 78}\n  {title}\n{'=' * 78}")


def event(kind: EventType, source: str, **payload) -> EnvironmentalEvent:
    return EnvironmentalEvent(event_type=kind, source=source, payload=payload)


def build_planner() -> BDIPlanner:
    source = CatalogCandidateSource([catalog_agent_a(extended=True), catalog_agent_b(extended=True)])
    return BDIPlanner.default(UtilityOptimizer(), source)


def describe(agent: BDIAgent) -> str:
    intention = agent.current_intention("des_select_sofa")
    if intention is None:
        desire = agent.desires.get("des_select_sofa")
        return f"no intention (desire is {desire.status.value}) - nothing is recommended"
    p = intention.params
    return (f"{p['product_id']} from {p['source_agent_id']} at Rs {p['price']:,.0f}, "
            f"utility {p['utility']:.2f} [{intention.status.value}]")


def show_new_transitions(agent: BDIAgent, since: int, kinds: set[str] | None = None) -> None:
    for t in agent.transitions[since:]:
        if kinds is None or t.kind in kinds:
            print("  " + t.format().replace("\n", "\n  "))
            print()


def step(agent: BDIAgent, title: str, ev: EnvironmentalEvent, show: set[str] | None = None) -> None:
    before = describe(agent)
    since = len(agent.transitions)
    log = agent.handle_event(ev)
    print(f"\n> {title}")
    print(f"    intention before: {before}")
    print(f"    intention after : {describe(agent)}")
    print(f"    agent decision  : {log.rule_name if log else 'no replanning needed'}")
    if show is not None:
        show_new_transitions(agent, since, show)


def build_store(use_atlas: bool):
    if use_atlas:
        connection = MongoConnection(load_settings())
        connection.connect()
        repository = BaseRepository(connection.collection(Collections.AGENT_STATES), id_prefix="agent")
        label = f"MongoDB Atlas, database '{connection.settings.database}', collection 'agent_states'"
        return RepositoryAgentStateStore(repository), repository, connection, label
    repository = InMemoryRepository("agent_states", id_prefix="agent")
    return RepositoryAgentStateStore(repository), repository, None, "in-memory repository (use --atlas for MongoDB)"


def main(use_atlas: bool = False, keep: bool = False) -> int:
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    store, repository, connection, store_label = build_store(use_atlas)
    failures: list[str] = []

    def check(condition: bool, text: str) -> None:
        print(f"    [{'OK' if condition else 'FAIL'}] {text}")
        if not condition:
            failures.append(text)

    try:
        repository.delete(AGENT_ID)  # start the demo from a clean record
        sep("SmartNest - Assignment 4.1: BDI Architecture")
        print(f"  Persistence: {store_label}")

        # ── Setup ─────────────────────────────────────────────────────────
        agent = BDIAgent(AGENT_ID, build_planner(), state_store=store)
        agent.set_requirements(REQUIREMENTS)
        agent.add_desire(select_sofa_desire(priority=80))
        agent.add_desire(stay_within_budget_desire(priority=90))

        sep("1. Beliefs and desires")
        for key, value in sorted(agent.beliefs.as_dict().items()):
            print(f"  belief  {key:24} = {value}")
        for desire in agent.desires.all():
            print(f"  desire  {desire.desire_id:18} priority {desire.priority:>3}  {desire.status.value:8} {desire.description}")

        # ── Initial intention ─────────────────────────────────────────────
        sep("2. Initial deliberation: the agent commits to a sofa")
        since = len(agent.transitions)
        agent.deliberate("initial deliberation")
        show_new_transitions(agent, since, {"desires_reconsidered", "intention_adopted", "intention_progress"})
        intention = agent.current_intention("des_select_sofa")
        print("  Intention plan:")
        for action in intention.actions:
            print(f"    - {action.kind:18} [{action.status.value}]{'  (waits for user confirmation)' if action.manual else ''}")
        check(intention.params["product_id"] == "SOFA_A2", "the optimizer's first choice is SOFA_A2 (as in Assignment 3.2)")
        original_id = intention.intention_id

        # ── The scenario ──────────────────────────────────────────────────
        sep("3. SCENARIO: SOFA_A2 goes out of stock")
        stream = EventStream()  # events arrive through the existing EventStream
        stream.emit(event(EventType.PRODUCT_AVAILABILITY, "product_monitor",
                          product_id="SOFA_A2", availability="out_of_stock"))
        since = len(agent.transitions)
        logs = agent.process_events(stream)
        print("  Complete transition log for this event:\n")
        show_new_transitions(agent, since)

        revised = agent.current_intention("des_select_sofa")
        old = agent.intentions.get(original_id)
        print("  BEFORE:", f"SOFA_A2 (intention {original_id})")
        print("  AFTER :", describe(agent))
        print(f"  Old intention is now: {old.status.value} (replaced_by {old.replaced_by})")
        print(f"  Belief SOFA_A2.availability = {agent.beliefs.value('SOFA_A2', 'availability')!r}")

        # Independent verification, outside the agent.
        every_candidate = []
        for catalog in (catalog_agent_a(extended=True), catalog_agent_b(extended=True)):
            every_candidate += [{**p, "source_agent_id": catalog.agent_id} for p in catalog.get_all_products()]
        without_a2 = [dict(p, availability="out_of_stock") if p["product_id"] == "SOFA_A2" else p
                      for p in every_candidate]
        expected = UtilityOptimizer().optimize(without_a2, UserRequirements(
            budget=30000, max_width_cm=210, max_depth_cm=110, min_seating=3,
            preferred_style="minimalist", preferred_colour="neutral")).selected
        print("\n  Independent check (UtilityOptimizer run directly, SOFA_A2 out of stock):")
        check(len(logs) == 1 and logs[0].rule_id == "BDI_REVISED", "the agent reported an intention revision")
        check(old.status.value == "replaced", "the old intention was replaced")
        check(revised.params["product_id"] != "SOFA_A2", "the new intention is not SOFA_A2")
        check(revised.params["product_id"] == expected.product_id,
              f"the new intention ({revised.params['product_id']}) is the optimizer's highest-utility feasible option ({expected.product_id}, U={expected.utility:.2f})")

        # ── Unrelated updates ─────────────────────────────────────────────
        sep("4. Unrelated updates do NOT cause replanning")
        intention_before = agent.current_intention("des_select_sofa").intention_id
        unrelated = [
            ("another sofa (SOFA_A7) goes out of stock - not the chosen one",
             event(EventType.PRODUCT_AVAILABILITY, "product_monitor", product_id="SOFA_A7", availability="out_of_stock")),
            ("an unrelated product ('Washing machine') goes out of stock",
             event(EventType.PRODUCT_AVAILABILITY, "product_monitor", product_name="Washing machine", availability="out_of_stock")),
            ("total budget raised 2,00,000 -> 2,50,000 (sofa budget still caps the sofa)",
             event(EventType.BUDGET_UPDATE, "budget_monitor", total_budget=250000)),
            ("the same sofa budget is reported again (no change)",
             event(EventType.BUDGET_UPDATE, "budget_monitor", sofa_budget=30000)),
        ]
        for title, ev in unrelated:
            since = len(agent.transitions)
            log = agent.handle_event(ev)
            new = agent.transitions[since:]
            kinds = [t.kind for t in new]
            if "belief_updated" not in kinds:
                reason = "the belief already had exactly this value, so nothing changed"
            elif "update_ignored" in kinds:
                reason = next(t for t in new if t.kind == "update_ignored").explanation
            else:  # only a monitoring goal looked at it
                reason = ("only the budget-monitoring goal re-checked it; "
                          + next(t for t in new if t.kind == "desires_reconsidered").explanation.split("|")[-1].strip())
            kept = agent.current_intention("des_select_sofa").intention_id == intention_before
            print(f"\n> {title}\n    replanned: {'yes' if log else 'no'} | intention kept: {kept}\n    why: {reason}")
            check(log is None and kept, "no replanning, intention unchanged")

        # ── Dynamic replanning ────────────────────────────────────────────
        sep("5. Real changes DO cause replanning")
        step(agent, "User now prefers a contemporary style",
             event(EventType.PREFERENCE_UPDATE, "user", preferred_style="contemporary"), {"intention_replaced"})
        step(agent, "Room is measured again: maximum width 210 -> 200 cm",
             event(EventType.ROOM_UPDATE, "user", max_width_cm=200), {"intention_invalidated", "intention_replaced"})
        step(agent, "Sofa budget cut to Rs 20,000 - nothing fits",
             event(EventType.BUDGET_UPDATE, "user", sofa_budget=20000), {"desire_blocked"})
        check(agent.current_intention("des_select_sofa") is None
              and agent.desires.get("des_select_sofa").status.value == "blocked",
              "no feasible product -> no recommendation (desire blocked)")
        step(agent, "Sofa budget restored to Rs 30,000",
             event(EventType.BUDGET_UPDATE, "user", sofa_budget=30000), {"desire_unblocked", "intention_adopted"})
        check(agent.current_intention("des_select_sofa") is not None, "the agent recovered and re-committed")

        # ── Persistence and restart ───────────────────────────────────────
        sep("6. Persistence: save, restart, restore")
        agent.save_state()
        document = repository.get(AGENT_ID)
        print(f"  Saved record '{document['_id']}' in {store_label}:")
        print(f"    beliefs={len(document['beliefs'])}  desires={len(document['desires'])}  "
              f"intentions={len(document['intentions'])}  transitions={len(document['transitions'])}  "
              f"updated_at={document['updated_at']:%Y-%m-%d %H:%M:%S}")
        before_restart = describe(agent)
        n_beliefs, n_transitions = len(agent.beliefs), len(agent.transitions)
        del agent

        print("\n  ... agent process restarted ...\n")
        agent = BDIAgent(AGENT_ID, build_planner(), state_store=store)
        print(f"  state_restored = {agent.state_restored}")
        print(f"  intention before restart: {before_restart}")
        print(f"  intention after restore : {describe(agent)}")
        check(agent.state_restored, "state was restored from the store")
        check(describe(agent) == before_restart, "the same intention was restored")
        check(len(agent.beliefs) == n_beliefs, f"all {n_beliefs} beliefs were restored")
        check(len(agent.transitions) >= n_transitions, "the transition history was restored")
        check(agent.beliefs.value("SOFA_A2", "availability") == "out_of_stock",
              "it still remembers that SOFA_A2 was out of stock")

        step(agent, "After the restart: SOFA_A2 is back in stock",
             event(EventType.PRODUCT_AVAILABILITY, "product_monitor", product_id="SOFA_A2", availability="in_stock"),
             {"intention_replaced"})
        check(agent.current_intention("des_select_sofa").params["product_id"] == "SOFA_A2",
              "the restored agent reconsidered and returned to the better SOFA_A2")

        sep("7. Intention history for 'select a sofa'")
        for i in agent.intentions.history_for("des_select_sofa"):
            print(f"  {i.intention_id}  {i.params['product_id']:8} Rs {i.params['price']:>8,.0f}  "
                  f"U={i.params['utility']:>6.2f}  {i.status.value:10} {i.status_reason[:60]}")

        sep("Result")
        if failures:
            print(f"  {len(failures)} check(s) FAILED:")
            for text in failures:
                print(f"    - {text}")
            return 1
        print("  All checks passed: the intention changed, a feasible alternative was chosen, "
              "irrelevant updates were ignored, and state survived a restart.")
        return 0
    finally:
        if not keep:
            try:
                repository.delete(AGENT_ID)
            except Exception as exc:
                print(f"  (could not remove the demo record: {type(exc).__name__})")
        else:
            print(f"\n  Record '{AGENT_ID}' left in the store (--keep).")
        if connection is not None:
            connection.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SmartNest BDI demonstration")
    parser.add_argument("--atlas", action="store_true", help="persist to MongoDB Atlas (agent_states)")
    parser.add_argument("--keep", action="store_true", help="leave the saved record in the store")
    args = parser.parse_args()
    sys.exit(main(use_atlas=args.atlas, keep=args.keep))
