"""Shared helpers for the BDI tests."""

from __future__ import annotations

from typing import Any

from smartnest.bdi import (
    BDIAgent,
    BDIPlanner,
    CatalogCandidateSource,
    select_sofa_desire,
    stay_within_budget_desire,
)
from smartnest.cnp.product_catalog import catalog_agent_a, catalog_agent_b
from smartnest.core.models import EnvironmentalEvent, EventType
from smartnest.optimization import UserRequirements, UtilityOptimizer

REQUIREMENTS: dict[str, Any] = {
    "total_budget": 200000,
    "sofa_budget": 30000,
    "max_width_cm": 210,
    "max_depth_cm": 110,
    "min_seating": 3,
    "preferred_style": "minimalist",
    "preferred_colour": "neutral",
}
SELECT = "des_select_sofa"
BUDGET = "des_total_budget"


def candidate_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for catalog in (catalog_agent_a(extended=True), catalog_agent_b(extended=True)):
        records += [{**p, "source_agent_id": catalog.agent_id} for p in catalog.get_all_products()]
    return records


def make_planner(source=None, optimizer=None) -> BDIPlanner:
    return BDIPlanner.default(optimizer or UtilityOptimizer(),
                              source or CatalogCandidateSource([catalog_agent_a(True), catalog_agent_b(True)]))


def make_agent(store=None, requirements=REQUIREMENTS, start=True, agent_id="bdi_test",
               planner=None, **kwargs) -> BDIAgent:
    agent = BDIAgent(agent_id, planner or make_planner(), state_store=store, **kwargs)
    if requirements:
        agent.set_requirements(dict(requirements))
    agent.add_desire(select_sofa_desire())
    agent.add_desire(stay_within_budget_desire())
    if start:
        agent.deliberate("test start")
    return agent


def event(kind: EventType, source: str = "test", **payload: Any) -> EnvironmentalEvent:
    return EnvironmentalEvent(event_type=kind, source=source, payload=payload)


def out_of_stock(product_id: str) -> EnvironmentalEvent:
    return event(EventType.PRODUCT_AVAILABILITY, "product_monitor", product_id=product_id,
                 availability="out_of_stock")


def in_stock(product_id: str) -> EnvironmentalEvent:
    return event(EventType.PRODUCT_AVAILABILITY, "product_monitor", product_id=product_id,
                 availability="in_stock")


def reference_choice(requirements: dict[str, Any] | None = None, availability: dict[str, str] | None = None,
                     prices: dict[str, float] | None = None, records: list[dict] | None = None) -> str | None:
    """What the UtilityOptimizer picks when run directly — the independent oracle."""
    req = {**REQUIREMENTS, **(requirements or {})}
    budgets = [v for v in (req.get("sofa_budget"), req.get("total_budget")) if v]
    candidates = []
    for record in (records if records is not None else candidate_records()):
        record = dict(record)
        record["availability"] = (availability or {}).get(record["product_id"], record["availability"])
        record["price"] = (prices or {}).get(record["product_id"], record["price"])
        candidates.append(record)
    result = UtilityOptimizer().optimize(candidates, UserRequirements(
        budget=min(budgets), max_width_cm=req["max_width_cm"], max_depth_cm=req["max_depth_cm"],
        min_seating=req["min_seating"], preferred_style=req["preferred_style"],
        preferred_colour=req["preferred_colour"]))
    return result.selected.product_id if result.selected else None


def chosen(agent: BDIAgent, desire_id: str = SELECT) -> str | None:
    intention = agent.current_intention(desire_id)
    return intention.params["product_id"] if intention else None


def kinds_since(agent: BDIAgent, index: int) -> list[str]:
    return [t.kind for t in agent.transitions[index:]]
