"""Whole-home furnishing: rooms, categories, budget allocation and the multi-agent coordinator."""

from smartnest.furnishing.activity import ActivityLog, AgentBus, AgentCallError
from smartnest.furnishing.catalog import (
    LayeredCatalog, SimulatedCatalog, SnapshotCatalog, CategoryListings, ORIGIN_REAL, ORIGIN_SIMULATED,
)
from smartnest.furnishing.categories import CATEGORIES, CategoryProfile, category
from smartnest.furnishing.coordinator import FurnishingCoordinator, FurnishingRequest, Plan, PlanError
from smartnest.furnishing.rooms import (
    ROOM_KINDS, SCOPE_ROOMS, Room, Slot, parse_home_type, rooms_for, rooms_for_scope, slots_for,
)

__all__ = [
    "ActivityLog", "AgentBus", "AgentCallError", "CATEGORIES", "CategoryListings", "CategoryProfile",
    "FurnishingCoordinator", "FurnishingRequest", "LayeredCatalog", "ORIGIN_REAL", "ORIGIN_SIMULATED", "Plan",
    "PlanError", "Room", "SimulatedCatalog", "Slot", "SnapshotCatalog", "category", "parse_home_type", "rooms_for",
    "slots_for", "ROOM_KINDS", "SCOPE_ROOMS", "rooms_for_scope",
]
