"""Names of SmartNest's MongoDB collections.

Only the names are established here.  No schemas, indexes or data
migrations exist yet; each assignment adds its own when it starts
persisting data.

Planned use:

    users             accounts and preferences (API assignments)
    home_projects     a user's rooms, dimensions, budget and style preferences
    products          product catalogue entries (CatalogProvider / ProductCatalog, 4.2)
    agent_states      persisted agent state: BDI beliefs, desires, intentions (4.1)
    tasks             work items exchanged between agents (task_id correlation)
    messages          inter-agent MessageEnvelope history
    recommendations   results of the utility optimizer (3.2)
    negotiations      Contract Net Protocol runs: rounds, offers, outcome (3.1)
"""

from __future__ import annotations

from enum import StrEnum


class Collections(StrEnum):
    """Collection names.  A ``StrEnum``, so members are usable as plain strings."""

    USERS = "users"
    HOME_PROJECTS = "home_projects"
    PRODUCTS = "products"
    AGENT_STATES = "agent_states"
    TASKS = "tasks"
    MESSAGES = "messages"
    RECOMMENDATIONS = "recommendations"
    NEGOTIATIONS = "negotiations"


#: Every established collection name, in declaration order.
ALL_COLLECTIONS: tuple[str, ...] = tuple(c.value for c in Collections)
