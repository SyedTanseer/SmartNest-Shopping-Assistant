"""SmartNest knowledge representation (Assignment 4.2).

Structured, validated, provenance-tracked knowledge about the user's
requirements, home project and rooms, budget, product listings,
environmental observations and agents' conclusions — persisted through an
injected repository.  No database driver, LLM or agent code is imported
here.
"""

from smartnest.knowledge.models import (
    ATTRIBUTE_MODELS,
    AgentKnowledgeAttrs,
    BudgetAttrs,
    HistoryEntry,
    HomeProjectAttrs,
    KnowledgeError,
    KnowledgeKind,
    KnowledgeRecord,
    ObservationAttrs,
    ProductSpecAttrs,
    Provenance,
    Relationship,
    RequirementsAttrs,
    RoomAttrs,
    make_record,
)
from smartnest.knowledge.store import (
    KNOWLEDGE_COLLECTION,
    KnowledgeStore,
    KnowledgeStoreError,
)

__all__ = [
    "ATTRIBUTE_MODELS", "AgentKnowledgeAttrs", "BudgetAttrs", "HistoryEntry", "HomeProjectAttrs",
    "KNOWLEDGE_COLLECTION", "KnowledgeError", "KnowledgeKind", "KnowledgeRecord", "KnowledgeStore",
    "KnowledgeStoreError", "ObservationAttrs", "ProductSpecAttrs", "Provenance", "Relationship",
    "RequirementsAttrs", "RoomAttrs", "make_record",
]
