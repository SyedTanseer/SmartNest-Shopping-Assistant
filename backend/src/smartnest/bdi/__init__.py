"""SmartNest BDI architecture (Assignment 4.1).

    beliefs.py       BeliefBase     facts with source/time; updates report real changes; invalidation
    conditions.py    Condition      tests over beliefs (preconditions, activation conditions)
    desires.py       DesireManager  prioritised goals, conditions, status, conflict resolution
    intentions.py    IntentionManager  committed plans: validate, execute, suspend, cancel, replace
    planner.py       BDIPlanner     desire -> intention; uses the existing UtilityOptimizer
    bdi_agent.py     BDIAgent       the deliberation cycle, built on BaseAgent
    transitions.py   Transition     explained before/after history
    state_store.py   AgentStateStore  persistence interface (+ in-memory, repository-backed)

Deterministic Python only.  No database driver, socket, Gemini or other
external service is imported here; persistence is injected.
"""

from smartnest.bdi.bdi_agent import BDIAgent
from smartnest.bdi.beliefs import Belief, BeliefBase, BeliefChange
from smartnest.bdi.conditions import Condition, ConditionResult
from smartnest.bdi.desires import (
    Desire,
    DesireManager,
    DesireStatus,
    select_sofa_desire,
    stay_within_budget_desire,
)
from smartnest.bdi.intentions import (
    Action,
    ActionOutcome,
    ActionStatus,
    Intention,
    IntentionManager,
    IntentionStatus,
    ValidationResult,
    default_executor,
)
from smartnest.bdi.planner import (
    BDIPlanner,
    CandidateSource,
    CatalogCandidateSource,
    MaintainBudgetHandler,
    PlanOutcome,
    SelectProductHandler,
    effective_budget,
    normalise_requirement,
)
from smartnest.bdi.state_store import (
    AgentSnapshot,
    AgentStateStore,
    InMemoryAgentStateStore,
    RepositoryAgentStateStore,
    StateStoreError,
)
from smartnest.bdi.transitions import Transition

__all__ = [
    "Action", "ActionOutcome", "ActionStatus", "AgentSnapshot", "AgentStateStore", "BDIAgent",
    "BDIPlanner", "Belief", "BeliefBase", "BeliefChange", "CandidateSource", "CatalogCandidateSource",
    "Condition", "ConditionResult", "Desire", "DesireManager", "DesireStatus",
    "InMemoryAgentStateStore", "Intention", "IntentionManager", "IntentionStatus",
    "MaintainBudgetHandler", "PlanOutcome", "RepositoryAgentStateStore", "SelectProductHandler",
    "StateStoreError", "Transition", "ValidationResult", "default_executor", "effective_budget",
    "normalise_requirement", "select_sofa_desire", "stay_within_budget_desire",
]
