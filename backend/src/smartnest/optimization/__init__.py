"""SmartNest utility optimization (Assignment 3.2).

A self-contained, deterministic decision component:

    ScoringConfig        — validated weights and compatibility tables
    ConstraintValidator  — record validation + hard constraints
    UtilityModel         — U = 100 × (0.30P + 0.25Q + 0.20F + 0.15S + 0.10C)
    UtilityOptimizer     — exhaustive search, ranking, selection, explanation

It has no dependency on CNP, sockets, databases, LLMs or the UI, so CNP
agents (3.2), BDI agents (4.1) and API-backed catalogues (4.2) can all use
it unchanged.
"""

from smartnest.optimization.constraint_validator import (
    AVAILABILITY_LABELS,
    ConstraintValidator,
    EligibilityResult,
    InvalidRequirementsError,
    FIT_LABELS,
    ProductCandidate,
    SpatialAssessment,
    UserRequirements,
    assess_space,
    validate_record,
)
from smartnest.optimization.scoring_config import (
    COMPONENTS,
    DEFAULT_WEIGHTS,
    ScoringConfig,
    ScoringConfigError,
)
from smartnest.optimization.utility_model import UtilityBreakdown, UtilityModel
from smartnest.optimization.utility_optimizer import (
    CandidateEvaluation,
    OptimizationResult,
    UtilityOptimizer,
    ranking_key,
)

__all__ = [
    "AVAILABILITY_LABELS",
    "COMPONENTS",
    "DEFAULT_WEIGHTS",
    "FIT_LABELS",
    "CandidateEvaluation",
    "ConstraintValidator",
    "EligibilityResult",
    "InvalidRequirementsError",
    "OptimizationResult",
    "ProductCandidate",
    "ScoringConfig",
    "ScoringConfigError",
    "SpatialAssessment",
    "UserRequirements",
    "UtilityBreakdown",
    "UtilityModel",
    "UtilityOptimizer",
    "assess_space",
    "ranking_key",
    "validate_record",
]
