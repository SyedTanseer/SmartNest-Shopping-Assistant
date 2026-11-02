"""Gemini integration (Assignment 4.2): structured language capabilities via ``google-genai``.

    config.py    GeminiSettings / load_gemini_settings   (GEMINI_API_KEY, GEMINI_MODEL)
    client.py    LLMClient protocol + GenaiClient        (the only code that touches the SDK)
    service.py   GeminiService                           (validated extraction, explanations, plan changes)
    schemas.py   Pydantic schemas for structured outputs
    errors.py    GeminiError hierarchy + secret redaction
    testing.py   ScriptedLLMClient for tests and offline demos

Independent of BDI, CNP and the UtilityOptimizer: Gemini turns language
into validated structure and back; it never decides, scores or overrides
hard constraints.
"""

from smartnest.integrations.gemini.client import (
    GenaiClient,
    GroundingSource,
    LLMClient,
    LLMResponse,
    UrlRetrieval,
)
from smartnest.integrations.gemini.config import (
    DEFAULT_MODEL,
    GeminiSettings,
    gemini_configured,
    load_gemini_settings,
)
from smartnest.integrations.gemini.errors import (
    GeminiAPIError,
    GeminiBlockedError,
    GeminiConfigError,
    GeminiError,
    GeminiResponseError,
    GeminiTimeoutError,
    redact_secrets,
)
from smartnest.integrations.gemini.schemas import (
    ExtractedConstraint,
    PlanChangeRequest,
    PreferencesExtraction,
    ProductExtraction,
    ProductListExtraction,
    RecommendationExplanation,
    RecommendationFacts,
    RequirementChange,
    RequirementsExtraction,
)
from smartnest.integrations.gemini.service import GeminiService, InvalidInputError

__all__ = [
    "DEFAULT_MODEL", "ExtractedConstraint", "GeminiAPIError", "GeminiBlockedError", "GeminiConfigError",
    "GeminiError", "GeminiResponseError", "GeminiService", "GeminiSettings", "GeminiTimeoutError",
    "GenaiClient", "GroundingSource", "InvalidInputError", "LLMClient", "LLMResponse", "PlanChangeRequest",
    "PreferencesExtraction", "ProductExtraction", "ProductListExtraction", "RecommendationExplanation",
    "RecommendationFacts", "RequirementChange", "RequirementsExtraction", "UrlRetrieval",
    "gemini_configured", "load_gemini_settings", "redact_secrets",
]
