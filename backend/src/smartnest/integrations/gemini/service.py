"""GeminiService: validated, structured language-model capabilities.

What it does (all return validated Pydantic objects):

    extract_requirements        user message  -> budgets, size limits, seats, style, colour
    extract_preferences         user message  -> soft preferences and explicit constraints
    extract_products            retrieved text -> product listings (judged per item later)
    explain_recommendation      optimizer facts -> a short explanation
    interpret_change_request    user message + current plan -> requirement changes

What it deliberately does **not** do
------------------------------------
- It does not decide, score or rank anything.  Hard constraints and
  utility are computed elsewhere (the validators and the
  ``UtilityOptimizer``); this service only turns language into structure
  and structure into language.
- It does not trust the model.  Every output is parsed and validated; an
  invalid answer is re-asked once with the specific problem and then
  rejected with :class:`GeminiResponseError`.  Explanations are checked
  against the supplied facts and replaced by a deterministic template if
  they introduce a number or product that was not provided.
- It imports nothing from BDI, CNP or the optimizer.

Transient API errors (429, 5xx, timeouts) are retried with exponential
backoff; the sleep function is injectable so tests do not wait.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Callable, Sequence, TypeVar

from pydantic import BaseModel, ValidationError

from smartnest.integrations.gemini import prompts
from smartnest.integrations.gemini.client import LLMClient, LLMResponse
from smartnest.integrations.gemini.errors import GeminiAPIError, GeminiError, GeminiResponseError
from smartnest.integrations.gemini.schemas import (
    PlanChangeRequest,
    PreferencesExtraction,
    ProductListExtraction,
    RecommendationExplanation,
    RecommendationFacts,
    RequirementsExtraction,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

MAX_MESSAGE_CHARS = 4000
MAX_SOURCE_CHARS = 40000

_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE)
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


class InvalidInputError(GeminiError, ValueError):
    """The text handed to the service is unusable (empty or far too long)."""


def _strip_fences(text: str) -> str:
    match = _FENCE.match(text or "")
    return match.group(1) if match else (text or "").strip()


def _summarise(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        parts = []
        for err in exc.errors()[:5]:
            where = ".".join(str(p) for p in err["loc"]) or "output"
            parts.append(f"{where}: {err['msg'].removeprefix('Value error, ')}")
        return "; ".join(parts)
    return str(exc)[:200]


def _tidy_numbers(value: Any) -> Any:
    """Round floats to 2 places (and whole floats to ints) so the model quotes readable figures.

    Only the copy shown to the model is rounded; faithfulness is still checked against the
    exact facts, with a tolerance that accepts the rounded form.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        rounded = round(value, 2)
        return int(rounded) if rounded.is_integer() else rounded
    if isinstance(value, dict):
        return {k: _tidy_numbers(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_tidy_numbers(v) for v in value]
    return value


def _numbers_in(text: str) -> list[float]:
    found = []
    for raw in _NUMBER.findall(text):
        try:
            found.append(float(raw.replace(",", "")))
        except ValueError:  # pragma: no cover - the regex only yields numerics
            continue
    return found


def _collect_numbers(value: Any, into: set[float]) -> None:
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, (int, float)):
        into.add(float(value))
    elif isinstance(value, str):
        into.update(_numbers_in(value))
    elif isinstance(value, dict):
        for item in value.values():
            _collect_numbers(item, into)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            _collect_numbers(item, into)


#: Numbers an explanation may always use: small counts/ranks ("3 seats", "#2") and the scale of the
#: score itself ("62.69 out of 100").
_SCALE_NUMBERS = frozenset({100.0})


def _supported(number: float, allowed: set[float]) -> bool:
    if number in _SCALE_NUMBERS or (number <= 20 and float(number).is_integer()):
        return True
    return any(abs(number - a) <= max(0.011, abs(a) * 0.0005) for a in allowed)


class GeminiService:
    """Natural-language capabilities on top of an :class:`LLMClient`.

    Args:
        client: Any ``LLMClient`` (the real ``GenaiClient`` or a test double).
        max_attempts: Tries per structured answer (default 2: one re-ask).
        max_api_retries: Retries for transient API errors (default 2).
        sleep: Backoff sleep function (injectable).
    """

    def __init__(self, client: LLMClient, *, max_attempts: int = 2, max_api_retries: int = 2,
                 backoff_seconds: float = 1.0, sleep: Callable[[float], None] = time.sleep) -> None:
        if max_attempts < 1 or max_api_retries < 0:
            raise ValueError("max_attempts must be >= 1 and max_api_retries >= 0")
        self._client = client
        self._max_attempts = max_attempts
        self._max_api_retries = max_api_retries
        self._backoff = backoff_seconds
        self._sleep = sleep

    @property
    def client(self) -> LLMClient:
        return self._client

    # ══ Public capabilities ══════════════════════════════════════════════

    def extract_requirements(self, message: str) -> RequirementsExtraction:
        """Budgets, size limits, seats, style and colour stated in ``message``."""
        text = self._check_message(message)
        return self._structured(prompts.requirements_prompt(text), RequirementsExtraction,
                                prompts.SYSTEM_EXTRACTION, what="requirement extraction")

    def extract_preferences(self, message: str) -> PreferencesExtraction:
        """Soft preferences and explicit constraints stated in ``message``."""
        text = self._check_message(message)
        return self._structured(prompts.preferences_prompt(text), PreferencesExtraction,
                                prompts.SYSTEM_EXTRACTION, what="preference extraction")

    def extract_products(self, source_text: str, *, allowed_urls: Sequence[str] = (),
                         limit: int = 10) -> ProductListExtraction:
        """Product listings found in ``source_text``.

        ``allowed_urls`` are the only URLs the model may use as ``source_url``;
        callers must still verify them (see the product normalizer).
        """
        if not isinstance(source_text, str) or not source_text.strip():
            raise InvalidInputError("there is no text to extract products from")
        if limit < 1:
            raise InvalidInputError("limit must be at least 1")
        text = source_text.strip()[:MAX_SOURCE_CHARS]
        return self._structured(prompts.products_prompt(text, list(allowed_urls), limit), ProductListExtraction,
                                prompts.SYSTEM_EXTRACTION, what="product extraction")

    def interpret_change_request(self, message: str, current_requirements: dict[str, Any]) -> PlanChangeRequest:
        """What the user wants to change about the current plan."""
        text = self._check_message(message)
        return self._structured(prompts.change_request_prompt(text, current_requirements), PlanChangeRequest,
                                prompts.SYSTEM_EXTRACTION, what="change-request interpretation")

    def explain_recommendation(self, facts: RecommendationFacts) -> RecommendationExplanation:
        """A short explanation of the recommendation, grounded in ``facts``.

        If the model fails, or its answer mentions a number or product that is
        not in ``facts``, a deterministic template is returned instead
        (``source == "template"``), so the user always gets a faithful explanation.
        """
        payload = _tidy_numbers(facts.model_dump(exclude_none=True))
        try:
            explanation = self._structured(prompts.explanation_prompt(payload), RecommendationExplanation,
                                           prompts.SYSTEM_WRITING, what="recommendation explanation")
        except GeminiError as exc:
            logger.warning("Explanation unavailable (%s); using the template", type(exc).__name__)
            return self.template_explanation(facts)
        problem = self._explanation_problem(explanation, facts)
        if problem:
            logger.warning("Explanation rejected (%s); using the template", problem)
            return self.template_explanation(facts)
        return explanation.model_copy(update={"source": "gemini"})

    # ══ Deterministic explanation (no LLM) ═══════════════════════════════

    @staticmethod
    def template_explanation(facts: RecommendationFacts) -> RecommendationExplanation:
        sentences = [
            f"{facts.product_name} is recommended at ₹{facts.price:,.0f} with an overall score of "
            f"{facts.utility:.2f} out of 100."
        ]
        if facts.inspected_count:
            sentences.append(f"It scored highest of the {facts.feasible_count} options that meet your requirements "
                             f"(out of {facts.inspected_count} checked).")
        if facts.runner_up_name and facts.runner_up_utility is not None:
            sentences.append(f"The runner-up was {facts.runner_up_name} at {facts.runner_up_utility:.2f}.")
        if facts.retailer:
            sentences.append(f"It is listed by {facts.retailer}.")
        if facts.completeness_percent is not None and facts.completeness_percent < 100:
            missing = f" Not available, so not scored: {', '.join(facts.unavailable_attributes)}." \
                if facts.unavailable_attributes else ""
            sentences.append(f"The score rests on {facts.completeness_percent:g}% of the applicable criteria.{missing}")
        if facts.url_type == "google_shopping":
            sentences.append("The link is a Google Shopping page, not a direct retailer listing.")
        if facts.fit_status == "Fit verified":
            sentences.append("Its stated size was checked against your room limits and fits.")
        elif facts.fit_status == "Fit not verified":
            sentences.append("Its size is not stated, so whether it fits your room is not verified.")
        elif facts.fit_status == "No room limits specified":
            sentences.append("No room limits were given, so fit was not assessed.")
        return RecommendationExplanation(text=" ".join(sentences), source="template",
                                         highlights=[f"score {facts.utility:.2f}/100", f"₹{facts.price:,.0f}"])

    @staticmethod
    def _explanation_problem(explanation: RecommendationExplanation, facts: RecommendationFacts) -> str:
        allowed: set[float] = set()
        _collect_numbers(facts.model_dump(exclude_none=True), allowed)
        for number in _numbers_in(explanation.text):
            if not _supported(number, allowed):
                return f"number {number:g} is not in the facts"
        name = facts.product_name.lower()
        if name not in explanation.text.lower() and facts.product_id.lower() not in explanation.text.lower():
            return "the recommended product is not mentioned"
        return ""

    # ══ Internals ════════════════════════════════════════════════════════

    @staticmethod
    def _check_message(message: str) -> str:
        if not isinstance(message, str) or not message.strip():
            raise InvalidInputError("the message is empty")
        if len(message) > MAX_MESSAGE_CHARS:
            raise InvalidInputError(f"the message is too long ({len(message)} > {MAX_MESSAGE_CHARS} characters)")
        return message.strip()

    def _call(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """One model call, retrying transient API failures with backoff."""
        for attempt in range(self._max_api_retries + 1):
            try:
                return self._client.generate(prompt, **kwargs)
            except GeminiAPIError as exc:
                if not exc.retryable or attempt == self._max_api_retries:
                    raise
                delay = self._backoff * (2 ** attempt)
                logger.warning("Transient Gemini error (%s); retrying in %.1fs", exc.status or "no status", delay)
                self._sleep(delay)
        raise AssertionError("unreachable")  # pragma: no cover

    def _structured(self, prompt: str, schema: type[T], system: str, *, what: str) -> T:
        problem = ""
        for attempt in range(1, self._max_attempts + 1):
            full_prompt = prompt if not problem else (
                f"{prompt}\n\nYour previous reply was rejected: {problem}\n"
                "Reply again with only valid JSON that matches the schema.")
            response = self._call(full_prompt, system=system, schema=schema, temperature=0.0)
            try:
                return schema.model_validate_json(_strip_fences(response.text))
            except (ValidationError, ValueError) as exc:
                problem = _summarise(exc)
                if response.finish_reason == "MAX_TOKENS":
                    problem += " (the reply was cut off)"
                logger.warning("%s: invalid model output on attempt %d: %s", what, attempt, problem)
        raise GeminiResponseError(f"{what} failed: the model's output was invalid after "
                                  f"{self._max_attempts} attempt(s): {problem}")
