"""Gemini client: a small protocol, and the real adapter over ``google-genai``.

``LLMClient`` is the only thing the rest of SmartNest depends on.  The
real implementation, :class:`GenaiClient`, wraps the official
``google-genai`` SDK's ``models.generate_content`` call (documented as
"legacy but fully supported"; the newer Interactions API can be added
behind the same protocol without touching callers).

Design rules
------------
- **Never combine tools with a response schema.**  Grounded search calls
  return free text plus source metadata; structured extraction is a
  separate, tool-free call.  This works on every model regardless of
  whether tool + structured-output combinations are supported.
- **Sources come only from response metadata.**  Grounding chunks and
  URL-context retrievals are returned exactly as the API reported them;
  nothing is derived from the model's prose.
- **All SDK failures become :class:`GeminiError` subclasses**, with the API
  key redacted from every message.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel

from smartnest.integrations.gemini.config import GeminiSettings
from smartnest.integrations.gemini.errors import (
    GeminiAPIError,
    GeminiBlockedError,
    GeminiError,
    GeminiTimeoutError,
)

#: HTTP statuses worth retrying.
RETRYABLE_STATUSES = frozenset({408, 429, 500, 502, 503, 504})


@dataclass(frozen=True)
class GroundingSource:
    """A web source reported by Google Search grounding."""

    uri: str
    title: str | None = None
    domain: str | None = None


@dataclass(frozen=True)
class UrlRetrieval:
    """The outcome of the URL-context tool fetching one page."""

    url: str
    status: str  # "success", "error", "paywall", "unsafe", "unspecified"

    @property
    def succeeded(self) -> bool:
        return self.status == "success"


@dataclass
class LLMResponse:
    """What a Gemini call returned."""

    text: str
    sources: list[GroundingSource] = field(default_factory=list)
    retrievals: list[UrlRetrieval] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)
    finish_reason: str | None = None
    model: str | None = None
    usage: dict[str, int] = field(default_factory=dict)


class LLMClient(Protocol):
    """The interface SmartNest uses to talk to a language model."""

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        schema: type[BaseModel] | None = None,
        grounding: bool = False,
        url_context: bool = False,
        temperature: float = 0.0,
        max_output_tokens: int | None = None,
    ) -> LLMResponse: ...


class GenaiClient:
    """:class:`LLMClient` over the official ``google-genai`` SDK.

    Args:
        settings: Validated Gemini settings (key, model, timeout).
        sdk_client: Inject a prebuilt SDK client (used by tests).
    """

    def __init__(self, settings: GeminiSettings, sdk_client: Any = None) -> None:
        self._settings = settings
        if sdk_client is None:
            from google import genai
            from google.genai import types

            sdk_client = genai.Client(
                api_key=settings.api_key,
                http_options=types.HttpOptions(timeout=int(settings.timeout_seconds * 1000)),
            )
        self._client = sdk_client

    @property
    def model(self) -> str:
        return self._settings.model

    # ── Calling ──────────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        schema: type[BaseModel] | None = None,
        grounding: bool = False,
        url_context: bool = False,
        temperature: float = 0.0,
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        from google.genai import types

        tools = []
        if grounding:
            tools.append(types.Tool(google_search=types.GoogleSearch()))
        if url_context:
            tools.append(types.Tool(url_context=types.UrlContext()))
        if schema is not None and tools:
            raise ValueError("structured output and tools are never combined in one call")

        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            tools=tools or None,
            response_mime_type="application/json" if schema is not None else None,
            response_schema=schema,
        )
        try:
            response = self._client.models.generate_content(
                model=self._settings.model, contents=prompt, config=config)
        except GeminiError:
            raise
        except Exception as exc:
            raise self._translate(exc) from None
        return self._convert(response)

    # ── Error translation ────────────────────────────────────────────────

    def _translate(self, exc: Exception) -> GeminiError:
        redact = self._settings.redact
        name = type(exc).__name__
        try:
            from google.genai import errors as genai_errors
            is_api_error = isinstance(exc, genai_errors.APIError)
        except Exception:  # pragma: no cover - SDK import problem
            is_api_error = False

        if is_api_error:
            code = getattr(exc, "code", None)
            detail = redact(str(getattr(exc, "message", "") or exc))
            return GeminiAPIError(f"Gemini API error {code}: {detail}", status=code,
                                  retryable=code in RETRYABLE_STATUSES)
        if isinstance(exc, TimeoutError) or "timeout" in name.lower():
            return GeminiTimeoutError()
        if "connect" in name.lower() or isinstance(exc, (ConnectionError, OSError)):
            return GeminiAPIError(f"could not reach the Gemini API ({name})", retryable=True)
        return GeminiAPIError(f"unexpected Gemini client error ({name}): {redact(str(exc))}")

    # ── Response conversion ──────────────────────────────────────────────

    def _convert(self, response: Any) -> LLMResponse:
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            feedback = getattr(response, "prompt_feedback", None)
            reason = getattr(feedback, "block_reason", None)
            raise GeminiBlockedError(
                f"Gemini returned no answer" + (f" (blocked: {getattr(reason, 'name', reason)})" if reason else ""))
        candidate = candidates[0]

        parts = getattr(getattr(candidate, "content", None), "parts", None) or []
        text = "".join(p.text for p in parts if getattr(p, "text", None) and not getattr(p, "thought", False))

        finish = getattr(candidate, "finish_reason", None)
        finish_name = getattr(finish, "name", None) or (str(finish) if finish is not None else None)

        sources: list[GroundingSource] = []
        queries: list[str] = []
        grounding = getattr(candidate, "grounding_metadata", None)
        if grounding is not None:
            for chunk in getattr(grounding, "grounding_chunks", None) or []:
                web = getattr(chunk, "web", None)
                if web is not None and getattr(web, "uri", None):
                    sources.append(GroundingSource(uri=web.uri, title=getattr(web, "title", None),
                                                   domain=getattr(web, "domain", None)))
            queries = [q for q in (getattr(grounding, "web_search_queries", None) or []) if q]

        retrievals: list[UrlRetrieval] = []
        context = getattr(candidate, "url_context_metadata", None)
        if context is not None:
            for item in getattr(context, "url_metadata", None) or []:
                url = getattr(item, "retrieved_url", None)
                status = getattr(item, "url_retrieval_status", None)
                status_name = (getattr(status, "name", None) or str(status or "")).replace(
                    "URL_RETRIEVAL_STATUS_", "").lower() or "unspecified"
                if url:
                    retrievals.append(UrlRetrieval(url=url, status=status_name))

        usage_meta = getattr(response, "usage_metadata", None)
        usage = {}
        if usage_meta is not None:
            for key in ("prompt_token_count", "candidates_token_count", "total_token_count"):
                value = getattr(usage_meta, key, None)
                if isinstance(value, int):
                    usage[key] = value

        return LLMResponse(text=text, sources=sources, retrievals=retrievals, search_queries=queries,
                           finish_reason=finish_name, model=getattr(response, "model_version", None) or self.model,
                           usage=usage)
