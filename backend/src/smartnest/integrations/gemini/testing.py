"""Test doubles for the Gemini layer.

``ScriptedLLMClient`` replays queued responses and records every call,
so tests and the offline demonstration can exercise the real service and
product pipeline without network access or credentials.  It is **not** a
model: it returns exactly what it was given.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Union

from pydantic import BaseModel

from smartnest.integrations.gemini.client import GroundingSource, LLMResponse, UrlRetrieval
from smartnest.integrations.gemini.errors import GeminiError

Scripted = Union[LLMResponse, str, dict, BaseModel, Exception, Callable[..., Any]]


def json_response(payload: Any) -> LLMResponse:
    """An ``LLMResponse`` whose text is ``payload`` as JSON."""
    if isinstance(payload, BaseModel):
        text = payload.model_dump_json()
    elif isinstance(payload, str):
        text = payload
    else:
        text = json.dumps(payload)
    return LLMResponse(text=text, finish_reason="STOP")


def grounded_response(text: str, source_uris: list[str], *, titles: list[str] | None = None,
                      retrievals: list[tuple[str, str]] | None = None,
                      queries: list[str] | None = None) -> LLMResponse:
    """An ``LLMResponse`` shaped like a Google-Search-grounded answer."""
    titles = titles or [None] * len(source_uris)
    return LLMResponse(
        text=text,
        sources=[GroundingSource(uri=u, title=t) for u, t in zip(source_uris, titles)],
        retrievals=[UrlRetrieval(url=u, status=s) for u, s in (retrievals or [])],
        search_queries=queries or [],
        finish_reason="STOP",
    )


class ScriptedLLMClient:
    """Replays queued responses in order; records ``calls`` for assertions.

    Each queued item may be an ``LLMResponse``, a JSON string/dict/Pydantic
    model (wrapped as a response), an ``Exception`` (raised), or a callable
    ``(prompt, **kwargs) -> item``.
    """

    def __init__(self, *responses: Scripted) -> None:
        self._queue: list[Scripted] = list(responses)
        self.calls: list[dict[str, Any]] = []

    def queue(self, *responses: Scripted) -> None:
        self._queue.extend(responses)

    @property
    def remaining(self) -> int:
        return len(self._queue)

    def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        self.calls.append({"prompt": prompt, **kwargs})
        if not self._queue:
            raise AssertionError("ScriptedLLMClient has no response left for this call")
        item = self._queue.pop(0)
        if callable(item) and not isinstance(item, (LLMResponse, BaseModel)):
            item = item(prompt, **kwargs)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, LLMResponse):
            return item
        return json_response(item)


def raise_error(error: GeminiError) -> Exception:
    return error
