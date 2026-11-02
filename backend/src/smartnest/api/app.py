"""SmartNest HTTP API (FastAPI).

The API is a thin layer: it validates input, finds the session, calls the furnishing layer and returns JSON.  All
planning, ranking and budget arithmetic stays in the agents below it.

    GET    /api/health                              component status, SerpApi quota, catalog origin
    GET    /api/catalog                             what product data exists per category (real/simulated, age)
    POST   /api/sessions                            start a session (optionally with a first message or fields)
    GET    /api/sessions/{id}                       plan, rooms, budget analytics, shortlist, chat history
    POST   /api/sessions/{id}/messages              chat: the assistant parses it, replans, replies
    POST   /api/sessions/{id}/events                structured change: budget, preferences, stock, price, choice...
    GET    /api/sessions/{id}/rooms                 room-wise view
    GET    /api/sessions/{id}/budget                budget analytics
    GET    /api/sessions/{id}/items/{slot}/options  every listing found for one item
    POST   /api/sessions/{id}/shortlist             toggle a product on the shortlist
    POST   /api/sessions/{id}/compare               side-by-side data for 2-4 products
    GET    /api/sessions/{id}/activity?since=N      the agents' activity log + roster + signature stats

It makes no product search request: listings come from saved real snapshots (or, only when none exist and it is
allowed, from clearly labelled simulated samples).  A SerpApi key is never needed to run the API.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from smartnest.api import views
from smartnest.api.sessions import SessionNotFound, SessionRegistry
from smartnest.furnishing import (
    CATEGORIES,
    FurnishingCoordinator,
    FurnishingRequest,
    LayeredCatalog,
    PlanError,
    SimulatedCatalog,
    SnapshotCatalog,
)
from smartnest.furnishing.assistant import ChatAssistant, HomeSession, MAX_MESSAGE_CHARS
from smartnest.furnishing.catalog import DEFAULT_CATALOG_DIR

logger = logging.getLogger(__name__)


# ── Request bodies ───────────────────────────────────────────────────────────

class CreateSession(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str | None = Field(default=None, max_length=MAX_MESSAGE_CHARS)
    total_budget: float | None = Field(default=None, gt=0, le=1e9)
    bedrooms: int | None = Field(default=None, ge=1, le=5)
    scope: list[str] = Field(default_factory=list, max_length=5)
    audience: Literal["child", "teen", "adult"] | None = None
    age: int | None = Field(default=None, ge=0, le=99)
    themes: list[str] = Field(default_factory=list, max_length=8)
    items: list[str] = Field(default_factory=list, max_length=12)       # category ids or plain names ("bed", "telescope")
    style: str | None = Field(default=None, max_length=40)
    colour: str | None = Field(default=None, max_length=40)


class ChatBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)


class EventBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["out_of_stock", "price_change", "budget_change", "preference_change", "remove_item",
                  "choose_product", "clear_choice", "cart_add", "cart_set", "cart_remove", "cart_clear",
                  "cart_add_plan", "refresh_category"]
    category: str | None = Field(default=None, max_length=40)
    themes: list[str] | None = Field(default=None, max_length=8)
    qty: int | None = Field(default=None, ge=0, le=10)
    product_id: str | None = Field(default=None, max_length=80)
    slot_id: str | None = Field(default=None, max_length=80)
    price: float | None = Field(default=None, gt=0, le=1e9)
    total_budget: float | None = Field(default=None, gt=0, le=1e9)
    style: str | None = Field(default=None, max_length=40)
    colour: str | None = Field(default=None, max_length=40)


class SearchBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    category: str = Field(min_length=2, max_length=40)


class ShortlistBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    product_id: str = Field(min_length=1, max_length=80)


class CompareBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    product_ids: list[str] = Field(min_length=2, max_length=4)


# ── Settings ─────────────────────────────────────────────────────────────────

class ApiSettings(BaseModel):
    catalog_dir: Path = DEFAULT_CATALOG_DIR
    allow_simulated: bool | Literal["auto"] = "auto"
    use_gemini: bool | Literal["auto"] = "auto"
    use_atlas: bool = False
    live_search: bool = False              # SMARTNEST_ALLOW_LIVE_SEARCH=1: allow the (quota-limited) live search button
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    max_sessions: int = 200
    requests_per_minute: int = 120

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None, env_file: Path | str | None = None) -> "ApiSettings":
        """Settings from ``SMARTNEST_*`` variables: the real environment wins over ``backend/.env``.

        An explicit ``environ`` (tests) is used as given, with no file, so a test never depends on the developer's .env.
        """
        if environ is None:
            from dotenv import dotenv_values
            path = Path(env_file) if env_file is not None else Path(__file__).resolve().parents[3] / ".env"
            env = {**(dotenv_values(path) if path.is_file() else {}), **os.environ}
        else:
            env = environ

        def flag(name: str, default: Any) -> Any:
            raw = (env.get(name) or "").strip().lower()
            return "auto" if raw == "auto" else (raw in ("1", "true", "yes")) if raw else default

        values: dict[str, Any] = {
            "allow_simulated": flag("SMARTNEST_ALLOW_SIMULATED", "auto"),
            "use_gemini": flag("SMARTNEST_USE_GEMINI", "auto"),
            "use_atlas": flag("SMARTNEST_USE_ATLAS", False),
            "live_search": flag("SMARTNEST_ALLOW_LIVE_SEARCH", False),
        }
        if env.get("SMARTNEST_CATALOG_DIR"):
            values["catalog_dir"] = Path(env["SMARTNEST_CATALOG_DIR"])
        if env.get("SMARTNEST_CORS_ORIGINS"):
            values["cors_origins"] = [o.strip() for o in env["SMARTNEST_CORS_ORIGINS"].split(",") if o.strip()]
        return cls(**values)


class RateLimiter:
    """Sliding one-minute window per client address (protects Gemini usage and the CPU)."""

    def __init__(self, per_minute: int) -> None:
        self._limit, self._hits, self._lock = per_minute, defaultdict(deque), threading.Lock()

    def allow(self, client: str) -> bool:
        now = time.monotonic()
        with self._lock:
            hits = self._hits[client]
            while hits and now - hits[0] > 60:
                hits.popleft()
            if len(hits) >= self._limit:
                return False
            hits.append(now)
            if len(self._hits) > 5000:
                self._hits.clear()
            return True


# ── Factory ──────────────────────────────────────────────────────────────────

def client_address(request: Request) -> str:
    """The caller's address for rate limiting.

    On Vercel every request arrives through the platform's proxy, so ``request.client`` is the proxy and would put all
    visitors in one bucket.  Vercel sets ``X-Forwarded-For`` itself (it replaces any value the visitor sent), so there
    (and only there, when ``VERCEL`` is set) its first address is the visitor.  Anywhere else the header is untrusted.
    """
    if os.environ.get("VERCEL"):
        first = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
        if first:
            return first
    return request.client.host if request.client else "?"


def _build_gemini(settings: ApiSettings):
    if settings.use_gemini is False:
        return None
    try:
        from smartnest.integrations.gemini import GeminiService, GenaiClient, gemini_configured, load_gemini_settings
        if not gemini_configured():
            return None
        return GeminiService(GenaiClient(load_gemini_settings()))
    except Exception as exc:
        logger.warning("Gemini unavailable: %s", type(exc).__name__)
        return None


def _build_repository(settings: ApiSettings):
    if not settings.use_atlas:
        return None
    try:
        from smartnest.database import BaseRepository, Collections, MongoConnection, load_settings
        connection = MongoConnection(load_settings())
        connection.connect()
        return BaseRepository(connection.collection(Collections.HOME_PROJECTS.value), id_prefix="home")
    except Exception as exc:
        logger.warning("MongoDB unavailable, sessions will not persist: %s", type(exc).__name__)
        return None


def _build_searcher(settings: ApiSettings, snapshots: SnapshotCatalog):
    if not settings.live_search:
        return None
    try:
        from smartnest.furnishing.live import LiveSearcher
        from smartnest.integrations.products.serpapi import load_serpapi_settings
        return LiveSearcher(load_serpapi_settings(), snapshots)
    except Exception as exc:
        logger.warning("live search unavailable: %s", type(exc).__name__)
        return None


def create_app(settings: ApiSettings | None = None, *, catalog=None, gemini: Any = "auto",
               repository: Any = "auto", searcher: Any = "auto") -> FastAPI:
    settings = settings or ApiSettings.from_env()
    snapshots = SnapshotCatalog(settings.catalog_dir)
    if catalog is None:
        has_real = any(snapshots.listings(c).records for c in CATEGORIES)
        allow_sim = (not has_real) if settings.allow_simulated == "auto" else bool(settings.allow_simulated)
        catalog = LayeredCatalog(snapshots, SimulatedCatalog(), allow_simulated=allow_sim)
    llm = _build_gemini(settings) if gemini == "auto" else gemini
    live = _build_searcher(settings, snapshots) if searcher == "auto" else searcher
    repo = _build_repository(settings) if repository == "auto" else repository
    registry = SessionRegistry(lambda: FurnishingCoordinator(catalog), repository=repo, max_sessions=settings.max_sessions)
    assistant = ChatAssistant(llm)
    limiter = RateLimiter(settings.requests_per_minute)

    app = FastAPI(title="SmartNest API", version="1.0.0")
    app.state.registry, app.state.assistant, app.state.catalog = registry, assistant, catalog
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["GET", "POST"],
                       allow_headers=["Content-Type"], allow_credentials=False)

    @app.middleware("http")
    async def throttle(request: Request, call_next):
        if request.method == "POST" and not limiter.allow(client_address(request)):
            return JSONResponse({"detail": "too many requests, slow down"}, status_code=429)
        return await call_next(request)

    @app.exception_handler(PlanError)
    async def plan_error(_: Request, exc: PlanError):
        return JSONResponse({"detail": str(exc)}, status_code=422)

    @app.exception_handler(SessionNotFound)
    async def not_found(_: Request, exc: SessionNotFound):
        return JSONResponse({"detail": "session not found"}, status_code=404)

    @app.exception_handler(Exception)
    async def unexpected(_: Request, exc: Exception):
        logger.exception("unhandled error")
        return JSONResponse({"detail": "internal error"}, status_code=500)

    def session_of(session_id: str) -> HomeSession:
        return registry.get(session_id)

    # ── Status ───────────────────────────────────────────────────────────

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        quota = None
        try:
            from smartnest.integrations.products.serpapi.client import UsageLedger
            from smartnest.integrations.products.serpapi.config import SerpApiSettings
            path = SerpApiSettings.__dataclass_fields__["usage_file"].default
            ledger = UsageLedger(path, 10 ** 9)
            limit = _lifetime_limit()
            quota = {"used": ledger.used, "limit": limit}
        except Exception:
            pass
        return {"status": "ok", "gemini": llm is not None, "persistence": "mongodb" if repo is not None else "memory",
                "serpapi_quota": quota, "sessions": len(registry),
                "live_search": live.status() if live is not None else {"enabled": False},
                "catalog": {"simulated_allowed": getattr(catalog, "allow_simulated", False)}}

    @app.get("/api/catalog")
    def catalog_info() -> dict[str, Any]:
        out = []
        for cat_id, profile in CATEGORIES.items():
            listings = catalog.listings(cat_id)
            out.append({"category": cat_id, "label": profile.label, "origin": listings.origin,
                        "listings": len(listings.records), "retrieved_at": listings.retrieved_at,
                        "query": listings.query, "notes": listings.notes})
        return {"categories": out}

    # ── Sessions ─────────────────────────────────────────────────────────

    @app.post("/api/sessions", status_code=201)
    def create_session(body: CreateSession) -> dict[str, Any]:
        session = registry.create()
        with session.lock:
            if body.total_budget is not None and (body.bedrooms is not None or body.scope or body.items):
                from smartnest.furnishing.naming import resolve_names
                items = tuple(resolve_names(body.items))
                if body.items and not items:
                    raise PlanError("none of the items could be understood as things to buy")
                session.start(FurnishingRequest(
                    body.total_budget, body.bedrooms, body.style.strip().lower() if body.style else None,
                    body.colour.strip().lower() if body.colour else None, None, tuple(body.scope), body.audience,
                    body.age, tuple(t.strip().lower() for t in body.themes), items))
                session.say("assistant", assistant.describe_plan(session.plan, intro=True), "new_request")
            elif body.message:
                assistant.respond(session, body.message)
        registry.save(session)
        return views.session_view(session)

    @app.get("/api/sessions/{session_id}")
    def get_session(session_id: str) -> dict[str, Any]:
        return views.session_view(session_of(session_id))

    @app.post("/api/sessions/{session_id}/messages")
    def chat(session_id: str, body: ChatBody) -> dict[str, Any]:
        session = session_of(session_id)
        reply = assistant.respond(session, body.message)
        registry.save(session)
        return {"reply": {"text": reply.text, "kind": reply.kind, "changed": reply.changed, "needs": reply.needs,
                          "wording": reply.wording}, "session": views.session_view(session)}

    @app.post("/api/sessions/{session_id}/events")
    def apply_event(session_id: str, body: EventBody) -> dict[str, Any]:
        session = session_of(session_id)
        event = body.model_dump(exclude_none=True)
        with session.lock:
            if session.plan is None:
                raise PlanError("there is no plan yet")
            session.apply(event)
        registry.save(session)
        return views.session_view(session)

    @app.post("/api/sessions/{session_id}/search")
    def live_search(session_id: str, body: SearchBody) -> dict[str, Any]:
        """Search the live shopping API for ONE category of this plan.  Spends one SerpApi request."""
        from smartnest.furnishing.live import LiveSearchError
        session = session_of(session_id)
        if live is None:
            return JSONResponse({"detail": "Live search is switched off. Set SMARTNEST_ALLOW_LIVE_SEARCH=1 "
                                           "(and a SerpApi key) to enable it."}, status_code=409)
        with session.lock:
            if session.plan is None:
                raise PlanError("there is no plan yet")
            if body.category not in {s.profile.id for s in session.coordinator._slots}:
                raise PlanError(f"{body.category!r} is not part of this plan")
            try:
                found = live.search(body.category, session.request.themes if session.request else ())
            except LiveSearchError as exc:
                return JSONResponse({"detail": str(exc), "code": exc.code}, status_code=409 if exc.code == "quota" else 502)
            if not found.records:
                return JSONResponse({"detail": f"The search found no usable {body.category.replace('_', ' ')} listings.",
                                     "code": "empty"}, status_code=404)
            session.apply({"type": "refresh_category", "category": body.category})
        registry.save(session)
        return {"found": len(found.records), "query": found.query, "session": views.session_view(session)}

    @app.get("/api/sessions/{session_id}/rooms")
    def rooms(session_id: str) -> dict[str, Any]:
        return {"rooms": views.room_view(session_of(session_id))}

    @app.get("/api/sessions/{session_id}/budget")
    def budget(session_id: str) -> dict[str, Any]:
        data = views.budget_view(session_of(session_id))
        if data is None:
            raise PlanError("there is no plan yet")
        return data

    @app.get("/api/sessions/{session_id}/items/{slot_id}/options")
    def options(session_id: str, slot_id: str) -> dict[str, Any]:
        return views.options_view(session_of(session_id), slot_id)

    @app.post("/api/sessions/{session_id}/shortlist")
    def shortlist(session_id: str, body: ShortlistBody) -> dict[str, Any]:
        session = session_of(session_id)
        with session.lock:
            items = session.coordinator.toggle_shortlist(body.product_id)
        registry.save(session)
        return {"shortlist": items}

    @app.post("/api/sessions/{session_id}/compare")
    def compare(session_id: str, body: CompareBody) -> dict[str, Any]:
        return {"products": views.compare_view(session_of(session_id), body.product_ids)}

    @app.get("/api/sessions/{session_id}/activity")
    def activity(session_id: str, since: int = 0) -> dict[str, Any]:
        session = session_of(session_id)
        data = views.agents_view(session)
        data["events"] = [e.to_dict() for e in session.coordinator.log.events(max(since, 0))][-400:]
        return data

    return app


def _lifetime_limit() -> int | None:
    try:
        from smartnest.integrations.products.serpapi.config import load_serpapi_settings
        return load_serpapi_settings().lifetime_limit
    except Exception:
        return None



def get_app() -> FastAPI:
    """``uvicorn smartnest.api.app:get_app --factory``."""
    return create_app()
