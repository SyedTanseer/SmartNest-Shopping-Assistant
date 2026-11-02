"""Structured, validated knowledge records.

Everything SmartNest learns — from the user, from an LLM, from a web
page, from an environmental event, from its own reasoning — can be kept
as a :class:`KnowledgeRecord`.  Every record carries

    knowledge_id     unique identifier
    kind, entity     what sort of fact it is and what it is about
    attributes       the facts themselves, validated per kind
    relationships    typed links to other records
    provenance       where it came from (source, URL, method, retrieval time,
                     which fields were inferred rather than stated)
    confidence       0..1
    version/history  every update bumps the version and keeps the old values
    valid            records are never silently deleted; they are invalidated
                     with a reason

Kinds
-----
    requirements   what the user wants (budgets, size limits, seats, style, colour)
    home_project   a furnishing project
    room           a room and its dimensions
    budget         a budget and its allocation
    product_spec   a product listing: price, dimensions, rating, retailer, URL
    observation    something observed in the environment (a price change, ...)
    agent_knowledge a conclusion or decision of an agent (a recommendation, ...)

This module knows nothing about databases, LLMs or agents.  Records
serialise to plain JSON-compatible dicts, so any repository can store them.
"""

from __future__ import annotations

import math
import threading
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


_clock_lock = threading.Lock()
_last_timestamp = datetime.min.replace(tzinfo=timezone.utc)


def utc_now() -> datetime:
    """Timezone-aware UTC time that never repeats or goes backwards within this process.

    The system clock can tick coarsely (about 15 ms on Windows), so two records created
    back to back would otherwise get identical timestamps and "most recently modified"
    would be ambiguous.
    """
    global _last_timestamp
    with _clock_lock:
        now = datetime.now(timezone.utc)
        if now <= _last_timestamp:
            now = _last_timestamp + timedelta(microseconds=1)
        _last_timestamp = now
        return now


class KnowledgeError(ValueError):
    """A knowledge record is invalid or an operation on it is not allowed."""


# ── Provenance and relationships ─────────────────────────────────────────────

SourceType = Literal["user", "llm", "web", "api", "catalogue", "agent", "event", "system"]


class Provenance(BaseModel):
    """Where a piece of knowledge came from."""

    source: str                      # e.g. "user", "gemini", "gemini_search", "product_monitor"
    source_type: SourceType
    url: str | None = None           # the page / document it was read from, if any
    retrieved_at: datetime = Field(default_factory=utc_now)
    method: str = ""                 # e.g. "google_search_grounding", "requirement_extraction"
    inferred_fields: list[str] = Field(default_factory=list)  # attributes guessed, not stated
    notes: str = ""

    @field_validator("source")
    @classmethod
    def _source_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("provenance source must not be empty")
        return value.strip()


class Relationship(BaseModel):
    """A typed link to another record (``part_of``, ``sold_by``, ``about``, ...)."""

    relation: str
    target_id: str
    target_kind: str | None = None

    @field_validator("relation", "target_id")
    @classmethod
    def _not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value.strip()


class HistoryEntry(BaseModel):
    """The values an update replaced."""

    version: int
    replaced_at: datetime = Field(default_factory=utc_now)
    replaced_by: str = ""
    previous: dict[str, Any] = Field(default_factory=dict)  # attribute -> old value (None = was absent)


class KnowledgeKind(str, Enum):
    REQUIREMENTS = "requirements"
    HOME_PROJECT = "home_project"
    ROOM = "room"
    BUDGET = "budget"
    PRODUCT_SPEC = "product_spec"
    OBSERVATION = "observation"
    AGENT_KNOWLEDGE = "agent_knowledge"


# ── Attribute models (one per kind) ──────────────────────────────────────────

def _finite_positive(value: float | None, name: str, allow_zero: bool = False) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{name} must be a finite number {'>= 0' if allow_zero else '> 0'}")
    return value


def _clean_text(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


class _Attrs(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class RequirementsAttrs(_Attrs):
    """What the user wants.  Every field optional; unknown fields are rejected."""

    category: str | None = None
    currency: Literal["INR"] = "INR"
    total_budget: float | None = None
    sofa_budget: float | None = None
    max_width_cm: float | None = None
    max_depth_cm: float | None = None
    min_seating: int | None = Field(default=None, strict=True)
    preferred_style: str | None = None
    preferred_colour: str | None = None
    preferred_materials: list[str] = Field(default_factory=list)
    notes: str | None = None

    @field_validator("total_budget", "sofa_budget", "max_width_cm", "max_depth_cm")
    @classmethod
    def _positive(cls, v: float | None, info: Any) -> float | None:
        return _finite_positive(v, info.field_name)

    @field_validator("min_seating")
    @classmethod
    def _seats(cls, v: int | None) -> int | None:
        if v is not None and (isinstance(v, bool) or v < 1):
            raise ValueError("min_seating must be at least 1")
        return v

    @field_validator("category", "preferred_style", "preferred_colour", "notes")
    @classmethod
    def _text(cls, v: str | None) -> str | None:
        v = _clean_text(v)
        return v.lower() if v is not None else None

    @field_validator("preferred_materials")
    @classmethod
    def _materials(cls, v: list[str]) -> list[str]:
        return sorted({m.strip().lower() for m in v if m and m.strip()})

    #: knowledge attribute -> BDI belief predicate (subject "user")
    BELIEF_FIELDS: ClassVar[tuple[str, ...]] = (
        "total_budget", "sofa_budget", "max_width_cm", "max_depth_cm",
        "min_seating", "preferred_style", "preferred_colour",
    )

    def belief_values(self) -> dict[str, Any]:
        """The ``user.*`` belief values this record supports (only fields that are set)."""
        data = self.model_dump()
        return {k: data[k] for k in self.BELIEF_FIELDS if data.get(k) is not None}


class HomeProjectAttrs(_Attrs):
    name: str
    description: str | None = None
    location: str | None = None

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        if not v:
            raise ValueError("name must not be empty")
        return v


class RoomAttrs(_Attrs):
    name: str
    width_cm: float
    length_cm: float
    height_cm: float | None = None
    purpose: str | None = None

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        if not v:
            raise ValueError("name must not be empty")
        return v

    @field_validator("width_cm", "length_cm", "height_cm")
    @classmethod
    def _positive(cls, v: float | None, info: Any) -> float | None:
        return _finite_positive(v, info.field_name)


class BudgetAttrs(_Attrs):
    total_budget: float
    currency: Literal["INR"] = "INR"
    allocations: dict[str, float] = Field(default_factory=dict)

    @field_validator("total_budget")
    @classmethod
    def _total(cls, v: float) -> float:
        return _finite_positive(v, "total_budget")

    @model_validator(mode="after")
    def _allocations(self) -> "BudgetAttrs":
        for name, amount in self.allocations.items():
            _finite_positive(amount, f"allocation '{name}'")
        if sum(self.allocations.values()) > self.total_budget + 1e-9:
            raise ValueError("allocations exceed the total budget")
        return self


class ProductSpecAttrs(_Attrs):
    """A product listing.  Only identity is mandatory; completeness is judged downstream."""

    product_id: str
    name: str
    category: str | None = None
    price: float | None = None
    currency: Literal["INR"] = "INR"
    width_cm: float | None = None
    depth_cm: float | None = None
    height_cm: float | None = None
    seating_capacity: int | None = Field(default=None, strict=True)
    material: str | None = None
    colour: str | None = None
    style: str | None = None
    availability: str | None = None
    rating: float | None = None
    review_count: int | None = Field(default=None, strict=True)
    retailer: str | None = None
    url: str | None = None                 # a direct retailer URL only
    url_type: Literal["retailer", "google_shopping", "missing", "invalid"] | None = None
    product_page_url: str | None = None    # the Google Shopping page, when that is all there is (never a retailer link)
    image_url: str | None = None
    verified_at: datetime | None = None

    @field_validator("product_id", "name")
    @classmethod
    def _required_text(cls, v: str) -> str:
        if not v:
            raise ValueError("must not be empty")
        return v

    @field_validator("price", "width_cm", "depth_cm", "height_cm")
    @classmethod
    def _positive(cls, v: float | None, info: Any) -> float | None:
        return _finite_positive(v, info.field_name)

    @field_validator("rating")
    @classmethod
    def _rating(cls, v: float | None) -> float | None:
        if v is not None and (not math.isfinite(v) or not 0 <= v <= 5):
            raise ValueError("rating must be between 0 and 5")
        return v

    @field_validator("seating_capacity", "review_count")
    @classmethod
    def _counts(cls, v: int | None, info: Any) -> int | None:
        if v is not None and (isinstance(v, bool) or v < (1 if info.field_name == "seating_capacity" else 0)):
            raise ValueError(f"{info.field_name} is out of range")
        return v

    @field_validator("material", "colour", "style", "availability", "category")
    @classmethod
    def _lower(cls, v: str | None) -> str | None:
        v = _clean_text(v)
        return v.lower() if v is not None else None


class ObservationAttrs(_Attrs):
    observation_type: str
    subject: str
    details: dict[str, Any] = Field(default_factory=dict)
    observed_at: datetime = Field(default_factory=utc_now)

    @field_validator("observation_type", "subject")
    @classmethod
    def _required(cls, v: str) -> str:
        if not v:
            raise ValueError("must not be empty")
        return v


class AgentKnowledgeAttrs(_Attrs):
    agent_id: str
    topic: str
    statement: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)

    @field_validator("agent_id", "topic")
    @classmethod
    def _required(cls, v: str) -> str:
        if not v:
            raise ValueError("must not be empty")
        return v


ATTRIBUTE_MODELS: dict[KnowledgeKind, type[_Attrs]] = {
    KnowledgeKind.REQUIREMENTS: RequirementsAttrs,
    KnowledgeKind.HOME_PROJECT: HomeProjectAttrs,
    KnowledgeKind.ROOM: RoomAttrs,
    KnowledgeKind.BUDGET: BudgetAttrs,
    KnowledgeKind.PRODUCT_SPEC: ProductSpecAttrs,
    KnowledgeKind.OBSERVATION: ObservationAttrs,
    KnowledgeKind.AGENT_KNOWLEDGE: AgentKnowledgeAttrs,
}

_ID_PREFIX = {
    KnowledgeKind.REQUIREMENTS: "req",
    KnowledgeKind.HOME_PROJECT: "prj",
    KnowledgeKind.ROOM: "room",
    KnowledgeKind.BUDGET: "bud",
    KnowledgeKind.PRODUCT_SPEC: "prd",
    KnowledgeKind.OBSERVATION: "obs",
    KnowledgeKind.AGENT_KNOWLEDGE: "agk",
}


def _validate_attributes(kind: KnowledgeKind, attributes: dict[str, Any]) -> dict[str, Any]:
    """Validate against the kind's model; returns JSON-compatible, normalised attributes."""
    try:
        parsed = ATTRIBUTE_MODELS[kind].model_validate(attributes)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or 'attributes'}: {err['msg']}" for err in exc.errors()
        )
        raise KnowledgeError(f"invalid {kind.value} attributes: {problems}") from None
    return parsed.model_dump(mode="json", exclude_none=True)


# ── The record ───────────────────────────────────────────────────────────────

class KnowledgeRecord(BaseModel):
    """One piece of knowledge, with provenance, confidence, versions and invalidation."""

    model_config = ConfigDict(extra="ignore")  # tolerate repository bookkeeping fields

    knowledge_id: str = ""
    kind: KnowledgeKind
    entity: str
    attributes: dict[str, Any] = Field(default_factory=dict)
    relationships: list[Relationship] = Field(default_factory=list)
    provenance: Provenance
    confidence: float = Field(default=1.0, strict=True)
    version: int = 1
    recorded_at: datetime = Field(default_factory=utc_now)
    modified_at: datetime = Field(default_factory=utc_now)
    valid: bool = True
    invalidated_at: datetime | None = None
    invalidation_reason: str = ""
    history: list[HistoryEntry] = Field(default_factory=list)

    @field_validator("entity")
    @classmethod
    def _entity(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("entity must not be empty")
        return v.strip()

    @field_validator("confidence")
    @classmethod
    def _confidence(cls, v: float) -> float:
        if isinstance(v, bool) or not math.isfinite(v) or not 0.0 <= v <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        return v

    @model_validator(mode="after")
    def _finish(self) -> "KnowledgeRecord":
        if not self.knowledge_id:
            self.knowledge_id = f"{_ID_PREFIX[self.kind]}_{uuid.uuid4().hex[:12]}"
        self.attributes = _validate_attributes(self.kind, self.attributes)
        return self

    # ── Typed view ───────────────────────────────────────────────────────

    def typed(self) -> _Attrs:
        """The attributes as their validated model (e.g. ``RequirementsAttrs``)."""
        return ATTRIBUTE_MODELS[self.kind].model_validate(self.attributes)

    # ── Updates and invalidation ─────────────────────────────────────────

    def update(self, changes: dict[str, Any], provenance: Provenance,
               confidence: float | None = None) -> dict[str, tuple[Any, Any]]:
        """Change attributes (``None`` removes one).  Returns ``{attribute: (old, new)}``.

        A real change bumps the version and keeps the replaced values in
        ``history``.  An update that changes nothing returns ``{}`` and leaves
        the record untouched.  The merged result is re-validated; on failure
        the record is unchanged.
        """
        if not self.valid:
            raise KnowledgeError(f"{self.knowledge_id} was invalidated ({self.invalidation_reason}); "
                                 "record new knowledge instead of updating it")
        merged = {**self.attributes, **changes}
        merged = {k: v for k, v in merged.items() if v is not None}
        validated = _validate_attributes(self.kind, merged)

        diff: dict[str, tuple[Any, Any]] = {}
        for key in set(self.attributes) | set(validated):
            old, new = self.attributes.get(key), validated.get(key)
            if old != new:
                diff[key] = (old, new)
        if not diff:
            return {}

        self.history.append(HistoryEntry(version=self.version, replaced_by=provenance.source,
                                         previous={k: old for k, (old, _) in diff.items()}))
        self.attributes = validated
        self.provenance = provenance
        if confidence is not None:
            self.confidence = KnowledgeRecord._confidence(confidence)
        self.version += 1
        self.modified_at = utc_now()
        return diff

    def invalidate(self, reason: str) -> bool:
        """Mark the knowledge as no longer true.  ``False`` if it already was invalid."""
        if not self.valid:
            return False
        if not reason.strip():
            raise KnowledgeError("an invalidation reason is required")
        self.valid = False
        self.invalidated_at = utc_now()
        self.invalidation_reason = reason.strip()
        self.modified_at = self.invalidated_at
        return True

    def related(self, relation: str | None = None) -> list[Relationship]:
        return [r for r in self.relationships if relation is None or r.relation == relation]

    # ── Serialisation ────────────────────────────────────────────────────

    def to_document(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> "KnowledgeRecord":
        try:
            return cls.model_validate(document)
        except ValidationError as exc:
            raise KnowledgeError(f"stored knowledge record is invalid: {exc.error_count()} error(s)") from None


# ── Builders ─────────────────────────────────────────────────────────────────

def make_record(kind: KnowledgeKind, entity: str, attributes: dict[str, Any], provenance: Provenance,
                confidence: float = 1.0, relationships: list[Relationship] | None = None,
                knowledge_id: str = "") -> KnowledgeRecord:
    """Create and validate a record."""
    try:
        return KnowledgeRecord(knowledge_id=knowledge_id, kind=kind, entity=entity, attributes=attributes,
                               provenance=provenance, confidence=confidence,
                               relationships=relationships or [])
    except ValidationError as exc:
        problems = "; ".join(
            (f"{'.'.join(str(p) for p in e['loc'])}: " if e["loc"] else "") + e["msg"].removeprefix("Value error, ")
            for e in exc.errors()
        )
        raise KnowledgeError(problems) from None
