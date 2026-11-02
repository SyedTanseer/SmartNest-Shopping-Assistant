"""Pydantic schemas for Gemini's structured outputs.

Every model output is parsed into one of these and **validated**; an
output that does not validate is rejected (after one re-ask), never
patched up.  The schemas describe what the model may *report*.  They do
not decide anything: hard constraints and utility belong to the
validators and the optimizer, which consume these values later.

Two strictness levels:

    strict   requirements, preferences, change requests, explanations —
             a single object; a bad value invalidates the whole answer.
    lenient  product listings — a list in which one bad entry must not
             discard the others.  Fields are loosely typed here and judged
             per item by the product normalizer.
"""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _text(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    value = value.strip()
    return value or None


def _positive(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite number greater than 0, got {value!r}")
    return value


def _unit_interval(value: float, name: str = "confidence") -> float:
    if isinstance(value, bool) or not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1, got {value!r}")
    return value


# ══ Requirements ═════════════════════════════════════════════════════════════

class RequirementsExtraction(BaseModel):
    """Quantitative requirements stated in a user message (INR and centimetres)."""

    model_config = ConfigDict(extra="ignore")

    category: str | None = Field(default=None, description="What the user wants, e.g. 'sofa'.")
    sofa_budget: float | None = Field(default=None, description="Maximum price for the item in INR.")
    total_budget: float | None = Field(default=None, description="Total budget for the whole project in INR.")
    max_width_cm: float | None = Field(default=None, description="Maximum width in centimetres.")
    max_depth_cm: float | None = Field(default=None, description="Maximum depth in centimetres.")
    min_seating: int | None = Field(default=None, description="Minimum number of seats.")
    preferred_style: str | None = Field(default=None, description="Style the user asked for, e.g. 'minimalist'.")
    preferred_colour: str | None = Field(default=None, description="Colour the user asked for.")
    preferred_materials: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, description="0 to 1: how sure you are about the extraction.")
    assumptions: list[str] = Field(default_factory=list, description="Anything you had to assume.")
    unclear: list[str] = Field(default_factory=list, description="Things the user did not specify clearly.")

    @field_validator("sofa_budget", "total_budget", "max_width_cm", "max_depth_cm")
    @classmethod
    def _numbers(cls, v: float | None, info: Any) -> float | None:
        return _positive(v, info.field_name)

    @field_validator("min_seating")
    @classmethod
    def _seats(cls, v: int | None) -> int | None:
        if v is not None and (isinstance(v, bool) or v < 1):
            raise ValueError(f"min_seating must be at least 1, got {v!r}")
        return v

    @field_validator("category", "preferred_style", "preferred_colour")
    @classmethod
    def _strings(cls, v: str | None) -> str | None:
        v = _text(v)
        return v.lower() if v else None

    @field_validator("preferred_materials")
    @classmethod
    def _materials(cls, v: list[str]) -> list[str]:
        return sorted({m.strip().lower() for m in v if isinstance(m, str) and m.strip()})

    @field_validator("confidence")
    @classmethod
    def _confidence(cls, v: float) -> float:
        return _unit_interval(v)

    def stated_fields(self) -> dict[str, Any]:
        """Only the requirement fields the user actually specified."""
        names = ("sofa_budget", "total_budget", "max_width_cm", "max_depth_cm", "min_seating",
                 "preferred_style", "preferred_colour")
        return {n: getattr(self, n) for n in names if getattr(self, n) is not None}


# ══ Preferences and constraints ══════════════════════════════════════════════

class ExtractedConstraint(BaseModel):
    """A limit the user stated, with the words that justify it."""

    model_config = ConfigDict(extra="ignore")

    field: Literal["price", "width_cm", "depth_cm", "seating", "style", "colour", "material", "brand", "other"]
    comparator: Literal["<=", ">=", "==", "!="]
    value: float | str
    unit: str | None = None
    hard: bool = Field(default=True, description="True if the user insists; False if it is a wish.")
    quote: str | None = Field(default=None, description="The words in the user's message that state this.")


class PreferencesExtraction(BaseModel):
    """Soft preferences and explicit constraints from a user message."""

    model_config = ConfigDict(extra="ignore")

    liked_styles: list[str] = Field(default_factory=list)
    disliked_styles: list[str] = Field(default_factory=list)
    liked_colours: list[str] = Field(default_factory=list)
    disliked_colours: list[str] = Field(default_factory=list)
    preferred_materials: list[str] = Field(default_factory=list)
    avoided_materials: list[str] = Field(default_factory=list)
    must_haves: list[str] = Field(default_factory=list)
    nice_to_haves: list[str] = Field(default_factory=list)
    constraints: list[ExtractedConstraint] = Field(default_factory=list)
    confidence: float = 0.0
    unclear: list[str] = Field(default_factory=list)

    @field_validator("liked_styles", "disliked_styles", "liked_colours", "disliked_colours",
                     "preferred_materials", "avoided_materials")
    @classmethod
    def _terms(cls, v: list[str]) -> list[str]:
        return sorted({t.strip().lower() for t in v if isinstance(t, str) and t.strip()})

    @field_validator("confidence")
    @classmethod
    def _confidence(cls, v: float) -> float:
        return _unit_interval(v)


# ══ Products (lenient, judged per item by the normalizer) ════════════════════

class ProductExtraction(BaseModel):
    """One product as stated in retrieved text.  Unknown values must be null."""

    model_config = ConfigDict(extra="ignore")

    name: str | None = None
    price: float | None = Field(default=None, description="Price as a number, in the currency given.")
    currency: str | None = Field(default=None, description="ISO code, e.g. INR.")
    retailer: str | None = None
    source_url: str | None = Field(default=None, description="Must be copied from the allowed URL list.")
    image_url: str | None = None
    width_cm: float | None = None
    depth_cm: float | None = None
    height_cm: float | None = None
    dimensions_unit: Literal["cm", "mm", "in"] | None = Field(
        default=None, description="Unit of the width/depth/height values as printed on the page.")
    seating_capacity: int | None = None
    material: str | None = None
    colour: str | None = None
    style: str | None = None
    style_inferred: bool = Field(default=False, description="True if style was inferred, not stated.")
    availability: Literal["in_stock", "out_of_stock", "unknown"] | None = None
    rating: float | None = Field(default=None, description="Average customer rating out of 5.")
    review_count: int | None = None


class ProductListExtraction(BaseModel):
    model_config = ConfigDict(extra="ignore")

    products: list[ProductExtraction] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


# ══ Conversational plan changes ══════════════════════════════════════════════

ChangeableField = Literal["sofa_budget", "total_budget", "max_width_cm", "max_depth_cm", "min_seating",
                          "preferred_style", "preferred_colour"]


class RequirementChange(BaseModel):
    model_config = ConfigDict(extra="ignore")

    field: ChangeableField
    action: Literal["set", "clear"]
    value: float | str | None = None

    @field_validator("value")
    @classmethod
    def _value(cls, v: float | str | None) -> float | str | None:
        if isinstance(v, float) and not math.isfinite(v):
            raise ValueError("value must be finite")
        return v.strip() if isinstance(v, str) else v


class PlanChangeRequest(BaseModel):
    """What the user wants to change about the current plan."""

    model_config = ConfigDict(extra="ignore")

    intent: Literal["modify_requirements", "confirm_choice", "reject_choice", "ask_question", "other"]
    changes: list[RequirementChange] = Field(default_factory=list)
    clarification_needed: str | None = None
    confidence: float = 0.0

    @field_validator("confidence")
    @classmethod
    def _confidence(cls, v: float) -> float:
        return _unit_interval(v)

    @field_validator("clarification_needed")
    @classmethod
    def _clarification(cls, v: str | None) -> str | None:
        return _text(v)


# ══ Explanations ═════════════════════════════════════════════════════════════

class RecommendationFacts(BaseModel):
    """Authoritative facts the explanation must stay within.  Computed by the optimizer, not by Gemini."""

    product_id: str
    product_name: str
    price: float
    utility: float                                   # 0-100, computed by the UtilityOptimizer
    component_scores: dict[str, float] = Field(default_factory=dict)
    retailer: str | None = None
    url: str | None = None
    feasible_count: int = 0
    inspected_count: int = 0
    runner_up_name: str | None = None
    runner_up_utility: float | None = None
    budget: float | None = None
    requirements: dict[str, Any] = Field(default_factory=dict)
    rejected_examples: list[str] = Field(default_factory=list)
    fit_status: str | None = None    # "Fit verified" / "Fit not verified" / "No room limits specified"
    availability_status: str | None = None     # "In stock" (a confirmed recommendation is always in stock)
    url_type: str | None = None                # "retailer" | "google_shopping"
    completeness_percent: float | None = None  # share of the applicable scoring criteria that had data
    unavailable_attributes: list[str] = Field(default_factory=list)   # needed for scoring but not stated


class RecommendationExplanation(BaseModel):
    """A short explanation of a recommendation, written for the user."""

    model_config = ConfigDict(extra="ignore")

    text: str
    highlights: list[str] = Field(default_factory=list)
    source: Literal["gemini", "template"] = "gemini"

    @field_validator("text")
    @classmethod
    def _text_present(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("explanation text must not be empty")
        return v
