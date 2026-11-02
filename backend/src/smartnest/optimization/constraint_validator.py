"""Record validation and hard-constraint checking (Assignment 3.2).

Two separate questions are answered for every candidate, in this order:

1. **Is the record valid?**  Every field needed to compute utility must be
   present, of the right type, finite and within its domain.  A missing
   or malformed field produces a controlled *validation failure*
   (status ``"invalid"``) — it is never silently replaced by a default.

2. **Does it satisfy the hard constraints?**  Budget, seating, confirmed
   availability and - *only when the user gave them* - the room limits
   (width, depth, height).  Every violated constraint is recorded as an
   explicit reason (status ``"rejected"``).  Boundaries are inclusive: a
   price equal to the budget or a width equal to the room limit is allowed.

Room limits are optional, and so are a product's width, depth and height
(they are never invented).  Each product gets a *spatial assessment*:

    not_required  the user gave no room limits, so fit is not a question
    verified      every limit the user gave was checked against a stated size and holds
    unknown       at least one needed size is missing and no stated size breaks a limit;
                  the product is NOT incompatible, and it is NOT claimed to fit
    incompatible  a stated size exceeds a limit the user gave (a hard-constraint violation)

A product whose fit is ``unknown`` gets status ``"unverified"``: it can be
scored, but it is kept apart from fit-verified products and is never
selected as the recommendation.

Optional attributes and availability
------------------------------------
Only the product's id, name, a positive price and source are mandatory.  Rating, review count, style, colour,
material, seating, sizes and availability may be missing; a missing value stays ``None`` (never defaulted).  Missing
attributes are not penalised here: the utility model simply does not score them (see ``utility_model``).

Availability has three explicit states: ``in_stock``, ``out_of_stock`` and ``unknown``.  Confirmed out of stock is
rejected.  Unknown is neither rejected nor treated as in stock: like an unverifiable fit, it makes the product
``"unverified"`` - scored and listed, but in a separate category and never presented as a confirmed-available
recommendation.

Only candidates that pass the checks are ``"eligible"`` for utility scoring.
This module never computes utility, so no score can override a hard
constraint.

The models here are plain, JSON-serialisable Pydantic models with no
dependency on sockets, CNP, databases or LLMs.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Literal

from pydantic import BaseModel, ConfigDict

from smartnest.optimization.scoring_config import ScoringConfig


class InvalidRequirementsError(ValueError):
    """Raised when user requirements are malformed (caller error)."""


# ── Numeric helpers ──────────────────────────────────────────────────────────

def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _numeric_error(name: str, value: Any, *, positive: bool = False,
                   non_negative: bool = False, integer: bool = False) -> str:
    """Describe why ``value`` is not an acceptable number, or return ``""``."""
    if value is None:
        return f"{name} is missing"
    if not _is_number(value):
        return f"{name} must be a number, got {value!r}"
    if not math.isfinite(value):
        return f"{name} must be finite, got {value!r}"
    if integer and float(value) != int(value):
        return f"{name} must be a whole number, got {value!r}"
    if positive and value <= 0:
        return f"{name} must be greater than 0, got {value!r}"
    if non_negative and value < 0:
        return f"{name} must not be negative, got {value!r}"
    return ""


def _clean_text(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip().lower()
    return None


# ── User requirements ────────────────────────────────────────────────────────

class UserRequirements(BaseModel):
    """What the user asked for.  Validated on construction.

    ``preferred_style`` / ``preferred_colour`` may be ``None`` meaning
    "no preference" (every product then scores the exact-match value for
    that component).  All numeric limits must be finite and positive —
    in particular a zero budget is rejected, because affordability
    (1 − price/budget) would divide by zero.

    The room limits ``max_width_cm`` / ``max_depth_cm`` / ``max_height_cm`` are
    optional (``None`` = the user did not give one).  A limit that is given must
    be finite and positive.  Dimensional constraints are enforced only for the
    limits given.
    """

    model_config = ConfigDict(frozen=True)

    budget: float
    max_width_cm: float | None = None
    max_depth_cm: float | None = None
    max_height_cm: float | None = None
    min_seating: int = 1
    preferred_style: str | None = None
    preferred_colour: str | None = None

    def __init__(self, **data: Any) -> None:
        errors = [
            _numeric_error("budget", data.get("budget"), positive=True),
            *(_numeric_error(key, data.get(key), positive=True)
              for key in ("max_width_cm", "max_depth_cm", "max_height_cm") if data.get(key) is not None),
            _numeric_error("min_seating", data.get("min_seating", 1), positive=True, integer=True),
        ]
        for key in ("preferred_style", "preferred_colour"):
            value = data.get(key)
            if value is not None and not isinstance(value, str):
                errors.append(f"{key} must be a string or None, got {value!r}")
        errors = [e for e in errors if e]
        if errors:
            raise InvalidRequirementsError("; ".join(errors))
        data["min_seating"] = int(data.get("min_seating", 1))
        data["preferred_style"] = _clean_text(data.get("preferred_style"))
        data["preferred_colour"] = _clean_text(data.get("preferred_colour"))
        super().__init__(**data)

    @property
    def has_spatial_limits(self) -> bool:
        """Whether the user gave at least one room limit."""
        return any(v is not None for v in (self.max_width_cm, self.max_depth_cm, self.max_height_cm))

    @classmethod
    def from_cnp(cls, requirements: dict[str, Any], *, require_room_limits: bool = True) -> "UserRequirements":
        """Build from a CNP coordinator/bidder requirements dict.

        Accepts the 3.1 key names (``max_budget``, ``room_width`` /
        ``max_width``, ``room_depth`` / ``max_depth``, ``seating_capacity``,
        ``preferred_style``, ``preferred_colour`` / ``preferred_color``).

        CNP negotiations disclose the room limits progressively, and a bidder and the
        coordinator must score under the same weights or honest bids would be flagged as
        discrepancies.  So by default width and depth are still required here, exactly as in
        Assignment 3.2; pass ``require_room_limits=False`` to accept a dict without them.
        """
        def first(*keys: str) -> Any:
            for k in keys:
                if requirements.get(k) is not None:
                    return requirements[k]
            return None

        if require_room_limits:
            missing = [f"{name} is missing" for name, keys in (("max_width_cm", ("max_width_cm", "max_width", "room_width")),
                                                               ("max_depth_cm", ("max_depth_cm", "max_depth", "room_depth")))
                       if first(*keys) is None]
            if missing:
                raise InvalidRequirementsError("; ".join(missing))
        return cls(
            budget=first("budget", "max_budget"),
            max_width_cm=first("max_width_cm", "max_width", "room_width"),
            max_depth_cm=first("max_depth_cm", "max_depth", "room_depth"),
            max_height_cm=first("max_height_cm", "max_height", "room_height"),
            min_seating=first("min_seating", "seating_capacity") or 1,
            preferred_style=first("preferred_style") or None,
            preferred_colour=first("preferred_colour", "preferred_color") or None,
        )


# ── Product candidate ────────────────────────────────────────────────────────

Availability = Literal["in_stock", "out_of_stock", "unknown"]

#: How each availability state is shown to people.
AVAILABILITY_LABELS: dict[str, str] = {
    "in_stock": "In stock",
    "out_of_stock": "Out of stock",
    "unknown": "Availability unknown - check stock on the retailer's listing",
}

#: Availability values that mean "confirmed out of stock" (anything else not confirmed in stock is unknown).
OUT_OF_STOCK_VALUES: frozenset[str] = frozenset({"out_of_stock", "sold_out", "unavailable", "discontinued"})


class ProductCandidate(BaseModel):
    """A validated product record, normalised for scoring.

    Only ``product_id``, ``name``, ``price`` and ``source_agent_id`` are mandatory.  Every other attribute may
    be ``None`` = *not stated* (never guessed, never defaulted): sizes, seating, material, colour, style, rating,
    review count.  ``availability`` is always one of three explicit states once validated - a record that does
    not state it is ``"unknown"``, which is not the same as in stock.
    """

    model_config = ConfigDict(frozen=True)

    product_id: str
    name: str
    price: float
    width_cm: float | None = None
    depth_cm: float | None = None
    height_cm: float | None = None
    seating_capacity: int | None = None
    material: str | None = None
    colour: str | None = None
    style: str | None = None
    availability: str = "unknown"
    rating: float | None = None
    review_count: int | None = None
    source_agent_id: str

    def with_price(self, price: float) -> "ProductCandidate":
        """Copy with a different (e.g. negotiated) price."""
        return self.model_copy(update={"price": float(price)})


# Required text fields: record key(s) -> candidate field.
_REQUIRED_TEXT: dict[str, tuple[str, ...]] = {
    "name": ("name",),
    "source_agent_id": ("source_agent_id",),
}
# Optional text fields (required again when ``require_complete`` is set).
_OPTIONAL_TEXT: dict[str, tuple[str, ...]] = {
    "material": ("material",),
    "colour": ("colour", "color"),
    "style": ("style",),
}


def validate_record(record: Any, *, require_complete: bool = False) -> tuple[ProductCandidate | None, list[str]]:
    """Validate a raw product record.

    Mandatory: ``product_id``, ``name``, a positive ``price`` and ``source_agent_id``.  Everything else is optional
    and stays ``None`` when absent; a value that *is* present must be valid (a rating of 9 or a width of 0 is an
    error, not "missing").  ``availability`` absent means ``"unknown"``.

    Args:
        require_complete: Legacy strict mode (used by CNP negotiations, whose participants exchange complete
            records): material, colour, style, seating, rating, review count and availability must all be stated.

    Returns:
        ``(candidate, [])`` when valid, ``(None, errors)`` otherwise.
        ``errors`` lists every problem found, not just the first.
    """
    if isinstance(record, ProductCandidate):
        return record, []
    if not isinstance(record, dict):
        return None, [f"record must be a mapping, got {type(record).__name__}"]

    errors: list[str] = []
    values: dict[str, Any] = {}

    pid = record.get("product_id")
    if not isinstance(pid, str) or not pid.strip():
        errors.append(f"product_id is missing or invalid ({pid!r})")
    else:
        values["product_id"] = pid.strip()

    def text(field_name: str, keys: tuple[str, ...], required: bool) -> None:
        raw = next((record[k] for k in keys if record.get(k) is not None), None)
        if raw is None:
            if required:
                errors.append(f"{field_name} is missing")
            else:
                values[field_name] = None
        elif not isinstance(raw, str) or not raw.strip():
            errors.append(f"{field_name} must be a non-empty string, got {raw!r}")
        else:
            # Names and IDs keep their case; matching fields are normalised.
            values[field_name] = raw.strip() if field_name in ("name", "source_agent_id") else raw.strip().lower()

    for field_name, keys in _REQUIRED_TEXT.items():
        text(field_name, keys, True)
    for field_name, keys in _OPTIONAL_TEXT.items():
        text(field_name, keys, require_complete)
    text("availability", ("availability",), require_complete)
    if values.get("availability") is None and not any(e.startswith("availability") for e in errors):
        values["availability"] = "unknown"  # not stated: an explicit state, never "in stock"

    err = _numeric_error("price", record.get("price"), positive=True)
    if err:
        errors.append(err)
    else:
        values["price"] = record["price"]

    optional_numbers = {
        "seating_capacity": dict(positive=True, integer=True),
        "rating": dict(non_negative=True),
        "review_count": dict(non_negative=True, integer=True),
        "width_cm": dict(positive=True),
        "depth_cm": dict(positive=True),
        "height_cm": dict(positive=True),
    }
    always_optional = {"width_cm", "depth_cm", "height_cm"}  # optional even in complete mode (see UserRequirements)
    for key, rule in optional_numbers.items():
        if record.get(key) is None and (key in always_optional or not require_complete):
            values[key] = None
            continue
        err = _numeric_error(key, record.get(key), **rule)
        if err:
            errors.append(err)
        else:
            values[key] = float(record[key]) if key.endswith("_cm") else record[key]

    if values.get("rating") is not None and values["rating"] > 5:
        errors.append(f"rating must be between 0 and 5, got {values['rating']!r}")

    if errors:
        return None, errors

    for key in ("seating_capacity", "review_count"):
        if values.get(key) is not None:
            values[key] = int(values[key])
    return ProductCandidate(**values), []


# ── Eligibility ──────────────────────────────────────────────────────────────

Status = Literal["eligible", "unverified", "rejected", "invalid"]
SpatialStatus = Literal["verified", "unknown", "incompatible", "not_required"]

#: How each spatial status is shown to people.
FIT_LABELS: dict[str, str] = {
    "verified": "Fit verified",
    "unknown": "Fit not verified",
    "incompatible": "Dimensionally incompatible",
    "not_required": "No room limits specified",
}


class SpatialAssessment(BaseModel):
    """Whether a product fits the room limits the user gave (see the module docstring)."""

    model_config = ConfigDict(frozen=True)

    status: SpatialStatus
    violations: list[str] = []     # stated sizes that exceed a limit
    verified_axes: list[str] = []  # limits checked against a stated size and satisfied
    unknown_axes: list[str] = []   # limits that could not be checked: the product's size is not stated
    notes: list[str] = []          # human-readable explanation of the unknown axes

    @property
    def label(self) -> str:
        return FIT_LABELS[self.status]


class EligibilityResult(BaseModel):
    """Outcome of validating one record and checking its hard constraints."""

    product_id: str
    source_agent_id: str | None = None
    status: Status
    reasons: list[str] = []
    candidate: ProductCandidate | None = None
    spatial: SpatialAssessment | None = None
    availability: Availability | None = None
    gaps: list[str] = []   # what could not be verified for an "unverified" product: "fit", "availability", "seating"

    @property
    def eligible(self) -> bool:
        return self.status == "eligible"


def assess_space(product: ProductCandidate, req: UserRequirements) -> SpatialAssessment:
    """Compare the product's stated sizes with the room limits the user gave.

    Only limits the user gave are checked.  A missing product size is "unknown", never a
    violation and never a pass; one exceeded limit makes the product incompatible even if
    another size is unknown.
    """
    if not req.has_spatial_limits:
        return SpatialAssessment(status="not_required")
    violations: list[str] = []
    verified: list[str] = []
    unknown: list[str] = []
    notes: list[str] = []
    for axis, size, limit in (("width", product.width_cm, req.max_width_cm),
                              ("depth", product.depth_cm, req.max_depth_cm),
                              ("height", product.height_cm, req.max_height_cm)):
        if limit is None:
            continue
        if size is None:
            unknown.append(axis)
            notes.append(f"{axis} is not stated, so the {limit:g} cm limit cannot be checked")
        elif size > limit:
            violations.append(f"{axis} {size:g} cm exceeds maximum {axis} {limit:g} cm by {size - limit:g} cm")
        else:
            verified.append(axis)
    status = "incompatible" if violations else ("unknown" if unknown else "verified")
    return SpatialAssessment(status=status, violations=violations, verified_axes=verified,
                             unknown_axes=unknown, notes=notes)


class ConstraintValidator:
    """Checks records against the mandatory (hard) constraints.

    Args:
        config: Supplies ``available_states`` (which availability values
            count as confirmed in stock).  Defaults to :class:`ScoringConfig`.
    """

    def __init__(self, config: ScoringConfig | None = None) -> None:
        self._available = frozenset((config or ScoringConfig()).available_states)

    def availability_state(self, product: ProductCandidate) -> Availability:
        """``in_stock`` (a configured available state), ``out_of_stock`` (confirmed), or ``unknown`` (anything else)."""
        value = (product.availability or "unknown").strip().lower()
        if value in self._available:
            return "in_stock"
        if value in OUT_OF_STOCK_VALUES:
            return "out_of_stock"
        return "unknown"

    def check_constraints(self, product: ProductCandidate, req: UserRequirements) -> list[str]:
        """Return every hard constraint ``product`` violates (empty if none).

        Only *known* violations count: a missing room size, seating capacity or availability is not a violation
        (see :meth:`verification_gaps`).  A product confirmed out of stock is rejected.
        """
        reasons: list[str] = []
        if product.price > req.budget:
            reasons.append(
                f"price ₹{product.price:,.0f} exceeds budget ₹{req.budget:,.0f} "
                f"by ₹{product.price - req.budget:,.0f}"
            )
        reasons.extend(assess_space(product, req).violations)
        if product.seating_capacity is not None and product.seating_capacity < req.min_seating:
            reasons.append(
                f"seating capacity {product.seating_capacity} is below the minimum {req.min_seating}"
            )
        if self.availability_state(product) == "out_of_stock":
            reasons.append(f"confirmed out of stock (availability: {product.availability})")
        return reasons

    def verification_gaps(self, product: ProductCandidate, req: UserRequirements) -> tuple[list[str], list[str]]:
        """What cannot be verified: ``(gaps, notes)``, e.g. ``(["fit", "availability"], [...])``."""
        gaps: list[str] = []
        notes: list[str] = []
        spatial = assess_space(product, req)
        if spatial.status == "unknown":
            gaps.append("fit")
            notes.extend(spatial.notes)
        if product.seating_capacity is None and req.min_seating > 1:
            gaps.append("seating")
            notes.append(f"seating capacity is not stated, so the minimum of {req.min_seating} seats cannot be checked")
        if self.availability_state(product) == "unknown":
            gaps.append("availability")
            notes.append("availability is not confirmed - check stock on the retailer's listing before buying")
        return gaps, notes

    def evaluate(self, record: Any, req: UserRequirements, *, require_complete: bool = False) -> EligibilityResult:
        """Validate ``record`` and check its hard constraints.

        Status: ``invalid`` (malformed), ``rejected`` (a known violation, incl. confirmed out of stock),
        ``unverified`` (nothing violated, but fit, seating or stock could not be verified) or ``eligible``.
        """
        candidate, errors = validate_record(record, require_complete=require_complete)
        if candidate is None:
            raw = record if isinstance(record, dict) else {}
            pid = raw.get("product_id")
            return EligibilityResult(
                product_id=pid if isinstance(pid, str) and pid.strip() else "<unknown>",
                source_agent_id=raw.get("source_agent_id") if isinstance(raw.get("source_agent_id"), str) else None,
                status="invalid",
                reasons=[f"invalid record: {e}" for e in errors],
            )
        violations = self.check_constraints(candidate, req)
        spatial = assess_space(candidate, req)
        gaps, notes = ([], []) if violations else self.verification_gaps(candidate, req)
        if violations:
            status: Status = "rejected"
        elif gaps:
            status = "unverified"  # satisfies everything that could be checked; something is not verified
        else:
            status = "eligible"
        return EligibilityResult(
            product_id=candidate.product_id,
            source_agent_id=candidate.source_agent_id,
            status=status,
            reasons=notes if status == "unverified" else violations,
            candidate=candidate,
            spatial=spatial,
            availability=self.availability_state(candidate),
            gaps=gaps,
        )

    def evaluate_all(self, records: Iterable[Any], req: UserRequirements) -> list[EligibilityResult]:
        return [self.evaluate(r, req) for r in records]
