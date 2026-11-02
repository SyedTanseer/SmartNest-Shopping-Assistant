"""Scoring configuration for the SmartNest utility model (Assignment 3.2).

Everything that shapes a utility score and is not a product or a user
requirement lives here, so it can be changed without touching the
mathematics:

    - Top-level weights of the five utility components.
    - Sub-weights inside the quality and spatial components.
    - The review count at which review volume stops adding quality.
    - Deterministic style-compatibility and colour-family tables.
    - Which availability values count as "confirmed available".

The configuration is a validated, JSON-serialisable Pydantic model.
Invalid configurations raise :class:`ScoringConfigError` at construction,
so a bad weight vector can never reach the optimizer.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

#: Component keys, in the order of the utility formula
#: U = 100 × (w_price·P + w_quality·Q + w_spatial·F + w_style·S + w_colour·C)
COMPONENTS: tuple[str, ...] = ("price", "quality", "spatial", "style", "colour")

#: Tolerance used when checking that a weight vector sums to 1.
WEIGHT_SUM_TOLERANCE = 1e-9


class ScoringConfigError(ValueError):
    """Raised when a scoring configuration is invalid."""


def _check_unit_weights(name: str, weights: dict[str, float], expected: tuple[str, ...]) -> None:
    keys = set(weights)
    if keys != set(expected):
        missing = sorted(set(expected) - keys)
        unknown = sorted(keys - set(expected))
        raise ScoringConfigError(
            f"{name} must have exactly the keys {list(expected)}"
            + (f"; missing {missing}" if missing else "")
            + (f"; unknown {unknown}" if unknown else "")
        )
    for key, value in weights.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ScoringConfigError(f"{name}[{key!r}] must be a number, got {value!r}")
        if not math.isfinite(value):
            raise ScoringConfigError(f"{name}[{key!r}] must be finite, got {value!r}")
        if value < 0:
            raise ScoringConfigError(f"{name}[{key!r}] must be non-negative, got {value!r}")
    total = sum(weights.values())
    if abs(total - 1.0) > WEIGHT_SUM_TOLERANCE:
        raise ScoringConfigError(f"{name} must sum to 1, got {total!r}")


def _normalise_table(name: str, table: dict[str, list[str]]) -> dict[str, list[str]]:
    """Lower-case and de-duplicate a compatibility table."""
    out: dict[str, list[str]] = {}
    for key, values in table.items():
        if not isinstance(key, str) or not key.strip():
            raise ScoringConfigError(f"{name} keys must be non-empty strings, got {key!r}")
        if not isinstance(values, (list, tuple, set, frozenset)):
            raise ScoringConfigError(f"{name}[{key!r}] must be a list of strings")
        cleaned: list[str] = []
        for v in values:
            if not isinstance(v, str) or not v.strip():
                raise ScoringConfigError(f"{name}[{key!r}] contains an invalid entry {v!r}")
            v = v.strip().lower()
            if v not in cleaned:
                cleaned.append(v)
        out[key.strip().lower()] = sorted(cleaned)
    return out


DEFAULT_WEIGHTS: dict[str, float] = {
    "price": 0.30,
    "quality": 0.25,
    "spatial": 0.20,
    "style": 0.15,
    "colour": 0.10,
}

#: preferred style → styles that are recognised as compatible (score 0.5).
DEFAULT_STYLE_COMPATIBILITY: dict[str, list[str]] = {
    "minimalist": ["scandinavian", "contemporary", "modern", "japandi"],
    "scandinavian": ["minimalist", "japandi", "contemporary"],
    "contemporary": ["modern", "minimalist", "scandinavian"],
    "modern": ["contemporary", "minimalist", "mid-century"],
    "mid-century": ["modern", "retro"],
    "traditional": ["classic", "chesterfield", "rustic"],
    "industrial": ["rustic", "modern"],
}

#: preferred colour → colours in the same recognised family (score 0.5).
DEFAULT_COLOUR_FAMILIES: dict[str, list[str]] = {
    "neutral": ["beige", "cream", "grey", "gray", "white", "off-white", "ivory", "taupe", "charcoal"],
    "grey": ["gray", "charcoal", "neutral", "white"],
    "beige": ["cream", "ivory", "taupe", "neutral"],
    "blue": ["navy", "teal", "sky blue"],
    "green": ["olive", "sage", "emerald"],
    "red": ["burgundy", "maroon", "terracotta"],
}


class ScoringConfig(BaseModel):
    """Validated, serialisable configuration for :class:`UtilityModel`.

    Attributes:
        weights: Top-level component weights; keys must be exactly
            :data:`COMPONENTS`, values non-negative and finite, summing to 1.
        quality_rating_weight / quality_review_weight: Q = a·(rating/5) +
            b·min(reviews/saturation, 1); a + b must equal 1.
        review_saturation: Review count at which the review term reaches 1.
        spatial_width_weight / spatial_depth_weight: F = a·(w/W) + b·(d/D);
            a + b must equal 1.
        exact_match_score / compatible_score / incompatible_score: Scores
            used by both the style and colour components (in [0, 1],
            exact ≥ compatible ≥ incompatible).
        style_compatibility: preferred style → compatible styles.
        colour_families: preferred colour → colours in the same family.
        available_states: Availability values that count as confirmed.
    """

    # validate_default: defaults go through the same normalisation as input.
    model_config = ConfigDict(frozen=True, validate_default=True)

    weights: dict[str, float] = Field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    quality_rating_weight: float = 0.80
    quality_review_weight: float = 0.20
    review_saturation: int = 100
    spatial_width_weight: float = 0.50
    spatial_depth_weight: float = 0.50
    exact_match_score: float = 1.0
    compatible_score: float = 0.5
    incompatible_score: float = 0.0
    style_compatibility: dict[str, list[str]] = Field(
        default_factory=lambda: {k: list(v) for k, v in DEFAULT_STYLE_COMPATIBILITY.items()}
    )
    colour_families: dict[str, list[str]] = Field(
        default_factory=lambda: {k: list(v) for k, v in DEFAULT_COLOUR_FAMILIES.items()}
    )
    available_states: list[str] = Field(default_factory=lambda: ["in_stock", "low_stock"])

    # Pydantic would turn our ScoringConfigError into a ValidationError; we
    # validate in __init__ instead so callers get one clear error type.
    def __init__(self, **data: Any) -> None:
        try:
            super().__init__(**data)
        except ScoringConfigError:
            raise
        except ValidationError as exc:
            messages = []
            for err in exc.errors():
                original = (err.get("ctx") or {}).get("error")
                location = ".".join(str(p) for p in err.get("loc", ()))
                text = str(original) if original else err.get("msg", "invalid value")
                messages.append(f"{location}: {text}" if location and not original else text)
            raise ScoringConfigError("; ".join(messages)) from exc

    @field_validator("weights", mode="before")
    @classmethod
    def _raw_weights(cls, value: Any) -> dict[str, float]:
        # Check before pydantic coerces "0.3" → 0.3 or True → 1.0.
        if not isinstance(value, dict):
            raise ScoringConfigError("weights must be a mapping of component → weight")
        _check_unit_weights("weights", value, COMPONENTS)
        return {k: float(v) for k, v in value.items()}

    @field_validator(
        "quality_rating_weight", "quality_review_weight", "spatial_width_weight",
        "spatial_depth_weight", "exact_match_score", "compatible_score",
        "incompatible_score", "review_saturation", mode="before",
    )
    @classmethod
    def _raw_number(cls, value: Any, info: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ScoringConfigError(f"{info.field_name} must be a number, got {value!r}")
        if not math.isfinite(value):
            raise ScoringConfigError(f"{info.field_name} must be finite, got {value!r}")
        return value

    @field_validator("style_compatibility", "colour_families", mode="before")
    @classmethod
    def _tables(cls, value: Any, info: Any) -> dict[str, list[str]]:
        if not isinstance(value, dict):
            raise ScoringConfigError(f"{info.field_name} must be a mapping")
        return _normalise_table(info.field_name, value)

    @field_validator("available_states", mode="before")
    @classmethod
    def _states(cls, value: Any) -> list[str]:
        if not isinstance(value, (list, tuple, set)) or not value:
            raise ScoringConfigError("available_states must be a non-empty list")
        if any(not isinstance(v, str) or not v.strip() for v in value):
            raise ScoringConfigError("available_states entries must be non-empty strings")
        return sorted({v.strip().lower() for v in value})

    @model_validator(mode="after")
    def _validate(self) -> "ScoringConfig":
        _check_unit_weights("weights", dict(self.weights), COMPONENTS)
        _check_unit_weights(
            "quality sub-weights",
            {"rating": self.quality_rating_weight, "reviews": self.quality_review_weight},
            ("rating", "reviews"),
        )
        _check_unit_weights(
            "spatial sub-weights",
            {"width": self.spatial_width_weight, "depth": self.spatial_depth_weight},
            ("width", "depth"),
        )
        if isinstance(self.review_saturation, bool) or self.review_saturation < 1:
            raise ScoringConfigError("review_saturation must be a positive integer")
        scores = (self.exact_match_score, self.compatible_score, self.incompatible_score)
        if any(not math.isfinite(s) or not 0.0 <= s <= 1.0 for s in scores):
            raise ScoringConfigError("match scores must be finite and within [0, 1]")
        if not self.exact_match_score >= self.compatible_score >= self.incompatible_score:
            raise ScoringConfigError("match scores must satisfy exact ≥ compatible ≥ incompatible")
        return self

    def effective_weights(self, spatial: bool = True) -> dict[str, float]:
        """The weights actually used for one product.

        With ``spatial=True`` these are the configured weights.  With ``spatial=False`` (room
        limits not specified, or the product's size unknown) the SpaceFit component is removed and
        the remaining weights are scaled up in proportion so they sum to 1 again; the ``spatial``
        key is absent.  With the default weights that is 0.30/0.80, 0.25/0.80, 0.15/0.80, 0.10/0.80
        = 0.375, 0.3125, 0.1875, 0.125.

        Raises:
            ScoringConfigError: if no other component has any weight to take over.
        """
        if spatial:
            return dict(self.weights)
        return self.weights_for([k for k in COMPONENTS if k != "spatial"])

    def weights_for(self, components: "Iterable[str]") -> dict[str, float]:
        """Weights for exactly the given components, scaled up proportionally so they sum to 1.

        Used when only some components are supported by data: each kept weight is divided by the total of the kept
        weights, so ratios between them are preserved.  Components not listed are absent from the result.

        Raises:
            ScoringConfigError: for an unknown component, an empty selection, or a selection whose configured
                weights are all zero (nothing could take over the excluded weight).
        """
        keep = list(dict.fromkeys(components))
        unknown = [k for k in keep if k not in COMPONENTS]
        if unknown:
            raise ScoringConfigError(f"unknown component(s): {unknown}")
        if len(keep) == len(COMPONENTS) and set(keep) == set(COMPONENTS):
            return dict(self.weights)
        kept = {k: self.weights[k] for k in keep}
        total = sum(kept.values())
        if not kept or not math.isfinite(total) or total <= 0:
            raise ScoringConfigError(
                "the excluded components cannot be redistributed: no remaining component has a weight")
        effective = {k: v / total for k, v in kept.items()}
        if abs(sum(effective.values()) - 1.0) > WEIGHT_SUM_TOLERANCE:  # defensive; cannot happen
            raise ScoringConfigError("redistributed weights do not sum to 1")
        return effective

    def with_weights(self, **weights: float) -> "ScoringConfig":
        """Return a copy with some top-level weights replaced (re-validated)."""
        merged = {**self.weights, **weights}
        return ScoringConfig(**{**self.model_dump(), "weights": merged})

    @staticmethod
    def normalised(weights: dict[str, float]) -> dict[str, float]:
        """Helper: scale non-negative weights so they sum to 1.

        Useful for callers that think in relative importances (e.g. 3:2:2:2:1).
        The result still goes through normal validation when used.
        """
        total = sum(weights.values())
        if not math.isfinite(total) or total <= 0:
            raise ScoringConfigError("weights must have a positive, finite sum to normalise")
        return {k: v / total for k, v in weights.items()}
