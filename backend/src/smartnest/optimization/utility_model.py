"""Explicit mathematical utility model (Assignment 3.2).

    U(p) = 100 × (0.30·P + 0.25·Q + 0.20·F + 0.15·S + 0.10·C)      when every component is supported by data
    U(p) = 100 × (0.375·P + 0.3125·Q + 0.1875·S + 0.125·C)          e.g. when SpaceFit is excluded (below)

with every component normalised to [0, 1]:

    P  price affordability   clamp(1 − price / budget, 0, 1)
    Q  quality               0.80·(rating / 5) + 0.20·min(reviews / 100, 1)
    F  spatial utilisation   0.50·(width / max_width) + 0.50·(depth / max_depth)
    S  style compatibility   1.0 exact · 0.5 recognised-compatible · 0.0 otherwise
    C  colour compatibility  1.0 exact · 0.5 same colour family  · 0.0 otherwise

The weights, sub-weights, review saturation and compatibility tables come
from :class:`ScoringConfig`, so the coefficients above are the defaults,
not constants baked into the code.

Only components supported by actual data are scored (dynamic weights):
    P (price) is always available.  Q needs the product's rating (the review count refines it when stated).
    S and C apply only when the user stated a style / colour preference, and then need the product's style /
    colour; with no stated preference there is nothing to match, so they are not applicable and not scored (a
    constant would add points that say nothing about the product).  F needs a verified fit (below).  A component
    whose data is missing is EXCLUDED - never scored as 0, 0.5 or anything else - and the remaining weights are
    scaled up in proportion so the effective weights still sum to 1.  Each breakdown records which components were
    excluded and why (``excluded``), which attributes were unavailable (``unavailable``) and an information
    completeness indicator: the share of the configured weight of the *applicable* components that was actually
    scored (1.0 = everything that applies was scored).

    Consequence worth knowing: a product scored on few components can rank by price alone.  Completeness is shown
    next to every score, and it breaks exact utility ties, so thinly-evidenced products are never silently equal
    to well-documented ones.

When SpaceFit (F) is excluded:
    F is used only for a product whose fit is *verified* against room limits the user gave
    (width and/or depth).  It is excluded - not scored as 0, 0.5 or anything else - when the
    user gave no room limits, when the product's size needed for the check is not stated, or
    when the only limit is a height limit.  The other weights are then scaled up in proportion
    so the effective weights still sum to 1 (``ScoringConfig.effective_weights``).  The
    ``spatial`` key is then absent from the breakdown, and ``spatial_included`` is False.

Spatial-utilisation assumptions (F):
    - The limits are the free space available for the sofa (the wall
      length and the floor depth in front of it), so a product that uses
      more of that space — without exceeding it — fills the area better
      and scores higher.  A product at exactly the limit scores 1.0.
    - Width and depth matter equally by default (0.5 / 0.5).  If the user gave only one of
      the two limits, only that axis is scored (its sub-weight is rescaled to 1).
    - The score is only defined for products within the limits; hard
      constraints are checked first, and calling the spatial scorer with
      an oversized product raises an error instead of returning > 1.
    - Leaving clearance for walkways is a user decision: the user should
      enter the clearance-adjusted space as the limit.

The model is pure arithmetic: no I/O, no randomness, no network, no LLM,
so identical inputs always give identical outputs.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict

from smartnest.optimization.constraint_validator import ProductCandidate, UserRequirements, assess_space
from smartnest.optimization.scoring_config import COMPONENTS, ScoringConfig

#: Decimal places kept in scores; also the resolution at which ties are detected.
SCORE_PRECISION = 6

#: Short symbols used in reports, keyed by component name.
SYMBOLS: dict[str, str] = {
    "price": "P", "quality": "Q", "spatial": "F", "style": "S", "colour": "C",
}


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    if math.isnan(value):
        raise ValueError("cannot clamp NaN")
    return max(low, min(high, value))


class UtilityBreakdown(BaseModel):
    """Full, serialisable explanation of one product's utility.

    Attributes:
        components: Normalised component scores in [0, 1].  Has no ``spatial`` key when
            SpaceFit was excluded.
        weights: The *effective* weights used (they sum to 1; ``spatial`` absent when excluded).
        contributions: Points each component adds to U (100 × w × score).
        total_utility: U on the 0–100 scale.
        spatial_included: Whether SpaceFit counted towards U.
        completeness: Information completeness in [0, 1]: configured weight of the scored components divided by the
            configured weight of the applicable ones.
        excluded: Components that were not scored, with the reason (e.g. ``{"quality": "rating not stated"}``).
        unavailable: Attributes that were needed but not stated (e.g. ``["rating", "style"]``).
    """

    model_config = ConfigDict(frozen=True)

    components: dict[str, float]
    weights: dict[str, float]
    contributions: dict[str, float]
    total_utility: float
    spatial_included: bool = True
    completeness: float = 1.0
    excluded: dict[str, str] = {}
    unavailable: list[str] = []

    @property
    def completeness_note(self) -> str:
        """One sentence on how much of the score rests on stated data, naming what was unavailable."""
        scored = ", ".join(SYMBOLS[k] + "=" + k for k in self.components)
        text = f"Scored on {self.completeness:.0%} of the applicable criteria ({scored})."
        if self.unavailable:
            text += f" Not available, so not scored: {', '.join(self.unavailable)}."
        return text

    @property
    def normalised_utility(self) -> float:
        """U / 100, in [0, 1]."""
        return round(self.total_utility / 100.0, SCORE_PRECISION)


class UtilityModel:
    """Computes the normalised components and total utility of a product.

    Args:
        config: Scoring configuration (validated).  Defaults to
            :class:`ScoringConfig` with the assignment's weights.
    """

    def __init__(self, config: ScoringConfig | None = None) -> None:
        self._config = config or ScoringConfig()

    @property
    def config(self) -> ScoringConfig:
        return self._config

    @property
    def weights(self) -> dict[str, float]:
        return dict(self._config.weights)

    # ── Components ───────────────────────────────────────────────────────

    def price_score(self, price: float, budget: float) -> float:
        """P = clamp(1 − price / budget, 0, 1)."""
        if not (math.isfinite(budget) and budget > 0):
            raise ValueError(f"budget must be finite and > 0, got {budget!r}")
        if not math.isfinite(price):
            raise ValueError(f"price must be finite, got {price!r}")
        return clamp(1.0 - price / budget)

    def quality_score_available(self, rating: float, review_count: int | None) -> float:
        """Q from what is stated: with a review count it is the usual Q; without one, the rating alone.

        (The rating sub-weight is rescaled to 1 - a missing review count is not scored as zero reviews.)
        """
        if review_count is not None:
            return self.quality_score(rating, review_count)
        if not (math.isfinite(rating) and 0.0 <= rating <= 5.0):
            raise ValueError(f"rating must be within [0, 5], got {rating!r}")
        return clamp(rating / 5.0)

    def quality_score(self, rating: float, review_count: int) -> float:
        """Q = a·(rating / 5) + b·min(reviews / saturation, 1)."""
        if not (math.isfinite(rating) and 0.0 <= rating <= 5.0):
            raise ValueError(f"rating must be within [0, 5], got {rating!r}")
        if review_count < 0:
            raise ValueError(f"review_count must be non-negative, got {review_count!r}")
        cfg = self._config
        review_term = min(review_count / cfg.review_saturation, 1.0)
        return clamp(cfg.quality_rating_weight * (rating / 5.0) + cfg.quality_review_weight * review_term)

    def spatial_score(self, width: float, depth: float, max_width: float, max_depth: float) -> float:
        """F = a·(width / max_width) + b·(depth / max_depth), within the limits only."""
        for name, v in (("max_width", max_width), ("max_depth", max_depth)):
            if not (math.isfinite(v) and v > 0):
                raise ValueError(f"{name} must be finite and > 0, got {v!r}")
        for name, v in (("width", width), ("depth", depth)):
            if not (math.isfinite(v) and v > 0):
                raise ValueError(f"{name} must be finite and > 0, got {v!r}")
        if width > max_width or depth > max_depth:
            raise ValueError(
                "spatial score is only defined for products within the room limits "
                f"({width}×{depth} cm vs {max_width}×{max_depth} cm)"
            )
        cfg = self._config
        return clamp(cfg.spatial_width_weight * (width / max_width)
                     + cfg.spatial_depth_weight * (depth / max_depth))

    def spatial_fit_score(self, product: ProductCandidate, req: UserRequirements) -> float | None:
        """F for a product whose fit is verified, using only the limits the user gave.

        Returns ``None`` (no score at all) when F is not defined: no width/depth limit was given,
        or a size needed for a given limit is not stated.  Never invents a value.
        """
        axes = []
        if req.max_width_cm is not None:
            axes.append((self._config.spatial_width_weight, product.width_cm, req.max_width_cm, "width"))
        if req.max_depth_cm is not None:
            axes.append((self._config.spatial_depth_weight, product.depth_cm, req.max_depth_cm, "depth"))
        if not axes or any(size is None for _, size, _, _ in axes):
            return None
        for _, size, limit, name in axes:
            if not (math.isfinite(limit) and limit > 0):
                raise ValueError(f"max_{name} must be finite and > 0, got {limit!r}")
            if not (math.isfinite(size) and size > 0):
                raise ValueError(f"{name} must be finite and > 0, got {size!r}")
            if size > limit:
                raise ValueError("spatial score is only defined for products within the room limits "
                                 f"({name} {size} cm vs limit {limit} cm)")
        total_weight = sum(w for w, *_ in axes)
        if total_weight <= 0:
            return None
        return clamp(sum(w * (size / limit) for w, size, limit, _ in axes) / total_weight)

    def _match_score(self, value: str, preferred: str | None, table: dict[str, list[str]]) -> float:
        cfg = self._config
        if preferred is None:
            return cfg.exact_match_score  # no preference → nothing to penalise
        value = value.strip().lower()
        if not value:
            raise ValueError("value must be a non-empty string")
        if value == preferred:
            return cfg.exact_match_score
        if value in table.get(preferred, []):
            return cfg.compatible_score
        return cfg.incompatible_score

    def style_score(self, style: str, preferred_style: str | None) -> float:
        """S via the configured style-compatibility table."""
        return self._match_score(style, preferred_style, self._config.style_compatibility)

    def colour_score(self, colour: str, preferred_colour: str | None) -> float:
        """C via the configured colour-family table."""
        return self._match_score(colour, preferred_colour, self._config.colour_families)

    # ── Total ─────────────────────────────────────────────────────────────

    def score(self, product: ProductCandidate, req: UserRequirements) -> UtilityBreakdown:
        """Compute U from the components the product's stated data supports, and return the breakdown.

        Intended for a product that has no known constraint violation.  Components whose data is missing are
        excluded (not scored as 0) and their weight is redistributed; see the module docstring.
        """
        assessment = assess_space(product, req)
        if assessment.status == "incompatible":
            raise ValueError("spatial score is only defined for products within the room limits: "
                             + "; ".join(assessment.violations))

        components: dict[str, float] = {"price": self.price_score(product.price, req.budget)}
        excluded: dict[str, str] = {}
        unavailable: list[str] = []
        applicable = ["price", "quality"]

        # Quality: needs a rating.
        if product.rating is not None:
            components["quality"] = self.quality_score_available(product.rating, product.review_count)
        else:
            excluded["quality"] = "rating not stated"
            unavailable.append("rating")
        if product.review_count is None:
            unavailable.append("review count")

        # Spatial fit: applicable only if the user gave a width or depth limit; scored only if verified.
        if req.max_width_cm is not None or req.max_depth_cm is not None:
            applicable.append("spatial")
            spatial = self.spatial_fit_score(product, req) if assessment.status == "verified" else None
            if spatial is not None:
                components["spatial"] = spatial
            else:
                axes = [a for a in assessment.unknown_axes if a in ("width", "depth")]
                excluded["spatial"] = "size not stated: " + ", ".join(axes) if axes else "fit not verified"
                unavailable.extend(axes)
        else:
            excluded["spatial"] = "no width or depth limit given"

        # Style and colour: applicable only with a stated preference, and then they need the product's value.
        for key, value, preferred, scorer in (("style", product.style, req.preferred_style, self.style_score),
                                              ("colour", product.colour, req.preferred_colour, self.colour_score)):
            if preferred is None:
                excluded[key] = f"no {key} preference given"   # not applicable: nothing to match against
                continue
            applicable.append(key)
            if value is None:
                excluded[key] = f"{key} not stated"
                unavailable.append(key)
            else:
                components[key] = scorer(value, preferred)

        keys = [k for k in COMPONENTS if k in components]
        weights = self._config.weights_for(keys)
        contributions = {k: 100.0 * weights[k] * components[k] for k in keys}
        total = sum(contributions.values())
        if not math.isfinite(total) or not -1e-9 <= total <= 100.0 + 1e-9:
            raise ArithmeticError(f"utility out of range: {total!r}")  # defensive; should be impossible

        configured = self._config.weights
        applicable_weight = sum(configured[k] for k in applicable)
        scored_weight = sum(configured[k] for k in applicable if k in components)
        completeness = scored_weight / applicable_weight if applicable_weight > 0 else 1.0
        return UtilityBreakdown(
            components={k: round(components[k], SCORE_PRECISION) for k in keys},
            weights={k: weights[k] for k in keys},
            contributions={k: round(contributions[k], SCORE_PRECISION) for k in keys},
            total_utility=round(clamp(total, 0.0, 100.0), SCORE_PRECISION),
            spatial_included="spatial" in components,
            completeness=round(clamp(completeness), SCORE_PRECISION),
            excluded=excluded,
            unavailable=list(dict.fromkeys(unavailable)),
        )
