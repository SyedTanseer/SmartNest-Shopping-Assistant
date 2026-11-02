"""Value types for product discovery."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class ProductQuery:
    """What to look for."""

    category: str = "sofa"
    max_price: float | None = None
    min_seating: int | None = None
    style: str | None = None
    colour: str | None = None
    max_width_cm: float | None = None
    max_depth_cm: float | None = None
    country: str = "India"
    currency: str = "INR"
    limit: int = 8

    def __post_init__(self) -> None:
        if not self.category.strip():
            raise ValueError("category must not be empty")
        if self.limit < 1:
            raise ValueError("limit must be at least 1")

    def describe(self) -> str:
        """The request in plain words, used to prompt the search."""
        words = []
        if self.min_seating:
            words.append(f"{self.min_seating}-seater")
        if self.style:
            words.append(self.style)
        if self.colour:
            words.append(f"{self.colour}-coloured")
        words.append(self.category)
        text = " ".join(words)
        limits = []
        if self.max_price:
            limits.append(f"priced at or under ₹{self.max_price:,.0f}")
        if self.max_width_cm:
            limits.append(f"no wider than {self.max_width_cm:g} cm")
        if self.max_depth_cm:
            limits.append(f"no deeper than {self.max_depth_cm:g} cm")
        if limits:
            text += ", " + ", ".join(limits)
        return f"{text}, sold online in {self.country} (prices in {self.currency}); find up to {self.limit} different products"

    def key(self) -> tuple:
        return (self.category, self.max_price, self.min_seating, self.style, self.colour,
                self.max_width_cm, self.max_depth_cm, self.country)


@dataclass
class RejectedListing:
    """A listing that was found but not accepted, and why."""

    name: str | None
    url: str | None
    reasons: list[str]
    details: dict | None = None   # what is known about the product (price, merchant, missing fields...)


@dataclass
class DiscoveryResult:
    """Everything one discovery run produced."""

    provider_id: str
    query: ProductQuery
    retrieved_at: datetime = field(default_factory=utc_now)
    listings: list[dict] = field(default_factory=list)          # accepted, optimizer-compatible records
    rejected: list[RejectedListing] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)            # resolved source URLs reported by the search
    search_queries: list[str] = field(default_factory=list)
    extracted_count: int = 0
    notes: list[str] = field(default_factory=list)
    api_calls: int = 0              # paid SerpApi requests made by this run (0 when served from cache)
    cache_hits: int = 0             # responses served from the MongoDB search cache
    stale: bool = False             # True if any data came from an expired cache entry (SerpApi failed)

    def summary(self) -> str:
        return (f"{len(self.sources)} source page(s) -> {self.extracted_count} product(s) extracted -> "
                f"{len(self.listings)} accepted, {len(self.rejected)} rejected")


class ProductDiscoveryError(Exception):
    """A provider could not complete a discovery run (API failure, timeout, ...)."""

    def __init__(self, provider_id: str, message: str) -> None:
        super().__init__(f"{provider_id}: {message}")
        self.provider_id = provider_id
