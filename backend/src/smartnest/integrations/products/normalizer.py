"""Turning extracted listings into optimizer-compatible product records.

The extraction step is untrusted: it is a language model reading text.
This module is where a listing either becomes a trustworthy record or is
**rejected with reasons** — nothing is defaulted or guessed:

    - the price must be stated, positive, in INR, and (when the retrieved
      text is available) actually appear in that text;
    - the product URL must be one of the URLs the search really reported;
    - width and depth must be stated with a unit and be physically plausible;
    - seats, material, colour, style, rating and review count must be present
      (the optimizer needs them — see ``smartnest.optimization.validate_record``);
    - the finished record must pass the optimizer's own record validation.

Availability is the one field that may be "unknown": the optimizer then
rejects the product itself with a clear reason, which is more informative
than dropping it here.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

from smartnest.integrations.gemini.schemas import ProductExtraction
from smartnest.integrations.products.models import RejectedListing
from smartnest.integrations.products.url_validation import (
    AllowedUrls,
    check_url,
    domain_of,
    url_fingerprint,
)
from smartnest.optimization import validate_record

_UNIT_TO_CM = {"cm": 1.0, "mm": 0.1, "in": 2.54}
_CURRENCY_ALIASES = {"inr": "INR", "₹": "INR", "rs": "INR", "rs.": "INR", "rupee": "INR", "rupees": "INR"}

#: Plausible finished sizes (cm) per category; anything outside is a data error, not a product.
PLAUSIBLE_CM = {
    "sofa": {"width": (40.0, 500.0), "depth": (30.0, 300.0)},
}
_DEFAULT_BOUNDS = {"width": (5.0, 1000.0), "depth": (5.0, 1000.0)}


@dataclass
class NormalizedListing:
    record: dict
    warnings: list[str] = field(default_factory=list)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _clean(value: str | None) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def price_in_text(price: float, text: str) -> bool:
    """Whether ``price`` appears in ``text`` as printed (``24999``, ``24,999``, ``24 999``, ``24999.00``)."""
    whole = int(round(price))
    plain = str(whole)
    grouped = f"{whole:,}"
    indian = _indian_grouping(whole)
    spaced = grouped.replace(",", " ")
    for form in {plain, grouped, indian, spaced}:
        # Not part of a longer number: no digit (or "digit," / "digit.") directly before, and no digit
        # (or ",digit") directly after.  Trailing sentence punctuation such as "24,999," is fine.
        if re.search(rf"(?<!\d)(?<!\d,)(?<!\d\.){re.escape(form)}(?:\.\d+)?(?!\d)(?!,\d)", text):
            return True
    return False


def _indian_grouping(n: int) -> str:
    s = str(n)
    if len(s) <= 3:
        return s
    head, tail = s[:-3], s[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    return ",".join(groups + [tail])


def normalize_listing(
    item: ProductExtraction,
    *,
    allowed: AllowedUrls,
    provider_id: str,
    retrieved_at: datetime,
    category: str = "sofa",
    source_text: str | None = None,
    image_ok: Callable[[str], bool] | None = None,
) -> NormalizedListing | RejectedListing:
    """Validate one extracted listing.  Returns the record, or a rejection listing every problem found."""
    reasons: list[str] = []
    warnings: list[str] = []

    name = _clean(item.name)
    if name is None:
        reasons.append("name is missing")

    # Price ------------------------------------------------------------------
    price = item.price
    if not _is_number(price) or price <= 0:
        reasons.append("price is missing or not a positive number")
        price = None
    currency = _CURRENCY_ALIASES.get((item.currency or "").strip().lower(), (item.currency or "").strip().upper())
    if not currency:
        reasons.append("currency is not stated")
    elif currency != "INR":
        reasons.append(f"price is in {currency}, not INR")
    if price is not None and source_text is not None and not price_in_text(price, source_text):
        reasons.append(f"price {price:g} does not appear in the retrieved text")

    # URL --------------------------------------------------------------------
    url = None
    raw_url = _clean(item.source_url)
    if raw_url is None:
        reasons.append("source URL is missing")
    else:
        problem = check_url(raw_url)
        if problem:
            reasons.append(f"source URL rejected: {problem}")
        else:
            url = allowed.lookup(raw_url)
            if url is None:
                reasons.append("source URL was not among the sources the search reported")

    # Dimensions ---------------------------------------------------------------
    bounds = PLAUSIBLE_CM.get(category.lower(), _DEFAULT_BOUNDS)
    width = depth = None
    if item.width_cm is None or item.depth_cm is None:
        reasons.append("width and depth must both be stated")
    elif item.dimensions_unit is None:
        reasons.append("the unit of the dimensions is not stated")
    elif not _is_number(item.width_cm) or not _is_number(item.depth_cm) or item.width_cm <= 0 or item.depth_cm <= 0:
        reasons.append("width and depth must be positive numbers")
    else:
        factor = _UNIT_TO_CM[item.dimensions_unit]
        width, depth = round(item.width_cm * factor, 2), round(item.depth_cm * factor, 2)
        if not bounds["width"][0] <= width <= bounds["width"][1]:
            reasons.append(f"width {width:g} cm is implausible for a {category}")
        if not bounds["depth"][0] <= depth <= bounds["depth"][1]:
            reasons.append(f"depth {depth:g} cm is implausible for a {category}")

    # Other attributes the optimizer needs --------------------------------------
    seats = item.seating_capacity
    if seats is None or isinstance(seats, bool) or seats < 1:
        reasons.append("seating capacity is missing or invalid")
    material, colour, style = _clean(item.material), _clean(item.colour), _clean(item.style)
    for label, value in (("material", material), ("colour", colour), ("style", style)):
        if value is None:
            reasons.append(f"{label} is missing")
    rating = item.rating
    if not _is_number(rating) or not 0 <= rating <= 5:
        reasons.append("rating is missing or outside 0-5")
    reviews = item.review_count
    if reviews is None or isinstance(reviews, bool) or reviews < 0:
        reasons.append("review count is missing or invalid")

    availability = item.availability or "unknown"
    if availability == "unknown":
        warnings.append("availability is not stated; the optimizer will not accept it as available")

    if reasons:
        return RejectedListing(name=name, url=raw_url, reasons=reasons)

    # Optional image: never a reason to reject the product ---------------------------
    image_url = _clean(item.image_url)
    if image_url is not None:
        if check_url(image_url) is not None:
            warnings.append("image URL dropped: not a valid public https URL")
            image_url = None
        elif image_ok is not None and not image_ok(image_url):
            warnings.append("image URL dropped: it did not serve an image")
            image_url = None

    domain = domain_of(url)
    record = {
        "product_id": f"WEB_{re.sub(r'[^A-Z0-9]', '', domain.upper())[:10]}_{url_fingerprint(url)}",
        "name": name,
        "price": float(price),
        "width_cm": width,
        "depth_cm": depth,
        "height_cm": round(item.height_cm * _UNIT_TO_CM[item.dimensions_unit], 2) if _is_number(item.height_cm) else None,
        "seating_capacity": int(seats),
        "material": material,
        "colour": colour,
        "style": style,
        "availability": availability,
        "rating": float(rating),
        "review_count": int(reviews),
        "source_agent_id": provider_id,
        # provenance and display fields (ignored by the optimizer)
        "category": category.lower(),
        "currency": "INR",
        "retailer": _clean(item.retailer) or domain,
        "url": url,
        "image_url": image_url,
        "verified_at": retrieved_at.isoformat(),
        "inferred_fields": ["style"] if item.style_inferred else [],
    }
    _, problems = validate_record(record)
    if problems:
        return RejectedListing(name=name, url=url, reasons=[f"failed record validation: {p}" for p in problems])
    return NormalizedListing(record=record, warnings=warnings)
