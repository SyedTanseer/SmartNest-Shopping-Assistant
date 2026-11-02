"""Turning SerpApi Google Shopping data into optimizer-compatible records.

Nothing here invents data.  The **missing-data policy** is:

    * A field SerpApi does not state stays ``None``.  No default is ever assigned.
    * A record is offered to the optimizer only if every field the optimizer's own
      record validation requires is present (price, width, depth, seating, material,
      colour, style, rating, review count).  Otherwise it goes to ``rejected`` with the
      list of missing fields and the product details we do have, so nothing is hidden.
    * Availability is the one field that may be the explicit state ``"unknown"``
      (SerpApi does not state stock); the optimizer then refuses it with a clear reason.
    * Values are read only from stated structured fields or explicit text:
      ``about_the_product.features`` entries (``{"title", "value"}``) for width, depth,
      material, colour and style; "N seater" wording in the title/snippet for seating.
      Ambiguous or conflicting statements are treated as missing.

URL classification (never present a Google page as a retailer listing):

    retailer         an https link on a non-Google domain (a genuine purchase page candidate)
    google_shopping  a Google domain (product page, ad redirect, ...)
    invalid          present but not a usable public https URL
    missing          not provided
"""

from __future__ import annotations

import hashlib
import math
import re
from urllib.parse import quote
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable

from smartnest.integrations.products.url_validation import check_url, domain_of
from smartnest.optimization import validate_record

#: Mandatory for a listing to be offered to the optimizer: a name, a valid positive INR price and a usable
#: listing URL (a direct retailer URL or, failing that, the Google Shopping product page - labelled as such).
REQUIRED_FIELDS = ("name", "price", "listing_url")
#: Optional: stated or ``None``.  A missing one is never defaulted; the optimizer simply does not score it
#: (rating, review count, style, colour), or cannot verify the matching constraint (sizes, seating, stock).
OPTIONAL_FIELDS = ("rating", "review_count", "style", "colour", "material", "seating_capacity",
                   "width_cm", "depth_cm", "height_cm")

URL_RETAILER = "retailer"
URL_GOOGLE = "google_shopping"
URL_INVALID = "invalid"
URL_MISSING = "missing"

_GOOGLE_HOST = re.compile(
    r"(^|\.)(google\.(com|co\.[a-z]{2}|com\.[a-z]{2}|[a-z]{2,3})|googleadservices\.com|googlesyndication\.com|g\.co|goo\.gl)$")
_CURRENCIES = (("₹", "INR"), ("$", "USD"), ("€", "EUR"), ("£", "GBP"))
_INR_WORD = re.compile(r"(?<![a-z])(inr|rs\.?)(?![a-z])", re.IGNORECASE)
_AMOUNT = re.compile(r"\d[\d,]*(?:\.\d+)?")
_UNIT_TO_CM = {"cm": 1.0, "mm": 0.1, "m": 100.0, "in": 2.54, "inch": 2.54, "inches": 2.54}
_NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _text(value: Any) -> str | None:
    return value.strip() or None if isinstance(value, str) else None


# ── Price ─────────────────────────────────────────────────────────────────

def parse_price(text: Any, extracted: Any = None) -> tuple[float | None, str | None, list[str]]:
    """``(amount, currency, problems)`` from a display price (``"₹24,999.00"``) and/or ``extracted_price``.

    The currency is read only from the display text; without it the currency is ``None``
    (never assumed to be INR).
    """
    problems: list[str] = []
    currency = None
    amount = None
    if isinstance(text, str) and text.strip():
        raw = text.strip()
        for symbol, code in _CURRENCIES:
            if symbol in raw:
                currency = code
                break
        if currency is None and _INR_WORD.search(raw):
            currency = "INR"
        match = _AMOUNT.search(raw)
        if match and re.search(r"(?<!\w)-\s*[^\d\s]?\s*\d", raw):
            problems.append("price is negative")
            return None, currency, problems
        if match:
            try:
                amount = float(match.group().replace(",", ""))
            except ValueError:
                amount = None
    if _is_number(extracted) and extracted > 0:
        if amount is not None and abs(amount - float(extracted)) > 0.5:
            problems.append(f"price text ({text!r}) disagrees with extracted_price ({extracted!r})")
            return None, currency, problems
        amount = float(extracted)
    if amount is None or amount <= 0:
        problems.append("price is missing or not a positive number")
        return None, currency, problems
    return amount, currency, problems


# ── URLs ──────────────────────────────────────────────────────────────────

def classify_url(url: Any) -> tuple[str, str | None]:
    """``(kind, url)``; the URL is returned only for ``retailer`` and ``google_shopping``.

    SerpApi's Google ``product_link`` can contain raw spaces (a search phrase); those are
    percent-encoded so the link is usable.  Relative links (``/goto?...``) are not URLs we can use.
    """
    if not isinstance(url, str) or not url.strip():
        return URL_MISSING, None
    url = url.strip()
    host_and_rest = re.match(r"(https?://[^/?#\s]+)(?=[/?#]|$)(.*)$", url, re.DOTALL)
    if host_and_rest:  # encode spaces etc. in the path/query only; a space in the host stays invalid
        url = host_and_rest.group(1) + quote(host_and_rest.group(2), safe=":/?&=%,#+@;~!*()'$")
    if check_url(url) is not None:
        return URL_INVALID, None
    if _GOOGLE_HOST.search(domain_of(url)):
        return URL_GOOGLE, url
    return URL_RETAILER, url


# ── Specifications ────────────────────────────────────────────────────────

def _dimension_cm(value: str, label: str | None = None) -> float | None:
    """A dimension in cm from text such as ``"210 cm"``; with ``label`` only the number after that word."""
    haystack = value
    if label:
        found = re.search(rf"{label}\s*[:=\-]?\s*(.*)", value, re.IGNORECASE)
        if not found:
            return None
        haystack = found.group(1)
    match = re.match(r"\s*(\d+(?:\.\d+)?)\s*(cm|mm|inches|inch|in|m)\b", haystack, re.IGNORECASE)
    if not match:
        return None
    result = round(float(match.group(1)) * _UNIT_TO_CM[match.group(2).lower()], 2)
    return result if result > 0 else None


def unwrap_details(details: Any) -> dict | None:
    """The ``product_results`` object of a ``google_immersive_product`` response (stores, features...)."""
    if not isinstance(details, dict):
        return None
    inner = details.get("product_results")
    return inner if isinstance(inner, dict) else details


def _features(details: dict | None) -> list[tuple[str, str]]:
    details = unwrap_details(details)
    about = (details or {}).get("about_the_product")
    feats = about.get("features") if isinstance(about, dict) else None
    out: list[tuple[str, str]] = []
    if isinstance(feats, list):
        for f in feats:
            if isinstance(f, dict) and isinstance(f.get("title"), str) and isinstance(f.get("value"), str):
                out.append((f["title"].strip().lower(), f["value"].strip()))
    return out


def seating_from_text(*texts: str | None) -> int | None:
    """Seat count from explicit "N seater" wording; ``None`` if absent or if the texts disagree."""
    found: set[int] = set()
    pattern = re.compile(r"(?<![\d+])\b(\d|one|two|three|four|five|six)\s*[- ]?\s*seat(?:er|ers|s|ing)?\b", re.IGNORECASE)
    for text in texts:
        if not text:
            continue
        for m in pattern.finditer(text):
            token = m.group(1).lower()
            found.add(int(token) if token.isdigit() else _NUMBER_WORDS[token])
    found.discard(0)
    return found.pop() if len(found) == 1 else None


def specs_from(item: dict, details: dict | None) -> dict[str, Any]:
    """Stated specifications (width/depth in cm, seating, material, colour, style); absent ones are ``None``."""
    feats = _features(details)

    def pick(*names: str) -> str | None:
        for title, value in feats:
            if title in names and value:
                return value
        return None

    width = depth = None
    w, d = pick("width", "overall width"), pick("depth", "overall depth")
    if w:
        width = _dimension_cm(w)
    if d:
        depth = _dimension_cm(d)
    if width is None or depth is None:
        for title, value in feats:
            if "dimension" in title or title in ("size", "product size"):
                width = width if width is not None else _dimension_cm(value, "width")
                depth = depth if depth is not None else _dimension_cm(value, "depth")

    seat_feature = next((v for t, v in feats if "seating" in t or t in ("seats", "seater")), None)
    seating = seating_from_text(item.get("title"), item.get("snippet"), *(item.get("extensions") or []))
    if seat_feature:
        if seat_feature.strip().isdigit() and int(seat_feature) > 0:
            stated = int(seat_feature)
            seating = stated if seating in (None, stated) else None
        else:
            seating = seating_from_text(seat_feature) if seating is None else seating
    return {
        "width_cm": width, "depth_cm": depth, "seating_capacity": seating,
        "material": pick("material", "upholstery material"),
        "colour": pick("color", "colour"),
        "style": pick("style", "design style"),
    }


def availability_from(*texts: Iterable[str] | str | None) -> str:
    """``in_stock`` / ``out_of_stock`` only when a text states it explicitly and consistently, else ``unknown``."""
    flat: list[str] = []
    for t in texts:
        if isinstance(t, str):
            flat.append(t)
        elif t:
            flat.extend(x for x in t if isinstance(x, str))
    blob = " ".join(flat).lower()
    out = bool(re.search(r"out of stock|sold out|currently unavailable", blob))
    inn = bool(re.search(r"(?<!out of )\bin stock\b", blob))
    if inn and not out:
        return "in_stock"
    if out and not inn:
        return "out_of_stock"
    return "unknown"


# ── Offers (product-details response) ─────────────────────────────────────

@dataclass
class Offer:
    merchant: str
    price: float
    url_kind: str
    url: str | None
    details: list[str] = field(default_factory=list)


def best_offer(details: dict | None) -> tuple[Offer | None, int]:
    """The cheapest INR offer with a direct retailer URL, and how many INR offers had none.

    Offers in other currencies, with unreadable prices, or whose link is a Google page are
    never selected as "the retailer listing".
    """
    details = unwrap_details(details)
    stores = ((details or {}).get("stores") if isinstance((details or {}).get("stores"), list) else None) or []
    best: Offer | None = None
    without_direct = 0
    for store in stores:
        if not isinstance(store, dict):
            continue
        price, currency, problems = parse_price(store.get("price"), store.get("extracted_price"))
        merchant = _text(store.get("name"))
        if problems or currency != "INR" or merchant is None:
            continue
        kind, url = classify_url(store.get("link"))
        if kind != URL_RETAILER:
            without_direct += 1
            continue
        offer = Offer(merchant, price, kind, url,
                      [s for s in (store.get("details_and_offers") or []) if isinstance(s, str)] +
                      [s for s in (store.get("tag"),) if isinstance(s, str)])
        if best is None or offer.price < best.price:
            best = offer
    return best, without_direct


# ── Records ───────────────────────────────────────────────────────────────

@dataclass
class Built:
    """One SerpApi product: a complete ``record`` for the optimizer, or the ``missing`` fields and why."""

    name: str | None
    record: dict | None = None
    reasons: list[str] = field(default_factory=list)
    summary: dict = field(default_factory=dict)  # what we do know, for display when incomplete


def product_id_for(item: dict, merchant: str | None) -> str:
    raw = _text(item.get("product_id")) or ""
    clean = re.sub(r"[^A-Za-z0-9]", "", raw)[:24]
    if clean:
        return f"SERP_{clean}"
    digest = hashlib.sha256(f"{item.get('title')}|{merchant}".encode()).hexdigest()[:12].upper()
    return f"SERP_{digest}"


def build_listing(item: dict, details: dict | None, *, retrieved_at: datetime, provider_id: str,
                  category: str = "sofa") -> Built:
    """Combine a shopping result and (optionally) its product details into a record, or explain why not."""
    details = unwrap_details(details)
    name = _text(item.get("title"))
    reasons: list[str] = []
    if name is None:
        reasons.append("name is missing")

    price, currency, problems = parse_price(item.get("price"), item.get("extracted_price"))
    merchant = _text(item.get("source"))
    price_source = "search_result"
    url_kind, url = URL_MISSING, None

    # The search result carries a Google Shopping page, and a direct link only if SerpApi supplies one.
    page_kind, page_url = classify_url(item.get("product_link"))
    product_page_url = page_url if page_kind == URL_GOOGLE else None
    direct_kind, direct_url = classify_url(item.get("link"))
    if direct_kind == URL_RETAILER:
        url_kind, url = URL_RETAILER, direct_url
    elif page_kind == URL_RETAILER:  # product_link unexpectedly on a retailer domain
        url_kind, url = URL_RETAILER, page_url
    elif direct_kind == URL_GOOGLE or page_kind == URL_GOOGLE:
        url_kind = URL_GOOGLE
    elif URL_INVALID in (direct_kind, page_kind):
        url_kind = URL_INVALID

    offer, no_direct = best_offer(details)
    if offer is not None:  # a verified retailer offer replaces the search-level merchant/price
        price, currency, merchant, price_source = offer.price, "INR", offer.merchant, "retailer_offer"
        url_kind, url = offer.url_kind, offer.url
        problems = []

    reasons.extend(problems)
    if price is not None and currency != "INR":
        reasons.append(f"price currency is {currency or 'not stated'}, not INR")
        price = None

    specs = specs_from(item, details)
    rating = item.get("rating") if _is_number(item.get("rating")) and 0 <= item.get("rating") <= 5 else None
    reviews = item.get("reviews")
    reviews = int(reviews) if _is_number(reviews) and reviews >= 0 and float(reviews).is_integer() else None
    availability = availability_from(item.get("snippet"), item.get("extensions"),
                                     offer.details if offer else None)
    image = next((u for u in (item.get("thumbnail"), item.get("serpapi_thumbnail")) if check_url(u) is None), None)

    record = {
        "product_id": product_id_for(item, merchant),
        "name": name, "price": price, **{k: specs[k] for k in ("width_cm", "depth_cm", "seating_capacity",
                                                                "material", "colour", "style")},
        "availability": availability, "rating": rating, "review_count": reviews,
        "source_agent_id": provider_id,
        # provenance and display fields (ignored by the optimizer)
        "category": category.lower(), "currency": "INR" if currency == "INR" else currency,
        "retailer": merchant, "merchant": merchant,
        "url": url, "url_type": url_kind, "product_page_url": product_page_url,
        "price_source": price_source, "image_url": image, "height_cm": None,
        "verified_at": retrieved_at.isoformat(), "inferred_fields": [],
        "discovery_method": "serpapi_google_shopping",
    }
    # The listing URL: a direct retailer link if there is one, otherwise the Google Shopping page, labelled as such.
    if url_kind == URL_RETAILER and url:
        record["listing_url"], record["listing_url_type"] = url, URL_RETAILER
    elif product_page_url:
        record["listing_url"], record["listing_url_type"] = product_page_url, URL_GOOGLE
    else:
        record["listing_url"], record["listing_url_type"] = None, url_kind  # "missing" or "invalid"
    missing = [f for f in REQUIRED_FIELDS if record.get(f) is None]
    unstated = [f for f in OPTIONAL_FIELDS if record.get(f) is None]
    summary = {"name": name, "price": price, "currency": currency, "merchant": merchant, "url": url,
               "url_type": url_kind, "product_page_url": product_page_url, "rating": rating,
               "review_count": reviews, "missing": missing, "unstated": unstated,
               "listing_url": record["listing_url"], "listing_url_type": record["listing_url_type"],
               "unstated_dimensions": [d for d in ("width_cm", "depth_cm", "height_cm") if record.get(d) is None]}
    if no_direct and url is None:  # informational: the product is kept, but no direct purchase link exists
        summary["note"] = f"{no_direct} offer(s) had only Google links, no direct retailer URL"
    if missing:
        reasons.append("not stated by SerpApi: " + ", ".join(
            "a usable listing URL" if m == "listing_url" else m for m in missing))
    if reasons:
        return Built(name, None, reasons, summary)
    _, problems = validate_record(record)
    if problems:
        return Built(name, None, [f"failed record validation: {p}" for p in problems], summary)
    return Built(name, record, [], summary)
