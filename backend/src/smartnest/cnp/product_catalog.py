"""Simulated product catalogues for CNP bidding agents.

Each bidding agent has its own independent catalogue.  The catalogues
are clearly labelled as simulated — they contain no real retailer URLs.

The ``ProductCatalog`` class is separate from the Assignment 2.1
``CatalogProvider`` because CNP agents need richer product attributes
(material, color, style, seating capacity, depth, rating, reviews) and
don't need thread-safe price/availability updates.

Future compatibility:
    The ``ProductCatalog`` interface allows replacement with real
    product APIs (Assignment 4.2) without changing the bidding agent.
"""

from __future__ import annotations

import threading
from typing import Any

# Availability values that mean the product cannot be offered at all.
_UNAVAILABLE = frozenset({"out_of_stock", "unavailable", "discontinued"})


class ProductCatalog:
    """A simulated product catalogue for a CNP bidding agent.

    Thread-safe: a bidding agent's server handles each coordinator
    connection in its own thread, so reads are protected by a lock.

    Args:
        agent_id: The agent that owns this catalogue.
        products: List of product dicts.
    """

    def __init__(
        self,
        agent_id: str,
        products: list[dict[str, Any]],
    ) -> None:
        self._agent_id = agent_id
        self._lock = threading.RLock()
        self._products: dict[str, dict[str, Any]] = {}
        for p in products:
            pid = p["product_id"]
            self._products[pid] = dict(p)

    @property
    def agent_id(self) -> str:
        return self._agent_id

    def get_all_products(self) -> list[dict[str, Any]]:
        """Return all products in the catalogue."""
        with self._lock:
            return [dict(p) for p in self._products.values()]

    def get_product(self, product_id: str) -> dict[str, Any] | None:
        """Return a single product by ID, or ``None``."""
        with self._lock:
            p = self._products.get(product_id)
            return dict(p) if p else None

    def get_in_stock_products(self) -> list[dict[str, Any]]:
        """Return only products that are in stock."""
        with self._lock:
            return [
                dict(p) for p in self._products.values()
                if p.get("availability") == "in_stock"
            ]

    def find_eligible_products(
        self,
        max_budget: float,
        room_width: float | None,
        preferred_style: str = "",
        seating_capacity: int | None = None,
        max_width: float | None = None,
    ) -> list[dict[str, Any]]:
        """Find products that satisfy all given constraints.

        Args:
            max_budget: Maximum allowed price.
            room_width: Available room width (products must not exceed).
                ``None`` means the width has not been announced, so no
                width filter is applied.
            preferred_style: If given, filter by style (case-insensitive).
            seating_capacity: If given, filter by minimum seating capacity.
            max_width: Explicit max width override (defaults to room_width).

        Returns:
            List of eligible product dicts (copies).  Products that can't
            be bought (out of stock, unavailable, discontinued) are never
            eligible; low-stock or backorder items are, and the
            coordinator's evaluator scores the delivery risk.
        """
        width_limit = max_width if max_width is not None else room_width
        results: list[dict[str, Any]] = []

        with self._lock:
            products = [dict(p) for p in self._products.values()]

        for p in products:
            if str(p.get("availability", "")).lower() in _UNAVAILABLE or not p.get("availability"):
                continue
            if p.get("price", 0) > max_budget:
                continue
            if width_limit is not None and p.get("width_cm", 0) > width_limit:
                continue
            if seating_capacity and p.get("seating_capacity", 0) < seating_capacity:
                continue
            if preferred_style and p.get("style", "").lower() != preferred_style.lower():
                continue
            results.append(p)

        return results


# ── Default simulated catalogues ────────────────────────────────────────────
#
# ALL DATA BELOW IS SIMULATED for educational purposes.  These are not real
# retailer listings, prices or reviews, and no purchase URLs are included.
#
# Each product carries every field the Assignment 3.2 utility model needs:
# product_id, name, price, width_cm, depth_cm, seating_capacity, material,
# color, style, availability, rating (0–5), review_count, source_agent_id.
#
# ``catalog_agent_a()`` / ``catalog_agent_b()`` return the three sofas each
# that the Assignment 3.1 negotiation scenario is built on.  Passing
# ``extended=True`` adds four more sofas per agent for the Assignment 3.2
# optimisation scenario (14 candidates in total), chosen to exercise every
# hard constraint and its boundaries:
#
#   SOFA_A4  depth 115 cm  > 110 cm limit           → rejected (depth)
#   SOFA_A5  2 seats        < 3 required            → rejected (seating)
#   SOFA_A6  ₹30,000 / 210 cm / 110 cm / 3 seats    → exactly at every limit
#   SOFA_A7  traditional, burgundy                  → feasible, low S and C
#   SOFA_B4  contemporary, taupe                    → feasible, partial S and C
#   SOFA_B5  ₹30,001                                → rejected (₹1 over budget)
#   SOFA_B6  out of stock                           → rejected (availability)
#   SOFA_B7  211 cm                                 → rejected (1 cm too wide)

_AGENT_A = "research_agent_a"
_AGENT_B = "research_agent_b"


def _sofa(agent: str, product_id: str, name: str, price: float, width: float, depth: float,
          material: str, color: str, style: str, seats: int, availability: str,
          rating: float, reviews: int) -> dict[str, Any]:
    return {
        "product_id": product_id,
        "name": name,
        "price": price,
        "width_cm": width,
        "depth_cm": depth,
        "material": material,
        "color": color,
        "style": style,
        "seating_capacity": seats,
        "availability": availability,
        "rating": rating,
        "review_count": reviews,
        "source_agent_id": agent,
    }


_A_CORE = [
    _sofa(_AGENT_A, "SOFA_A1", "Minimalist Three-Seater Sofa A1", 24999, 230, 95,
          "fabric", "grey", "minimalist", 3, "in_stock", 4.2, 210),
    _sofa(_AGENT_A, "SOFA_A2", "Compact Minimalist Sofa A2", 27499, 195, 92,
          "leather", "neutral", "minimalist", 3, "in_stock", 4.5, 88),
    _sofa(_AGENT_A, "SOFA_A3", "Premium Minimalist Sofa A3", 29499, 205, 98,
          "fabric", "beige", "minimalist", 3, "in_stock", 4.7, 140),
]

_A_EXTENDED = [
    _sofa(_AGENT_A, "SOFA_A4", "Scandi Lounge Sofa A4", 25499, 200, 115,
          "fabric", "neutral", "scandinavian", 3, "in_stock", 4.6, 310),
    _sofa(_AGENT_A, "SOFA_A5", "Compact Loveseat A5", 18999, 160, 85,
          "fabric", "neutral", "minimalist", 2, "in_stock", 4.4, 75),
    _sofa(_AGENT_A, "SOFA_A6", "Exact-Fit Minimalist Sofa A6", 30000, 210, 110,
          "fabric", "cream", "minimalist", 3, "in_stock", 4.1, 100),
    _sofa(_AGENT_A, "SOFA_A7", "Classic Chesterfield Sofa A7", 26499, 205, 100,
          "leather", "burgundy", "traditional", 3, "in_stock", 4.5, 60),
]

_B_CORE = [
    _sofa(_AGENT_B, "SOFA_B1", "Scandi Three-Seater Sofa B1", 22999, 225, 90,
          "fabric", "white", "minimalist", 3, "in_stock", 4.0, 65),
    _sofa(_AGENT_B, "SOFA_B2", "Urban Minimalist Sofa B2", 28499, 200, 100,
          "leather", "grey", "minimalist", 3, "in_stock", 4.3, 120),
    _sofa(_AGENT_B, "SOFA_B3", "Slim Profile Sofa B3", 26999, 190, 88,
          "fabric", "neutral", "minimalist", 3, "in_stock", 4.4, 45),
]

_B_EXTENDED = [
    _sofa(_AGENT_B, "SOFA_B4", "Contemporary Track-Arm Sofa B4", 27999, 208, 105,
          "fabric", "taupe", "contemporary", 3, "in_stock", 4.8, 95),
    _sofa(_AGENT_B, "SOFA_B5", "Signature Minimalist Sofa B5", 30001, 200, 95,
          "leather", "neutral", "minimalist", 3, "in_stock", 4.9, 400),
    _sofa(_AGENT_B, "SOFA_B6", "Cloud Minimalist Sofa B6", 25999, 198, 96,
          "fabric", "neutral", "minimalist", 3, "out_of_stock", 4.6, 150),
    _sofa(_AGENT_B, "SOFA_B7", "Edge Minimalist Sofa B7", 24999, 211, 100,
          "fabric", "grey", "minimalist", 3, "in_stock", 4.2, 30),
]


def catalog_agent_a(extended: bool = False) -> ProductCatalog:
    """Simulated catalogue for Product Research Agent A.

    Args:
        extended: Include the four extra Assignment 3.2 sofas.

    All entries are simulated data for educational purposes.
    No real retailer listings or URLs are included.
    """
    products = _A_CORE + (_A_EXTENDED if extended else [])
    return ProductCatalog(agent_id=_AGENT_A, products=[dict(p) for p in products])


def catalog_agent_b(extended: bool = False) -> ProductCatalog:
    """Simulated catalogue for Product Research Agent B.

    Args:
        extended: Include the four extra Assignment 3.2 sofas.

    All entries are simulated data for educational purposes.
    No real retailer listings or URLs are included.
    """
    products = _B_CORE + (_B_EXTENDED if extended else [])
    return ProductCatalog(agent_id=_AGENT_B, products=[dict(p) for p in products])
