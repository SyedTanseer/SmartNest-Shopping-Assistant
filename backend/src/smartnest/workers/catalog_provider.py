"""Catalog provider abstraction for product monitoring.

This module defines the interface that :class:`ProductMonitorWorker`
expects from a product catalogue.  In Assignment 2.1 we use a simulated
provider, but the abstraction means future assignments can plug in real
retailer APIs (Assignment 4.2) without rewriting the worker.

Design notes:
    - The provider interface is a simple abstract class, not a Pydantic
      model, because providers are stateful services, not data.
    - Product state is stored as a plain dict so it serializes naturally
      to MongoDB documents in future assignments.
    - Thread safety is the provider's responsibility — it protects its
      internal state with a lock.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from time import monotonic
from typing import Any


class CatalogProvider(ABC):
    """Abstract interface for a product catalogue provider."""

    @abstractmethod
    def get_all_products(self) -> list[dict[str, Any]]:
        """Return a snapshot of all monitored products and their current state.

        Each product dict should contain at least:
            - ``product_name``: str
            - ``price``: float (current price in INR)
            - ``availability``: str ("in_stock" or "out_of_stock")
        """
        ...

    @abstractmethod
    def get_product(self, product_name: str) -> dict[str, Any] | None:
        """Return a single product by name, or ``None`` if not found."""
        ...

    def get_change_time(self, product_name: str) -> float | None:
        """Return the ``time.monotonic()`` instant the product last changed.

        Optional: providers that can't know when a change happened (e.g. a
        remote API without timestamps) return ``None``, and the monitor
        falls back to measuring its own detection time.
        """
        return None


class ProductState:
    """A single product's state snapshot.

    Stored as a plain dict for serialization.  This is a convenience
    wrapper for type-safe access.
    """

    def __init__(
        self,
        product_name: str,
        price: float,
        availability: str = "in_stock",
        **extra: Any,
    ) -> None:
        self.product_name = product_name
        self.price = price
        self.availability = availability
        self.extra = extra

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable dict representation."""
        d = {
            "product_name": self.product_name,
            "price": self.price,
            "availability": self.availability,
        }
        d.update(self.extra)
        return d


class SimulatedCatalogProvider(CatalogProvider):
    """A simulated catalogue that can be programmatically updated.

    This provider holds an in-memory list of products and allows
    external code to update prices and availability to simulate
    catalogue changes.  All state access is protected by a lock.

    Args:
        initial_products: List of product dicts with at least
            ``product_name``, ``price``, and ``availability``.
    """

    def __init__(self, initial_products: list[dict[str, Any]] | None = None) -> None:
        self._lock = threading.RLock()
        self._products: dict[str, dict[str, Any]] = {}
        self._changed_at: dict[str, float] = {}

        for p in initial_products or []:
            name = p["product_name"]
            self._products[name] = dict(p)

    # ── CatalogProvider interface ─────────────────────────────────────────

    def get_all_products(self) -> list[dict[str, Any]]:
        """Return a snapshot of all products (thread-safe copy)."""
        with self._lock:
            return [dict(p) for p in self._products.values()]

    def get_product(self, product_name: str) -> dict[str, Any] | None:
        """Return a single product by name (thread-safe copy), or ``None``."""
        with self._lock:
            p = self._products.get(product_name)
            return dict(p) if p else None

    def get_change_time(self, product_name: str) -> float | None:
        """Monotonic time of the product's last simulated change, if any."""
        with self._lock:
            return self._changed_at.get(product_name)

    # ── Simulation controls ──────────────────────────────────────────────

    def update_price(self, product_name: str, new_price: float) -> bool:
        """Update a product's price.  Returns ``True`` if the product exists."""
        with self._lock:
            p = self._products.get(product_name)
            if p is None:
                return False
            if p.get("price") != new_price:
                p["price"] = new_price
                self._changed_at[product_name] = monotonic()
            return True

    def update_availability(self, product_name: str, availability: str) -> bool:
        """Update a product's availability.  Returns ``True`` if it exists."""
        with self._lock:
            p = self._products.get(product_name)
            if p is None:
                return False
            if p.get("availability") != availability:
                p["availability"] = availability
                self._changed_at[product_name] = monotonic()
            return True

    def add_product(self, product: dict[str, Any]) -> None:
        """Add a new product to the catalogue."""
        with self._lock:
            self._products[product["product_name"]] = dict(product)
            self._changed_at[product["product_name"]] = monotonic()

    def remove_product(self, product_name: str) -> bool:
        """Remove a product (e.g. delisted).  Returns ``True`` if it existed."""
        with self._lock:
            if self._products.pop(product_name, None) is None:
                return False
            self._changed_at[product_name] = monotonic()
            return True
