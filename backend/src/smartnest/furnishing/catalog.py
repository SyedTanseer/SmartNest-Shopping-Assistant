"""Where the furnishing layer gets product listings.

Three interchangeable catalogs, all returning the optimizer's record format plus a statement of *origin*:

    SnapshotCatalog    REAL listings captured earlier from SerpApi Google Shopping and saved to disk
                       (``backend/data/catalog/<category>.json``, git-ignored).  Serving a snapshot costs no
                       SerpApi request.  Every snapshot keeps the time it was retrieved.
    SimulatedCatalog   listings invented for development and tests.  They are built in the exact shape of a
                       SerpApi response and parsed by the same code, but every record is marked
                       ``simulated`` and has a sample-only URL.  They are never presented as real products.
    LayeredCatalog     real snapshot first; simulated only if allowed and only for categories with no snapshot.

The origin travels with every listing (``data_origin``) so an interface can label it.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Protocol

from smartnest.furnishing.categories import CATEGORIES, category
from smartnest.integrations.products.serpapi import parsing

ORIGIN_REAL = "serpapi"
ORIGIN_SIMULATED = "simulated"
SIMULATED_PROVIDER = "simulated_sample"

DEFAULT_CATALOG_DIR = Path(__file__).resolve().parents[3] / "data" / "catalog"


@dataclass
class CategoryListings:
    category: str
    records: list[dict]
    origin: str                                  # ORIGIN_REAL / ORIGIN_SIMULATED / "none"
    retrieved_at: str | None = None
    query: str | None = None
    rejected: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def simulated(self) -> bool:
        return self.origin == ORIGIN_SIMULATED


class ProductCatalog(Protocol):
    def listings(self, category_id: str) -> CategoryListings: ...


def _words(text: str | None) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _key(query: str) -> str:
    """A file-name-safe key for a search: its sorted words (``[a-z0-9_]`` only, so it can never be a path)."""
    return "_".join(sorted(_words(query)))[:80]


# ── Real snapshots ─────────────────────────────────────────────────────────

class SnapshotCatalog:
    """Real SerpApi listings saved to ``<directory>/<category>.json``."""

    def __init__(self, directory: Path | str = DEFAULT_CATALOG_DIR, *, max_age: timedelta | None = None) -> None:
        self.directory = Path(directory)
        self.max_age = max_age

    def path_for(self, category_id: str, query: str | None = None) -> Path:
        """``<category>.json``; for a theme-sensitive category searched WITH themes, one file per search
        (``<category>__<words>.json``), so a new theme never overwrites the data of an old one."""
        profile = category(category_id)  # validates the id, so it can never become a path traversal
        if query and profile.theme_sensitive and _words(query) != _words(profile.search_term):
            return self.directory / f"{category_id}__{_key(query)}.json"
        return self.directory / f"{category_id}.json"

    def save(self, listings: CategoryListings) -> Path:
        path = self.path_for(listings.category, listings.query)
        path.parent.mkdir(parents=True, exist_ok=True)
        doc = {"category": listings.category, "origin": ORIGIN_REAL, "retrieved_at": listings.retrieved_at,
               "query": listings.query, "records": listings.records, "rejected": listings.rejected,
               "notes": listings.notes}
        path.write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
        return path

    def listings_for(self, category_id: str, themes: tuple[str, ...] = ()) -> CategoryListings:
        """Listings that answer THIS request.

        For a theme-sensitive category (wall decor, rugs) several searches may be saved (one per theme).  The best
        one for the request wins: an exact match for the wanted search, else one that mentions a requested theme (the
        more the better), else a generic search with no theme words.  A "monster cartoon wall poster" search is NOT an
        answer to a princess-themed request, so when nothing covers the theme the item gets no listings (and can be
        searched live) instead of silently showing the wrong thing.
        """
        profile = category(category_id)
        if not profile.theme_sensitive:
            return self.listings(category_id)
        wanted_words = _words(profile.search_query(tuple(themes)))
        requested = {w for t in themes for w in _words(t)}
        term_words = _words(profile.search_term)
        best: tuple[int, str, CategoryListings] | None = None
        seen: list[str] = []
        for path in self._saved(category_id):
            got = self._load(path, category_id)
            if not got.records:
                continue
            seen.append(got.query or path.stem)
            query_words = _words(got.query)
            theme_words = query_words - term_words
            if query_words == wanted_words:
                score = 1000
            elif theme_words & requested:
                score = 10 + len(theme_words & requested)
            elif not theme_words:
                score = 1
            else:
                continue
            if best is None or (score, path.name) > (best[0], best[1]):
                best = (score, path.name, got)
        if best is not None:
            return best[2]
        wanted = profile.search_query(tuple(themes))
        if not seen:
            return CategoryListings(category_id, [], "none", None, wanted,
                                    notes=[f"no saved snapshot for {category_id}"])
        return CategoryListings(category_id, [], "none", None, wanted, notes=[
            f"the saved {profile.label.lower()} listings were searched for "
            + " and ".join(f"\u201c{q}\u201d" for q in seen)
            + f", which does not cover your theme ({', '.join(themes) or 'none'})"])

    def _saved(self, category_id: str) -> list[Path]:
        """Every saved search file for a category: the base file and any ``<category>__<words>.json``."""
        if not self.directory.is_dir():
            return []
        category(category_id)                                 # validates the id before it is used in a glob
        files = [self.directory / f"{category_id}.json", *sorted(self.directory.glob(f"{category_id}__*.json"))]
        return [f for f in files if f.is_file()]

    def listings(self, category_id: str) -> CategoryListings:
        return self._load(self.path_for(category_id), category_id)

    def _load(self, path: Path, category_id: str) -> CategoryListings:
        if not path.is_file():
            return CategoryListings(category_id, [], "none", notes=[f"no saved snapshot for {category_id}"])
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
            records = doc["records"]
            if not isinstance(records, list):
                raise ValueError("records is not a list")
        except (OSError, ValueError, KeyError) as exc:
            return CategoryListings(category_id, [], "none", notes=[f"snapshot unreadable: {type(exc).__name__}"])
        notes = list(doc.get("notes") or [])
        retrieved = doc.get("retrieved_at")
        if self.max_age and retrieved:
            try:
                age = datetime.now(timezone.utc) - datetime.fromisoformat(retrieved)
                if age > self.max_age:
                    notes.append(f"snapshot is {age.days} day(s) old")
            except ValueError:
                pass
        return CategoryListings(category_id, records, ORIGIN_REAL, retrieved, doc.get("query"),
                                list(doc.get("rejected") or []), notes)


# ── Simulated listings ─────────────────────────────────────────────────────

_BRANDS = ("Nestwood", "Aurora Home", "Sample Living", "Casa Verde", "Oakline", "Urbanly", "Homekraft", "Studio Nine")
_STYLE_WORDS = ("Modern", "Classic", "Compact", "Premium", "Essential", "Signature")
_FINISH = ("Walnut", "Oak", "Grey", "Teak", "White", "Charcoal")

# category -> (title template, price range, share of results with a rating, [seats])
_SHAPES: dict[str, tuple[str, tuple[int, int], tuple[int, ...]]] = {
    "sofa": ("{brand} {style} {seats} Seater Sofa ({finish})", (9000, 70000), (2, 3, 3, 3, 4, 5)),
    "tv_unit": ("{brand} {style} TV Unit ({finish})", (3500, 30000), ()),
    "bed": ("{brand} {style} Queen Size Bed ({finish})", (9000, 60000), ()),
    "mattress": ("{brand} {style} Queen Size Mattress", (4500, 40000), ()),
    "wardrobe": ("{brand} {style} {doors} Door Wardrobe ({finish})", (8000, 65000), (2, 3, 4)),
    "dining_table": ("{brand} {style} {seats} Seater Dining Table Set ({finish})", (7000, 60000), (4, 4, 6, 6)),
    "refrigerator": ("{brand} {litres} L {style} Double Door Refrigerator", (13000, 70000), (190, 250, 300, 350)),
    "kids_bed": ("{brand} {style} Kids Single Bed ({finish})", (5000, 30000), ()),
    "study_table": ("{brand} {style} Study Table ({finish})", (2500, 20000), ()),
    "bookshelf": ("{brand} {style} {seats} Shelf Bookshelf ({finish})", (2500, 25000), (3, 4, 5)),
    "wall_decor": ("{brand} {seats} Wall Poster Set", (199, 2500), ("Monster", "Cartoon", "Abstract", "Floral", "Dinosaur")),
    "study_chair": ("{brand} {style} Study Chair ({finish})", (1800, 15000), ()),
    "table_lamp": ("{brand} {style} Table Lamp ({finish})", (399, 4000), ()),
    "rug": ("{brand} {seats} Kids Play Rug", (499, 9000), ("Cartoon", "Monster", "Space", "Floral", "Geometric")),
    "bean_bag": ("{brand} {style} Bean Bag ({finish})", (799, 6000), ()),
    "toy_storage": ("{brand} {style} Toy Storage Organizer ({finish})", (799, 9000), ()),
}


def _raw_items(category_id: str, rng: random.Random, count: int) -> list[dict]:
    template, (low, high), variants = _SHAPES[category_id]
    items = []
    for position in range(1, count + 1):
        brand, style, finish = rng.choice(_BRANDS), rng.choice(_STYLE_WORDS), rng.choice(_FINISH)
        variant = rng.choice(variants) if variants else None
        title = template.format(brand=brand, style=style, finish=finish, seats=variant, doors=variant, litres=variant)
        price = int(round(rng.uniform(low, high) / 100) * 100) - 1
        item = {
            "position": position, "title": title, "product_id": f"{category_id}-{rng.getrandbits(48):x}",
            "source": f"{brand} Store",
            "link": f"https://sample-retailer.example/{category_id}/{position}",
            "price": f"Rs {price:,}", "extracted_price": price,
            "thumbnail": f"https://sample-images.example/{category_id}/{position}.jpg",
        }
        if rng.random() < 0.15:                       # real Google Shopping states a rating for ~15% of results
            item["rating"] = round(rng.uniform(3.6, 4.8), 1)
            item["reviews"] = rng.randint(3, 400)
        roll = rng.random()                           # real search results rarely state stock; keep some that do
        if roll < 0.35:
            item["extensions"] = ["In stock"]
        elif roll < 0.40:
            item["extensions"] = ["Out of stock"]
        items.append(item)
    return items


class SimulatedCatalog:
    """Deterministic made-up listings, built and parsed like real SerpApi data and labelled ``simulated``."""

    def __init__(self, seed: int = 7, per_category: int = 10, retrieved_at: datetime | None = None) -> None:
        self.seed = seed
        self.per_category = per_category
        self._at = retrieved_at or datetime(2026, 1, 1, tzinfo=timezone.utc)

    def listings_for(self, category_id: str, themes: tuple[str, ...] = ()) -> CategoryListings:
        return self.listings(category_id)                   # invented samples: the theme does not change them

    def listings(self, category_id: str) -> CategoryListings:
        category(category_id)
        if category_id not in _SHAPES:
            return CategoryListings(category_id, [], "none", notes=[f"no simulation for {category_id}"])
        rng = random.Random(f"{self.seed}:{category_id}")
        records, rejected = [], []
        for item in _raw_items(category_id, rng, self.per_category):
            built = parsing.build_listing(item, None, retrieved_at=self._at, provider_id=SIMULATED_PROVIDER,
                                          category=category_id)
            if built.record is None:
                rejected.append({"name": built.name, "reasons": built.reasons})
                continue
            built.record.update({"data_origin": ORIGIN_SIMULATED, "discovery_method": "simulated_sample"})
            records.append(built.record)
        return CategoryListings(category_id, records, ORIGIN_SIMULATED, self._at.isoformat(),
                                query=category(category_id).search_term, rejected=rejected,
                                notes=["SIMULATED sample listings - not real products"])


class LayeredCatalog:
    """Real snapshot first; simulated listings only if ``allow_simulated`` and only when no snapshot exists."""

    def __init__(self, real: ProductCatalog, simulated: ProductCatalog | None = None, *,
                 allow_simulated: bool = False) -> None:
        self._real = real
        self._sim = simulated
        self.allow_simulated = allow_simulated and simulated is not None

    def listings_for(self, category_id: str, themes: tuple[str, ...] = ()) -> CategoryListings:
        """Like :meth:`listings`, for a theme-sensitive category: real data only if it covers the requested theme."""
        fetch = getattr(self._real, "listings_for", None)
        real = fetch(category_id, themes) if fetch else self._real.listings(category_id)
        return self._or_simulated(category_id, real)

    def listings(self, category_id: str) -> CategoryListings:
        return self._or_simulated(category_id, self._real.listings(category_id))

    def _or_simulated(self, category_id: str, real: CategoryListings) -> CategoryListings:
        if real.records:
            return self._tag(real)
        if self.allow_simulated:
            sim = self._sim.listings(category_id)           # type: ignore[union-attr]
            sim.notes.insert(0, "no real snapshot for this category; showing simulated sample listings")
            return sim
        return real

    @staticmethod
    def _tag(listings: CategoryListings) -> CategoryListings:
        for record in listings.records:
            record.setdefault("data_origin", ORIGIN_REAL)
        return listings
