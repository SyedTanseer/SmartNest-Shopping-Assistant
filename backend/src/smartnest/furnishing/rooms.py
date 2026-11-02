"""From "2BHK" to the rooms and the items each room needs.

The Space Planning agent's deterministic core.  A home type expands to rooms; each room lists the categories it
needs.  Room *dimensions are optional*: when the user states them they become size limits on that room's
products, when they do not, no limit is invented (see the optional-dimensions policy).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from smartnest.furnishing.categories import CategoryProfile, category


@dataclass(frozen=True)
class Slot:
    """One thing to buy: a category in a room."""

    slot_id: str            # e.g. "bedroom_1.bed"
    room_id: str            # e.g. "bedroom_1"
    room_label: str         # e.g. "Bedroom 1"
    profile: CategoryProfile
    min_seating: int = 1
    max_width_cm: float | None = None
    max_depth_cm: float | None = None

    @property
    def label(self) -> str:
        return f"{self.room_label} {self.profile.label.lower()}"

    def to_dict(self) -> dict:
        return {"slot_id": self.slot_id, "room_id": self.room_id, "room_label": self.room_label,
                "category": self.profile.id, "min_seating": self.min_seating,
                "max_width_cm": self.max_width_cm, "max_depth_cm": self.max_depth_cm}

    @classmethod
    def from_dict(cls, d: dict) -> "Slot":
        return cls(d["slot_id"], d["room_id"], d["room_label"], category(d["category"]),
                   int(d.get("min_seating") or 1), d.get("max_width_cm"), d.get("max_depth_cm"))


@dataclass(frozen=True)
class Room:
    room_id: str
    label: str
    categories: tuple[str, ...]
    width_cm: float | None = None
    depth_cm: float | None = None


_LIVING = Room("living_room", "Living room", ("sofa", "tv_unit"))
_KITCHEN = Room("kitchen", "Kitchen", ("refrigerator", "dining_table"))


#: Rooms that can be furnished on their own (no "BHK" needed).  Key = the scope word used in requests.
SCOPE_ROOMS: dict[str, Room] = {
    "living_room": _LIVING,
    "kitchen": _KITCHEN,
    "bedroom": Room("bedroom_1", "Bedroom", ("bed", "mattress", "wardrobe")),
    "kids_bedroom": Room("kids_bedroom", "Kid's bedroom",
                         ("kids_bed", "mattress", "wardrobe", "study_table", "study_chair", "bookshelf", "wall_decor",
                          "table_lamp", "rug", "bean_bag", "toy_storage")),
    "study": Room("study", "Study", ("study_table", "study_chair", "bookshelf", "table_lamp")),
}
ROOM_KINDS: tuple[str, ...] = tuple(SCOPE_ROOMS)


def rooms_for_scope(scope: tuple[str, ...] | list[str], dimensions: dict[str, dict[str, float]] | None = None) -> list[Room]:
    """The rooms for a request that names rooms instead of a home type (e.g. just a child's bedroom)."""
    dims = dimensions or {}
    out: list[Room] = []
    for kind in scope:
        if kind not in SCOPE_ROOMS:
            raise ValueError(f"unknown room {kind!r}; known: {', '.join(ROOM_KINDS)}")
        r = SCOPE_ROOMS[kind]
        if r in out:
            continue
        d = dims.get(r.room_id) or {}
        out.append(Room(r.room_id, r.label, r.categories, d.get("width_cm"), d.get("depth_cm")))
    if not out:
        raise ValueError("no rooms were given")
    return out


def rooms_for_items(items: tuple[str, ...] | list[str], label: str = "Shopping list") -> list[Room]:
    """One pseudo-room holding exactly the things the user asked for (no home or room template involved)."""
    unique = tuple(dict.fromkeys(items))
    if not unique:
        raise ValueError("no items were given")
    return [Room("shopping_list", label, unique)]


def _bedroom(n: int) -> Room:
    return Room(f"bedroom_{n}", f"Bedroom {n}", ("bed", "mattress", "wardrobe"))


_BHK = re.compile(r"\b([1-5])\s*[- ]?\s*bhk\b", re.IGNORECASE)
_BEDROOMS = re.compile(r"\b([1-5])\s*[- ]?\s*(?:bed\s*rooms?|bedrooms?)\b", re.IGNORECASE)


def parse_home_type(text: str) -> int | None:
    """The number of bedrooms in "2BHK" / "2 bhk" / "2 bedroom apartment", else ``None``."""
    for pattern in (_BHK, _BEDROOMS):
        match = pattern.search(text or "")
        if match:
            return int(match.group(1))
    return None


def rooms_for(bedrooms: int, dimensions: dict[str, dict[str, float]] | None = None) -> list[Room]:
    """The rooms of a home with ``bedrooms`` bedrooms; ``dimensions`` (room id -> width_cm/depth_cm) is optional."""
    if not 1 <= bedrooms <= 5:
        raise ValueError("bedrooms must be between 1 and 5")
    rooms = [_LIVING, *(_bedroom(n) for n in range(1, bedrooms + 1)), _KITCHEN]
    dims = dimensions or {}
    return [Room(r.room_id, r.label, r.categories, (dims.get(r.room_id) or {}).get("width_cm"),
                 (dims.get(r.room_id) or {}).get("depth_cm")) for r in rooms]


def slots_for(rooms: list[Room]) -> list[Slot]:
    slots = []
    for room in rooms:
        for cat in room.categories:
            profile = category(cat)
            slots.append(Slot(f"{room.room_id}.{cat}", room.room_id, room.label, profile,
                              min_seating=3 if cat == "sofa" else (4 if cat == "dining_table" else 1),
                              max_width_cm=room.width_cm, max_depth_cm=room.depth_cm))
    return slots
