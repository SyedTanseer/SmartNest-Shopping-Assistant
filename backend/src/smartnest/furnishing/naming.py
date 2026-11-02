"""From the words a person (or Gemini) uses to category ids.

    "bed", "couch", "queen size bed"   -> the curated category (bed, sofa, bed)
    "standing desk", "telescope"       -> a custom category (``x_standing_desk``, ``x_telescope``)
    "bed and wardrobe"                 -> both, split
    "ignore previous instructions ..." -> nothing usable (not a short noun phrase)

A name maps to a CURATED category only when its extra words are plain modifiers (size, material, colour, style, counts).
"queen size bed" is a bed, but "standing desk" is not a study table, so it gets its own category and its own search.

Gemini is only ever asked for NAMES.  This module is the gate between those names and the planner: short noun phrases
only, a hard cap on how many, and a strict slug for anything new, because a custom id becomes a snapshot file name and a
search phrase sent to a paid API.
"""

from __future__ import annotations

import re
from typing import Iterable

from smartnest.furnishing.categories import CATEGORIES, custom_id
from smartnest.furnishing.intent import CATEGORY_WORDS, categories_in

MAX_ITEMS = 12
MAX_NAME_CHARS = 40
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 '&,/\-]{1,%d}$" % (MAX_NAME_CHARS - 1))
_SPLIT = re.compile(r"\s+and\s+|,|&|/", re.IGNORECASE)
_WHOLE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 '&,/\-]{0,%d}$" % (4 * MAX_NAME_CHARS - 1))   # no dots, colons, slashes-first

#: A name containing one of these is an instruction or a leak attempt, not something to buy.  The model is only asked
#: for names, but its output is data from outside our code, so it is screened like any other input.
BLOCKED = frozenset("""
ignore instruction instructions reveal system prompt password secret api token assistant gemini delete drop execute
script http https www override jailbreak
""".split())

#: Words that describe a thing without changing WHAT it is.
MODIFIERS = frozenset("""
queen king single double twin size sized wooden wood solid engineered modern minimalist minimal classic contemporary
small large big compact tall slim leather fabric metal steel iron glass plastic white black grey gray brown beige blue
green red cream walnut oak teak door doors seater seat seats shaped corner premium basic simple cheap good best nice new
foldable folding storage with and for a an the my your family home bedroom living room kitchen kids kid
""".split())


def _canonical(name: str) -> str | None:
    """The curated id if ``name`` is a synonym plus modifiers only, else ``None``."""
    known = categories_in(name)
    if len(known) != 1:
        return None
    words = re.findall(r"[a-z0-9]+", name.lower())
    synonyms = sorted(CATEGORY_WORDS.get(known[0], ()), key=len, reverse=True)
    for syn in synonyms:
        tokens = re.findall(r"[a-z0-9]+", syn.lower())
        if not tokens or not all(any(w == t or w == t + "s" for w in words) for t in tokens):
            continue
        rest = list(words)
        for t in tokens:
            for i, w in enumerate(rest):
                if w == t or w == t + "s":
                    del rest[i]
                    break
        if all(w in MODIFIERS or w.isdigit() or len(w) == 1 for w in rest):
            return known[0]
    return None


def _resolve_one(part: str) -> list[str]:
    part = part.strip().strip(".!?;:")
    if not _NAME.match(part):
        return []
    words = re.findall(r"[a-z0-9]+", part.lower())
    if len(words) > 4 or BLOCKED & set(words):
        return []
    canonical = _canonical(part)
    if canonical:
        return [canonical]
    one = custom_id(part)
    return [one] if one else []


def resolve_name(name: object) -> list[str]:
    """Category ids named by ``name`` ("bed and wardrobe" gives both); ``[]`` if unusable."""
    if not isinstance(name, str) or not _WHOLE.match(name.strip().rstrip(".!?;:")):
        return []                                       # judged as a whole BEFORE splitting: "../../etc/passwd" is not 2 items
    out: list[str] = []
    for part in _SPLIT.split(name):
        for cid in _resolve_one(part):
            if cid not in out:
                out.append(cid)
    return out


def resolve_names(names: Iterable[object], *, limit: int = MAX_ITEMS) -> list[str]:
    """Category ids for many names: de-duplicated, order kept, capped at ``limit``."""
    out: list[str] = []
    for name in names:
        for cid in resolve_name(name):
            if cid not in out:
                out.append(cid)
    return out[:limit]


def is_known(category_id: str) -> bool:
    return category_id in CATEGORIES
