"""Reading a *brief*: which room, for whom, with what theme, under what budget.

Real requests are not always "2BHK, ₹2 lakh".  People say "decorate my 10 year old son's bedroom under 50000, he loves
cartoons and monsters".  This module pulls out the pieces deterministically:

    scope      the rooms to furnish ("kids_bedroom", "living_room"...) when no home type (BHK) is given
    audience   child / teen / adult, and the age if stated
    themes     interest words that should steer wall decor, bedding...  (a *preference*, never a requirement)
    budget     "under 50000", "within ₹45k", "budget of 2 lakh" (see also intent.parse_money)

Nothing here guesses: a value is returned only when the text states it.
"""

from __future__ import annotations

import re

_I = re.IGNORECASE

# ── Budget phrasings that carry no currency sign: "under 50000", "up to 45,000" ────────────────────────────────
_LIMIT = re.compile(
    r"\b(?:under|below|within|upto|up to|at most|maximum of|maximum|max|less than|not more than|no more than|"
    r"around|about|budget of|budget is|budget:)\s*(?:rs\.?|inr|₹)?\s*(\d[\d,]*(?:\.\d+)?)(?!\s*(?:k|lakhs?|lacs?|crores?|cr)\b)",
    _I)
MIN_BARE_BUDGET = 1000.0          # "under 3 seats" is not a budget
_AFTER = re.compile(
    r"(\d[\d,]*(?:\.\d+)?)\s*(?:rupees|rs\.?|inr)?\s*(?:to spend|to use|as (?:my )?budget|is my budget|in total|in all)\b", _I)
_NUMBER = re.compile(r"(?<![\w.])(\d[\d,]{3,}(?:\.\d+)?)(?![\w])")
_UNIT_AFTER = re.compile(r"^\s*(?:sq|square|ft|feet|foot|cm|mm|inch|inches|kg|litre|liter|l\b|year|yr|bhk|watt|w\b|%)", _I)


def parse_limit_phrase(text: str) -> float | None:
    for match in [*_LIMIT.finditer(text or ""), *_AFTER.finditer(text or "")]:
        try:
            value = float(match.group(1).replace(",", ""))
        except ValueError:
            continue
        if value >= MIN_BARE_BUDGET:
            return value
    return None


def guess_bare_budget(text: str) -> float | None:
    """A single standalone number of 1,000 or more in a short message that names a room or a home ("my study, 25000").

    Used only when nothing else states a budget; refused if there is more than one candidate or a unit follows
    ("1200 sq ft", "10 year").
    """
    text = text or ""
    if len(text.split()) > 40:
        return None
    candidates = []
    for m in _NUMBER.finditer(text):
        if _UNIT_AFTER.match(text[m.end():m.end() + 12]):
            continue
        try:
            value = float(m.group(1).replace(",", ""))
        except ValueError:
            continue
        if value >= MIN_BARE_BUDGET:
            candidates.append(value)
    return candidates[0] if len(candidates) == 1 else None


# ── Rooms ─────────────────────────────────────────────────────────────────────────────────────────────────────
_KID_WORDS = re.compile(r"\b(son|sons|daughter|daughters|kid|kids|child|children|childs|toddler|baby|nursery|boy|girl|"
                        r"boys|girls|little one)\b", _I)
_TEEN_WORDS = re.compile(r"\b(teen|teens|teenager|teenage)\b", _I)
_AGE = re.compile(r"\b(\d{1,2})\s*[- ]?\s*(?:years?|yrs?)[\s-]*old\b|\baged?\s+(\d{1,2})\b|\b(\d{1,2})\s*(?:yo|y/o)\b", _I)

_LIVING = re.compile(r"\b(living\s*room|drawing\s*room|family\s*room|lounge|hall)\b", _I)
_KITCHEN = re.compile(r"\bkitchen\b", _I)
_STUDY = re.compile(r"\b(study\s*room|home\s*office|office\s*room|work\s*room|workroom|"
                    r"(?:my|the|a)\s+study(?!\s*(?:table|desk|chair|lamp|material|tables|desks)))\b", _I)
_BEDROOM = re.compile(r"\b(bed\s*room|bedroom|master\s*room|guest\s*room|nursery|kids?'?s?\s*room|"
                      r"(?:son|daughter|child|kid)'?s?\s+room)\b", _I)
_ROOM_ONLY = re.compile(r"\b(?:my|his|her|their|our)\s+(?:[\w']+\s+){0,5}?room\b", _I)


def parse_age(text: str) -> int | None:
    match = _AGE.search(text or "")
    if not match:
        return None
    age = int(next(g for g in match.groups() if g))
    return age if 0 <= age <= 99 else None


def parse_audience(text: str) -> tuple[str | None, int | None]:
    """``(audience, age)``: ``child`` (up to 12), ``teen`` (13-19) or ``adult``; ``None`` when not stated."""
    age = parse_age(text)
    if age is not None:
        return ("child" if age <= 12 else "teen" if age <= 19 else "adult"), age
    if _TEEN_WORDS.search(text or ""):
        return "teen", None
    if _KID_WORDS.search(text or ""):
        return "child", None
    return None, None


def parse_scope(text: str) -> list[str]:
    """Room kinds named in ``text`` (``kids_bedroom``, ``bedroom``, ``living_room``, ``kitchen``, ``study``)."""
    text = text or ""
    scope: list[str] = []
    audience, _ = parse_audience(text)
    others = []
    if _LIVING.search(text):
        others.append("living_room")
    if _KITCHEN.search(text):
        others.append("kitchen")
    if _STUDY.search(text):
        others.append("study")
    # "my room" / "his room" alone means a bedroom, but only when no other room was named ("my living room" is not it)
    if _BEDROOM.search(text) or (not others and _ROOM_ONLY.search(text)):
        scope.append("kids_bedroom" if audience in ("child", "teen") else "bedroom")
    return scope + others


# ── Themes ────────────────────────────────────────────────────────────────────────────────────────────────────
THEME_WORDS = ("cartoon", "monster", "dinosaur", "dino", "space", "planet", "superhero", "marvel", "batman", "spiderman",
               "car", "truck", "train", "football", "cricket", "sport", "princess", "unicorn", "fairy", "jungle", "safari",
               "ocean", "sea", "robot", "minecraft", "pokemon", "anime", "music", "art", "nature", "animal", "bird",
               "butterfly", "rainbow", "galaxy", "pirate", "dragon", "lego", "disney", "frozen", "harry potter",
               "star wars", "travel", "map", "floral", "geometric", "vintage", "retro", "boho", "gaming")

_CUE = re.compile(
    r"(?:interested in|into|loves?|likes?|fan of|fans of|obsessed with|passionate about|crazy about|mad about|"
    r"theme(?:d)?(?: around| of| on| is| as)?|themed|favou?rites? (?:are|is))\s+(?:the\s+|a\s+|some\s+)?"
    r"([a-z][a-z ,&'\-]{2,70}?)(?=[.!?;]|\bso\b|\bbut\b|\bbecause\b|\bas\b|$)", _I)

_STOP = frozenset("""a an the and or of in on to for with as such so that this these those would could should be is are was
were very much really especially too also quite just nice good great addition thing things stuff something anything
interested interesting like likes love loves into fan fans his her their my our your he she they it its i we you
please find look looking want wants need needs under below within around about decorate decorating decor
bedroom room home house kid kids child son daughter boy girl old year years""".split())


def _singular(word: str) -> str:
    return word[:-1] if len(word) > 4 and word.endswith("s") and not word.endswith("ss") else word


def parse_themes(text: str, *, limit: int = 5) -> list[str]:
    """Interest / theme words the text states, in order, lower-case and singular ("cartoons" -> "cartoon")."""
    text = text or ""
    found: list[str] = []

    def add(word: str) -> None:
        word = _singular(word.lower().strip(" '-"))
        if word and word not in found and len(word) >= 3 and word not in _STOP:
            found.append(word)

    for match in _CUE.finditer(text):
        for chunk in re.split(r",|&|\band\b", match.group(1)):
            words = [w for w in re.findall(r"[a-z]+", chunk.lower()) if w not in _STOP and len(w) >= 3]
            for w in words[:2]:
                add(w)
    lowered = text.lower()
    for word in THEME_WORDS:
        if re.search(rf"\b{re.escape(word)}s?\b", lowered):
            add(word)
    return found[:limit]


def theme_matches(name: str, themes: list[str] | tuple[str, ...]) -> list[str]:
    """Which ``themes`` the listing title itself mentions (the evidence is the title; nothing is inferred)."""
    lowered = (name or "").lower()
    return [t for t in themes if re.search(rf"\b{re.escape(t)}s?\b", lowered)]
