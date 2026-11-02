"""Small shared helpers for the BDI package."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone


def utc_now() -> datetime:
    """Timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    """Short unique identifier such as ``int_3f2a9c1b7d04``."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"
