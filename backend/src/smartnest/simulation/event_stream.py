"""Convenience re-export of EventStream.

Kept as a separate module so the simulation package has a clear entry
point for the event stream, even though the implementation lives in
``smartnest.core.events``.
"""

from smartnest.core.events import EventStream

__all__ = ["EventStream"]
