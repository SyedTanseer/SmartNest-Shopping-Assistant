"""In-memory stand-ins for PyMongo objects, for unit tests.

``FakeCollection`` is the shared ``InMemoryCollection`` (equality-only
filters, ``$set`` / ``$setOnInsert``); it lets the repository's own logic
(id generation, timestamps, validation, error translation) be tested
without a server.  It is deliberately NOT a Mongo emulator — real query
semantics are covered by the optional Atlas integration tests.
"""

from __future__ import annotations

from smartnest.database.repositories.in_memory import InMemoryCollection as FakeCollection  # noqa: F401


class BrokenCollection:
    """A collection whose every operation raises ``error``."""

    name = "broken"

    def __init__(self, error: Exception) -> None:
        self._error = error

    def __getattr__(self, item: str):
        def fail(*args, **kwargs):
            raise self._error
        return fail
