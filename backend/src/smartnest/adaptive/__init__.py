"""Adaptive behaviour (Assignment 4.2): live information flowing through knowledge, validation and the BDI agent.

``AdaptiveSession`` is the only module that knows about Gemini, product
discovery, the knowledge store and the BDI agent together.  Nothing else
imports it.
"""

from smartnest.adaptive.session import AdaptiveSession, Recommendation, SessionTurn, loosens

__all__ = ["AdaptiveSession", "Recommendation", "SessionTurn", "loosens"]
