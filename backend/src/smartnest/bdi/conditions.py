"""Conditions: small, serialisable tests evaluated against the agent's beliefs.

Conditions are how the BDI agent *recognises* that something it relied on
is no longer true.  They are used for

    - desire activation conditions ("this goal only makes sense when …"),
    - intention preconditions ("this plan is only valid while …"),
    - action preconditions ("this step may only run when …").

A condition compares one belief with a constant or with another belief::

    Condition(subject="SOFA_A2", predicate="availability",
              operator="in", value=["in_stock", "low_stock"])

    Condition(subject="SOFA_A2", predicate="price", operator="lte",
              ref_subject="user", ref_predicate="sofa_budget")

A belief that is unknown or invalidated makes the condition *fail* with an
explicit explanation — the agent never assumes a missing fact is fine.
Conditions are plain data (Pydantic), so they persist with the agent state.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol

from pydantic import BaseModel

Operator = Literal["eq", "ne", "in", "not_in", "lt", "lte", "gt", "gte", "exists", "not_exists"]

_SYMBOL = {
    "eq": "==", "ne": "!=", "in": "in", "not_in": "not in",
    "lt": "<", "lte": "<=", "gt": ">", "gte": ">=",
}


class BeliefReader(Protocol):
    """Anything that can look up a currently valid belief value."""

    def value(self, subject: str, predicate: str, default: Any = None) -> Any: ...


class ConditionResult(BaseModel):
    """Outcome of evaluating one condition."""

    ok: bool
    explanation: str


class Condition(BaseModel):
    """``<subject>.<predicate> <operator> <value or referenced belief>``."""

    subject: str
    predicate: str
    operator: Operator = "eq"
    value: Any = None
    ref_subject: str | None = None
    ref_predicate: str | None = None
    description: str = ""

    @property
    def label(self) -> str:
        return f"{self.subject}.{self.predicate}"

    def references(self) -> list[tuple[str, str]]:
        """Every (subject, predicate) belief this condition reads."""
        refs = [(self.subject, self.predicate)]
        if self.ref_subject and self.ref_predicate:
            refs.append((self.ref_subject, self.ref_predicate))
        return refs

    def evaluate(self, beliefs: BeliefReader) -> ConditionResult:
        actual = beliefs.value(self.subject, self.predicate)

        if self.operator == "exists":
            ok = actual is not None
            return ConditionResult(ok=ok, explanation=f"{self.label} is {'known' if ok else 'unknown'}")
        if self.operator == "not_exists":
            ok = actual is None
            return ConditionResult(ok=ok, explanation=f"{self.label} is {'unknown' if ok else 'known'}")

        if actual is None:
            return ConditionResult(ok=False, explanation=f"{self.label} is unknown")

        if self.ref_subject and self.ref_predicate:
            expected = beliefs.value(self.ref_subject, self.ref_predicate)
            expected_label = f"{self.ref_subject}.{self.ref_predicate}"
            if expected is None:
                return ConditionResult(ok=False, explanation=f"{expected_label} is unknown")
            shown = f"{expected_label} ({expected!r})"
        else:
            expected = self.value
            shown = repr(expected)

        try:
            ok = self._compare(actual, expected)
        except TypeError:
            return ConditionResult(
                ok=False,
                explanation=f"{self.label} = {actual!r} cannot be compared with {shown}",
            )
        symbol = _SYMBOL[self.operator]
        if ok:
            explanation = f"{self.label} = {actual!r} (requires {symbol} {shown}): satisfied"
        else:
            explanation = f"{self.label} = {actual!r} but requires {symbol} {shown}"
        return ConditionResult(ok=ok, explanation=explanation)

    def _compare(self, actual: Any, expected: Any) -> bool:
        op = self.operator
        if op == "eq":
            return actual == expected
        if op == "ne":
            return actual != expected
        if op == "in":
            return actual in expected
        if op == "not_in":
            return actual not in expected
        if isinstance(actual, bool) or isinstance(expected, bool):
            raise TypeError("booleans are not ordered")
        if op == "lt":
            return actual < expected
        if op == "lte":
            return actual <= expected
        if op == "gt":
            return actual > expected
        if op == "gte":
            return actual >= expected
        raise TypeError(f"unsupported operator {op}")  # pragma: no cover
