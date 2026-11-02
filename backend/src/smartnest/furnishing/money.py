"""Rupee formatting with Indian digit grouping."""

from __future__ import annotations


def inr(amount: float) -> str:
    """₹ with Indian digit grouping: 1234567 -> ₹12,34,567."""
    n = int(round(amount))
    sign, s = ("-" if n < 0 else ""), str(abs(n))
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        s = ",".join(parts + [tail])
    return f"{sign}₹{s}"
