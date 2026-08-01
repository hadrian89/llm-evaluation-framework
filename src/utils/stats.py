"""Small statistics helpers used by the performance evaluator."""

from __future__ import annotations

import math


def percentile(values: list[float], pct: float) -> float:
    """Linear-interpolation percentile (matches numpy's default method)."""
    if not values:
        return 0.0
    data = sorted(values)
    if len(data) == 1:
        return data[0]
    rank = (pct / 100) * (len(data) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return data[int(rank)]
    weight = rank - lower
    return data[lower] * (1 - weight) + data[upper] * weight


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
