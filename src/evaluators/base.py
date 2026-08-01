"""Base class shared by all evaluators."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.models import EvalCase, MetricResult


class BaseEvaluator(ABC):
    """Every evaluator scores a single `EvalCase` and returns a `MetricResult`.

    Subclasses set `name` and a default `threshold`, then implement
    `_score()`. `evaluate()` wraps that to apply the pass/fail cutoff
    consistently and to catch scoring errors without killing a whole
    batch run.
    """

    name: str = "base"
    default_threshold: float = 0.7
    higher_is_better: bool = True

    def __init__(self, threshold: float | None = None) -> None:
        self.threshold = self.default_threshold if threshold is None else threshold

    @abstractmethod
    def _score(self, case: EvalCase) -> MetricResult:
        """Compute the metric. Implementations should not set `passed`;
        `evaluate()` derives it from the threshold for consistency."""

    def evaluate(self, case: EvalCase) -> MetricResult:
        result = self._score(case)
        if self.higher_is_better:
            passed = result.score >= self.threshold
        else:
            passed = result.score <= self.threshold
        result.passed = passed
        result.threshold = self.threshold
        return result

    def evaluate_batch(self, cases: list[EvalCase]) -> list[MetricResult]:
        return [self.evaluate(case) for case in cases]
