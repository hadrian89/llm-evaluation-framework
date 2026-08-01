"""Latency, token usage, and cost tracking.

`PerformanceEvaluator` scores individual cases against a per-query
budget (so it composes with the rest of the pipeline), and exposes
`aggregate()` for batch-level p50/p95/p99 latency and total cost,
which the reporter surfaces separately since those numbers only make
sense across a whole run.
"""

from __future__ import annotations

from src.evaluators.base import BaseEvaluator
from src.models import EvalCase, MetricResult
from src.utils.stats import mean, percentile


class PerformanceEvaluator(BaseEvaluator):
    name = "performance"
    default_threshold = 0.7

    def __init__(
        self,
        threshold: float | None = None,
        max_latency_ms: float = 3000,
        max_cost_usd: float = 0.05,
    ) -> None:
        super().__init__(threshold)
        self.max_latency_ms = max_latency_ms
        self.max_cost_usd = max_cost_usd

    @staticmethod
    def _budget_score(value: float, budget: float) -> float:
        if budget <= 0:
            return 1.0
        if value <= budget:
            return 1.0
        return max(0.0, 1 - (value - budget) / budget)

    def _score(self, case: EvalCase) -> MetricResult:
        latency = case.latency_ms
        cost = case.cost_usd
        if latency is None and cost is None:
            return MetricResult(
                name=self.name, score=1.0, passed=False, reason="no latency/cost data recorded", details={}
            )

        latency_score = self._budget_score(latency, self.max_latency_ms) if latency is not None else 1.0
        cost_score = self._budget_score(cost, self.max_cost_usd) if cost is not None else 1.0
        score = min(latency_score, cost_score)

        return MetricResult(
            name=self.name,
            score=score,
            passed=False,
            reason=f"latency={latency}ms cost=${cost}",
            details={
                "latency_ms": latency,
                "cost_usd": cost,
                "prompt_tokens": case.prompt_tokens,
                "completion_tokens": case.completion_tokens,
                "max_latency_ms": self.max_latency_ms,
                "max_cost_usd": self.max_cost_usd,
            },
        )

    @staticmethod
    def aggregate(cases: list[EvalCase]) -> dict:
        """Batch-level latency percentiles, token usage, and cost totals."""
        latencies = [c.latency_ms for c in cases if c.latency_ms is not None]
        costs = [c.cost_usd for c in cases if c.cost_usd is not None]
        prompt_tokens = [c.prompt_tokens for c in cases if c.prompt_tokens is not None]
        completion_tokens = [c.completion_tokens for c in cases if c.completion_tokens is not None]

        return {
            "num_cases": len(cases),
            "latency_ms": {
                "p50": round(percentile(latencies, 50), 2),
                "p95": round(percentile(latencies, 95), 2),
                "p99": round(percentile(latencies, 99), 2),
                "mean": round(mean(latencies), 2),
            }
            if latencies
            else {},
            "cost_usd": {
                "total": round(sum(costs), 6),
                "mean_per_query": round(mean(costs), 6),
            }
            if costs
            else {},
            "tokens": {
                "total_prompt_tokens": sum(prompt_tokens),
                "total_completion_tokens": sum(completion_tokens),
                "mean_prompt_tokens": round(mean(prompt_tokens), 2),
                "mean_completion_tokens": round(mean(completion_tokens), 2),
            }
            if (prompt_tokens or completion_tokens)
            else {},
        }
