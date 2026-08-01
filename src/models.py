"""Shared data models used across evaluators, pipeline, and API layers."""

from __future__ import annotations

import statistics
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvalCase:
    """A single question/answer instance to evaluate."""

    question: str
    answer: str
    contexts: list[str] = field(default_factory=list)
    ground_truth: str | None = None
    latency_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None
    model: str | None = None
    case_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "question": self.question,
            "answer": self.answer,
            "contexts": self.contexts,
            "ground_truth": self.ground_truth,
            "latency_ms": self.latency_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": self.cost_usd,
            "model": self.model,
            "metadata": self.metadata,
        }


@dataclass
class MetricResult:
    """The outcome of a single evaluator run against a single case."""

    name: str
    score: float
    passed: bool
    threshold: float | None = None
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "score": round(self.score, 4),
            "passed": self.passed,
            "threshold": self.threshold,
            "reason": self.reason,
            "details": self.details,
        }


@dataclass
class EvalResult:
    """All metric results for a single case."""

    case: EvalCase
    metrics: list[MetricResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(m.passed for m in self.metrics)

    def metric(self, name: str) -> MetricResult | None:
        return next((m for m in self.metrics if m.name == name), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case": self.case.to_dict(),
            "metrics": [m.to_dict() for m in self.metrics],
            "passed": self.passed,
        }


@dataclass
class EvaluationReport:
    """Aggregate report across many evaluated cases."""

    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: float = field(default_factory=time.time)
    results: list[EvalResult] = field(default_factory=list)

    @property
    def num_cases(self) -> int:
        return len(self.results)

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.passed) / len(self.results)

    def metric_names(self) -> list[str]:
        names: list[str] = []
        for r in self.results:
            for m in r.metrics:
                if m.name not in names:
                    names.append(m.name)
        return names

    def metric_summary(self, name: str) -> dict[str, float]:
        scores = [m.score for r in self.results for m in r.metrics if m.name == name]
        if not scores:
            return {}
        summary = {
            "mean": statistics.fmean(scores),
            "min": min(scores),
            "max": max(scores),
            "count": len(scores),
        }
        if len(scores) > 1:
            summary["stdev"] = statistics.pstdev(scores)
        else:
            summary["stdev"] = 0.0
        return summary

    def summary(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "num_cases": self.num_cases,
            "pass_rate": round(self.pass_rate, 4),
            "metrics": {name: self.metric_summary(name) for name in self.metric_names()},
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.summary(),
            "results": [r.to_dict() for r in self.results],
        }
