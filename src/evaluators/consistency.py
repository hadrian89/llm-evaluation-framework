"""Multi-run consistency (response variance) analysis.

Given several responses generated from the *same* input (e.g. by
re-running a model at temperature > 0), this scores how similar those
responses are to one another via mean pairwise lexical similarity.
Unlike the other evaluators this operates on a *group* of cases rather
than a single one, so it intentionally does not subclass
`BaseEvaluator`.
"""

from __future__ import annotations

from itertools import combinations

from src.models import EvalCase, MetricResult
from src.utils.stats import mean
from src.utils.text import cosine_similarity


class ConsistencyEvaluator:
    name = "consistency"
    default_threshold = 0.7

    def __init__(self, threshold: float | None = None) -> None:
        self.threshold = self.default_threshold if threshold is None else threshold

    def evaluate_answers(self, question: str, answers: list[str]) -> MetricResult:
        answers = [a for a in answers if a and a.strip()]
        if len(answers) < 2:
            return MetricResult(
                name=self.name,
                score=1.0,
                passed=True,
                threshold=self.threshold,
                reason="fewer than 2 runs supplied; nothing to compare",
                details={"num_runs": len(answers)},
            )

        pairwise = [cosine_similarity(a, b) for a, b in combinations(answers, 2)]
        score = mean(pairwise)
        return MetricResult(
            name=self.name,
            score=score,
            passed=score >= self.threshold,
            threshold=self.threshold,
            reason=f"mean pairwise similarity across {len(answers)} runs",
            details={
                "num_runs": len(answers),
                "pairwise_similarities": [round(s, 4) for s in pairwise],
                "min_similarity": round(min(pairwise), 4),
            },
        )

    def evaluate_cases(self, cases: list[EvalCase]) -> MetricResult:
        """Convenience overload when runs are already `EvalCase` objects
        that share the same `question`."""
        if not cases:
            return MetricResult(
                name=self.name, score=0.0, passed=False, threshold=self.threshold, reason="no cases", details={}
            )
        return self.evaluate_answers(cases[0].question, [c.answer for c in cases])
