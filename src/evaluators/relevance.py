"""Answer relevance and context precision/recall evaluation.

Three focused evaluators, each producing its own named metric so they
compose cleanly with the pipeline runner and the config thresholds in
`eval_config.yaml` (`answer_relevance`, `context_precision`,
`context_recall`).
"""

from __future__ import annotations

import json

from src.evaluators.base import BaseEvaluator
from src.evaluators.llm_client import LLMClient
from src.models import EvalCase, MetricResult
from src.utils.text import cosine_similarity

RELEVANCE_JUDGE_PROMPT = """You are an evaluator judging how relevant an ANSWER is to a QUESTION.
Score from 0 (completely irrelevant) to 1 (fully addresses the question).
Respond with ONLY compact JSON: {{"score": <0-1 float>}}

QUESTION: {question}
ANSWER: {answer}
"""


class AnswerRelevanceEvaluator(BaseEvaluator):
    """How well the answer addresses the question (embedding/LLM similarity)."""

    name = "answer_relevance"
    default_threshold = 0.7

    def __init__(self, threshold: float | None = None, llm_client: LLMClient | None = None) -> None:
        super().__init__(threshold)
        self.llm_client = llm_client

    def _score(self, case: EvalCase) -> MetricResult:
        if not case.answer.strip():
            return MetricResult(name=self.name, score=0.0, passed=False, reason="empty answer", details={})

        if self.llm_client is not None:
            raw = self.llm_client.complete(
                RELEVANCE_JUDGE_PROMPT.format(question=case.question, answer=case.answer)
            )
            try:
                score = float(json.loads(raw).get("score", 0.0))
            except (json.JSONDecodeError, TypeError, ValueError):
                score = cosine_similarity(case.question, case.answer)
        else:
            score = cosine_similarity(case.question, case.answer)

        return MetricResult(
            name=self.name,
            score=score,
            passed=False,
            reason="similarity between question and answer",
            details={"method": "llm_judge" if self.llm_client else "cosine_similarity"},
        )


class ContextPrecisionEvaluator(BaseEvaluator):
    """Fraction of retrieved contexts that are actually relevant to the question."""

    name = "context_precision"
    default_threshold = 0.6

    def __init__(self, threshold: float | None = None, relevance_cutoff: float = 0.15) -> None:
        super().__init__(threshold)
        self.relevance_cutoff = relevance_cutoff

    def _score(self, case: EvalCase) -> MetricResult:
        if not case.contexts:
            return MetricResult(name=self.name, score=0.0, passed=False, reason="no contexts", details={})
        per_context = [cosine_similarity(case.question, ctx) for ctx in case.contexts]
        relevant = sum(1 for s in per_context if s >= self.relevance_cutoff)
        score = relevant / len(case.contexts)
        return MetricResult(
            name=self.name,
            score=score,
            passed=False,
            reason=f"{relevant}/{len(case.contexts)} contexts relevant to the question",
            details={"per_context_scores": [round(s, 4) for s in per_context]},
        )


class ContextRecallEvaluator(BaseEvaluator):
    """Fraction of the ground-truth content that is covered by the retrieved contexts."""

    name = "context_recall"
    default_threshold = 0.6

    def _score(self, case: EvalCase) -> MetricResult:
        if not case.ground_truth:
            return MetricResult(
                name=self.name, score=0.0, passed=False, reason="no ground_truth provided", details={}
            )
        if not case.contexts:
            return MetricResult(name=self.name, score=0.0, passed=False, reason="no contexts", details={})

        from src.utils.text import split_sentences, token_overlap_ratio

        combined_context = "\n".join(case.contexts)
        gt_sentences = split_sentences(case.ground_truth) or [case.ground_truth]
        covered = [token_overlap_ratio(s, combined_context) >= 0.4 for s in gt_sentences]
        score = sum(covered) / len(gt_sentences)
        return MetricResult(
            name=self.name,
            score=score,
            passed=False,
            reason=f"{sum(covered)}/{len(gt_sentences)} ground-truth statements covered by context",
            details={"num_ground_truth_statements": len(gt_sentences)},
        )
