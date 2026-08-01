"""Evaluation pipeline orchestrator.

`EvaluationRunner` runs a configurable list of per-case evaluators over
an `EvalDataset` and assembles an `EvaluationReport`. Use
`build_default_runner()` to get the standard evaluator set wired up
from `eval_config.yaml` and whichever LLM judge (if any) is configured
in `Settings`.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from src.datasets.builder import EvalDataset
from src.evaluators.base import BaseEvaluator
from src.evaluators.faithfulness import FaithfulnessEvaluator
from src.evaluators.hallucination import HallucinationEvaluator
from src.evaluators.llm_client import LLMClient, build_llm_client
from src.evaluators.performance import PerformanceEvaluator
from src.evaluators.relevance import (
    AnswerRelevanceEvaluator,
    ContextPrecisionEvaluator,
    ContextRecallEvaluator,
)
from src.evaluators.safety import OffTopicEvaluator, PIIEvaluator, ToxicityEvaluator
from src.models import EvalCase, EvalResult, EvaluationReport


class EvaluationRunner:
    def __init__(self, evaluators: Iterable[BaseEvaluator]) -> None:
        self.evaluators: list[BaseEvaluator] = list(evaluators)

    def run_case(self, case: EvalCase) -> EvalResult:
        metrics = [evaluator.evaluate(case) for evaluator in self.evaluators]
        return EvalResult(case=case, metrics=metrics)

    def run(self, dataset: EvalDataset | list[EvalCase]) -> EvaluationReport:
        cases = list(dataset)
        results = [self.run_case(case) for case in cases]
        report = EvaluationReport(results=results)
        return report


def build_default_runner(
    config: dict[str, Any] | None = None,
    llm_client: LLMClient | None = None,
) -> EvaluationRunner:
    """Assemble the standard evaluator suite from `eval_config.yaml`-shaped config."""
    config = config or {}
    thresholds = config.get("thresholds", {})
    performance_cfg = config.get("performance", {})
    safety_cfg = config.get("safety", {})
    judge_cfg = config.get("judge", {})

    if llm_client is None:
        llm_client = build_llm_client(judge_cfg.get("provider", "heuristic"))

    evaluators: list[BaseEvaluator] = [
        FaithfulnessEvaluator(threshold=thresholds.get("faithfulness"), llm_client=llm_client),
        AnswerRelevanceEvaluator(threshold=thresholds.get("answer_relevance"), llm_client=llm_client),
        ContextPrecisionEvaluator(threshold=thresholds.get("context_precision")),
        ContextRecallEvaluator(threshold=thresholds.get("context_recall")),
        HallucinationEvaluator(
            threshold=thresholds.get("hallucination_fabrication_rate"), llm_client=llm_client
        ),
        PIIEvaluator(threshold=thresholds.get("safety_pii"), categories=safety_cfg.get("pii_categories")),
        ToxicityEvaluator(threshold=thresholds.get("safety_toxicity")),
        OffTopicEvaluator(threshold=thresholds.get("safety_off_topic")),
        PerformanceEvaluator(
            max_latency_ms=performance_cfg.get("latency_p95_ms_max", 3000),
            max_cost_usd=performance_cfg.get("cost_per_query_usd_max", 0.05),
        ),
    ]
    return EvaluationRunner(evaluators)
