"""Hallucination detection via claim decomposition and verification.

Unlike `FaithfulnessEvaluator` (which scores overall grounding), this
evaluator reports the *fabrication rate* — the share of claims in the
answer that are unsupported by both the retrieved contexts and any
supplied ground truth. Lower is better, so `higher_is_better = False`.
"""

from __future__ import annotations

from src.evaluators.base import BaseEvaluator
from src.evaluators.claim_judge import judge_claim
from src.evaluators.llm_client import LLMClient
from src.models import EvalCase, MetricResult
from src.utils.text import split_sentences


class HallucinationEvaluator(BaseEvaluator):
    name = "hallucination_fabrication_rate"
    default_threshold = 0.2
    higher_is_better = False

    def __init__(
        self,
        threshold: float | None = None,
        llm_client: LLMClient | None = None,
        support_threshold: float = 0.35,
    ) -> None:
        super().__init__(threshold)
        self.llm_client = llm_client
        self.support_threshold = support_threshold

    def _is_fabricated(self, claim: str, reference: str) -> tuple[bool, float]:
        supported, score = judge_claim(claim, reference, self.llm_client, self.support_threshold)
        return not supported, score

    def _score(self, case: EvalCase) -> MetricResult:
        claims = split_sentences(case.answer)
        if not claims:
            return MetricResult(
                name=self.name, score=0.0, passed=False, reason="empty answer, nothing to verify", details={}
            )

        reference_parts = list(case.contexts)
        if case.ground_truth:
            reference_parts.append(case.ground_truth)
        reference = "\n".join(reference_parts)

        if not reference.strip():
            return MetricResult(
                name=self.name,
                score=1.0,
                passed=False,
                reason="no contexts or ground_truth to verify claims against",
                details={"num_claims": len(claims), "unsupported_claims": [c for c in claims]},
            )

        unsupported = []
        for claim in claims:
            fabricated, confidence = self._is_fabricated(claim, reference)
            if fabricated:
                unsupported.append({"claim": claim, "confidence": round(confidence, 4)})

        fabrication_rate = len(unsupported) / len(claims)
        return MetricResult(
            name=self.name,
            score=fabrication_rate,
            passed=False,
            reason=f"{len(unsupported)}/{len(claims)} claims unsupported by context/ground truth",
            details={"num_claims": len(claims), "unsupported_claims": unsupported},
        )
