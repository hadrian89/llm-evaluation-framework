"""Faithfulness (grounding) evaluation.

Decomposes the answer into individual claims (sentences) and checks
each one against the retrieved contexts, either lexically (default,
offline) or via an LLM judge when one is configured. The metric score
is the fraction of claims that are grounded in the provided contexts —
this doubles as a proxy for source-attribution accuracy.
"""

from __future__ import annotations

from src.evaluators.base import BaseEvaluator
from src.evaluators.claim_judge import judge_claim
from src.evaluators.llm_client import LLMClient
from src.models import EvalCase, MetricResult
from src.utils.text import best_match_score, split_sentences, token_overlap_ratio


class FaithfulnessEvaluator(BaseEvaluator):
    name = "faithfulness"
    default_threshold = 0.7

    def __init__(
        self,
        threshold: float | None = None,
        llm_client: LLMClient | None = None,
        claim_support_threshold: float = 0.35,
    ) -> None:
        super().__init__(threshold)
        self.llm_client = llm_client
        self.claim_support_threshold = claim_support_threshold

    def _judge_claim(self, claim: str, context: str) -> tuple[bool, float]:
        return judge_claim(claim, context, self.llm_client, self.claim_support_threshold)

    def _score(self, case: EvalCase) -> MetricResult:
        if not case.contexts:
            return MetricResult(
                name=self.name,
                score=0.0,
                passed=False,
                reason="no contexts provided; cannot assess grounding",
                details={"claims": []},
            )
        claims = split_sentences(case.answer)
        if not claims:
            return MetricResult(
                name=self.name, score=0.0, passed=False, reason="empty answer", details={"claims": []}
            )

        combined_context = "\n".join(case.contexts)
        claim_details = []
        supported_count = 0
        for claim in claims:
            best_score, best_idx = best_match_score(claim, case.contexts, scorer=token_overlap_ratio)
            supported, judge_score = self._judge_claim(claim, combined_context)
            if supported:
                supported_count += 1
            claim_details.append(
                {
                    "claim": claim,
                    "supported": supported,
                    "score": round(judge_score, 4),
                    "best_matching_context_index": best_idx,
                    "lexical_overlap": round(best_score, 4),
                }
            )

        grounding_score = supported_count / len(claims)
        return MetricResult(
            name=self.name,
            score=grounding_score,
            passed=False,
            reason=f"{supported_count}/{len(claims)} claims grounded in provided context",
            details={"claims": claim_details, "num_claims": len(claims)},
        )
