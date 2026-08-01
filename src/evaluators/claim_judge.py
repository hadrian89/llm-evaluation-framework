"""Shared claim-vs-source verification used by faithfulness and hallucination.

Both evaluators need the same primitive — "is this claim supported by
this source text?" — so it lives in one place with one JSON schema
(`{"verdict": "supported"|"unsupported", "score": <0-1>}`). Keeping a
single schema means every `LLMClient` implementation, including the
offline `HeuristicClient`, only needs to answer one shape of question.
"""

from __future__ import annotations

import json

from src.evaluators.llm_client import LLMClient
from src.utils.text import token_overlap_ratio

CLAIM_JUDGE_PROMPT = """You are a strict fact-checking judge.
Given a CLAIM and a SOURCE passage, decide whether the SOURCE supports the CLAIM.
Respond with ONLY compact JSON: {{"verdict": "supported" | "unsupported", "score": <0-1 float>}}

CLAIM: {claim}
SOURCE: {source}
"""


def judge_claim(
    claim: str,
    source: str,
    llm_client: LLMClient | None,
    support_threshold: float = 0.35,
) -> tuple[bool, float]:
    """Returns (is_supported, score)."""
    if llm_client is None:
        score = token_overlap_ratio(claim, source)
        return score >= support_threshold, score

    raw = llm_client.complete(CLAIM_JUDGE_PROMPT.format(claim=claim, source=source))
    try:
        parsed = json.loads(raw)
        score = float(parsed.get("score", 0.0))
        supported = str(parsed.get("verdict", "unsupported")).lower() == "supported"
        return supported, score
    except (json.JSONDecodeError, TypeError, ValueError):
        score = token_overlap_ratio(claim, source)
        return score >= support_threshold, score
