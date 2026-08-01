"""Pluggable LLM-as-judge client.

Evaluators that want a language model's judgment (e.g. faithfulness,
relevance) depend only on the `LLMClient` protocol below. Concrete
clients are guarded imports so the framework has zero hard dependency
on any specific SDK, and a deterministic `HeuristicClient` keeps
everything working with no API key configured at all.
"""

from __future__ import annotations

import json
import re
from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """Minimal interface evaluators need from an LLM judge."""

    def complete(self, prompt: str, *, temperature: float = 0.0) -> str:
        ...


class HeuristicClient:
    """A judge that never calls out to the network.

    It answers the narrow JSON-scoring prompts our evaluators send by
    falling back to lexical heuristics, so the framework is fully
    functional offline. This is intentionally conservative — real
    deployments should configure `OpenAIClient`/`AnthropicClient` for
    higher-fidelity judgments.
    """

    name = "heuristic"

    def complete(self, prompt: str, *, temperature: float = 0.0) -> str:
        from src.utils.text import cosine_similarity, token_overlap_ratio

        match = re.search(
            r"CLAIM:\s*(.*?)\nSOURCE:\s*(.*)", prompt, re.DOTALL
        )
        if match:
            # Claim verification uses stopword-filtered overlap (not raw
            # cosine similarity) so common words like "is"/"the"/"of" can't
            # mask an unsupported factual claim — same rigor as the
            # no-client fallback in `judge_claim`.
            claim, source = match.group(1).strip(), match.group(2).strip()
            score = token_overlap_ratio(claim, source)
            verdict = "supported" if score >= 0.35 else "unsupported"
            return json.dumps({"verdict": verdict, "score": round(score, 3)})

        match = re.search(
            r"QUESTION:\s*(.*?)\nANSWER:\s*(.*)", prompt, re.DOTALL
        )
        if match:
            question, answer = match.group(1).strip(), match.group(2).strip()
            score = cosine_similarity(question, answer)
            return json.dumps({"score": round(score, 3)})

        return json.dumps({"score": 0.0})


class OpenAIClient:
    """Thin wrapper around the OpenAI SDK. Requires `pip install openai`."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "OpenAIClient requires the 'openai' package: pip install openai"
            ) from exc
        self._client = OpenAI(api_key=api_key)
        self.model = model

    def complete(self, prompt: str, *, temperature: float = 0.0) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""


class AnthropicClient:
    """Thin wrapper around the Anthropic SDK. Requires `pip install anthropic`."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-5") -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "AnthropicClient requires the 'anthropic' package: pip install anthropic"
            ) from exc
        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def complete(self, prompt: str, *, temperature: float = 0.0) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in response.content if hasattr(block, "text"))


def build_llm_client(
    provider: str = "heuristic",
    *,
    api_key: str | None = None,
    model: str | None = None,
) -> LLMClient:
    """Factory used by settings-driven code paths (pipeline, API, examples)."""
    provider = (provider or "heuristic").lower()
    if provider == "openai":
        if not api_key:
            raise ValueError("OpenAI provider requires an api_key")
        return OpenAIClient(api_key=api_key, model=model or "gpt-4o-mini")
    if provider == "anthropic":
        if not api_key:
            raise ValueError("Anthropic provider requires an api_key")
        return AnthropicClient(api_key=api_key, model=model or "claude-sonnet-5")
    return HeuristicClient()
