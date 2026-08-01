"""Synthetic test case generation from source documents.

Offline mode (default) turns each source passage into a templated
Q&A pair using the passage itself as both context and ground truth —
useful for smoke-testing a pipeline or bootstrapping a golden set
before human review. Pass an `LLMClient` for higher-quality, more
natural questions.
"""

from __future__ import annotations

import json

from src.datasets.golden_set import GoldenSetManager
from src.evaluators.llm_client import LLMClient
from src.utils.text import split_sentences

QUESTION_GEN_PROMPT = """Generate one clear, specific question that is fully answered by the PASSAGE below.
Also provide a concise ground-truth answer drawn only from the passage.
Respond with ONLY compact JSON: {{"question": "...", "answer": "..."}}

PASSAGE: {passage}
"""

_TEMPLATES = [
    "What does the following passage say about {topic}?",
    "According to the source material, what is stated regarding {topic}?",
    "Can you summarize what is known about {topic}?",
]


def _guess_topic(passage: str) -> str:
    sentences = split_sentences(passage)
    first = sentences[0] if sentences else passage
    words = first.split()
    return " ".join(words[:6]).rstrip(".,;:") or "this topic"


class SyntheticGenerator:
    def __init__(self, llm_client: LLMClient | None = None, seed: int = 0) -> None:
        self.llm_client = llm_client
        self._seed = seed

    def _generate_offline(self, passage: str, index: int) -> tuple[str, str]:
        topic = _guess_topic(passage)
        template = _TEMPLATES[index % len(_TEMPLATES)]
        question = template.format(topic=topic)
        answer = passage.strip()
        return question, answer

    def _generate_with_llm(self, passage: str) -> tuple[str, str]:
        raw = self.llm_client.complete(QUESTION_GEN_PROMPT.format(passage=passage))
        try:
            parsed = json.loads(raw)
            return parsed["question"], parsed["answer"]
        except (json.JSONDecodeError, KeyError, TypeError):
            return self._generate_offline(passage, 0)

    def generate(self, passages: list[str], tags: list[str] | None = None) -> GoldenSetManager:
        manager = GoldenSetManager()
        for i, passage in enumerate(passages):
            passage = passage.strip()
            if not passage:
                continue
            if self.llm_client is not None:
                question, answer = self._generate_with_llm(passage)
            else:
                question, answer = self._generate_offline(passage, i)
            manager.add(
                question=question,
                ground_truth=answer,
                contexts=[passage],
                tags=tags or ["synthetic"],
            )
        return manager

    def generate_from_document(
        self, document: str, tags: list[str] | None = None, chunk_size: int = 3
    ) -> GoldenSetManager:
        """Chunk a longer document into `chunk_size`-sentence passages first."""
        sentences = split_sentences(document)
        chunks = [
            " ".join(sentences[i : i + chunk_size]) for i in range(0, len(sentences), chunk_size)
        ]
        return self.generate(chunks, tags=tags)
