"""Golden Q&A set management.

A "golden set" is the curated reference dataset (question, expected
answer/ground truth, supporting contexts) used as the basis for
evaluation runs. It's distinct from `EvalCase`, which additionally
carries the *actual* answer produced by the system under test — golden
entries get turned into `EvalCase`s once you have those answers, via
`to_eval_cases()`.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.models import EvalCase


@dataclass
class GoldenSetEntry:
    question: str
    ground_truth: str
    contexts: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    entry_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "question": self.question,
            "ground_truth": self.ground_truth,
            "contexts": self.contexts,
            "tags": self.tags,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GoldenSetEntry:
        return cls(
            question=data["question"],
            ground_truth=data["ground_truth"],
            contexts=data.get("contexts", []) or [],
            tags=data.get("tags", []) or [],
            entry_id=data.get("entry_id") or uuid.uuid4().hex[:12],
            metadata=data.get("metadata", {}) or {},
        )


class GoldenSetManager:
    def __init__(self, entries: list[GoldenSetEntry] | None = None) -> None:
        self.entries: list[GoldenSetEntry] = entries or []

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    def add(
        self,
        question: str,
        ground_truth: str,
        contexts: list[str] | None = None,
        tags: list[str] | None = None,
        **metadata: Any,
    ) -> GoldenSetEntry:
        entry = GoldenSetEntry(
            question=question,
            ground_truth=ground_truth,
            contexts=contexts or [],
            tags=tags or [],
            metadata=metadata,
        )
        self.entries.append(entry)
        return entry

    def by_tag(self, tag: str) -> list[GoldenSetEntry]:
        return [e for e in self.entries if tag in e.tags]

    def get(self, entry_id: str) -> GoldenSetEntry | None:
        return next((e for e in self.entries if e.entry_id == entry_id), None)

    @classmethod
    def load(cls, path: str | Path) -> GoldenSetManager:
        path = Path(path)
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        entries = [GoldenSetEntry.from_dict(e) for e in data.get("entries", [])]
        return cls(entries)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"entries": [e.to_dict() for e in self.entries]}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def to_eval_cases(self, answers: dict[str, str], **defaults: Any) -> list[EvalCase]:
        """Pair golden entries with the system-under-test's actual answers.

        `answers` maps `entry_id -> generated answer`. Entries without a
        matching answer are skipped.
        """
        cases = []
        for entry in self.entries:
            if entry.entry_id not in answers:
                continue
            cases.append(
                EvalCase(
                    question=entry.question,
                    answer=answers[entry.entry_id],
                    contexts=entry.contexts,
                    ground_truth=entry.ground_truth,
                    metadata={"tags": entry.tags, "entry_id": entry.entry_id, **entry.metadata},
                    **defaults,
                )
            )
        return cases
