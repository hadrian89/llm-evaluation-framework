"""Evaluation dataset construction from in-memory records, JSON, or CSV."""

from __future__ import annotations

import csv
import json
import random
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from src.models import EvalCase


class EvalDataset:
    """An ordered, filterable collection of `EvalCase` objects."""

    def __init__(self, cases: Iterable[EvalCase] | None = None) -> None:
        self._cases: list[EvalCase] = list(cases) if cases else []

    def __len__(self) -> int:
        return len(self._cases)

    def __iter__(self) -> Iterator[EvalCase]:
        return iter(self._cases)

    def __getitem__(self, idx: int) -> EvalCase:
        return self._cases[idx]

    @property
    def cases(self) -> list[EvalCase]:
        return list(self._cases)

    def add(self, case: EvalCase) -> None:
        self._cases.append(case)

    def extend(self, cases: Iterable[EvalCase]) -> None:
        self._cases.extend(cases)

    def filter(self, predicate) -> EvalDataset:
        return EvalDataset([c for c in self._cases if predicate(c)])

    def sample(self, n: int, seed: int | None = None) -> EvalDataset:
        rng = random.Random(seed)
        n = min(n, len(self._cases))
        return EvalDataset(rng.sample(self._cases, n))

    @classmethod
    def from_records(cls, records: list[dict[str, Any]]) -> EvalDataset:
        cases = []
        for record in records:
            record = dict(record)
            question = record.pop("question")
            answer = record.pop("answer")
            contexts = record.pop("contexts", []) or []
            if isinstance(contexts, str):
                contexts = [contexts]
            cases.append(
                EvalCase(
                    question=question,
                    answer=answer,
                    contexts=contexts,
                    ground_truth=record.pop("ground_truth", None),
                    latency_ms=record.pop("latency_ms", None),
                    prompt_tokens=record.pop("prompt_tokens", None),
                    completion_tokens=record.pop("completion_tokens", None),
                    cost_usd=record.pop("cost_usd", None),
                    model=record.pop("model", None),
                    metadata=record,
                )
            )
        return cls(cases)

    @classmethod
    def from_json(cls, path: str | Path) -> EvalDataset:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("cases", [])
        return cls.from_records(data)

    @classmethod
    def from_csv(cls, path: str | Path, context_delimiter: str = "|") -> EvalDataset:
        records = []
        with Path(path).open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                row = dict(row)
                if row.get("contexts"):
                    row["contexts"] = [c.strip() for c in row["contexts"].split(context_delimiter) if c.strip()]
                records.append(row)
        return cls.from_records(records)

    def to_json(self, path: str | Path) -> None:
        payload = {"cases": [c.to_dict() for c in self._cases]}
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def to_records(self) -> list[dict[str, Any]]:
        return [c.to_dict() for c in self._cases]
