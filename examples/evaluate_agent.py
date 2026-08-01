#!/usr/bin/env python
"""Evaluate an agent across repeated runs of the same input.

Demonstrates the multi-run consistency check plus the standard
per-case metrics (faithfulness, hallucination, safety) applied to
each individual run. Fully offline by default:

    python examples/evaluate_agent.py --dataset examples/data/agent_sample.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.settings import get_settings, load_eval_config  # noqa: E402
from src.evaluators.consistency import ConsistencyEvaluator  # noqa: E402
from src.evaluators.llm_client import build_llm_client  # noqa: E402
from src.models import EvalCase  # noqa: E402
from src.pipeline.runner import build_default_runner  # noqa: E402


def load_agent_runs(path: str) -> tuple[list[EvalCase], str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    question = data["question"]
    contexts = data.get("contexts", [])
    ground_truth = data.get("ground_truth")
    cases = [
        EvalCase(
            question=question,
            answer=run["answer"],
            contexts=contexts,
            ground_truth=ground_truth,
            latency_ms=run.get("latency_ms"),
            prompt_tokens=run.get("prompt_tokens"),
            completion_tokens=run.get("completion_tokens"),
            cost_usd=run.get("cost_usd"),
            metadata={"tool_calls": run.get("tool_calls", [])},
        )
        for run in data["runs"]
    ]
    return cases, question


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="examples/data/agent_sample.json")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    settings = get_settings()
    config = load_eval_config(args.config or settings.eval_config_path)
    llm_client = build_llm_client(settings.llm_judge_provider, api_key=settings.openai_api_key)

    cases, question = load_agent_runs(args.dataset)
    print(f"Question: {question}")
    print(f"Runs: {len(cases)}\n")

    runner = build_default_runner(config, llm_client=llm_client)
    report = runner.run(cases)
    for i, result in enumerate(report.results, start=1):
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] Run {i}: {result.case.answer[:90]}")
        for metric in result.metrics:
            if not metric.passed:
                print(f"          ✗ {metric.name}={metric.score:.3f} (threshold {metric.threshold})")

    consistency_threshold = config.get("thresholds", {}).get("consistency")
    consistency = ConsistencyEvaluator(threshold=consistency_threshold).evaluate_cases(cases)
    print(f"\nConsistency across {len(cases)} runs: {consistency.score:.3f} "
          f"({'PASS' if consistency.passed else 'FAIL'}, threshold {consistency.threshold})")
    if not consistency.passed:
        print(f"  min pairwise similarity: {consistency.details['min_similarity']:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
