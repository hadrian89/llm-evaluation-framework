#!/usr/bin/env python
"""A/B compare two models' outputs on the same set of questions.

Fully offline by default using bundled sample datasets:

    python examples/compare_models.py --model-a gpt-4 --model-b claude-3

Pass --dataset-a/--dataset-b to compare your own captured outputs
(JSON in the same shape as examples/data/rag_sample.json).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.settings import get_settings, load_eval_config  # noqa: E402
from src.datasets.builder import EvalDataset  # noqa: E402
from src.evaluators.llm_client import build_llm_client  # noqa: E402
from src.pipeline.comparator import ModelComparator  # noqa: E402
from src.pipeline.runner import build_default_runner  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-a", default="model_a")
    parser.add_argument("--model-b", default="model_b")
    parser.add_argument("--dataset-a", default="examples/data/compare_baseline.json")
    parser.add_argument("--dataset-b", default="examples/data/compare_candidate.json")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    settings = get_settings()
    config = load_eval_config(args.config or settings.eval_config_path)
    llm_client = build_llm_client(settings.llm_judge_provider, api_key=settings.openai_api_key)

    dataset_a = EvalDataset.from_json(args.dataset_a)
    dataset_b = EvalDataset.from_json(args.dataset_b)

    runner = build_default_runner(config, llm_client=llm_client)
    comparator = ModelComparator(runner)
    comparison = comparator.compare(dataset_a, dataset_b, label_a=args.model_a, label_b=args.model_b)

    print(f"{args.model_a} pass rate: {comparison.report_a.pass_rate:.1%}")
    print(f"{args.model_b} pass rate: {comparison.report_b.pass_rate:.1%}\n")

    print(f"{'metric':32s} {args.model_a:>12s} {args.model_b:>12s} {'winner':>10s}")
    for c in comparison.comparisons:
        print(f"{c.metric:32s} {c.mean_a:12.3f} {c.mean_b:12.3f} {c.winner:>10s}")

    wins_b = sum(1 for c in comparison.comparisons if c.winner == args.model_b)
    wins_a = sum(1 for c in comparison.comparisons if c.winner == args.model_a)
    print(f"\n{args.model_a} won {wins_a} metrics, {args.model_b} won {wins_b} metrics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
