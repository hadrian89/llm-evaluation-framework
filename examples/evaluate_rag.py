#!/usr/bin/env python
"""Evaluate a RAG system's outputs against the full evaluator suite.

Runs fully offline by default (heuristic scoring, no API key needed):

    python examples/evaluate_rag.py --dataset examples/data/rag_sample.json

Set LLM_JUDGE_PROVIDER=openai (and OPENAI_API_KEY) for LLM-as-judge scoring.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.settings import get_settings, load_eval_config  # noqa: E402
from src.datasets.builder import EvalDataset  # noqa: E402
from src.evaluators.llm_client import build_llm_client  # noqa: E402
from src.pipeline.reporter import save_html, save_json  # noqa: E402
from src.pipeline.runner import build_default_runner  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="examples/data/rag_sample.json")
    parser.add_argument("--config", default=None, help="Path to eval_config.yaml (defaults to settings)")
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--fail-under", type=float, default=None, help="Exit 1 if pass_rate is below this")
    args = parser.parse_args()

    settings = get_settings()
    config = load_eval_config(args.config or settings.eval_config_path)

    provider = settings.llm_judge_provider
    api_key = settings.openai_api_key if provider == "openai" else settings.anthropic_api_key
    llm_client = build_llm_client(provider, api_key=api_key, model=settings.llm_judge_model)

    dataset = EvalDataset.from_json(args.dataset)
    runner = build_default_runner(config, llm_client=llm_client)
    report = runner.run(dataset)

    summary = report.summary()
    print(f"Run {report.run_id} — {summary['num_cases']} cases — pass rate {summary['pass_rate']:.1%}")
    for name, stats in summary["metrics"].items():
        print(f"  {name:35s} mean={stats['mean']:.3f}  min={stats['min']:.3f}  max={stats['max']:.3f}")

    report_dir = Path(args.report_dir)
    json_path = save_json(report, report_dir / f"{report.run_id}.json")
    html_path = save_html(report, report_dir / f"{report.run_id}.html")
    print(f"\nReports written to {json_path} and {html_path}")

    if args.fail_under is not None and report.pass_rate < args.fail_under:
        print(f"\nFAIL: pass rate {report.pass_rate:.1%} is below threshold {args.fail_under:.1%}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
