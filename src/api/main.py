"""FastAPI application exposing the evaluation pipeline over HTTP."""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI

from src import __version__
from src.api.schemas import (
    CompareRequest,
    CompareResponse,
    ConsistencyRequest,
    ConsistencyResponse,
    EvalCaseIn,
    EvalResultOut,
    EvaluateRequest,
    EvaluateResponse,
    HealthResponse,
    MetricComparisonOut,
    MetricResultOut,
)
from src.config.settings import get_settings, load_eval_config
from src.evaluators.consistency import ConsistencyEvaluator
from src.evaluators.llm_client import build_llm_client
from src.models import EvalCase
from src.pipeline.comparator import ModelComparator
from src.pipeline.runner import EvaluationRunner, build_default_runner

app = FastAPI(
    title="LLM Evaluation Framework API",
    description="Faithfulness, relevance, hallucination, safety, performance, and consistency evaluation.",
    version=__version__,
)


@lru_cache
def get_runner() -> EvaluationRunner:
    settings = get_settings()
    config = load_eval_config(settings.eval_config_path)
    judge_cfg = config.get("judge", {})
    provider = judge_cfg.get("provider", settings.llm_judge_provider)
    api_key = settings.openai_api_key if provider == "openai" else settings.anthropic_api_key
    llm_client = build_llm_client(provider, api_key=api_key, model=settings.llm_judge_model)
    return build_default_runner(config, llm_client=llm_client)


def _to_eval_case(case_in: EvalCaseIn) -> EvalCase:
    return EvalCase(**case_in.model_dump())


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(version=__version__)


@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate(request: EvaluateRequest) -> EvaluateResponse:
    runner = get_runner()
    cases = [_to_eval_case(c) for c in request.cases]
    report = runner.run(cases)
    summary = report.summary()
    return EvaluateResponse(
        run_id=report.run_id,
        num_cases=report.num_cases,
        pass_rate=report.pass_rate,
        metrics=summary["metrics"],
        results=[
            EvalResultOut(
                case_id=r.case.case_id,
                question=r.case.question,
                answer=r.case.answer,
                passed=r.passed,
                metrics=[MetricResultOut(**m.to_dict()) for m in r.metrics],
            )
            for r in report.results
        ],
    )


@app.post("/compare", response_model=CompareResponse)
def compare(request: CompareRequest) -> CompareResponse:
    runner = get_runner()
    comparator = ModelComparator(runner)
    cases_a = [_to_eval_case(c) for c in request.cases_a]
    cases_b = [_to_eval_case(c) for c in request.cases_b]
    result = comparator.compare(cases_a, cases_b, label_a=request.label_a, label_b=request.label_b)
    data = result.to_dict()
    return CompareResponse(
        label_a=data["label_a"],
        label_b=data["label_b"],
        pass_rate_a=data["pass_rate_a"],
        pass_rate_b=data["pass_rate_b"],
        comparisons=[MetricComparisonOut(**c) for c in data["comparisons"]],
    )


@app.post("/consistency", response_model=ConsistencyResponse)
def consistency(request: ConsistencyRequest) -> ConsistencyResponse:
    settings = get_settings()
    config = load_eval_config(settings.eval_config_path)
    threshold = config.get("thresholds", {}).get("consistency")
    evaluator = ConsistencyEvaluator(threshold=threshold)
    result = evaluator.evaluate_answers(request.question, request.answers)
    return ConsistencyResponse(
        score=result.score,
        passed=result.passed,
        threshold=result.threshold or evaluator.default_threshold,
        reason=result.reason,
        details=result.details,
    )
