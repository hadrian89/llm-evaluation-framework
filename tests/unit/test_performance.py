from src.evaluators.performance import PerformanceEvaluator
from src.models import EvalCase


def test_within_budget_scores_one():
    case = EvalCase(question="Q", answer="A", latency_ms=500, cost_usd=0.001)
    result = PerformanceEvaluator(max_latency_ms=3000, max_cost_usd=0.05).evaluate(case)
    assert result.score == 1.0
    assert result.passed


def test_over_latency_budget_degrades_score():
    case = EvalCase(question="Q", answer="A", latency_ms=6000, cost_usd=0.001)
    result = PerformanceEvaluator(max_latency_ms=3000, max_cost_usd=0.05).evaluate(case)
    assert result.score < 1.0


def test_over_cost_budget_degrades_score():
    case = EvalCase(question="Q", answer="A", latency_ms=500, cost_usd=0.2)
    result = PerformanceEvaluator(max_latency_ms=3000, max_cost_usd=0.05).evaluate(case)
    assert result.score < 1.0


def test_no_data_recorded():
    case = EvalCase(question="Q", answer="A")
    result = PerformanceEvaluator().evaluate(case)
    assert result.score == 1.0
    assert "no latency/cost" in result.reason


def test_aggregate_percentiles():
    cases = [
        EvalCase(question="Q", answer="A", latency_ms=ms, cost_usd=0.001, prompt_tokens=10, completion_tokens=5)
        for ms in [100, 200, 300, 400, 500]
    ]
    agg = PerformanceEvaluator.aggregate(cases)
    assert agg["num_cases"] == 5
    assert agg["latency_ms"]["p50"] == 300
    assert agg["cost_usd"]["total"] == 0.005
    assert agg["tokens"]["total_prompt_tokens"] == 50


def test_aggregate_empty_cases():
    agg = PerformanceEvaluator.aggregate([])
    assert agg["num_cases"] == 0
    assert agg["latency_ms"] == {}
