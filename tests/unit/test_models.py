import pytest

from src.models import EvalCase, EvalResult, EvaluationReport, MetricResult


def test_eval_result_passed_requires_all_metrics():
    case = EvalCase(question="Q", answer="A")
    passing = MetricResult(name="m1", score=1.0, passed=True)
    failing = MetricResult(name="m2", score=0.0, passed=False)

    assert EvalResult(case=case, metrics=[passing]).passed
    assert not EvalResult(case=case, metrics=[passing, failing]).passed


def test_eval_result_metric_lookup():
    case = EvalCase(question="Q", answer="A")
    metric = MetricResult(name="faithfulness", score=0.9, passed=True)
    result = EvalResult(case=case, metrics=[metric])
    assert result.metric("faithfulness") is metric
    assert result.metric("missing") is None


def test_evaluation_report_summary():
    case = EvalCase(question="Q", answer="A")
    results = [
        EvalResult(case=case, metrics=[MetricResult(name="m", score=0.8, passed=True)]),
        EvalResult(case=case, metrics=[MetricResult(name="m", score=0.4, passed=False)]),
    ]
    report = EvaluationReport(results=results)
    assert report.num_cases == 2
    assert report.pass_rate == 0.5
    summary = report.metric_summary("m")
    assert summary["mean"] == pytest.approx(0.6)
    assert summary["min"] == 0.4
    assert summary["max"] == 0.8


def test_evaluation_report_empty():
    report = EvaluationReport()
    assert report.num_cases == 0
    assert report.pass_rate == 0.0
    assert report.metric_names() == []
