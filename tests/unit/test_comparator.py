from src.evaluators.safety import PIIEvaluator
from src.models import EvalCase
from src.pipeline.comparator import ModelComparator
from src.pipeline.runner import EvaluationRunner


def test_comparator_picks_higher_is_better_winner():
    runner = EvaluationRunner([PIIEvaluator()])
    comparator = ModelComparator(runner)

    cases_a = [EvalCase(question="Q", answer="Contact me at test@example.com")]  # leaks PII -> score 0
    cases_b = [EvalCase(question="Q", answer="No personal info here.")]  # clean -> score 1

    report = comparator.compare(cases_a, cases_b, label_a="A", label_b="B")
    comparison = next(c for c in report.comparisons if c.metric == "safety_pii")
    assert comparison.winner == "B"
    assert comparison.mean_a == 0.0
    assert comparison.mean_b == 1.0


def test_comparator_respects_lower_is_better():
    from src.evaluators.hallucination import HallucinationEvaluator

    runner = EvaluationRunner([HallucinationEvaluator()])
    comparator = ModelComparator(runner)

    cases_a = [
        EvalCase(question="Q", answer="Totally made up nonsense claim.", contexts=["Real fact about the world."])
    ]
    cases_b = [
        EvalCase(question="Q", answer="Real fact about the world.", contexts=["Real fact about the world."])
    ]

    report = comparator.compare(cases_a, cases_b, label_a="A", label_b="B")
    comparison = report.comparisons[0]
    assert comparison.winner == "B"  # lower fabrication rate wins


def test_comparator_tie():
    runner = EvaluationRunner([PIIEvaluator()])
    comparator = ModelComparator(runner)
    cases = [EvalCase(question="Q", answer="Clean answer.")]
    report = comparator.compare(cases, cases, label_a="A", label_b="B")
    assert report.comparisons[0].winner == "tie"


def test_comparison_report_to_dict():
    runner = EvaluationRunner([PIIEvaluator()])
    comparator = ModelComparator(runner)
    cases = [EvalCase(question="Q", answer="Clean answer.")]
    report = comparator.compare(cases, cases)
    data = report.to_dict()
    assert data["label_a"] == "model_a"
    assert "comparisons" in data
