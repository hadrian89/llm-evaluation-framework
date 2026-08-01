from src.evaluators.consistency import ConsistencyEvaluator
from src.models import EvalCase


def test_identical_answers_are_fully_consistent():
    evaluator = ConsistencyEvaluator()
    result = evaluator.evaluate_answers("Q", ["Paris is the capital.", "Paris is the capital."])
    assert result.score == 1.0
    assert result.passed


def test_divergent_answers_are_inconsistent():
    evaluator = ConsistencyEvaluator()
    result = evaluator.evaluate_answers(
        "Q", ["Paris is the capital of France.", "Bananas are a tropical fruit grown worldwide."]
    )
    assert result.score < 0.3
    assert not result.passed


def test_single_run_is_trivially_consistent():
    evaluator = ConsistencyEvaluator()
    result = evaluator.evaluate_answers("Q", ["Only one answer."])
    assert result.score == 1.0
    assert result.details["num_runs"] == 1


def test_evaluate_cases_wrapper():
    cases = [
        EvalCase(question="Q", answer="Paris is the capital."),
        EvalCase(question="Q", answer="Paris is the capital city."),
    ]
    result = ConsistencyEvaluator().evaluate_cases(cases)
    assert result.score > 0.5
