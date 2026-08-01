from src.evaluators.faithfulness import FaithfulnessEvaluator
from src.models import EvalCase


def test_grounded_answer_scores_high(grounded_case):
    evaluator = FaithfulnessEvaluator()
    result = evaluator.evaluate(grounded_case)
    assert result.score >= 0.7
    assert result.passed
    assert result.name == "faithfulness"


def test_hallucinated_answer_scores_low(hallucinated_case):
    evaluator = FaithfulnessEvaluator()
    result = evaluator.evaluate(hallucinated_case)
    assert result.score < 0.7
    assert not result.passed


def test_no_contexts_returns_zero():
    case = EvalCase(question="Q", answer="Some answer.", contexts=[])
    result = FaithfulnessEvaluator().evaluate(case)
    assert result.score == 0.0
    assert not result.passed


def test_empty_answer_returns_zero():
    case = EvalCase(question="Q", answer="", contexts=["Some context."])
    result = FaithfulnessEvaluator().evaluate(case)
    assert result.score == 0.0


def test_custom_threshold():
    case = EvalCase(
        question="Q",
        answer="Paris is the capital of France.",
        contexts=["Paris is the capital of France."],
    )
    strict = FaithfulnessEvaluator(threshold=1.1)
    result = strict.evaluate(case)
    assert not result.passed
