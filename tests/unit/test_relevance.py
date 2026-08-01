from src.evaluators.relevance import (
    AnswerRelevanceEvaluator,
    ContextPrecisionEvaluator,
    ContextRecallEvaluator,
)
from src.models import EvalCase


def test_answer_relevance_on_topic(grounded_case):
    result = AnswerRelevanceEvaluator().evaluate(grounded_case)
    assert result.score > 0.5
    assert result.passed


def test_answer_relevance_off_topic():
    case = EvalCase(question="What is the capital of France?", answer="Bananas are a good source of potassium.")
    result = AnswerRelevanceEvaluator().evaluate(case)
    assert result.score < 0.3
    assert not result.passed


def test_answer_relevance_empty_answer():
    case = EvalCase(question="Q", answer="")
    result = AnswerRelevanceEvaluator().evaluate(case)
    assert result.score == 0.0


def test_context_precision_all_relevant():
    case = EvalCase(
        question="What is the capital of France?",
        answer="Paris.",
        contexts=["Paris is the capital of France.", "France's capital city is Paris."],
    )
    result = ContextPrecisionEvaluator().evaluate(case)
    assert result.score == 1.0


def test_context_precision_some_irrelevant():
    case = EvalCase(
        question="What is the capital of France?",
        answer="Paris.",
        contexts=["Paris is the capital of France.", "Bananas are a tropical fruit grown worldwide."],
    )
    result = ContextPrecisionEvaluator().evaluate(case)
    assert 0 < result.score < 1


def test_context_precision_no_contexts():
    case = EvalCase(question="Q", answer="A", contexts=[])
    result = ContextPrecisionEvaluator().evaluate(case)
    assert result.score == 0.0


def test_context_recall_full_coverage():
    case = EvalCase(
        question="Q",
        answer="A",
        contexts=["Paris is the capital of France."],
        ground_truth="Paris is the capital of France.",
    )
    result = ContextRecallEvaluator().evaluate(case)
    assert result.score == 1.0


def test_context_recall_missing_ground_truth():
    case = EvalCase(question="Q", answer="A", contexts=["Some context."], ground_truth=None)
    result = ContextRecallEvaluator().evaluate(case)
    assert result.score == 0.0
    assert not result.passed
