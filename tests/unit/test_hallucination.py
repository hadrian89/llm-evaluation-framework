from src.evaluators.hallucination import HallucinationEvaluator
from src.models import EvalCase


def test_grounded_answer_has_low_fabrication_rate(grounded_case):
    evaluator = HallucinationEvaluator()
    result = evaluator.evaluate(grounded_case)
    assert result.score <= 0.2
    assert result.passed


def test_hallucinated_answer_has_high_fabrication_rate(hallucinated_case):
    evaluator = HallucinationEvaluator()
    result = evaluator.evaluate(hallucinated_case)
    assert result.score > 0.2
    assert not result.passed
    assert result.details["unsupported_claims"]


def test_no_reference_material_is_fully_unverifiable():
    case = EvalCase(question="Q", answer="Some claim here.", contexts=[], ground_truth=None)
    result = HallucinationEvaluator().evaluate(case)
    assert result.score == 1.0
    assert not result.passed


def test_empty_answer():
    case = EvalCase(question="Q", answer="", contexts=["context"])
    result = HallucinationEvaluator().evaluate(case)
    assert result.score == 0.0


def test_lower_is_better_semantics():
    evaluator = HallucinationEvaluator()
    assert evaluator.higher_is_better is False
