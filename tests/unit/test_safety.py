from src.evaluators.safety import OffTopicEvaluator, PIIEvaluator, SafetyEvaluator, ToxicityEvaluator
from src.models import EvalCase


def test_pii_detected(pii_case):
    result = PIIEvaluator().evaluate(pii_case)
    assert result.score == 0.0
    assert not result.passed
    assert "email" in result.details["found"]


def test_pii_clean(grounded_case):
    result = PIIEvaluator().evaluate(grounded_case)
    assert result.score == 1.0
    assert result.passed


def test_pii_category_filter():
    case = EvalCase(question="Q", answer="Email me at test@example.com", contexts=[])
    result = PIIEvaluator(categories=["ssn"]).evaluate(case)
    assert result.score == 1.0  # email ignored since only 'ssn' is checked


def test_toxicity_detected():
    case = EvalCase(question="Q", answer="You are so stupid, shut up.", contexts=[])
    result = ToxicityEvaluator().evaluate(case)
    assert result.score == 0.0
    assert not result.passed


def test_toxicity_clean(grounded_case):
    result = ToxicityEvaluator().evaluate(grounded_case)
    assert result.score == 1.0


def test_off_topic_scoring(grounded_case):
    result = OffTopicEvaluator().evaluate(grounded_case)
    assert result.score > 0


def test_safety_evaluator_runs_all_checks(grounded_case):
    results = SafetyEvaluator().evaluate(grounded_case)
    names = {r.name for r in results}
    assert names == {"safety_pii", "safety_toxicity", "safety_off_topic"}
