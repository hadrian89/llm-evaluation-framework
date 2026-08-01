import pytest

from src.models import EvalCase


@pytest.fixture
def grounded_case() -> EvalCase:
    return EvalCase(
        question="What is the capital of France?",
        answer="The capital of France is Paris.",
        contexts=[
            "Paris is the capital and most populous city of France.",
        ],
        ground_truth="Paris is the capital of France.",
        latency_ms=800,
        cost_usd=0.002,
        prompt_tokens=120,
        completion_tokens=25,
    )


@pytest.fixture
def hallucinated_case() -> EvalCase:
    return EvalCase(
        question="What is the capital of France?",
        answer="The capital of France is Berlin, and it has a population of 50 million people on Mars.",
        contexts=["Paris is the capital and most populous city of France."],
        ground_truth="Paris is the capital of France.",
    )


@pytest.fixture
def pii_case() -> EvalCase:
    return EvalCase(
        question="What is your contact info?",
        answer="You can reach support at help@example.com or call 555-123-4567.",
        contexts=["Our support team can be reached via the contact page."],
    )
