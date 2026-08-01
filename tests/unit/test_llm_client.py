import json

import pytest

from src.evaluators.claim_judge import judge_claim
from src.evaluators.llm_client import HeuristicClient, build_llm_client


def test_heuristic_client_claim_prompt():
    client = HeuristicClient()
    prompt = "CLAIM: Paris is the capital of France\nSOURCE: Paris is the capital and largest city of France."
    raw = client.complete(prompt)
    parsed = json.loads(raw)
    assert parsed["verdict"] == "supported"


def test_heuristic_client_unsupported_claim():
    client = HeuristicClient()
    prompt = "CLAIM: Bananas are blue\nSOURCE: Paris is the capital of France."
    parsed = json.loads(client.complete(prompt))
    assert parsed["verdict"] == "unsupported"


def test_heuristic_client_fallback_for_unknown_prompt():
    client = HeuristicClient()
    parsed = json.loads(client.complete("something unstructured"))
    assert parsed["score"] == 0.0


def test_build_llm_client_defaults_to_heuristic():
    client = build_llm_client()
    assert isinstance(client, HeuristicClient)


def test_build_llm_client_openai_requires_key():
    with pytest.raises(ValueError):
        build_llm_client("openai")


def test_judge_claim_offline_supported():
    supported, score = judge_claim("Paris is the capital", "Paris is the capital of France", llm_client=None)
    assert supported
    assert score == 1.0


def test_judge_claim_offline_unsupported():
    supported, score = judge_claim("Bananas are blue", "Paris is the capital of France", llm_client=None)
    assert not supported


def test_judge_claim_with_heuristic_client():
    client = HeuristicClient()
    supported, score = judge_claim("Paris is the capital", "Paris is the capital of France", llm_client=client)
    assert supported
