import pytest

from src.utils.pii import find_pii, toxicity_hits
from src.utils.stats import mean, percentile
from src.utils.text import (
    content_tokens,
    cosine_similarity,
    jaccard_similarity,
    split_sentences,
    token_overlap_ratio,
    tokenize,
)


def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("Hello, World!") == ["hello", "world"]


def test_content_tokens_removes_stopwords():
    tokens = content_tokens("The quick brown fox is in the box")
    assert "the" not in tokens
    assert "quick" in tokens


def test_split_sentences_basic():
    sentences = split_sentences("Paris is the capital. It is in France!")
    assert sentences == ["Paris is the capital.", "It is in France!"]


def test_split_sentences_empty_string():
    assert split_sentences("") == []
    assert split_sentences(None) == []


def test_cosine_similarity_identical_text_is_one():
    score = cosine_similarity("Paris is the capital of France", "Paris is the capital of France")
    assert score == pytest.approx(1.0)


def test_cosine_similarity_unrelated_text_is_low():
    score = cosine_similarity("Paris is the capital of France", "Bananas are yellow fruit")
    assert score < 0.2


def test_cosine_similarity_empty_string_is_zero():
    assert cosine_similarity("", "something") == 0.0
    assert cosine_similarity("something", "") == 0.0


def test_jaccard_similarity_bounds():
    score = jaccard_similarity("the cat sat on the mat", "the cat sat on a rug")
    assert 0 < score < 1


def test_token_overlap_ratio_full_support():
    assert token_overlap_ratio("Paris is the capital", "Paris is the capital of France") == 1.0


def test_token_overlap_ratio_no_support():
    assert token_overlap_ratio("bananas are yellow", "Paris is the capital of France") == 0.0


def test_percentile_matches_known_values():
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert percentile(values, 50) == 5.5
    assert percentile(values, 0) == 1
    assert percentile(values, 100) == 10


def test_percentile_empty_list():
    assert percentile([], 95) == 0.0


def test_mean_helper():
    assert mean([1, 2, 3]) == 2.0
    assert mean([]) == 0.0


def test_find_pii_detects_email_and_phone():
    found = find_pii("Contact me at test@example.com or 555-123-4567")
    assert "email" in found
    assert "phone" in found


def test_find_pii_clean_text():
    assert find_pii("The weather is nice today") == {}


def test_toxicity_hits_detects_terms():
    assert "stupid" in toxicity_hits("That was a stupid idea")


def test_toxicity_hits_clean_text():
    assert toxicity_hits("That was a great idea") == []
