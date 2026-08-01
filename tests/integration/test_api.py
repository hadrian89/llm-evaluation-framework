from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_evaluate_endpoint_returns_metrics():
    payload = {
        "cases": [
            {
                "question": "What is the capital of France?",
                "answer": "The capital of France is Paris.",
                "contexts": ["Paris is the capital and most populous city of France."],
                "ground_truth": "Paris is the capital of France.",
                "latency_ms": 500,
                "cost_usd": 0.001,
            }
        ]
    }
    response = client.post("/evaluate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["num_cases"] == 1
    assert "faithfulness" in data["metrics"]
    assert len(data["results"]) == 1
    assert len(data["results"][0]["metrics"]) > 0


def test_evaluate_endpoint_empty_cases():
    response = client.post("/evaluate", json={"cases": []})
    assert response.status_code == 200
    assert response.json()["num_cases"] == 0


def test_compare_endpoint():
    payload = {
        "label_a": "baseline",
        "label_b": "candidate",
        "cases_a": [{"question": "Q", "answer": "Contact me at test@example.com"}],
        "cases_b": [{"question": "Q", "answer": "No personal info shared here."}],
    }
    response = client.post("/compare", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["label_a"] == "baseline"
    assert data["label_b"] == "candidate"
    pii_comparison = next(c for c in data["comparisons"] if c["metric"] == "safety_pii")
    assert pii_comparison["winner"] == "candidate"


def test_consistency_endpoint():
    payload = {
        "question": "What is the capital of France?",
        "answers": ["Paris is the capital.", "Paris is the capital of France."],
    }
    response = client.post("/consistency", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["score"] > 0.5


def test_evaluate_endpoint_rejects_missing_required_field():
    response = client.post("/evaluate", json={"cases": [{"question": "Q"}]})
    assert response.status_code == 422
