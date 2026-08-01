from src.config.settings import load_eval_config
from src.pipeline.runner import EvaluationRunner, build_default_runner


def test_build_default_runner_produces_expected_metrics(grounded_case):
    config = load_eval_config("src/config/eval_config.yaml")
    runner = build_default_runner(config)
    report = runner.run([grounded_case])

    assert report.num_cases == 1
    expected_metrics = {
        "faithfulness",
        "answer_relevance",
        "context_precision",
        "context_recall",
        "hallucination_fabrication_rate",
        "safety_pii",
        "safety_toxicity",
        "safety_off_topic",
        "performance",
    }
    assert set(report.metric_names()) == expected_metrics


def test_grounded_case_passes_overall(grounded_case):
    config = load_eval_config("src/config/eval_config.yaml")
    runner = build_default_runner(config)
    report = runner.run([grounded_case])
    assert report.pass_rate == 1.0


def test_hallucinated_case_fails_hallucination_metric(hallucinated_case):
    config = load_eval_config("src/config/eval_config.yaml")
    runner = build_default_runner(config)
    report = runner.run([hallucinated_case])
    result = report.results[0]
    assert not result.metric("hallucination_fabrication_rate").passed


def test_empty_dataset_returns_empty_report():
    runner = build_default_runner({})
    report = runner.run([])
    assert report.num_cases == 0
    assert report.pass_rate == 0.0


def test_runner_with_custom_evaluator_list(grounded_case):
    from src.evaluators.safety import PIIEvaluator

    runner = EvaluationRunner([PIIEvaluator()])
    report = runner.run([grounded_case])
    assert report.metric_names() == ["safety_pii"]
