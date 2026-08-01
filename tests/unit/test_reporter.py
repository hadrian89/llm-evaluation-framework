from src.evaluators.safety import PIIEvaluator
from src.pipeline.reporter import render_html, save_html, save_json
from src.pipeline.runner import EvaluationRunner


def test_render_html_contains_run_info(grounded_case):
    runner = EvaluationRunner([PIIEvaluator()])
    report = runner.run([grounded_case])
    html = render_html(report)
    assert report.run_id in html
    assert "safety_pii" in html
    assert "<table>" in html


def test_save_html_writes_file(tmp_path, grounded_case):
    runner = EvaluationRunner([PIIEvaluator()])
    report = runner.run([grounded_case])
    path = save_html(report, tmp_path / "sub" / "report.html")
    assert path.exists()
    assert "<!doctype html>" in path.read_text(encoding="utf-8")


def test_save_json_writes_valid_json(tmp_path, grounded_case):
    import json

    runner = EvaluationRunner([PIIEvaluator()])
    report = runner.run([grounded_case])
    path = save_json(report, tmp_path / "report.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["run_id"] == report.run_id
    assert data["num_cases"] == 1
