"""HTML and JSON report generation for evaluation runs."""

from __future__ import annotations

import json
from pathlib import Path

from jinja2 import Template

from src.models import EvaluationReport

_HTML_TEMPLATE = Template(
    """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Evaluation Report — {{ report.run_id }}</title>
<style>
  body { font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif; margin: 2rem; color: #1a1a1a; background: #fafafa; }
  h1 { margin-bottom: 0.25rem; }
  .meta { color: #666; margin-bottom: 1.5rem; }
  .summary-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 12px; margin-bottom: 2rem; }
  .card { background: white; border: 1px solid #e2e2e2; border-radius: 8px; padding: 12px 16px; }
  .card .label { font-size: 0.75rem; text-transform: uppercase; color: #888; letter-spacing: .04em; }
  .card .value { font-size: 1.5rem; font-weight: 600; }
  .pass { color: #1a7f37; }
  .fail { color: #c62828; }
  table { border-collapse: collapse; width: 100%; background: white; }
  th, td { border: 1px solid #e2e2e2; padding: 6px 10px; text-align: left; font-size: 0.85rem; }
  th { background: #f2f2f2; position: sticky; top: 0; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 0.75rem; font-weight: 600; }
  .badge.pass { background: #e6f4ea; color: #1a7f37; }
  .badge.fail { background: #fdecea; color: #c62828; }
  .scroll { overflow-x: auto; max-height: 70vh; overflow-y: auto; }
</style>
</head>
<body>
  <h1>Evaluation Report</h1>
  <div class="meta">Run <code>{{ report.run_id }}</code> · {{ report.num_cases }} cases · pass rate
    <strong class="{{ 'pass' if report.pass_rate >= 0.8 else 'fail' }}">{{ '%.1f' % (report.pass_rate * 100) }}%</strong>
  </div>

  <div class="summary-grid">
    {% for name, s in summary.metrics.items() %}
    <div class="card">
      <div class="label">{{ name }}</div>
      <div class="value">{{ '%.3f' % s.mean }}</div>
      <div class="label">min {{ '%.2f' % s.min }} · max {{ '%.2f' % s.max }} · n={{ s.count }}</div>
    </div>
    {% endfor %}
  </div>

  <div class="scroll">
  <table>
    <thead>
      <tr>
        <th>#</th>
        <th>Question</th>
        <th>Answer</th>
        {% for name in metric_names %}<th>{{ name }}</th>{% endfor %}
        <th>Result</th>
      </tr>
    </thead>
    <tbody>
      {% for result in report.results %}
      <tr>
        <td>{{ loop.index }}</td>
        <td>{{ result.case.question[:120] }}</td>
        <td>{{ result.case.answer[:160] }}</td>
        {% for name in metric_names %}
        {% set m = result.metric(name) %}
        <td>{{ '%.3f' % m.score if m else '—' }}</td>
        {% endfor %}
        <td><span class="badge {{ 'pass' if result.passed else 'fail' }}">{{ 'PASS' if result.passed else 'FAIL' }}</span></td>
      </tr>
      {% endfor %}
    </tbody>
  </table>
  </div>
</body>
</html>
"""
)


def render_html(report: EvaluationReport) -> str:
    return _HTML_TEMPLATE.render(
        report=report,
        summary=report.summary(),
        metric_names=report.metric_names(),
    )


def save_html(report: EvaluationReport, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(report), encoding="utf-8")
    return path


def save_json(report: EvaluationReport, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return path
