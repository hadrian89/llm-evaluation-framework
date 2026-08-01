"""Optional LangSmith integration for pushing evaluation results as traces.

Guarded import: the `langsmith` package is an optional extra
(`pip install -e ".[langsmith]"`). Without it, or without an API key
configured, `LangSmithReporter.log_report()` is a documented no-op so
the rest of the pipeline never breaks because of a missing extra.
"""

from __future__ import annotations

import logging

from src.models import EvaluationReport

logger = logging.getLogger(__name__)


class LangSmithReporter:
    def __init__(self, api_key: str | None = None, project: str = "llm-evaluation-framework") -> None:
        self.api_key = api_key
        self.project = project
        self._client = None
        if api_key:
            try:
                from langsmith import Client

                self._client = Client(api_key=api_key)
            except ImportError:
                logger.warning(
                    "LangSmith integration requested but 'langsmith' is not installed; "
                    "install with pip install -e '.[langsmith]'. Skipping."
                )

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def log_report(self, report: EvaluationReport) -> None:
        if not self.enabled:
            logger.info("LangSmith not configured; skipping trace upload for run %s", report.run_id)
            return

        for result in report.results:
            self._client.create_feedback(
                run_id=result.case.metadata.get("langsmith_run_id"),
                key="evaluation",
                score=1.0 if result.passed else 0.0,
                value={m.name: m.score for m in result.metrics},
                comment=f"eval run {report.run_id}",
            )
