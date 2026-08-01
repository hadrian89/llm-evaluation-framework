"""Optional MLflow integration for logging evaluation runs as experiments.

Guarded import: `mlflow` is an optional extra
(`pip install -e ".[mlflow]"`). Without it, or without a tracking URI
configured, `MLflowReporter.log_report()` is a documented no-op.
"""

from __future__ import annotations

import logging

from src.models import EvaluationReport

logger = logging.getLogger(__name__)


class MLflowReporter:
    def __init__(self, tracking_uri: str | None = None, experiment: str = "llm-evaluation-framework") -> None:
        self.tracking_uri = tracking_uri
        self.experiment = experiment
        self._mlflow = None
        if tracking_uri:
            try:
                import mlflow

                mlflow.set_tracking_uri(tracking_uri)
                mlflow.set_experiment(experiment)
                self._mlflow = mlflow
            except ImportError:
                logger.warning(
                    "MLflow integration requested but 'mlflow' is not installed; "
                    "install with pip install -e '.[mlflow]'. Skipping."
                )

    @property
    def enabled(self) -> bool:
        return self._mlflow is not None

    def log_report(self, report: EvaluationReport) -> None:
        if not self.enabled:
            logger.info("MLflow not configured; skipping experiment log for run %s", report.run_id)
            return

        summary = report.summary()
        with self._mlflow.start_run(run_name=report.run_id):
            self._mlflow.log_param("num_cases", summary["num_cases"])
            self._mlflow.log_metric("pass_rate", summary["pass_rate"])
            for name, stats in summary["metrics"].items():
                for stat_name, value in stats.items():
                    self._mlflow.log_metric(f"{name}_{stat_name}", value)
