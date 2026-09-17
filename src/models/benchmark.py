"""
Benchmark orchestrator.

Coordinates the full pipeline for a set of models: hyperparameter tuning
via GridSearchCV, fitting, evaluation, and report generation. This is the
top-level entry point used by ``scripts`` and notebooks.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.evaluation.metrics import ClassificationMetrics, compute_classification_metrics
from src.models.base_model import BaseModel
from src.tuning.grid_search import GridSearchResult, tune_model

logger = logging.getLogger(__name__)


class BenchmarkError(Exception):
    """Raised when the benchmark orchestration fails."""


@dataclass
class ModelBenchmarkResult:
    """
    Result of benchmarking a single model.

    Attributes:
        model_name: Display name of the model.
        metrics: Test-set classification metrics.
        best_params: Winning hyperparameters from GridSearchCV.
        cv_score: Best cross-validated score found during tuning.
        tuning_time_seconds: Time spent on hyperparameter search.
        training_time_seconds: Time spent on the final fit with best params.
        fitted_model: The fitted :class:`BaseModel` wrapper instance.
    """

    model_name: str
    metrics: ClassificationMetrics
    best_params: Dict[str, Any]
    cv_score: float
    tuning_time_seconds: float
    training_time_seconds: float
    fitted_model: BaseModel


class Benchmark:
    """
    Orchestrates hyperparameter tuning, fitting, and evaluation across models.

    Args:
        models: List of unfit :class:`BaseModel` instances to benchmark.
        cv_folds: Number of stratified folds used during ``GridSearchCV``.
        scoring: Scoring metric used to select the best hyperparameters.
    """

    def __init__(
        self,
        models: List[BaseModel],
        cv_folds: int = 5,
        scoring: str = "f1",
    ) -> None:
        if not models:
            raise BenchmarkError("At least one model must be provided to Benchmark.")
        self.models = models
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.results: Dict[str, ModelBenchmarkResult] = {}

    def run(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, ModelBenchmarkResult]:
        """
        Run the full benchmark: tune, fit, and evaluate every registered model.

        Args:
            X_train: Training features (post feature engineering/selection/imbalance handling).
            y_train: Training labels.
            X_test: Held-out test features.
            y_test: Held-out test labels.

        Returns:
            Dictionary mapping model name to its :class:`ModelBenchmarkResult`.

        Raises:
            BenchmarkError: If every model fails to tune or fit.
        """
        failures: List[str] = []

        for model in self.models:
            logger.info("=== Benchmarking model: %s ===", model.name)
            try:
                result = self._run_single_model(model, X_train, y_train, X_test, y_test)
                self.results[model.name] = result
            except Exception as exc:  # noqa: BLE001
                logger.error("Benchmark failed for model '%s': %s", model.name, exc)
                failures.append(model.name)

        if len(failures) == len(self.models):
            raise BenchmarkError(f"All models failed to benchmark: {failures}")

        return self.results

    def _run_single_model(
        self,
        model: BaseModel,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> ModelBenchmarkResult:
        """Tune, fit, and evaluate a single model."""
        search_result: GridSearchResult = tune_model(
            model, X_train, y_train, cv_folds=self.cv_folds, scoring=self.scoring
        )

        start = time.perf_counter()
        model.set_estimator(search_result.best_estimator)
        training_time = time.perf_counter() - start

        y_pred = model.predict(X_test)
        y_proba = None
        try:
            y_proba = model.predict_proba(X_test)
        except Exception:  # noqa: BLE001
            logger.info("Model '%s' does not support predict_proba; skipping ROC-AUC.", model.name)

        metrics = compute_classification_metrics(y_test.to_numpy(), y_pred, y_proba)

        return ModelBenchmarkResult(
            model_name=model.name,
            metrics=metrics,
            best_params=search_result.best_params,
            cv_score=search_result.best_score,
            tuning_time_seconds=search_result.search_time_seconds,
            training_time_seconds=training_time,
            fitted_model=model,
        )

    def summary_dataframe(self) -> pd.DataFrame:
        """
        Build a comparison DataFrame across all benchmarked models.

        Returns:
            DataFrame indexed by model name with columns for accuracy,
            precision, recall, f1, roc_auc, cv_score, and tuning_time_seconds.

        Raises:
            BenchmarkError: If :meth:`run` has not been called yet.
        """
        if not self.results:
            raise BenchmarkError("No results available. Call run() before summary_dataframe().")

        rows = []
        for name, result in self.results.items():
            row = {
                "model": name,
                "accuracy": result.metrics.accuracy,
                "precision": result.metrics.precision,
                "recall": result.metrics.recall,
                "f1": result.metrics.f1,
                "roc_auc": result.metrics.roc_auc,
                "cv_score": result.cv_score,
                "tuning_time_seconds": result.tuning_time_seconds,
                "training_time_seconds": result.training_time_seconds,
            }
            rows.append(row)

        return pd.DataFrame(rows).set_index("model")

    def best_model(self, metric: str = "f1") -> ModelBenchmarkResult:
        """
        Return the best-performing model's result by a given metric.

        Args:
            metric: One of ``"accuracy"``, ``"precision"``, ``"recall"``,
                ``"f1"``, or ``"roc_auc"``.

        Returns:
            The :class:`ModelBenchmarkResult` with the highest value for ``metric``.

        Raises:
            BenchmarkError: If no results are available or ``metric`` is invalid.
        """
        if not self.results:
            raise BenchmarkError("No results available. Call run() before best_model().")

        valid_metrics = {"accuracy", "precision", "recall", "f1", "roc_auc"}
        if metric not in valid_metrics:
            raise BenchmarkError(f"Invalid metric '{metric}'. Must be one of {valid_metrics}.")

        scored = [
            (name, getattr(result.metrics, metric) or 0.0) for name, result in self.results.items()
        ]
        best_name, _ = max(scored, key=lambda item: item[1])
        return self.results[best_name]

    def save_report(self, output_path: str | Path = "reports/benchmark_results.json") -> Path:
        """
        Serialize benchmark results (metrics, params, timings) to a JSON file.

        Args:
            output_path: Destination path for the JSON report.

        Returns:
            Path to the saved JSON file.

        Raises:
            BenchmarkError: If no results are available to save.
        """
        if not self.results:
            raise BenchmarkError("No results available. Call run() before save_report().")

        payload = {
            name: {
                "metrics": result.metrics.to_dict(),
                "best_params": result.best_params,
                "cv_score": result.cv_score,
                "tuning_time_seconds": result.tuning_time_seconds,
                "training_time_seconds": result.training_time_seconds,
            }
            for name, result in self.results.items()
        }

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

        logger.info("Saved benchmark report to %s", out_path)
        return out_path
