"""Tests for GridSearchCV tuning wrapper and evaluation utilities."""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.datasets import make_classification

from src.evaluation.cross_validation import CrossValidationError, stratified_cv_scores
from src.evaluation.metrics import MetricsError, compute_classification_metrics
from src.models.random_forest import RandomForestModel
from src.tuning.grid_search import GridSearchResult, TuningError, tune_model


@pytest.fixture
def small_classification_data():
    X, y = make_classification(n_samples=200, n_features=6, n_informative=4, random_state=42)
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    y_series = pd.Series(y)
    return X_df, y_series


class TestTuneModel:
    def test_returns_grid_search_result(self, small_classification_data):
        X, y = small_classification_data
        model = RandomForestModel(
            param_grid_override={"n_estimators": [50, 100], "max_depth": [None, 5]}
        )
        result = tune_model(model, X, y, cv_folds=3, scoring="f1")
        assert isinstance(result, GridSearchResult)
        assert result.best_estimator is not None
        assert 0.0 <= result.best_score <= 1.0

    def test_best_params_within_grid(self, small_classification_data):
        X, y = small_classification_data
        grid = {"n_estimators": [50, 100]}
        model = RandomForestModel(param_grid_override=grid)
        result = tune_model(model, X, y, cv_folds=3, scoring="f1")
        assert result.best_params["n_estimators"] in grid["n_estimators"]

    def test_cv_results_is_dataframe(self, small_classification_data):
        X, y = small_classification_data
        model = RandomForestModel(param_grid_override={"n_estimators": [50]})
        result = tune_model(model, X, y, cv_folds=3, scoring="f1")
        assert isinstance(result.cv_results, pd.DataFrame)
        assert len(result.cv_results) >= 1


class TestClassificationMetrics:
    def test_perfect_predictions(self):
        y_true = [0, 1, 0, 1, 1]
        y_pred = [0, 1, 0, 1, 1]
        metrics = compute_classification_metrics(y_true, y_pred)
        assert metrics.accuracy == 1.0
        assert metrics.f1 == 1.0

    def test_length_mismatch_raises(self):
        with pytest.raises(MetricsError):
            compute_classification_metrics([0, 1, 1], [0, 1])

    def test_confusion_matrix_shape(self):
        y_true = [0, 1, 0, 1]
        y_pred = [0, 0, 0, 1]
        metrics = compute_classification_metrics(y_true, y_pred)
        assert len(metrics.confusion_matrix) == 2
        assert len(metrics.confusion_matrix[0]) == 2


class TestStratifiedCVScores:
    def test_returns_expected_metrics(self, small_classification_data):
        X, y = small_classification_data
        model = RandomForestModel().build_estimator()
        results = stratified_cv_scores(model, X, y, cv_folds=3)
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            assert metric in results
            assert "mean" in results[metric]
            assert "std" in results[metric]
