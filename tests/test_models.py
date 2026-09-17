"""Tests for model wrappers and the benchmark orchestrator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from src.models.base_model import ModelError
from src.models.benchmark import Benchmark, BenchmarkError
from src.models.random_forest import RandomForestModel
from src.models.svm_model import SVMModel
from src.models.xgboost_model import XGBoostModel


@pytest.fixture
def small_classification_data():
    X, y = make_classification(
        n_samples=300, n_features=8, n_informative=5, random_state=42
    )
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])
    y_series = pd.Series(y)
    split = 240
    return X_df.iloc[:split], X_df.iloc[split:], y_series.iloc[:split], y_series.iloc[split:]


class TestRandomForestModel:
    def test_fit_predict(self, small_classification_data):
        X_train, X_test, y_train, y_test = small_classification_data
        model = RandomForestModel()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        assert len(preds) == len(X_test)

    def test_predict_before_fit_raises(self, small_classification_data):
        _, X_test, _, _ = small_classification_data
        model = RandomForestModel()
        with pytest.raises(ModelError):
            model.predict(X_test)

    def test_param_grid_not_empty(self):
        model = RandomForestModel()
        grid = model.param_grid()
        assert len(grid) > 0

    def test_save_and_load(self, small_classification_data, tmp_path):
        X_train, _, y_train, _ = small_classification_data
        model = RandomForestModel()
        model.fit(X_train, y_train)
        path = model.save(tmp_path)
        assert path.exists()

        loaded = RandomForestModel.load(path, name="random_forest")
        assert loaded.is_fitted


class TestXGBoostModel:
    def test_fit_predict(self, small_classification_data):
        X_train, X_test, y_train, y_test = small_classification_data
        model = XGBoostModel()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        assert len(preds) == len(X_test)

    def test_predict_proba_shape(self, small_classification_data):
        X_train, X_test, y_train, _ = small_classification_data
        model = XGBoostModel()
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)
        assert proba.shape == (len(X_test), 2)


class TestSVMModel:
    def test_fit_predict(self, small_classification_data):
        X_train, X_test, y_train, y_test = small_classification_data
        model = SVMModel()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        assert len(preds) == len(X_test)

    def test_probability_enabled_by_default(self, small_classification_data):
        X_train, X_test, y_train, _ = small_classification_data
        model = SVMModel()
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)
        assert proba.shape[1] == 2


class TestBenchmark:
    def test_requires_at_least_one_model(self):
        with pytest.raises(BenchmarkError):
            Benchmark(models=[])

    def test_run_produces_results_for_each_model(self, small_classification_data):
        X_train, X_test, y_train, y_test = small_classification_data
        models = [
            RandomForestModel(param_grid_override={"n_estimators": [50]}),
            SVMModel(param_grid_override={"C": [1.0]}),
        ]
        benchmark = Benchmark(models=models, cv_folds=3, scoring="f1")
        results = benchmark.run(X_train, y_train, X_test, y_test)
        assert set(results.keys()) == {"random_forest", "svm"}

    def test_summary_dataframe_has_expected_columns(self, small_classification_data):
        X_train, X_test, y_train, y_test = small_classification_data
        models = [RandomForestModel(param_grid_override={"n_estimators": [50]})]
        benchmark = Benchmark(models=models, cv_folds=3, scoring="f1")
        benchmark.run(X_train, y_train, X_test, y_test)
        summary = benchmark.summary_dataframe()
        for col in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            assert col in summary.columns

    def test_best_model_selects_highest_metric(self, small_classification_data):
        X_train, X_test, y_train, y_test = small_classification_data
        models = [
            RandomForestModel(param_grid_override={"n_estimators": [50]}),
            SVMModel(param_grid_override={"C": [1.0]}),
        ]
        benchmark = Benchmark(models=models, cv_folds=3, scoring="f1")
        benchmark.run(X_train, y_train, X_test, y_test)
        best = benchmark.best_model(metric="f1")
        summary = benchmark.summary_dataframe()
        assert best.metrics.f1 == summary["f1"].max()

    def test_summary_before_run_raises(self):
        models = [RandomForestModel()]
        benchmark = Benchmark(models=models)
        with pytest.raises(BenchmarkError):
            benchmark.summary_dataframe()
