"""Tests for feature engineering, selection, and imbalance handling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import FeatureEngineeringError, engineer_features
from src.features.imbalance import ImbalanceHandlingError, apply_class_weights, apply_smote
from src.features.selection import (
    FeatureSelectionError,
    remove_low_variance_features,
    select_features,
    select_k_best_features,
)


@pytest.fixture
def numeric_df():
    rng = np.random.RandomState(42)
    return pd.DataFrame(
        {
            "feature_0": rng.randn(200),
            "feature_1": rng.randn(200),
            "feature_2": rng.randn(200),
        }
    )


@pytest.fixture
def imbalanced_classification_data():
    rng = np.random.RandomState(42)
    X = pd.DataFrame(rng.randn(500, 5), columns=[f"f{i}" for i in range(5)])
    y = pd.Series([0] * 450 + [1] * 50)
    return X, y


class TestEngineerFeatures:
    def test_adds_expected_feature_types(self, numeric_df):
        out = engineer_features(numeric_df, max_interaction_pairs=3)
        assert any(col.endswith("_squared") for col in out.columns)
        assert any("_x_" in col for col in out.columns)
        assert "row_mean" in out.columns

    def test_output_has_more_columns_than_input(self, numeric_df):
        out = engineer_features(numeric_df)
        assert out.shape[1] > numeric_df.shape[1]

    def test_raises_on_no_numeric_columns(self):
        df = pd.DataFrame({"cat": ["a", "b", "c"]})
        with pytest.raises(FeatureEngineeringError):
            engineer_features(df, numeric_columns=[])


class TestFeatureSelection:
    def test_remove_low_variance_drops_constant_column(self, numeric_df):
        df = numeric_df.copy()
        df["constant"] = 1.0
        filtered, dropped = remove_low_variance_features(df, threshold=0.0)
        assert "constant" in dropped
        assert "constant" not in filtered.columns

    def test_select_k_best_returns_k_columns(self, imbalanced_classification_data):
        X, y = imbalanced_classification_data
        reduced, cols = select_k_best_features(X, y, k=3)
        assert reduced.shape[1] == 3
        assert len(cols) == 3

    def test_select_features_importance_method(self, imbalanced_classification_data):
        X, y = imbalanced_classification_data
        reduced, cols = select_features(X, y, method="importance", k=3)
        assert reduced.shape[1] <= 3
        assert len(cols) == reduced.shape[1]

    def test_select_features_invalid_method_raises(self, imbalanced_classification_data):
        X, y = imbalanced_classification_data
        with pytest.raises(FeatureSelectionError):
            select_features(X, y, method="not_a_real_method")


class TestImbalanceHandling:
    def test_smote_balances_classes(self, imbalanced_classification_data):
        X, y = imbalanced_classification_data
        X_res, y_res = apply_smote(X, y)
        counts = y_res.value_counts()
        assert counts.iloc[0] == counts.iloc[1]

    def test_smote_output_shapes_match(self, imbalanced_classification_data):
        X, y = imbalanced_classification_data
        X_res, y_res = apply_smote(X, y)
        assert len(X_res) == len(y_res)

    def test_class_weights_sum_reasonable(self, imbalanced_classification_data):
        _, y = imbalanced_classification_data
        weights = apply_class_weights(y)
        assert set(weights.keys()) == {0, 1}
        assert weights[1] > weights[0]  # minority class gets higher weight

    def test_class_weights_raises_on_single_class(self):
        y = pd.Series([0, 0, 0, 0])
        with pytest.raises(ImbalanceHandlingError):
            apply_class_weights(y)
