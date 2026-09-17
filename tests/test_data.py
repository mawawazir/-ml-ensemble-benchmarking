"""Tests for data loading and preprocessing."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.load_data import DataLoadError, load_raw_data, train_test_split_data
from src.data.preprocess import PreprocessError, preprocess_pipeline


class TestLoadRawData:
    def test_synthesizes_reproducible_dataset(self):
        df1 = load_raw_data(n_samples=500, n_features=10)
        df2 = load_raw_data(n_samples=500, n_features=10)
        pd.testing.assert_frame_equal(df1, df2)

    def test_synthesized_dataset_has_target_column(self):
        df = load_raw_data(n_samples=200, n_features=5, target_column="target")
        assert "target" in df.columns

    def test_synthesized_dataset_shape(self):
        df = load_raw_data(n_samples=300, n_features=8)
        # +1 target, +2 categorical columns
        assert df.shape[0] == 300
        assert df.shape[1] == 8 + 1 + 2

    def test_raises_on_missing_file(self, tmp_path):
        missing_path = tmp_path / "does_not_exist.csv"
        with pytest.raises(DataLoadError):
            load_raw_data(path=missing_path)

    def test_raises_on_missing_target_column(self, tmp_path):
        csv_path = tmp_path / "data.csv"
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(csv_path, index=False)
        with pytest.raises(DataLoadError):
            load_raw_data(path=csv_path, target_column="target")


class TestTrainTestSplit:
    def test_split_shapes(self):
        df = load_raw_data(n_samples=1000, n_features=5)
        X_train, X_test, y_train, y_test = train_test_split_data(df, test_size=0.2)
        assert len(X_train) + len(X_test) == len(df)
        assert abs(len(X_test) / len(df) - 0.2) < 0.02

    def test_stratification_preserves_class_ratio(self):
        df = load_raw_data(n_samples=2000, n_features=5)
        _, _, y_train, y_test = train_test_split_data(df, test_size=0.2, stratify=True)
        train_ratio = y_train.mean()
        test_ratio = y_test.mean()
        assert abs(train_ratio - test_ratio) < 0.05

    def test_raises_on_missing_target(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        with pytest.raises(DataLoadError):
            train_test_split_data(df, target_column="target")


class TestPreprocessPipeline:
    def test_output_has_no_missing_values(self):
        df = load_raw_data(n_samples=500, n_features=5)
        X_train, X_test, y_train, y_test = train_test_split_data(df)
        X_train_out, X_test_out, _ = preprocess_pipeline(X_train, X_test)
        assert not X_train_out.isnull().any().any()
        assert not X_test_out.isnull().any().any()

    def test_categorical_columns_are_encoded(self):
        df = load_raw_data(n_samples=300, n_features=5)
        X_train, X_test, _, _ = train_test_split_data(df)
        X_train_out, _, _ = preprocess_pipeline(X_train, X_test)
        assert all(X_train_out.dtypes.apply(lambda d: d.kind in "fi"))

    def test_raises_on_empty_dataframe(self):
        empty_df = pd.DataFrame()
        with pytest.raises(PreprocessError):
            preprocess_pipeline(empty_df)
