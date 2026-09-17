"""
Preprocessing pipeline: missing value imputation, encoding, and scaling.

The pipeline is built with scikit-learn's ``ColumnTransformer`` /
``Pipeline`` primitives so that it can be fit on training data and applied
consistently (with no leakage) to validation and test data.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


class PreprocessError(Exception):
    """Raised when preprocessing cannot be applied to the given data."""


def _infer_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Infer numeric and categorical columns from a DataFrame's dtypes.

    Args:
        df: Input feature DataFrame.

    Returns:
        Tuple of ``(numeric_columns, categorical_columns)``.
    """
    numeric_cols = df.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    return numeric_cols, categorical_cols


def build_preprocessing_pipeline(
    df: pd.DataFrame,
    numeric_impute_strategy: str = "median",
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """
    Construct a fit-ready ``ColumnTransformer`` for numeric and categorical columns.

    Numeric columns are median-imputed and optionally standardized.
    Categorical columns are most-frequent-imputed and one-hot encoded.

    Args:
        df: A representative feature DataFrame used only to infer column types.
        numeric_impute_strategy: Strategy passed to ``SimpleImputer`` for numeric columns.
        scale_numeric: Whether to apply ``StandardScaler`` to numeric columns.

    Returns:
        An unfit ``ColumnTransformer`` ready to be used inside a pipeline.

    Raises:
        PreprocessError: If the DataFrame has no usable columns.
    """
    numeric_cols, categorical_cols = _infer_column_types(df)

    if not numeric_cols and not categorical_cols:
        raise PreprocessError("No numeric or categorical columns found to preprocess.")

    numeric_steps = [("imputer", SimpleImputer(strategy=numeric_impute_strategy))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(steps=numeric_steps)

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformers = []
    if numeric_cols:
        transformers.append(("numeric", numeric_pipeline, numeric_cols))
    if categorical_cols:
        transformers.append(("categorical", categorical_pipeline, categorical_cols))

    logger.info(
        "Built preprocessing pipeline: %d numeric columns, %d categorical columns",
        len(numeric_cols),
        len(categorical_cols),
    )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def preprocess_pipeline(
    X_train: pd.DataFrame,
    X_test: Optional[pd.DataFrame] = None,
    numeric_impute_strategy: str = "median",
    scale_numeric: bool = True,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], ColumnTransformer]:
    """
    Fit a preprocessing pipeline on training data and transform train/test sets.

    Args:
        X_train: Training feature DataFrame.
        X_test: Optional test feature DataFrame, transformed using statistics
            fit on ``X_train`` only (no leakage).
        numeric_impute_strategy: Imputation strategy for numeric columns.
        scale_numeric: Whether to standardize numeric columns.

    Returns:
        Tuple of ``(X_train_transformed, X_test_transformed, fitted_transformer)``.
        ``X_test_transformed`` is ``None`` if ``X_test`` was not provided.

    Raises:
        PreprocessError: If preprocessing fails to fit or transform.
    """
    transformer = build_preprocessing_pipeline(
        X_train, numeric_impute_strategy=numeric_impute_strategy, scale_numeric=scale_numeric
    )

    try:
        X_train_arr = transformer.fit_transform(X_train)
    except Exception as exc:  # noqa: BLE001
        raise PreprocessError(f"Failed to fit/transform training data: {exc}") from exc

    feature_names = _get_output_feature_names(transformer)
    X_train_out = pd.DataFrame(_to_dense(X_train_arr), columns=feature_names, index=X_train.index)

    X_test_out = None
    if X_test is not None:
        try:
            X_test_arr = transformer.transform(X_test)
        except Exception as exc:  # noqa: BLE001
            raise PreprocessError(f"Failed to transform test data: {exc}") from exc
        X_test_out = pd.DataFrame(_to_dense(X_test_arr), columns=feature_names, index=X_test.index)

    logger.info("Preprocessing complete: output shape %s", X_train_out.shape)
    return X_train_out, X_test_out, transformer


def _to_dense(arr):
    """Convert a possibly-sparse array to a dense numpy array."""
    if hasattr(arr, "toarray"):
        return arr.toarray()
    return arr


def _get_output_feature_names(transformer: ColumnTransformer) -> List[str]:
    """
    Retrieve output feature names from a fitted ``ColumnTransformer``.

    Falls back to positional names if scikit-learn's naming introspection
    is unavailable for a given transformer configuration.

    Args:
        transformer: A fitted ``ColumnTransformer``.

    Returns:
        List of output column names.
    """
    try:
        return list(transformer.get_feature_names_out())
    except Exception:  # noqa: BLE001
        n_outputs = transformer.transform(transformer.feature_names_in_[:1].reshape(1, -1)).shape[1]
        return [f"feature_{i}" for i in range(n_outputs)]
