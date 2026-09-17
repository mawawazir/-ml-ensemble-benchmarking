"""
Feature selection utilities.

Provides variance-threshold filtering, univariate statistical selection,
and model-based (tree importance) selection, combined into a single
convenience entry point used by the benchmarking pipeline.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    SelectFromModel,
    SelectKBest,
    VarianceThreshold,
    f_classif,
)

logger = logging.getLogger(__name__)

RANDOM_STATE = 42


class FeatureSelectionError(Exception):
    """Raised when feature selection cannot be performed on the given data."""


def remove_low_variance_features(
    X: pd.DataFrame, threshold: float = 0.0
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Drop features whose variance falls at or below ``threshold``.

    Args:
        X: Feature DataFrame (numeric).
        threshold: Variance threshold; features with variance <= threshold are dropped.

    Returns:
        Tuple of ``(filtered_dataframe, dropped_column_names)``.
    """
    selector = VarianceThreshold(threshold=threshold)
    try:
        selector.fit(X)
    except ValueError as exc:
        raise FeatureSelectionError(f"Variance threshold selection failed: {exc}") from exc

    kept_mask = selector.get_support()
    kept_cols = X.columns[kept_mask].tolist()
    dropped_cols = X.columns[~kept_mask].tolist()

    if dropped_cols:
        logger.info("Dropped %d low-variance features: %s", len(dropped_cols), dropped_cols)

    return X[kept_cols], dropped_cols


def select_k_best_features(
    X: pd.DataFrame, y: pd.Series, k: int = 20
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Select the top-``k`` features by ANOVA F-value with respect to the target.

    Args:
        X: Feature DataFrame (numeric).
        y: Target labels.
        k: Number of top features to retain (capped at available feature count).

    Returns:
        Tuple of ``(reduced_dataframe, selected_column_names)``.
    """
    k = min(k, X.shape[1])
    selector = SelectKBest(score_func=f_classif, k=k)

    try:
        selector.fit(X, y)
    except ValueError as exc:
        raise FeatureSelectionError(f"Univariate feature selection failed: {exc}") from exc

    selected_cols = X.columns[selector.get_support()].tolist()
    logger.info("Selected top-%d features by ANOVA F-value: %s", k, selected_cols)
    return X[selected_cols], selected_cols


def select_features_by_importance(
    X: pd.DataFrame,
    y: pd.Series,
    max_features: Optional[int] = None,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Select features using a Random Forest's Gini feature importances.

    Args:
        X: Feature DataFrame (numeric).
        y: Target labels.
        max_features: Maximum number of features to retain. If ``None``,
            scikit-learn's ``"mean"`` importance threshold heuristic is used.
        random_state: Seed for reproducibility.

    Returns:
        Tuple of ``(reduced_dataframe, selected_column_names)``.
    """
    estimator = RandomForestClassifier(
        n_estimators=200, random_state=random_state, n_jobs=-1
    )

    try:
        estimator.fit(X, y)
    except ValueError as exc:
        raise FeatureSelectionError(f"Model-based feature selection failed: {exc}") from exc

    selector = SelectFromModel(
        estimator,
        prefit=True,
        max_features=max_features,
        threshold="mean" if max_features is None else None,
    )
    selected_cols = X.columns[selector.get_support()].tolist()
    logger.info(
        "Selected %d features by Random Forest importance (max_features=%s)",
        len(selected_cols),
        max_features,
    )
    return X[selected_cols], selected_cols


def select_features(
    X: pd.DataFrame,
    y: pd.Series,
    method: str = "importance",
    k: int = 20,
    variance_threshold: float = 0.0,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Convenience entry point applying variance filtering plus one selection method.

    Args:
        X: Feature DataFrame.
        y: Target labels.
        method: One of ``"importance"`` (Random Forest based) or ``"kbest"``
            (ANOVA F-value based).
        k: Number of features to retain when ``method="kbest"``.
        variance_threshold: Threshold used for the initial low-variance filter.

    Returns:
        Tuple of ``(final_dataframe, selected_column_names)``.

    Raises:
        FeatureSelectionError: If ``method`` is not recognized.
    """
    X_filtered, _ = remove_low_variance_features(X, threshold=variance_threshold)

    if method == "importance":
        return select_features_by_importance(X_filtered, y, max_features=k)
    if method == "kbest":
        return select_k_best_features(X_filtered, y, k=k)

    raise FeatureSelectionError(f"Unknown feature selection method: '{method}'")
