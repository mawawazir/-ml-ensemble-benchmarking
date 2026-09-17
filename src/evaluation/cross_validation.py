"""
Cross-validation strategies used throughout the benchmark.

Stratified K-Fold is used consistently so that class proportions are
preserved across folds — important given the class imbalance present
in the target dataset.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

logger = logging.getLogger(__name__)

RANDOM_STATE = 42

DEFAULT_SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]


class CrossValidationError(Exception):
    """Raised when cross-validation cannot be run on the given data/estimator."""


def stratified_cv_scores(
    estimator: Any,
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int = 5,
    scoring: List[str] = None,
    n_jobs: int = -1,
) -> Dict[str, Any]:
    """
    Run stratified k-fold cross-validation and summarize scores per metric.

    Args:
        estimator: A fit-ready, scikit-learn-compatible estimator.
        X: Feature data.
        y: Target labels.
        cv_folds: Number of stratified folds.
        scoring: List of scikit-learn scoring strings. Defaults to
            ``["accuracy", "precision", "recall", "f1", "roc_auc"]``.
        n_jobs: Number of parallel jobs.

    Returns:
        Dictionary mapping each metric name to a dict with ``mean``, ``std``,
        and the raw per-fold ``scores`` list.

    Raises:
        CrossValidationError: If cross-validation fails to run.
    """
    scoring = scoring or DEFAULT_SCORING
    cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

    try:
        raw_results = cross_validate(
            estimator, X, y, cv=cv_strategy, scoring=scoring, n_jobs=n_jobs, return_train_score=False
        )
    except Exception as exc:  # noqa: BLE001
        raise CrossValidationError(f"Cross-validation failed: {exc}") from exc

    summary: Dict[str, Any] = {}
    for metric in scoring:
        key = f"test_{metric}"
        scores = raw_results.get(key)
        if scores is None:
            continue
        summary[metric] = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "scores": [float(s) for s in scores],
        }

    logger.info(
        "Stratified %d-fold CV summary: %s",
        cv_folds,
        {k: round(v["mean"], 4) for k, v in summary.items()},
    )
    return summary
