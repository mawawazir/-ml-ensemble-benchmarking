"""
GridSearchCV wrapper for automated hyperparameter tuning.

Wraps scikit-learn's ``GridSearchCV`` with a consistent interface used by
every model in the benchmark, and returns a structured result object
capturing the best estimator, parameters, and CV score history.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.models.base_model import BaseModel

logger = logging.getLogger(__name__)

RANDOM_STATE = 42


class TuningError(Exception):
    """Raised when hyperparameter tuning fails."""


@dataclass
class GridSearchResult:
    """
    Structured result of a ``GridSearchCV`` run.

    Attributes:
        best_estimator: The refit estimator with the best-found hyperparameters.
        best_params: Dictionary of the winning hyperparameter combination.
        best_score: Cross-validated score achieved by ``best_params``.
        cv_results: Full ``cv_results_`` dictionary from scikit-learn, as a DataFrame.
        search_time_seconds: Wall-clock time spent on the grid search.
        scoring: The scoring metric used to select the best parameters.
    """

    best_estimator: Any
    best_params: Dict[str, Any]
    best_score: float
    cv_results: pd.DataFrame
    search_time_seconds: float
    scoring: str = field(default="f1")


def tune_model(
    model: BaseModel,
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int = 5,
    scoring: str = "f1",
    n_jobs: int = -1,
    param_grid_override: Optional[Dict[str, list]] = None,
    verbose: int = 1,
) -> GridSearchResult:
    """
    Run ``GridSearchCV`` for a given model using stratified k-fold cross-validation.

    Args:
        model: A :class:`BaseModel` subclass instance (unfit).
        X: Training features.
        y: Training labels.
        cv_folds: Number of stratified folds for cross-validation.
        scoring: Scikit-learn scoring string used to rank hyperparameter combinations.
        n_jobs: Number of parallel jobs; ``-1`` uses all available cores.
        param_grid_override: Optional grid to use instead of the model's default grid.
        verbose: Verbosity level passed through to ``GridSearchCV``.

    Returns:
        A :class:`GridSearchResult` capturing the best estimator and search metadata.

    Raises:
        TuningError: If the grid search fails to run (e.g. invalid parameter grid).
    """
    estimator = model.build_estimator()
    grid = param_grid_override if param_grid_override is not None else model.param_grid()

    cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

    search = GridSearchCV(
        estimator=estimator,
        param_grid=grid,
        scoring=scoring,
        cv=cv_strategy,
        n_jobs=n_jobs,
        verbose=verbose,
        refit=True,
        return_train_score=False,
    )

    logger.info(
        "Starting GridSearchCV for '%s': %d candidate combinations, cv_folds=%d, scoring=%s",
        model.name,
        _count_combinations(grid),
        cv_folds,
        scoring,
    )

    start = time.perf_counter()
    try:
        search.fit(X, y)
    except Exception as exc:  # noqa: BLE001
        raise TuningError(f"GridSearchCV failed for model '{model.name}': {exc}") from exc
    elapsed = time.perf_counter() - start

    logger.info(
        "GridSearchCV for '%s' complete in %.2fs. Best %s=%.4f, best_params=%s",
        model.name,
        elapsed,
        scoring,
        search.best_score_,
        search.best_params_,
    )

    return GridSearchResult(
        best_estimator=search.best_estimator_,
        best_params=search.best_params_,
        best_score=search.best_score_,
        cv_results=pd.DataFrame(search.cv_results_),
        search_time_seconds=elapsed,
        scoring=scoring,
    )


def _count_combinations(grid: Dict[str, list]) -> int:
    """Compute the total number of hyperparameter combinations in a grid."""
    total = 1
    for values in grid.values():
        total *= max(len(values), 1)
    return total
