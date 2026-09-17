"""
Plotting utilities for the benchmark report.

All functions save a figure to disk and return the resolved path, so they
can be composed into an automated reporting pipeline (see
``src/models/benchmark.py``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # non-interactive backend, safe for headless/CI environments

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid")


class PlottingError(Exception):
    """Raised when a plot cannot be generated or saved."""


def _resolve_output_path(output_dir: str | Path, filename: str) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / filename


def plot_model_comparison(
    results: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    output_dir: str | Path = "reports/figures",
    filename: str = "model_comparison.png",
) -> Path:
    """
    Plot a grouped bar chart comparing models across several metrics.

    Args:
        results: DataFrame indexed or columned by model name, containing
            one column per metric (e.g. ``accuracy``, ``f1``, ``roc_auc``).
        metrics: Metric columns to plot. Defaults to all numeric columns.
        output_dir: Directory to save the figure into.
        filename: Output file name.

    Returns:
        Path to the saved PNG file.

    Raises:
        PlottingError: If ``results`` has no plottable numeric columns.
    """
    if metrics is None:
        metrics = results.select_dtypes(include=[np.number]).columns.tolist()

    if not metrics:
        raise PlottingError("No numeric metric columns available to plot.")

    plot_df = results[metrics].reset_index().melt(id_vars=results.index.name or "index", var_name="metric", value_name="score")
    id_col = results.index.name or "index"

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=plot_df, x="metric", y="score", hue=id_col, ax=ax)
    ax.set_title("Model Comparison Across Metrics")
    ax.set_ylabel("Score")
    ax.set_xlabel("Metric")
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()

    out_path = _resolve_output_path(output_dir, filename)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved model comparison plot to %s", out_path)
    return out_path


def plot_roc_curves(
    fitted_models: Dict[str, object],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: str | Path = "reports/figures",
    filename: str = "roc_curves.png",
) -> Path:
    """
    Plot overlaid ROC curves for each fitted model.

    Args:
        fitted_models: Mapping of model name to a fitted estimator exposing
            ``predict_proba`` or ``decision_function``.
        X_test: Held-out test features.
        y_test: Held-out test labels.
        output_dir: Directory to save the figure into.
        filename: Output file name.

    Returns:
        Path to the saved PNG file.

    Raises:
        PlottingError: If no models could be plotted.
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    plotted = 0

    for name, estimator in fitted_models.items():
        try:
            RocCurveDisplay.from_estimator(estimator, X_test, y_test, ax=ax, name=name)
            plotted += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping ROC curve for '%s': %s", name, exc)

    if plotted == 0:
        plt.close(fig)
        raise PlottingError("No ROC curves could be generated for any model.")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_title("ROC Curves — Model Comparison")
    ax.legend(loc="lower right")
    fig.tight_layout()

    out_path = _resolve_output_path(output_dir, filename)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved ROC curves plot to %s", out_path)
    return out_path


def plot_confusion_matrices(
    fitted_models: Dict[str, object],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: str | Path = "reports/figures",
    filename: str = "confusion_matrices.png",
) -> Path:
    """
    Plot a row of confusion matrices, one per model.

    Args:
        fitted_models: Mapping of model name to a fitted estimator.
        X_test: Held-out test features.
        y_test: Held-out test labels.
        output_dir: Directory to save the figure into.
        filename: Output file name.

    Returns:
        Path to the saved PNG file.

    Raises:
        PlottingError: If no confusion matrices could be plotted.
    """
    n_models = len(fitted_models)
    if n_models == 0:
        raise PlottingError("No models provided for confusion matrix plotting.")

    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5))
    if n_models == 1:
        axes = [axes]

    for ax, (name, estimator) in zip(axes, fitted_models.items()):
        try:
            ConfusionMatrixDisplay.from_estimator(estimator, X_test, y_test, ax=ax, colorbar=False)
            ax.set_title(name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping confusion matrix for '%s': %s", name, exc)
            ax.set_visible(False)

    fig.suptitle("Confusion Matrices — Model Comparison")
    fig.tight_layout()

    out_path = _resolve_output_path(output_dir, filename)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved confusion matrices plot to %s", out_path)
    return out_path


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    top_n: int = 20,
    title: str = "Feature Importance",
    output_dir: str | Path = "reports/figures",
    filename: str = "feature_importance.png",
) -> Path:
    """
    Plot a horizontal bar chart of the top-N most important features.

    Args:
        feature_names: List of feature names, aligned with ``importances``.
        importances: Array of importance scores (e.g. Gini importance or
            permutation importance means).
        top_n: Number of top features to display.
        title: Plot title.
        output_dir: Directory to save the figure into.
        filename: Output file name.

    Returns:
        Path to the saved PNG file.

    Raises:
        PlottingError: If ``feature_names`` and ``importances`` lengths differ.
    """
    if len(feature_names) != len(importances):
        raise PlottingError(
            f"feature_names length ({len(feature_names)}) != importances length ({len(importances)})"
        )

    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    df = df.sort_values("importance", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(df))))
    sns.barplot(data=df, y="feature", x="importance", ax=ax, color="steelblue")
    ax.set_title(title)
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    fig.tight_layout()

    out_path = _resolve_output_path(output_dir, filename)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved feature importance plot to %s", out_path)
    return out_path
