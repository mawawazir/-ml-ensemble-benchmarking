"""
Classification evaluation metrics.

Centralizes metric computation so every model in the benchmark is scored
identically, using the same definitions and the same positive-class
conventions.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


class MetricsError(Exception):
    """Raised when metrics cannot be computed for the given predictions."""


@dataclass
class ClassificationMetrics:
    """
    Container for a standard set of binary classification metrics.

    Attributes:
        accuracy: Overall accuracy.
        precision: Precision of the positive class.
        recall: Recall of the positive class.
        f1: F1-score of the positive class.
        roc_auc: Area under the ROC curve (``None`` if probabilities unavailable).
        confusion_matrix: 2x2 confusion matrix as a nested list.
    """

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: Optional[float]
    confusion_matrix: list

    def to_dict(self) -> Dict[str, Any]:
        """Return the metrics as a plain dictionary."""
        return asdict(self)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> ClassificationMetrics:
    """
    Compute accuracy, precision, recall, F1, ROC-AUC, and a confusion matrix.

    Args:
        y_true: Ground-truth binary labels.
        y_pred: Predicted binary labels.
        y_proba: Optional predicted probabilities for the positive class
            (or a 2-column probability matrix, in which case column 1 is used).

    Returns:
        A populated :class:`ClassificationMetrics` instance.

    Raises:
        MetricsError: If ``y_true`` and ``y_pred`` have mismatched lengths.
    """
    if len(y_true) != len(y_pred):
        raise MetricsError(
            f"Length mismatch between y_true ({len(y_true)}) and y_pred ({len(y_pred)})."
        )

    roc_auc: Optional[float] = None
    if y_proba is not None:
        proba_positive = y_proba[:, 1] if getattr(y_proba, "ndim", 1) == 2 else y_proba
        try:
            roc_auc = float(roc_auc_score(y_true, proba_positive))
        except ValueError as exc:
            logger.warning("Could not compute ROC-AUC: %s", exc)
            roc_auc = None

    metrics = ClassificationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=roc_auc,
        confusion_matrix=confusion_matrix(y_true, y_pred).tolist(),
    )

    logger.info(
        "Metrics — accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f roc_auc=%s",
        metrics.accuracy,
        metrics.precision,
        metrics.recall,
        metrics.f1,
        f"{metrics.roc_auc:.4f}" if metrics.roc_auc is not None else "N/A",
    )
    return metrics
