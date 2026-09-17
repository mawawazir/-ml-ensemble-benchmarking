"""
Class imbalance handling utilities.

Provides SMOTE-based oversampling for use during training, and helper
functions for computing class weights for estimators that support them
natively (Random Forest, SVM).
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight

logger = logging.getLogger(__name__)

RANDOM_STATE = 42


class ImbalanceHandlingError(Exception):
    """Raised when class imbalance handling cannot be applied."""


def apply_smote(
    X: pd.DataFrame,
    y: pd.Series,
    sampling_strategy: str | float = "auto",
    k_neighbors: int = 5,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Oversample the minority class using SMOTE.

    This must only be applied to the training split, never to validation
    or test data, to avoid leaking synthetic samples across the evaluation
    boundary.

    Args:
        X: Training feature DataFrame.
        y: Training target labels.
        sampling_strategy: Passed through to ``imblearn.SMOTE``. ``"auto"``
            resamples all classes but the majority to match the majority count.
        k_neighbors: Number of nearest neighbors used to synthesize new samples.
        random_state: Seed for reproducibility.

    Returns:
        Tuple of resampled ``(X_resampled, y_resampled)``.

    Raises:
        ImbalanceHandlingError: If SMOTE cannot be fit (e.g. too few minority
            samples for the requested ``k_neighbors``).
    """
    class_counts = y.value_counts()
    logger.info("Class distribution before SMOTE: %s", class_counts.to_dict())

    min_class_count = class_counts.min()
    effective_k = min(k_neighbors, max(min_class_count - 1, 1))
    if effective_k < k_neighbors:
        logger.warning(
            "Reducing SMOTE k_neighbors from %d to %d due to small minority class size (%d)",
            k_neighbors,
            effective_k,
            min_class_count,
        )

    try:
        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=effective_k,
            random_state=random_state,
        )
        X_resampled, y_resampled = smote.fit_resample(X, y)
    except ValueError as exc:
        raise ImbalanceHandlingError(f"SMOTE resampling failed: {exc}") from exc

    X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    y_resampled = pd.Series(y_resampled, name=y.name)

    logger.info(
        "Class distribution after SMOTE: %s", y_resampled.value_counts().to_dict()
    )
    return X_resampled, y_resampled


def apply_class_weights(y: pd.Series) -> Dict[int, float]:
    """
    Compute balanced class weights for use with estimators supporting ``class_weight``.

    Args:
        y: Target labels.

    Returns:
        Dictionary mapping each class label to its balanced weight.

    Raises:
        ImbalanceHandlingError: If weights cannot be computed (e.g. a single class present).
    """
    classes = np.unique(y)
    if len(classes) < 2:
        raise ImbalanceHandlingError("Cannot compute class weights with fewer than 2 classes.")

    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    weight_map = {int(cls): float(weight) for cls, weight in zip(classes, weights)}
    logger.info("Computed balanced class weights: %s", weight_map)
    return weight_map
