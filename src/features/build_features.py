"""
Feature engineering pipeline.

Derives new features from the base numeric columns (interactions,
polynomial terms, ratios, and binned aggregates) to improve model input
quality ahead of the modeling stage.
"""

from __future__ import annotations

import logging
from itertools import combinations
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngineeringError(Exception):
    """Raised when feature engineering cannot be applied to the given data."""


def engineer_features(
    df: pd.DataFrame,
    numeric_columns: Optional[List[str]] = None,
    max_interaction_pairs: int = 10,
    add_polynomial: bool = True,
    add_ratios: bool = True,
    add_statistical_aggregates: bool = True,
) -> pd.DataFrame:
    """
    Apply a feature engineering pipeline to a feature DataFrame.

    Adds, in order:
      1. Pairwise interaction terms (product) for a bounded set of numeric columns.
      2. Squared terms for numeric columns (polynomial degree 2).
      3. Ratio features between pairs of numeric columns (guarded against
         division by zero).
      4. Row-wise statistical aggregates (mean, std, min, max) across numeric columns.

    Args:
        df: Input feature DataFrame (should not include the target column).
        numeric_columns: Explicit list of numeric columns to engineer from.
            If ``None``, inferred automatically from dtypes.
        max_interaction_pairs: Cap on the number of pairwise interaction
            features generated, to avoid combinatorial explosion.
        add_polynomial: Whether to add squared terms.
        add_ratios: Whether to add pairwise ratio features.
        add_statistical_aggregates: Whether to add row-wise aggregate statistics.

    Returns:
        A new DataFrame containing the original columns plus engineered features.

    Raises:
        FeatureEngineeringError: If no numeric columns are available to engineer from.
    """
    out = df.copy()

    if numeric_columns is None:
        numeric_columns = out.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_columns:
        raise FeatureEngineeringError("No numeric columns available for feature engineering.")

    n_before = out.shape[1]

    if add_polynomial:
        for col in numeric_columns:
            out[f"{col}_squared"] = out[col] ** 2

    pairs = list(combinations(numeric_columns, 2))[:max_interaction_pairs]

    for col_a, col_b in pairs:
        out[f"{col_a}_x_{col_b}"] = out[col_a] * out[col_b]

    if add_ratios:
        for col_a, col_b in pairs:
            denom = out[col_b].replace(0, np.nan)
            out[f"{col_a}_div_{col_b}"] = (out[col_a] / denom).fillna(0.0)

    if add_statistical_aggregates:
        numeric_frame = out[numeric_columns]
        out["row_mean"] = numeric_frame.mean(axis=1)
        out["row_std"] = numeric_frame.std(axis=1).fillna(0.0)
        out["row_min"] = numeric_frame.min(axis=1)
        out["row_max"] = numeric_frame.max(axis=1)

    n_after = out.shape[1]
    logger.info(
        "Feature engineering added %d derived features (%d -> %d total columns)",
        n_after - n_before,
        n_before,
        n_after,
    )
    return out
