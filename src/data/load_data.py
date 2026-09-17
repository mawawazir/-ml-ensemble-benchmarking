"""
Data loading utilities.

Handles reading raw datasets from disk (or generating a reproducible
synthetic dataset when no raw data is present) and splitting data into
train/test partitions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

RANDOM_STATE = 42


class DataLoadError(Exception):
    """Raised when a dataset cannot be located or parsed."""


def load_raw_data(
    path: Optional[str | Path] = None,
    target_column: str = "target",
    n_samples: int = 50000,
    n_features: int = 20,
) -> pd.DataFrame:
    """
    Load a raw dataset from disk, or synthesize one if no path is given.

    When ``path`` is ``None`` or the file does not exist, a reproducible
    synthetic binary-classification dataset is generated. This keeps the
    pipeline runnable end-to-end without requiring a proprietary dataset
    to be committed to the repository.

    Args:
        path: Path to a CSV file containing raw data. If ``None``, data
            is synthesized instead.
        target_column: Name of the target column to create (when synthesizing)
            or expect (when loading from disk).
        n_samples: Number of rows to synthesize when generating data.
        n_features: Number of numeric features to synthesize.

    Returns:
        A DataFrame containing features and the target column.

    Raises:
        DataLoadError: If ``path`` is provided but cannot be read, or is
            missing the expected target column.
    """
    if path is not None:
        resolved = Path(path)
        if not resolved.exists():
            raise DataLoadError(f"Raw data file not found: {resolved}")
        try:
            df = pd.read_csv(resolved)
        except Exception as exc:  # noqa: BLE001 - surface as a domain-specific error
            raise DataLoadError(f"Failed to read CSV at {resolved}: {exc}") from exc

        if target_column not in df.columns:
            raise DataLoadError(
                f"Target column '{target_column}' not found in {resolved}. "
                f"Available columns: {list(df.columns)}"
            )
        logger.info("Loaded %d rows and %d columns from %s", len(df), df.shape[1], resolved)
        return df

    logger.info(
        "No raw data path provided; synthesizing %d samples with %d features "
        "(random_state=%d) for reproducible demonstration.",
        n_samples,
        n_features,
        RANDOM_STATE,
    )
    return _synthesize_dataset(n_samples=n_samples, n_features=n_features, target_column=target_column)


def _synthesize_dataset(
    n_samples: int,
    n_features: int,
    target_column: str,
    imbalance_ratio: float = 0.15,
) -> pd.DataFrame:
    """
    Generate a reproducible, moderately imbalanced synthetic classification dataset.

    Args:
        n_samples: Number of rows to generate.
        n_features: Number of numeric feature columns to generate.
        target_column: Name to assign to the binary label column.
        imbalance_ratio: Approximate proportion of the positive class.

    Returns:
        A synthesized DataFrame with numeric features, a couple of
        categorical-like columns, and a binary target.
    """
    from sklearn.datasets import make_classification

    rng = np.random.RandomState(RANDOM_STATE)

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=int(n_features * 0.6),
        n_redundant=int(n_features * 0.2),
        n_clusters_per_class=2,
        weights=[1 - imbalance_ratio, imbalance_ratio],
        flip_y=0.02,
        random_state=RANDOM_STATE,
    )

    columns = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=columns)

    # Add a couple of categorical-style columns to make preprocessing meaningful.
    df["category_region"] = rng.choice(["north", "south", "east", "west"], size=n_samples)
    df["category_channel"] = rng.choice(["online", "retail", "partner"], size=n_samples)

    # Inject a small, reproducible fraction of missing values.
    missing_mask = rng.rand(n_samples) < 0.03
    df.loc[missing_mask, "feature_0"] = np.nan

    df[target_column] = y
    return df


def train_test_split_data(
    df: pd.DataFrame,
    target_column: str = "target",
    test_size: float = 0.2,
    stratify: bool = True,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split a DataFrame into stratified train/test feature and label sets.

    Args:
        df: Full dataset including the target column.
        target_column: Name of the column to use as the prediction target.
        test_size: Fraction of data to reserve for testing.
        stratify: Whether to stratify the split by the target distribution.
        random_state: Seed for reproducibility.

    Returns:
        Tuple of ``(X_train, X_test, y_train, y_test)``.

    Raises:
        DataLoadError: If the target column is missing from ``df``.
    """
    if target_column not in df.columns:
        raise DataLoadError(f"Target column '{target_column}' not present in DataFrame.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    strat = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat
    )
    logger.info(
        "Split data into train=%d rows and test=%d rows (stratify=%s)",
        len(X_train),
        len(X_test),
        stratify,
    )
    return X_train, X_test, y_train, y_test
