"""
Abstract base class defining the common model interface.

Every benchmarked model wraps a scikit-learn-compatible estimator behind
this interface so that the tuning and benchmarking orchestrators can
treat all models uniformly.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RANDOM_STATE = 42


class ModelError(Exception):
    """Raised for model construction, fitting, or persistence failures."""


class BaseModel(ABC):
    """
    Abstract base class for all benchmarked models.

    Subclasses must implement :meth:`build_estimator` to return an unfit,
    scikit-learn-compatible estimator, and :meth:`param_grid` to return the
    hyperparameter grid used for ``GridSearchCV``.

    Attributes:
        name: Human-readable model name, used in reports and file names.
        params: Hyperparameters (or overrides) used to construct the estimator.
        estimator: The underlying fitted estimator, set after :meth:`fit`.
        is_fitted: Whether :meth:`fit` has been called successfully.
    """

    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None) -> None:
        self.name = name
        self.params: Dict[str, Any] = params or {}
        self.estimator: Any = None
        self.is_fitted: bool = False

    @abstractmethod
    def build_estimator(self) -> Any:
        """
        Construct and return an unfit, scikit-learn-compatible estimator.

        Returns:
            An estimator implementing ``fit``, ``predict``, and ideally
            ``predict_proba``.
        """
        raise NotImplementedError

    @abstractmethod
    def param_grid(self) -> Dict[str, list]:
        """
        Return the hyperparameter grid for ``GridSearchCV``.

        Returns:
            Dictionary mapping parameter names to lists of candidate values.
        """
        raise NotImplementedError

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseModel":
        """
        Fit the underlying estimator on training data.

        Args:
            X: Training features.
            y: Training labels.

        Returns:
            ``self``, to allow method chaining.

        Raises:
            ModelError: If fitting fails.
        """
        if self.estimator is None:
            self.estimator = self.build_estimator()

        try:
            self.estimator.fit(X, y)
        except Exception as exc:  # noqa: BLE001
            raise ModelError(f"Failed to fit model '{self.name}': {exc}") from exc

        self.is_fitted = True
        logger.info("Fitted model '%s' on %d samples", self.name, len(X))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate class predictions for ``X``.

        Args:
            X: Feature data to predict on.

        Returns:
            Array of predicted class labels.

        Raises:
            ModelError: If the model has not been fitted yet.
        """
        self._check_fitted()
        return self.estimator.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate class probability estimates for ``X``.

        Args:
            X: Feature data to predict on.

        Returns:
            Array of shape ``(n_samples, n_classes)`` with predicted probabilities.

        Raises:
            ModelError: If the model has not been fitted, or does not support
                probability estimates.
        """
        self._check_fitted()
        if not hasattr(self.estimator, "predict_proba"):
            raise ModelError(f"Model '{self.name}' does not support predict_proba().")
        return self.estimator.predict_proba(X)

    def set_estimator(self, estimator: Any) -> "BaseModel":
        """
        Replace the underlying estimator (e.g. with a fitted GridSearchCV's best_estimator_).

        Args:
            estimator: A fitted, scikit-learn-compatible estimator.

        Returns:
            ``self``, to allow method chaining.
        """
        self.estimator = estimator
        self.is_fitted = True
        return self

    def save(self, directory: str | Path) -> Path:
        """
        Persist the fitted estimator to disk using joblib.

        Args:
            directory: Directory in which to save the model file.

        Returns:
            Path to the saved model file.

        Raises:
            ModelError: If the model has not been fitted yet.
        """
        self._check_fitted()
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{self.name}.joblib"
        joblib.dump(self.estimator, out_path)
        logger.info("Saved model '%s' to %s", self.name, out_path)
        return out_path

    @classmethod
    def load(cls, path: str | Path, name: Optional[str] = None) -> "BaseModel":
        """
        Load a persisted estimator into a new model wrapper instance.

        Args:
            path: Path to a ``.joblib`` file produced by :meth:`save`.
            name: Optional override for the loaded model's display name.

        Returns:
            A model wrapper instance with the loaded estimator attached.

        Raises:
            ModelError: If the file cannot be found or loaded.
        """
        resolved = Path(path)
        if not resolved.exists():
            raise ModelError(f"Model file not found: {resolved}")
        try:
            estimator = joblib.load(resolved)
        except Exception as exc:  # noqa: BLE001
            raise ModelError(f"Failed to load model from {resolved}: {exc}") from exc

        instance = cls()
        if name is not None:
            instance.name = name
        instance.set_estimator(estimator)
        return instance

    def _check_fitted(self) -> None:
        if not self.is_fitted or self.estimator is None:
            raise ModelError(f"Model '{self.name}' has not been fitted yet.")

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "unfitted"
        return f"<{self.__class__.__name__} name='{self.name}' status={status}>"
