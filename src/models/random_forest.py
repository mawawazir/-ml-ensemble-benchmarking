"""Random Forest model wrapper."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from sklearn.ensemble import RandomForestClassifier

from src.models.base_model import BaseModel

RANDOM_STATE = 42


class RandomForestModel(BaseModel):
    """
    Random Forest classifier wrapper conforming to the :class:`BaseModel` interface.

    Args:
        params: Optional overrides for the estimator's constructor arguments.
        param_grid_override: Optional override for the hyperparameter grid
            used during ``GridSearchCV``.
    """

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        param_grid_override: Optional[Dict[str, List]] = None,
    ) -> None:
        super().__init__(name="random_forest", params=params)
        self._param_grid_override = param_grid_override

    def build_estimator(self) -> RandomForestClassifier:
        """
        Construct an unfit ``RandomForestClassifier``.

        Returns:
            A ``RandomForestClassifier`` configured with sensible defaults,
            overridden by any values in ``self.params``.
        """
        defaults: Dict[str, Any] = {
            "n_estimators": 200,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        }
        defaults.update(self.params)
        return RandomForestClassifier(**defaults)

    def param_grid(self) -> Dict[str, List]:
        """
        Return the Random Forest hyperparameter grid for ``GridSearchCV``.

        Returns:
            Dictionary mapping parameter names to candidate value lists.
        """
        if self._param_grid_override is not None:
            return self._param_grid_override

        return {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"],
        }
