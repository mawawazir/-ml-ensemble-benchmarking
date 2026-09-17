"""Support Vector Machine model wrapper."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from sklearn.svm import SVC

from src.models.base_model import BaseModel

RANDOM_STATE = 42


class SVMModel(BaseModel):
    """
    Support Vector Machine classifier wrapper conforming to the :class:`BaseModel` interface.

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
        super().__init__(name="svm", params=params)
        self._param_grid_override = param_grid_override

    def build_estimator(self) -> SVC:
        """
        Construct an unfit ``SVC`` classifier.

        Returns:
            An ``SVC`` configured with sensible defaults (probability
            estimates enabled for ROC-AUC scoring), overridden by any
            values in ``self.params``.
        """
        defaults: Dict[str, Any] = {
            "C": 1.0,
            "kernel": "rbf",
            "gamma": "scale",
            "class_weight": "balanced",
            "probability": True,
            "random_state": RANDOM_STATE,
        }
        defaults.update(self.params)
        return SVC(**defaults)

    def param_grid(self) -> Dict[str, List]:
        """
        Return the SVM hyperparameter grid for ``GridSearchCV``.

        Returns:
            Dictionary mapping parameter names to candidate value lists.
        """
        if self._param_grid_override is not None:
            return self._param_grid_override

        return {
            "C": [0.1, 1, 10, 100],
            "kernel": ["rbf", "linear"],
            "gamma": ["scale", "auto"],
        }
