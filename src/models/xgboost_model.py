"""XGBoost model wrapper."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from xgboost import XGBClassifier

from src.models.base_model import BaseModel

RANDOM_STATE = 42


class XGBoostModel(BaseModel):
    """
    XGBoost classifier wrapper conforming to the :class:`BaseModel` interface.

    Args:
        params: Optional overrides for the estimator's constructor arguments.
        param_grid_override: Optional override for the hyperparameter grid
            used during ``GridSearchCV``.
        scale_pos_weight: Optional class-imbalance weight, typically
            ``n_negative / n_positive``, passed to the estimator.
    """

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        param_grid_override: Optional[Dict[str, List]] = None,
        scale_pos_weight: Optional[float] = None,
    ) -> None:
        super().__init__(name="xgboost", params=params)
        self._param_grid_override = param_grid_override
        self.scale_pos_weight = scale_pos_weight

    def build_estimator(self) -> XGBClassifier:
        """
        Construct an unfit ``XGBClassifier``.

        Returns:
            An ``XGBClassifier`` configured with sensible defaults,
            overridden by any values in ``self.params``.
        """
        defaults: Dict[str, Any] = {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "eval_metric": "logloss",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        }
        if self.scale_pos_weight is not None:
            defaults["scale_pos_weight"] = self.scale_pos_weight

        defaults.update(self.params)
        return XGBClassifier(**defaults)

    def param_grid(self) -> Dict[str, List]:
        """
        Return the XGBoost hyperparameter grid for ``GridSearchCV``.

        Returns:
            Dictionary mapping parameter names to candidate value lists.
        """
        if self._param_grid_override is not None:
            return self._param_grid_override

        return {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 6, 9],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.8, 0.9, 1.0],
            "colsample_bytree": [0.8, 0.9, 1.0],
        }
