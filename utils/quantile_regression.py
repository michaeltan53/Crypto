# -*- coding: utf-8 -*-
"""
Quantile Regression Forest (QRF) based estimator for perturbation strength
and graded defense policy utilities.

- StrengthQuantileRegressor: fit/predict quantiles [0.1, 0.5, 0.9]
- DefensePolicy: map median S_hat to action levels using thresholds

This module is lightweight (uses scikit-learn) and is suitable for edge-side usage
with small trees and shallow depth.
"""

from typing import Tuple, Dict, List
import numpy as np

try:
    from sklearn.ensemble import RandomForestRegressor
except Exception:  # scikit-learn might not be installed; provide a fallback
    RandomForestRegressor = None


class StrengthQuantileRegressor:
    def __init__(self, n_estimators: int = 50, max_depth: int = 6, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if RandomForestRegressor is None:
            # Fallback: store simple statistics
            self.model = {'mean': float(np.mean(y)), 'std': float(np.std(y) + 1e-8)}
            return
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(X, y)

    def predict_quantiles(self, X: np.ndarray, quantiles: List[float] = [0.1, 0.5, 0.9]) -> np.ndarray:
        if RandomForestRegressor is None or not hasattr(self.model, 'estimators_'):
            # Gaussian fallback
            mean = self.model['mean']
            std = self.model['std']
            # approximate quantiles under normal assumption
            z = np.array([ -1.2816, 0.0, 1.2816 ])  # ~10%, 50%, 90%
            return np.clip(mean + z * std, 0.0, 1.0)[None, :]

        # Collect leaf-wise predictions across trees
        all_preds = np.stack([est.predict(X) for est in self.model.estimators_], axis=0)  # [T, N]
        # quantiles over trees (bagging-based QRF approximation)
        qs = np.quantile(all_preds, quantiles, axis=0).T  # [N, Q]
        return np.clip(qs, 0.0, 1.0)


class DefensePolicy:
    def __init__(self, s_log: float = 0.1, s_retry: float = 0.3):
        self.s_log = s_log
        self.s_retry = s_retry

    def decide(self, s_median: float) -> str:
        if s_median < self.s_log:
            return 'log'
        if s_median < self.s_retry:
            return 'retry_encrypted_channel'
        return 'block'

    def batch_decide(self, s_median_vec: np.ndarray) -> List[str]:
        return [self.decide(float(s)) for s in s_median_vec]

