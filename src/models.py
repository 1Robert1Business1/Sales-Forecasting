"""
Models — thin wrappers giving every forecaster a common fit/predict interface.

The rolling-origin CV harness in :mod:`src.evaluation` treats all models
uniformly: it calls ``fit(y_train, X_train)`` then ``predict(h, X_future)`` on a
fresh instance each fold. Keeping the wrappers thin means the harness — not the
model — owns the temporal splits and the scaling, so the comparison stays honest.

Allowed inputs (leakage discipline, carried from the EDA stage)
---------------------------------------------------------------
* Target: lagged ``sales`` only.
* Exogenous: the event FLAGS ``is_match_day`` and ``is_bank_holiday`` only — both
  are genuine known-future regressors (fixtures and bank holidays are published
  months ahead).
* The synthetic generator's ground-truth component columns
  (``baseline/trend/weekly/annual/events/mu``) are NEVER features. They exist for
  verification in the EDA notebook and nowhere else.

This module holds the baseline for Part A; SARIMA, Prophet and the LSTM are added
in Part B against this same interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd

from src.evaluation import SEASONAL_PERIOD


class ForecastModel(ABC):
    """Common interface. A fresh instance is created per CV fold."""

    #: short label used in results tables / plots
    name: str = "model"

    @abstractmethod
    def fit(
        self, y_train: pd.Series, X_train: Optional[pd.DataFrame] = None
    ) -> "ForecastModel":
        ...

    @abstractmethod
    def predict(
        self, h: int, X_future: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        ...


class SeasonalNaive(ForecastModel):
    """Seasonal-naive baseline: the forecast for each day is the value ``m`` days
    earlier — "this Saturday ≈ last Saturday". Zero parameters, and the bar every
    other model must clear to justify its complexity.

    This is also the in-sample reference the MASE scale is built from, so a
    correctly implemented MASE scores this model at roughly 1.0 under CV — the
    Part A sanity check.
    """

    name = "SeasonalNaive"

    def __init__(self, m: int = SEASONAL_PERIOD):
        self.m = m
        self._tail: Optional[np.ndarray] = None

    def fit(self, y_train: pd.Series, X_train: Optional[pd.DataFrame] = None) -> "SeasonalNaive":
        y = np.asarray(y_train, dtype=float)
        if y.shape[0] < self.m:
            raise ValueError(f"need >= m={self.m} training points, got {y.shape[0]}")
        self._tail = y[-self.m:]
        return self

    def predict(self, h: int, X_future: Optional[pd.DataFrame] = None) -> np.ndarray:
        if self._tail is None:
            raise RuntimeError("predict() called before fit()")
        reps = int(np.ceil(h / self.m))
        return np.tile(self._tail, reps)[:h]
