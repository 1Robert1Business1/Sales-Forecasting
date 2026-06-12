"""
Models — thin wrappers giving every forecaster a common fit/predict interface.

The rolling-origin CV harness in :mod:`src.evaluation` treats all models
uniformly: it calls ``fit(y_train, X_train)`` then ``predict(h, X_future)`` on a
fresh instance each fold. Keeping the wrappers thin means the harness — not the
model — owns the temporal splits and the scaling, so the comparison stays honest.

Allowed inputs (leakage discipline, carried from the EDA stage)
---------------------------------------------------------------
* Target: lagged ``sales`` only.
* Exogenous: the event flags ``is_match_day`` and ``is_bank_holiday`` plus
  ``match_importance``. All three are known-future:

  - ``is_match_day`` and ``is_bank_holiday`` — fixture lists and bank holidays
    are published months ahead.
  - ``match_importance`` is set by ``src/generate_data.py`` from a single
    ``Beta(2, 5)`` draw at fixture-generation time. It depends on no realised
    outcome, attendance, or sales. It encodes the inherent draw of the fixture
    (rivalry, league/cup stakes, billing) — attributes a rota-planning bar
    manager already has the week before kickoff. Pre-match, so legitimate
    exog.

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


# =============================================================================
# SARIMA / SARIMAX — statsmodels under the common interface
# =============================================================================

class SARIMAXModel(ForecastModel):
    """Thin wrapper over :class:`statsmodels.tsa.statespace.SARIMAX`.

    Default order is the airline model the ACF/PACF in the EDA notebook pointed
    at: ``(0, 1, 1)(0, 1, 1, 7)`` with no exogenous regressors. Passing event
    flags / match_importance through ``X_train`` turns it into SARIMAX with
    those as known-future regressors.
    """

    def __init__(
        self,
        order: tuple[int, int, int] = (0, 1, 1),
        seasonal_order: tuple[int, int, int, int] = (0, 1, 1, 7),
        trend: Optional[str] = None,
        maxiter: int = 200,
        label: Optional[str] = None,
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend
        self.maxiter = maxiter
        self.name = label or f"SARIMAX{order}{seasonal_order}"
        self._fit_res = None
        self._exog_cols: Optional[list[str]] = None

    def fit(self, y_train: pd.Series, X_train: Optional[pd.DataFrame] = None) -> "SARIMAXModel":
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        import warnings

        self._exog_cols = list(X_train.columns) if X_train is not None else None
        exog = X_train.values.astype(float) if X_train is not None else None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                y_train.values.astype(float),
                exog=exog,
                order=self.order,
                seasonal_order=self.seasonal_order,
                trend=self.trend,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self._fit_res = model.fit(disp=False, maxiter=self.maxiter)
        return self

    def predict(self, h: int, X_future: Optional[pd.DataFrame] = None) -> np.ndarray:
        if self._fit_res is None:
            raise RuntimeError("predict() called before fit()")
        exog = X_future.values.astype(float) if X_future is not None else None
        return np.asarray(self._fit_res.forecast(steps=h, exog=exog), dtype=float)

    def predict_interval(
        self,
        h: int,
        X_future: Optional[pd.DataFrame] = None,
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        """Mean forecast plus a (1 − α) prediction interval.

        Returns a DataFrame with columns ``mean``, ``mean_se``, ``lower``,
        ``upper``. Intervals come from the SARIMAX state-space forecast
        variance and assume **roughly normal** one-step errors — they are
        a plan-level cover, not a tail-risk guarantee. Actual tail risk on
        event nights (cup-final shocks, weather-driven swings, etc.) runs
        higher than the Gaussian assumption implies.
        """
        if self._fit_res is None:
            raise RuntimeError("predict_interval() called before fit()")
        exog = X_future.values.astype(float) if X_future is not None else None
        fc = self._fit_res.get_forecast(steps=h, exog=exog)
        summary = fc.summary_frame(alpha=alpha)
        return summary.rename(
            columns={"mean_ci_lower": "lower", "mean_ci_upper": "upper"}
        )[["mean", "mean_se", "lower", "upper"]]

    @property
    def aic(self) -> float:
        if self._fit_res is None:
            raise RuntimeError("aic accessed before fit()")
        return float(self._fit_res.aic)


# =============================================================================
# Prophet — interpretable holiday/event decomposition
# =============================================================================

class ProphetModel(ForecastModel):
    """Facebook/Meta Prophet, with the event flags as additive regressors.

    Positioned for interpretability rather than presumed accuracy. We do NOT
    call ``add_country_holidays`` because ``is_bank_holiday`` is already passed
    as a regressor — double-counting would be wrong.
    """

    name = "Prophet"

    def __init__(self, regressor_cols: Optional[list[str]] = None):
        self.regressor_cols = list(regressor_cols) if regressor_cols else []
        self._model = None

    @staticmethod
    def _ensure_cmdstan_path_valid() -> None:
        """Some Prophet wheels ship a trimmed CmdStan whose `makefile` is
        omitted (only the precompiled `prophet_model.bin` is needed at
        runtime). cmdstanpy's path validator rejects that, raising on
        `Prophet()` instantiation. Touching an empty `makefile` satisfies
        the check without changing what's actually executed.
        """
        import pathlib
        try:
            import prophet  # noqa: F401
        except ImportError:
            return
        base = pathlib.Path(prophet.__file__).parent / "stan_model"
        for cmdstan_dir in base.glob("cmdstan-*"):
            mk = cmdstan_dir / "makefile"
            if cmdstan_dir.is_dir() and not mk.exists():
                try:
                    mk.touch()
                except OSError:
                    pass

    def fit(self, y_train: pd.Series, X_train: Optional[pd.DataFrame] = None) -> "ProphetModel":
        import logging
        for n in ("prophet", "cmdstanpy"):
            logging.getLogger(n).setLevel(logging.WARNING)
        self._ensure_cmdstan_path_valid()
        from prophet import Prophet

        df_train = pd.DataFrame({"ds": y_train.index, "y": y_train.values})
        if X_train is not None and self.regressor_cols:
            for c in self.regressor_cols:
                df_train[c] = X_train[c].values

        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode="additive",
            interval_width=0.8,
        )
        for c in self.regressor_cols:
            m.add_regressor(c)
        m.fit(df_train)
        self._model = m
        return self

    def predict(self, h: int, X_future: Optional[pd.DataFrame] = None) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("predict() called before fit()")
        if X_future is None:
            future = self._model.make_future_dataframe(periods=h, include_history=False)
        else:
            future = pd.DataFrame({"ds": X_future.index})
            for c in self.regressor_cols:
                future[c] = X_future[c].values
        forecast = self._model.predict(future)
        return forecast["yhat"].values[-h:]


# =============================================================================
# LSTM — deliberately modest, one layer, early-stopped, train-only scaling
# =============================================================================

class LSTMModel(ForecastModel):
    """Single-layer LSTM with direct horizon-step output.

    Architectural choices are deliberately conservative — see the modelling
    notebook for the M-competition reasoning on why a high-capacity network is
    not expected to beat statistical methods on one short series. We are NOT
    tuning toward a win; we are honestly testing whether the cross-learning
    that pure-ML methods need can be approximated on a single venue.

    * One LSTM layer, ``units=32`` (few units).
    * Direct h-step Dense output (no recursion / no error compounding).
    * Lookback of 28 days (4 weeks).
    * Inputs per timestep: (scaled sales) + (whatever exog columns are passed).
    * Standard-scale the target on each fold's TRAINING data only.
    * shuffle=False; validation = last ``val_frac`` of windows (preserves order).
    * Early stopping on val_loss, ``restore_best_weights=True``.
    """

    name = "LSTM"

    def __init__(
        self,
        horizon: int = 7,
        lookback: int = 28,
        units: int = 32,
        dropout: float = 0.1,
        max_epochs: int = 60,
        patience: int = 8,
        batch_size: int = 32,
        val_frac: float = 0.15,
        learning_rate: float = 1e-3,
        seed: int = 42,
    ):
        self.horizon = horizon
        self.lookback = lookback
        self.units = units
        self.dropout = dropout
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.val_frac = val_frac
        self.learning_rate = learning_rate
        self.seed = seed
        self._scaler = None
        self._model = None
        self._train_features: Optional[np.ndarray] = None

    @staticmethod
    def _make_windows(
        features_2d: np.ndarray, lookback: int, horizon: int, target_col: int = 0
    ) -> tuple[np.ndarray, np.ndarray]:
        n = features_2d.shape[0]
        n_wins = n - lookback - horizon + 1
        if n_wins <= 0:
            raise ValueError(
                f"not enough data: n={n}, lookback={lookback}, horizon={horizon}"
            )
        X = np.zeros((n_wins, lookback, features_2d.shape[1]), dtype=np.float32)
        y = np.zeros((n_wins, horizon), dtype=np.float32)
        for i in range(n_wins):
            X[i] = features_2d[i : i + lookback]
            y[i] = features_2d[i + lookback : i + lookback + horizon, target_col]
        return X, y

    def fit(self, y_train: pd.Series, X_train: Optional[pd.DataFrame] = None) -> "LSTMModel":
        from sklearn.preprocessing import StandardScaler
        import os
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping
        from tensorflow.keras.layers import Dense, Input, LSTM
        from tensorflow.keras.models import Model
        from tensorflow.keras.optimizers import Adam

        tf.keras.utils.set_random_seed(self.seed)

        # Scale the target on TRAINING data only.
        self._scaler = StandardScaler()
        y_scaled = (
            self._scaler.fit_transform(y_train.values.reshape(-1, 1))
            .flatten()
            .astype(np.float32)
        )

        # Feature matrix: column 0 = scaled sales (target), cols 1.. = exog.
        if X_train is not None and X_train.shape[1] > 0:
            features = np.column_stack([y_scaled, X_train.values.astype(np.float32)])
        else:
            features = y_scaled.reshape(-1, 1)

        X_arr, y_arr = self._make_windows(features, self.lookback, self.horizon)

        inp = Input(shape=(self.lookback, features.shape[1]))
        x = LSTM(self.units, dropout=self.dropout)(inp)
        out = Dense(self.horizon)(x)
        model = Model(inp, out)
        model.compile(optimizer=Adam(self.learning_rate), loss="mse")

        model.fit(
            X_arr,
            y_arr,
            epochs=self.max_epochs,
            batch_size=self.batch_size,
            validation_split=self.val_frac,
            shuffle=False,
            verbose=0,
            callbacks=[
                EarlyStopping(
                    monitor="val_loss",
                    patience=self.patience,
                    restore_best_weights=True,
                )
            ],
        )
        self._model = model
        self._train_features = features
        return self

    def predict(self, h: int, X_future: Optional[pd.DataFrame] = None) -> np.ndarray:
        if h != self.horizon:
            raise ValueError(
                f"LSTM trained for horizon={self.horizon}, asked for h={h}"
            )
        if self._model is None or self._train_features is None:
            raise RuntimeError("predict() called before fit()")
        last = self._train_features[-self.lookback :].reshape(
            1, self.lookback, -1
        )
        pred_scaled = self._model.predict(last, verbose=0)[0]
        return (
            self._scaler.inverse_transform(pred_scaled.reshape(-1, 1))
            .flatten()
            .astype(float)
        )
