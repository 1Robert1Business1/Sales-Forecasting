"""
Evaluation — hand-rolled MASE and rolling-origin cross-validation.

Headline metric and CV harness for the sales-forecasting project. Implemented
directly on numpy / pandas against the statsmodels stack — there is NO Nixtla /
mlforecast dependency for the scoring itself. The implementation is cross-checked
against ``utilsforecast.losses.mase`` at the evaluation stage (see the modelling
notebook) to confirm the two agree to floating-point tolerance.

Design decisions that matter for trust
---------------------------------------
* **MASE denominator = in-sample, training-only.** The scale is the mean absolute
  error of the one-step seasonal-naive forecast (period ``m``) computed on the
  TRAINING portion of each fold ONLY. The test set never touches the scale, so
  there is no leakage of test information into the metric. MASE < 1 therefore
  means "better than the seasonal-naive baseline" on the same footing.

* **Rolling-origin, expanding-window CV.** Train up to a cutoff, forecast the next
  ``horizon`` days, roll the cutoff forward by ``step_size``, repeat. A random
  train/test split on a time series leaks the future into the past and is never
  used. One shared :class:`CVConfig` is built once and reused identically by every
  model, so the comparison is valid — same cutoffs, same folds, same scale per
  fold for all of them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd

SEASONAL_PERIOD = 7  # weekly


# =============================================================================
# Point-error metrics
# =============================================================================

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def seasonal_naive_scale(y_train: np.ndarray, m: int = SEASONAL_PERIOD) -> float:
    """In-sample one-step seasonal-naive MAE — the MASE denominator.

    scale = mean_{t > m} | y_train[t] - y_train[t - m] |

    Computed on the TRAINING series only. This is exactly the in-sample error
    of the "this period ≈ last period" forecast that MASE scales against.
    """
    y = np.asarray(y_train, dtype=float)
    if y.shape[0] <= m:
        raise ValueError(
            f"need more than m={m} training points to compute the seasonal-naive "
            f"scale; got {y.shape[0]}"
        )
    diffs = np.abs(y[m:] - y[:-m])
    scale = float(np.mean(diffs))
    if scale == 0.0:
        raise ValueError("seasonal-naive scale is zero (flat training series?)")
    return scale


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    m: int = SEASONAL_PERIOD,
) -> float:
    """Mean Absolute Scaled Error.

    MASE = MAE(y_true, y_pred) / seasonal_naive_scale(y_train, m)

    The scale comes from ``y_train`` (the fold's training portion) only —
    never from ``y_true`` / the test set.
    """
    scale = seasonal_naive_scale(y_train, m)
    return mae(y_true, y_pred) / scale


# =============================================================================
# Rolling-origin (expanding-window) cross-validation
# =============================================================================

@dataclass(frozen=True)
class CVConfig:
    """The single shared CV configuration, reused identically by every model.

    horizon    : forecast length per origin (7 = one week ahead, the ordering
                 decision the project is framed around).
    step_size  : how far the origin rolls between folds (7 = weekly origins).
    n_origins  : number of folds (>= 26 weekly origins as specified).
    m          : seasonal period for the MASE scale and the naive baseline.
    """

    horizon: int = 7
    step_size: int = 7
    n_origins: int = 26
    m: int = SEASONAL_PERIOD

    def splits(self, n_obs: int) -> list[tuple[int, int, int]]:
        """Return ``[(cutoff, test_start, test_end), ...]`` index triples.

        Expanding window: fold training data is always ``y[0:cutoff]``; the test
        block is ``y[cutoff:cutoff + horizon]``. The last fold's test block ends
        on the final observation; earlier origins step back by ``step_size``.
        """
        last_cutoff = n_obs - self.horizon
        first_cutoff = last_cutoff - (self.n_origins - 1) * self.step_size
        if first_cutoff <= self.m:
            raise ValueError(
                f"first cutoff {first_cutoff} too small for n_obs={n_obs}, "
                f"n_origins={self.n_origins}, horizon={self.horizon}"
            )
        return [
            (c, c, c + self.horizon)
            for c in range(first_cutoff, last_cutoff + 1, self.step_size)
        ]

    def describe(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        """Human-readable fold table (dates and training sizes)."""
        rows = []
        for i, (cutoff, ts, te) in enumerate(self.splits(len(index))):
            rows.append({
                "fold": i,
                "n_train": cutoff,
                "train_start": index[0].date(),
                "train_end": index[cutoff - 1].date(),
                "test_start": index[ts].date(),
                "test_end": index[te - 1].date(),
            })
        return pd.DataFrame(rows)


# A model factory returns a FRESH, unfitted model each fold (no state leakage
# across folds). Models follow the fit/predict interface in src/models.py.
ModelFactory = Callable[[], "object"]


def rolling_origin_cv(
    model_factory: ModelFactory,
    y: pd.Series,
    config: CVConfig,
    X: Optional[pd.DataFrame] = None,
    model_name: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run expanding-window CV for one model.

    Parameters
    ----------
    model_factory : callable returning a fresh model exposing
        ``fit(y_train, X_train)`` and ``predict(h, X_future)``.
    y : target series with a DatetimeIndex.
    config : the shared :class:`CVConfig`.
    X : optional exogenous regressors aligned to ``y`` (event flags).

    Returns
    -------
    (per_fold, predictions)
        per_fold    : one row per origin with mae / rmse / mase and dates.
        predictions : long frame (date, actual, prediction, fold, model) for
                      plotting forecast-vs-actual.
    """
    if not isinstance(y.index, pd.DatetimeIndex):
        raise TypeError("y must have a DatetimeIndex")
    name = model_name or getattr(model_factory, "model_name", "model")

    fold_rows: list[dict] = []
    pred_rows: list[pd.DataFrame] = []

    for fold, (cutoff, ts, te) in enumerate(config.splits(len(y))):
        y_train = y.iloc[:cutoff]
        y_test = y.iloc[ts:te]
        X_train = X.iloc[:cutoff] if X is not None else None
        X_future = X.iloc[ts:te] if X is not None else None

        model = model_factory()
        model.fit(y_train, X_train)
        y_pred = np.asarray(model.predict(config.horizon, X_future), dtype=float)

        fold_rows.append({
            "model": name,
            "fold": fold,
            "n_train": cutoff,
            "train_end": y.index[cutoff - 1].date(),
            "test_start": y.index[ts].date(),
            "test_end": y.index[te - 1].date(),
            "mae": mae(y_test.values, y_pred),
            "rmse": rmse(y_test.values, y_pred),
            "mase": mase(y_test.values, y_pred, y_train.values, config.m),
        })
        pred_rows.append(pd.DataFrame({
            "date": y_test.index,
            "actual": y_test.values,
            "prediction": y_pred,
            "fold": fold,
            "model": name,
        }))

    return pd.DataFrame(fold_rows), pd.concat(pred_rows, ignore_index=True)


def summarize(per_fold: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-fold scores to one row per model."""
    g = per_fold.groupby("model", sort=False)
    out = g.agg(
        n_folds=("fold", "count"),
        mase_mean=("mase", "mean"),
        mase_median=("mase", "median"),
        mase_std=("mase", "std"),
        mae_mean=("mae", "mean"),
        rmse_mean=("rmse", "mean"),
    ).reset_index()
    return out.sort_values("mase_mean").reset_index(drop=True)
