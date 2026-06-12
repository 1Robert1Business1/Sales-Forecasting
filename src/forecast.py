"""
forecast.py — reusable next-N-weeks sales forecast.

The single callable :func:`forecast_next_weeks` is the deployment surface
of the project: it takes a historical sales series, the matching event
flags, the known-future event flags for the horizon, and returns the
next-N-weeks daily forecast with prediction intervals. Built on the
**airline SARIMAX (0,1,1)(0,1,1,7)** model chosen on parsimony in
`notebooks/03_evaluation.ipynb` (Stage 5); the model class lives in
:mod:`src.models` and this module is a thin operational facade — no
model code is re-implemented here.

What the forecast does, and does not, give you
----------------------------------------------
The model captures the **structured** demand: the weekly rhythm, the
annual rhythm, and the systematic uplift a fixture or a bank holiday
puts on a day. It does **not** predict individual match-night
surprises — cup-final shocks, weather-driven swings, one-off closures —
because nothing in the inputs encodes them. The CV evidence is honest
about this: per-fold MASE peaks at ~1.6 on event-heavy weeks.

Prediction intervals are returned from the SARIMAX state-space variance
estimator. That assumes **roughly normal one-step errors**, which makes
the intervals an approximate plan-level cover, not a tail-risk
guarantee. The same caveat applies to safety-stock numbers derived from
the interval width.

Streamlit (noted, not built)
----------------------------
A Streamlit wrapper would accept a history CSV upload and either a
future-event-flag CSV or an in-form schedule editor, call
``forecast_next_weeks(...)``, and render the returned DataFrame as a
line chart with the lower/upper band. Listed in the brief's future-work
section; intentionally not built here.

Example
-------
Run it on the synthetic dataset that ships with the repo::

    python -m src.forecast

This fits on all-but-the-last-two-weeks of ``data/sales.csv`` and prints
the 14-day forecast next to the realised actuals, so the calibration
of the intervals is visible without opening anything else.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.models import SARIMAXModel

# The order chosen in Stage 5 (notebooks/03_evaluation.ipynb). Exposed as
# defaults so the function stays a one-liner for the common case; pass
# alternatives only for ablation studies.
DEFAULT_ORDER: tuple[int, int, int] = (0, 1, 1)
DEFAULT_SEASONAL_ORDER: tuple[int, int, int, int] = (0, 1, 1, 7)

# The known-future event flags SARIMAX takes as exogenous regressors.
# Each is justified as known-future in `src/models.py:ProphetModel` /
# `src/models.py:SARIMAXModel`: fixture lists and bank holidays are
# published months ahead; ``match_importance`` is a pre-match attribute
# of the fixture (rivalry / stakes / billing), not anything realised on
# the night.
REQUIRED_EXOG_COLS: tuple[str, ...] = (
    "is_match_day",
    "match_importance",
    "is_bank_holiday",
)


def forecast_next_weeks(
    history: pd.Series,
    history_exog: pd.DataFrame,
    future_exog: pd.DataFrame,
    n_weeks: int = 2,
    alpha: float = 0.05,
    order: tuple[int, int, int] = DEFAULT_ORDER,
    seasonal_order: tuple[int, int, int, int] = DEFAULT_SEASONAL_ORDER,
) -> pd.DataFrame:
    """Forecast the next N weeks of daily sales with prediction intervals.

    Parameters
    ----------
    history :
        Historical daily sales as a Series with a daily
        :class:`pandas.DatetimeIndex`. No missing days, no NaNs.
    history_exog :
        Event flags for every day in ``history``. Same index as
        ``history``. Required columns: see
        :data:`REQUIRED_EXOG_COLS`.
    future_exog :
        Event flags for the forecast horizon. The index must be a daily
        :class:`pandas.DatetimeIndex` starting the day *after* the last
        day of ``history`` and covering exactly ``n_weeks * 7`` days.
        Same columns as ``history_exog``.
    n_weeks :
        Forecast horizon in weeks. The model was selected on a one-
        week-ahead CV (Stage 4); beyond ~4 weeks the seasonal-naive
        anchor weakens and the forecast variance grows quickly.
    alpha :
        Significance level for the prediction interval.
        ``alpha = 0.05`` → 95 % interval. Read as a plan-level cover
        under roughly-normal errors; event-night tails run wider.
    order, seasonal_order :
        SARIMAX orders. Default to the Stage-5 airline choice; override
        only for ablation.

    Returns
    -------
    :class:`pandas.DataFrame`
        Columns ``date``, ``forecast``, ``lower``, ``upper``.

    Raises
    ------
    ValueError
        If inputs are mis-aligned, mis-typed, or have missing data.

    Examples
    --------
    >>> import pandas as pd
    >>> df = (pd.read_csv("data/sales.csv", comment="#", parse_dates=["date"])
    ...         .set_index("date").asfreq("D"))
    >>> fc = forecast_next_weeks(
    ...     history       = df["sales"].iloc[:-14],
    ...     history_exog  = df[list(REQUIRED_EXOG_COLS)].iloc[:-14],
    ...     future_exog   = df[list(REQUIRED_EXOG_COLS)].iloc[-14:],
    ...     n_weeks       = 2,
    ... )
    >>> fc.head()
    """
    horizon = int(n_weeks) * 7
    if horizon <= 0:
        raise ValueError(f"n_weeks must be positive, got {n_weeks}")

    _validate(history, history_exog, future_exog, horizon)

    model = SARIMAXModel(order=order, seasonal_order=seasonal_order)
    cols = list(REQUIRED_EXOG_COLS)
    model.fit(history, history_exog[cols])
    interval = model.predict_interval(horizon, future_exog[cols], alpha=alpha)

    return pd.DataFrame({
        "date": future_exog.index,
        "forecast": np.round(interval["mean"].values, 2),
        "lower":    np.round(interval["lower"].values, 2),
        "upper":    np.round(interval["upper"].values, 2),
    }).reset_index(drop=True)


def _validate(
    history: pd.Series,
    history_exog: pd.DataFrame,
    future_exog: pd.DataFrame,
    horizon: int,
) -> None:
    if not isinstance(history.index, pd.DatetimeIndex):
        raise TypeError("history must have a DatetimeIndex")
    if history.isna().any():
        raise ValueError("history contains NaNs")

    for name, frame in (("history_exog", history_exog), ("future_exog", future_exog)):
        missing = [c for c in REQUIRED_EXOG_COLS if c not in frame.columns]
        if missing:
            raise ValueError(f"{name} missing required columns: {missing}")
        if frame[list(REQUIRED_EXOG_COLS)].isna().any().any():
            raise ValueError(f"{name} contains NaNs in required columns")

    if len(history_exog) != len(history):
        raise ValueError(
            f"history_exog length {len(history_exog)} != history length {len(history)}"
        )
    if not (history_exog.index == history.index).all():
        raise ValueError("history_exog index does not match history index")

    if len(future_exog) != horizon:
        raise ValueError(
            f"future_exog length {len(future_exog)} != n_weeks * 7 = {horizon}"
        )
    expected_start = history.index[-1] + pd.Timedelta(days=1)
    if future_exog.index[0] != expected_start:
        raise ValueError(
            f"future_exog must start on {expected_start.date()}, "
            f"got {pd.Timestamp(future_exog.index[0]).date()}"
        )


# ============================================================================
# Runnable example — `python -m src.forecast`
# ============================================================================

def _example() -> None:
    """Fit on all-but-the-last-2-weeks of ``data/sales.csv`` and show the result."""
    root = Path(__file__).resolve().parents[1]
    df = (
        pd.read_csv(root / "data" / "sales.csv", comment="#", parse_dates=["date"])
        .set_index("date").asfreq("D")
    )

    n_weeks = 2
    horizon = n_weeks * 7
    history       = df["sales"].iloc[:-horizon]
    history_exog  = df[list(REQUIRED_EXOG_COLS)].iloc[:-horizon].astype(float)
    future_exog   = df[list(REQUIRED_EXOG_COLS)].iloc[-horizon:].astype(float)
    actuals       = df["sales"].iloc[-horizon:]

    fc = forecast_next_weeks(
        history=history,
        history_exog=history_exog,
        future_exog=future_exog,
        n_weeks=n_weeks,
    )

    out = fc.copy()
    out["actual"] = actuals.values
    out["abs_err"] = (out["actual"] - out["forecast"]).abs().round(2)
    mae = float(out["abs_err"].mean())

    print(f"Trained on {len(history):,} days "
          f"({history.index[0].date()} → {history.index[-1].date()}).")
    print(f"Forecasting the next {horizon} days "
          f"({future_exog.index[0].date()} → {future_exog.index[-1].date()}):\n")
    print(out.to_string(index=False))
    print(f"\nMean absolute error on these {horizon} days: £{mae:,.0f}/day")
    print("(This window includes Christmas Day + Boxing Day — an event-heavy "
          "test by construction.)")


if __name__ == "__main__":
    _example()
