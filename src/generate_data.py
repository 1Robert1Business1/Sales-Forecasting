"""
==============================================================================
 SYNTHETIC SALES GENERATOR — UK SPORTS BAR
 Stage 2 of the sales-forecasting build (see PROJECT_BRIEF.md).

 ----------------------------------------------------------------------------
 THIS DATA IS SYNTHETIC. IT IS NOT REAL.
 ----------------------------------------------------------------------------
 No real bar's takings are used or simulated. The series is generated from
 documented, separable structural components under a fixed random seed so the
 dataset is fully reproducible from this script.

 Structure
 ---------
     sales[t] = baseline
              + trend[t]
              + weekly_seasonality[day_of_week[t]]
              + annual_seasonality[day_of_year[t]]
              + event_effects[t]            (football + bank holidays, % of mu)
              + heteroscedastic_noise[t]    (sd proportional to mu)

 Every component is its own function and is also written into the output
 CSV as a column, so the EDA stage can STL-decompose `sales` and confirm
 what the decomposition recovers matches what was built in.

 Football fixtures
 -----------------
 PLAUSIBLE, NOT REAL. A weekday-conditioned Bernoulli draw populates a
 recurring season (August–May) with fixture days; each fixture's importance
 is drawn from a Beta(2, 5) so big matches are rarer than ordinary ones.
 Importance is mapped linearly into a 4–25 % uplift band — i.e. a midweek
 small-fixture night nudges trade ~4–6 %, while a derby / Champions League
 final pushes ~22–25 %. Real fixtures would carry actual club names,
 kickoff times and competitions — we are emulating only the statistical
 shape of "how often is the bar showing football and how big is the draw".

 Bank holidays
 -------------
 Drawn from `holidays.country_holidays("GB", subdiv="ENG")` so the calendar
 is the real England & Wales bank holiday calendar. Each holiday carries a
 per-name uplift (Christmas Day quiet, Boxing Day a football peak, others
 a moderate positive). Unknown names fall back to a default uplift.

 Evaluation tooling note (relevant to Stage 5)
 ---------------------------------------------
 MASE and rolling-origin (walk-forward) cross-validation are HAND-ROLLED
 in src/evaluation.py against the statsmodels stack — no Nixtla /
 mlforecast dependency for the headline scoring. At the evaluation stage
 the implementation is cross-checked against `utilsforecast` to confirm
 the two agree to floating-point tolerance.

 ----------------------------------------------------------------------------
 Run from the repo root:
     python -m src.generate_data
     python -m src.generate_data --plot      # also writes the sanity PNG
 ----------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import holidays  # noqa: E402

from src.config import RANDOM_SEED, set_seeds  # noqa: E402


# =============================================================================
# Generating parameters — committed alongside the seed for reproducibility.
# =============================================================================

@dataclass(frozen=True)
class GeneratorConfig:
    # Series span — 3 full calendar years gives two+ full annual cycles.
    start_date: date = date(2022, 1, 1)
    end_date: date = date(2024, 12, 31)

    # Level / trend
    baseline: float = 2500.0          # £/day
    trend_per_year: float = 90.0      # mild upward drift, £/day per year

    # Weekly deviations (£) from baseline. Sum ≈ 0 so they are pure shape.
    # Order: Mon, Tue, Wed, Thu, Fri, Sat, Sun.
    weekly_deviations: tuple = (-700.0, -800.0, -500.0, -200.0, 800.0, 1300.0, 100.0)

    # Annual seasonality — three Gaussian bumps on day-of-year, on the circle.
    dec_amplitude: float = 500.0
    dec_doy: int = 359
    dec_width: float = 25.0
    summer_amplitude: float = 300.0
    summer_doy: int = 172
    summer_width: float = 35.0
    jan_amplitude: float = -250.0
    jan_doy: int = 20
    jan_width: float = 18.0

    # Football season: months 8..12, 1..5.
    football_season_start_month: int = 8
    football_season_end_month: int = 5
    # Per-weekday match-day probability (Mon..Sun).
    fixture_probs: tuple = (0.25, 0.30, 0.25, 0.15, 0.10, 0.90, 0.70)
    # Match-day uplift band as a fraction of pre-event mu.
    fixture_uplift_min: float = 0.04
    fixture_uplift_max: float = 0.25
    # Beta distribution for fixture importance — α<β skews to small matches.
    fixture_importance_alpha: float = 2.0
    fixture_importance_beta: float = 5.0

    # Bank-holiday default uplift for any name not in HOLIDAY_UPLIFTS.
    default_holiday_uplift: float = 0.20

    # Heteroscedastic noise — sd proportional to mu.
    noise_cv: float = 0.08

    # Floor — even on the quietest day we are open (no zero-sales rows).
    min_sales: float = 50.0


# Per-name England & Wales bank-holiday uplifts (multipliers on mu).
# Both legacy and current `holidays`-library names are listed so the script
# is robust across library versions.
HOLIDAY_UPLIFTS = {
    "New Year's Day": 0.05,
    "Good Friday": 0.18,
    "Easter Monday": 0.22,
    "Early May Bank Holiday": 0.25,
    "May Day": 0.25,
    "Spring Bank Holiday": 0.28,
    "Late Summer Bank Holiday": 0.30,
    "Summer Bank Holiday": 0.30,
    "Christmas Day": -0.30,
    "Boxing Day": 0.40,
}


# =============================================================================
# Component functions — each returns an array aligned to `dates`.
# =============================================================================

def baseline_component(n_days: int, cfg: GeneratorConfig) -> np.ndarray:
    return np.full(n_days, cfg.baseline, dtype=float)


def trend_component(n_days: int, cfg: GeneratorConfig) -> np.ndarray:
    years_elapsed = np.arange(n_days) / 365.25
    return cfg.trend_per_year * years_elapsed


def weekly_component(dates: pd.DatetimeIndex, cfg: GeneratorConfig) -> np.ndarray:
    weekly = np.array(cfg.weekly_deviations, dtype=float)
    return weekly[np.asarray(dates.weekday)]


def _circular_distance(x: np.ndarray, centre: float, period: float = 365.25) -> np.ndarray:
    d = np.mod(x - centre, period)
    return np.minimum(d, period - d)


def annual_component(dates: pd.DatetimeIndex, cfg: GeneratorConfig) -> np.ndarray:
    doy = np.asarray(dates.dayofyear, dtype=float)
    dec = cfg.dec_amplitude * np.exp(
        -0.5 * (_circular_distance(doy, cfg.dec_doy) / cfg.dec_width) ** 2
    )
    summer = cfg.summer_amplitude * np.exp(
        -0.5 * (_circular_distance(doy, cfg.summer_doy) / cfg.summer_width) ** 2
    )
    january = cfg.jan_amplitude * np.exp(
        -0.5 * (_circular_distance(doy, cfg.jan_doy) / cfg.jan_width) ** 2
    )
    return dec + summer + january


def bank_holiday_effects(
    dates: pd.DatetimeIndex, cfg: GeneratorConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Return (uplift_pct, name) — `name` is "" on non-holiday days."""
    years = sorted({d.year for d in dates})
    cal = holidays.country_holidays("GB", subdiv="ENG", years=years)
    uplift = np.zeros(len(dates), dtype=float)
    name = np.array([""] * len(dates), dtype=object)
    for i, d in enumerate(dates.date):
        if d in cal:
            n = cal.get(d, "")
            name[i] = n
            uplift[i] = HOLIDAY_UPLIFTS.get(n, cfg.default_holiday_uplift)
    return uplift, name


def _in_football_season(dates: pd.DatetimeIndex, cfg: GeneratorConfig) -> np.ndarray:
    m = np.asarray(dates.month)
    return (m >= cfg.football_season_start_month) | (m <= cfg.football_season_end_month)


def football_fixtures(
    dates: pd.DatetimeIndex, cfg: GeneratorConfig, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a plausible recurring fixture schedule.

    Returns (is_match_day, importance, uplift_pct), aligned to `dates`.
    Importance ∈ [0, 1] from Beta(α, β); uplift_pct is a linear map into
    [fixture_uplift_min, fixture_uplift_max].
    """
    n = len(dates)
    is_match = np.zeros(n, dtype=bool)
    importance = np.zeros(n, dtype=float)
    uplift = np.zeros(n, dtype=float)

    probs = np.array(cfg.fixture_probs, dtype=float)
    in_season = _in_football_season(dates, cfg)
    weekdays = np.asarray(dates.weekday)
    months = np.asarray(dates.month)
    days = np.asarray(dates.day)

    for i in range(n):
        if not in_season[i]:
            continue
        # Real Premier League does not play on Christmas Day — suppress.
        if months[i] == 12 and days[i] == 25:
            continue
        if rng.random() < probs[weekdays[i]]:
            is_match[i] = True
            imp = rng.beta(cfg.fixture_importance_alpha, cfg.fixture_importance_beta)
            importance[i] = imp
            uplift[i] = cfg.fixture_uplift_min + imp * (
                cfg.fixture_uplift_max - cfg.fixture_uplift_min
            )
    return is_match, importance, uplift


# =============================================================================
# Assembly
# =============================================================================

def generate_series(cfg: GeneratorConfig, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate the full synthetic sales series with all components attached."""
    set_seeds(seed)
    rng = np.random.default_rng(seed)

    dates = pd.date_range(cfg.start_date, cfg.end_date, freq="D")
    n = len(dates)

    baseline = baseline_component(n, cfg)
    trend = trend_component(n, cfg)
    weekly = weekly_component(dates, cfg)
    annual = annual_component(dates, cfg)

    holiday_uplift, holiday_name = bank_holiday_effects(dates, cfg)
    is_match, match_importance, match_uplift = football_fixtures(dates, cfg, rng)

    mu_pre = baseline + trend + weekly + annual  # additive structural part
    total_event_pct = holiday_uplift + match_uplift
    events = mu_pre * total_event_pct
    mu = mu_pre + events

    noise_sd = cfg.noise_cv * np.maximum(mu, cfg.min_sales)
    noise = rng.normal(0.0, noise_sd)
    sales = np.maximum(mu + noise, cfg.min_sales)

    return pd.DataFrame({
        "date": dates.date,
        "sales": np.round(sales, 2),
        "baseline": baseline,
        "trend": np.round(trend, 4),
        "weekly": weekly,
        "annual": np.round(annual, 4),
        "events": np.round(events, 4),
        "mu": np.round(mu, 4),
        "is_match_day": is_match.astype(int),
        "match_importance": np.round(match_importance, 4),
        "is_bank_holiday": (holiday_uplift != 0).astype(int),
        "holiday_name": holiday_name,
    })


# =============================================================================
# I/O
# =============================================================================

DATA_DIR = ROOT / "data"
OUTPUT_PATH = DATA_DIR / "sales.csv"
PLOT_PATH = ROOT / "results" / "synthetic_data_sanity.png"


def save(df: pd.DataFrame, path: Path = OUTPUT_PATH, seed: int = RANDOM_SEED) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# SYNTHETIC DATA — generated by src/generate_data.py. NOT REAL.\n"
        f"# RANDOM_SEED={seed}\n"
        f"# date range: {df['date'].iloc[0]} to {df['date'].iloc[-1]} "
        f"({len(df)} daily rows)\n"
    )
    with path.open("w", newline="") as f:
        f.write(header)
        df.to_csv(f, index=False)
    return path


def render_plot(df: pd.DataFrame, out_path: Path = PLOT_PATH) -> Path:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    dates_pd = pd.to_datetime(df["date"])
    fig, axes = plt.subplots(3, 1, figsize=(13, 9.5), sharex=False)

    ax = axes[0]
    ax.plot(dates_pd, df["sales"], lw=0.7, alpha=0.85, label="sales (synthetic)")
    ax.plot(dates_pd, df["mu"], lw=1.2, color="C3", alpha=0.9, label="mean μ (pre-noise)")
    ax.set_title(
        "Synthetic daily sales — UK sports bar (3 years, seeded). NOT REAL DATA."
    )
    ax.set_ylabel("£ / day")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax = axes[1]
    mask = (dates_pd >= "2023-01-01") & (dates_pd <= "2023-03-31")
    sub = df.loc[mask].copy()
    sub_dates = pd.to_datetime(sub["date"])
    ax.plot(sub_dates, sub["sales"], "o-", ms=3, lw=0.8, label="sales")
    ax.plot(sub_dates, sub["mu"], color="C3", lw=1.2, label="mean μ")
    matches = sub[sub["is_match_day"] == 1]
    ax.scatter(pd.to_datetime(matches["date"]), matches["sales"],
               s=30, edgecolor="C2", facecolor="none", lw=1.2, label="match day")
    hols = sub[sub["is_bank_holiday"] == 1]
    ax.scatter(pd.to_datetime(hols["date"]), hols["sales"],
               s=70, color="C1", marker="^", label="bank holiday", zorder=5)
    ax.set_title("Zoom: Q1 2023 — weekly rhythm + events visible")
    ax.set_ylabel("£ / day")
    ax.legend(loc="upper left", ncol=4)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    ax = axes[2]
    ax.plot(dates_pd, df["trend"] + df["baseline"], label="baseline + trend", lw=1.2)
    ax.plot(dates_pd, df["annual"], label="annual season", lw=1.0, alpha=0.85)
    ax.plot(dates_pd, df["weekly"], label="weekly season", lw=0.5, alpha=0.5)
    ax.plot(dates_pd, df["events"], label="events (matches + bank hols)", lw=0.5, alpha=0.7)
    ax.set_title("Structural components — individually recoverable by EDA")
    ax.set_ylabel("£ deviation")
    ax.legend(loc="upper left", ncol=4)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


# =============================================================================
# CLI
# =============================================================================

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate the synthetic UK-sports-bar sales series."
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=OUTPUT_PATH,
        help=f"Output CSV path (default: {OUTPUT_PATH.relative_to(ROOT)})",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help=f"Also render a sanity-check PNG to {PLOT_PATH.relative_to(ROOT)}",
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED,
        help=f"Random seed (default: {RANDOM_SEED}).",
    )
    args = parser.parse_args(argv)

    cfg = GeneratorConfig()
    df = generate_series(cfg, seed=args.seed)
    save_path = save(df, args.output, seed=args.seed)

    print(f"Rows: {len(df):,} ({df['date'].iloc[0]} → {df['date'].iloc[-1]})")
    print(f"Mean daily sales: £{df['sales'].mean():,.0f}  "
          f"(min £{df['sales'].min():,.0f}, max £{df['sales'].max():,.0f})")
    print(f"Match days: {df['is_match_day'].sum():,}  "
          f"Bank holidays: {df['is_bank_holiday'].sum():,}")
    print(f"Wrote {save_path}")

    if args.plot:
        plot_path = render_plot(df)
        print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
