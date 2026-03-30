"""
src/metrics.py

Performance metrics for the Market Regimes and Trading Signals project.

This module computes common portfolio evaluation metrics for both the strategy
and the benchmark.

Design principles:
- simple and transparent
- no hidden assumptions
- reusable across full-sample and sub-sample analysis
- consistent with daily return inputs
"""

from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd

from src.config import RISK_FREE_RATE, TRADING_DAYS_PER_YEAR


# ---------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------

def validate_return_series(returns: pd.Series, series_name: str = "returns") -> pd.Series:
    """
    Validate and clean a return series.

    Parameters
    ----------
    returns : pd.Series
        Daily return series.
    series_name : str
        Name used in error messages.

    Returns
    -------
    pd.Series
        Cleaned return series with NaNs dropped.

    Raises
    ------
    TypeError
        If input is not a pandas Series.
    ValueError
        If the series is empty after dropping NaNs.
    """
    if not isinstance(returns, pd.Series):
        raise TypeError(f"{series_name} must be a pandas Series.")

    clean = returns.dropna().copy()

    if clean.empty:
        raise ValueError(f"{series_name} is empty after dropping NaNs.")

    return clean


def validate_cumulative_series(cumulative: pd.Series, series_name: str = "cumulative") -> pd.Series:
    """
    Validate and clean a cumulative value series.

    Parameters
    ----------
    cumulative : pd.Series
        Cumulative value series.
    series_name : str
        Name used in error messages.

    Returns
    -------
    pd.Series
        Cleaned cumulative series with NaNs dropped.
    """
    if not isinstance(cumulative, pd.Series):
        raise TypeError(f"{series_name} must be a pandas Series.")

    clean = cumulative.dropna().copy()

    if clean.empty:
        raise ValueError(f"{series_name} is empty after dropping NaNs.")

    return clean


# ---------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------

def total_return(cumulative: pd.Series) -> float:
    """
    Compute total return from a cumulative value series.

    Parameters
    ----------
    cumulative : pd.Series
        Cumulative value series.

    Returns
    -------
    float
        Total return.
    """
    cumulative = validate_cumulative_series(cumulative, "cumulative")
    return float(cumulative.iloc[-1] / cumulative.iloc[0] - 1.0)


def cagr(
    cumulative: pd.Series,
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Compute compound annual growth rate (CAGR).

    Parameters
    ----------
    cumulative : pd.Series
        Cumulative value series.
    trading_days_per_year : int
        Number of trading days per year.

    Returns
    -------
    float
        CAGR.
    """
    cumulative = validate_cumulative_series(cumulative, "cumulative")

    n_periods = len(cumulative) - 1
    if n_periods <= 0:
        return np.nan

    n_years = n_periods / trading_days_per_year
    if n_years <= 0:
        return np.nan

    start_value = cumulative.iloc[0]
    end_value = cumulative.iloc[-1]

    if start_value <= 0 or end_value <= 0:
        return np.nan

    return float((end_value / start_value) ** (1.0 / n_years) - 1.0)


def annualized_return(
    returns: pd.Series,
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Compute annualized return from daily returns.

    Parameters
    ----------
    returns : pd.Series
        Daily return series.
    trading_days_per_year : int
        Number of trading days per year.

    Returns
    -------
    float
        Annualized return.
    """
    returns = validate_return_series(returns)

    growth = (1.0 + returns).prod()
    n_periods = len(returns)

    if n_periods == 0 or growth <= 0:
        return np.nan

    return float(growth ** (trading_days_per_year / n_periods) - 1.0)


def annualized_volatility(
    returns: pd.Series,
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Compute annualized volatility from daily returns.

    Parameters
    ----------
    returns : pd.Series
        Daily return series.
    trading_days_per_year : int
        Number of trading days per year.

    Returns
    -------
    float
        Annualized volatility.
    """
    returns = validate_return_series(returns)
    return float(returns.std(ddof=1) * np.sqrt(trading_days_per_year))


def downside_volatility(
    returns: pd.Series,
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
    target_return: float = 0.0,
) -> float:
    """
    Compute annualized downside volatility.

    Parameters
    ----------
    returns : pd.Series
        Daily return series.
    trading_days_per_year : int
        Number of trading days per year.
    target_return : float
        Daily target return threshold.

    Returns
    -------
    float
        Annualized downside volatility.
    """
    returns = validate_return_series(returns)

    downside = returns[returns < target_return]
    if downside.empty:
        return 0.0

    return float(downside.std(ddof=1) * np.sqrt(trading_days_per_year))


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = RISK_FREE_RATE,
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Compute annualized Sharpe ratio.

    Assumes risk_free_rate is annualized.

    Parameters
    ----------
    returns : pd.Series
        Daily return series.
    risk_free_rate : float
        Annualized risk-free rate.
    trading_days_per_year : int
        Number of trading days per year.

    Returns
    -------
    float
        Sharpe ratio.
    """
    returns = validate_return_series(returns)

    daily_rf = risk_free_rate / trading_days_per_year
    excess_returns = returns - daily_rf

    vol = annualized_volatility(excess_returns, trading_days_per_year)
    if vol == 0 or np.isnan(vol):
        return np.nan

    ann_excess_return = excess_returns.mean() * trading_days_per_year
    return float(ann_excess_return / vol)


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = RISK_FREE_RATE,
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Compute annualized Sortino ratio.

    Parameters
    ----------
    returns : pd.Series
        Daily return series.
    risk_free_rate : float
        Annualized risk-free rate.
    trading_days_per_year : int
        Number of trading days per year.

    Returns
    -------
    float
        Sortino ratio.
    """
    returns = validate_return_series(returns)

    daily_rf = risk_free_rate / trading_days_per_year
    excess_returns = returns - daily_rf

    downside_vol = downside_volatility(excess_returns, trading_days_per_year, target_return=0.0)
    if downside_vol == 0 or np.isnan(downside_vol):
        return np.nan

    ann_excess_return = excess_returns.mean() * trading_days_per_year
    return float(ann_excess_return / downside_vol)


def max_drawdown(cumulative: pd.Series) -> float:
    """
    Compute maximum drawdown from a cumulative value series.

    Parameters
    ----------
    cumulative : pd.Series
        Cumulative value series.

    Returns
    -------
    float
        Maximum drawdown.
    """
    cumulative = validate_cumulative_series(cumulative, "cumulative")

    running_peak = cumulative.cummax()
    drawdown = cumulative / running_peak - 1.0
    return float(drawdown.min())


def calmar_ratio(cumulative: pd.Series) -> float:
    """
    Compute Calmar ratio as CAGR divided by absolute max drawdown.

    Parameters
    ----------
    cumulative : pd.Series
        Cumulative value series.

    Returns
    -------
    float
        Calmar ratio.
    """
    cumulative = validate_cumulative_series(cumulative, "cumulative")

    growth_rate = cagr(cumulative)
    mdd = max_drawdown(cumulative)

    if pd.isna(growth_rate) or pd.isna(mdd) or mdd == 0:
        return np.nan

    return float(growth_rate / abs(mdd))


# ---------------------------------------------------------------------
# Summary table builders
# ---------------------------------------------------------------------

def compute_performance_metrics(
    returns: pd.Series,
    cumulative: pd.Series,
    label: str | None = None,
) -> pd.Series:
    """
    Compute a full metric set for one return/cumulative pair.

    Parameters
    ----------
    returns : pd.Series
        Daily return series.
    cumulative : pd.Series
        Cumulative value series.
    label : str | None
        Optional label for naming the series.

    Returns
    -------
    pd.Series
        Performance metrics.
    """
    returns = validate_return_series(returns, "returns")
    cumulative = validate_cumulative_series(cumulative, "cumulative")

    metrics = pd.Series({
        "n_obs": len(returns),
        "avg_daily_return": returns.mean(),
        "total_return": total_return(cumulative),
        "annualized_return": annualized_return(returns),
        "cagr": cagr(cumulative),
        "annualized_volatility": annualized_volatility(returns),
        "downside_volatility": downside_volatility(returns),
        "sharpe_ratio": sharpe_ratio(returns),
        "sortino_ratio": sortino_ratio(returns),
        "max_drawdown": max_drawdown(cumulative),
        "calmar_ratio": calmar_ratio(cumulative),
    })

    if label is not None:
        metrics.name = label

    return metrics


def summarize_strategy_vs_benchmark(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize performance metrics for strategy and benchmark over the full sample.

    Expected columns
    ----------------
    Strategy:
    - strategy_ret_net
    - strategy_cum_net

    Benchmark:
    - benchmark_ret
    - benchmark_cum

    Parameters
    ----------
    df : pd.DataFrame
        Backtested dataframe.

    Returns
    -------
    pd.DataFrame
        Metric table with rows for strategy and benchmark.
    """
    required = [
        "strategy_ret_net",
        "strategy_cum_net",
        "benchmark_ret",
        "benchmark_cum",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    strategy_metrics = compute_performance_metrics(
        returns=df["strategy_ret_net"],
        cumulative=df["strategy_cum_net"],
        label="strategy",
    )

    benchmark_metrics = compute_performance_metrics(
        returns=df["benchmark_ret"],
        cumulative=df["benchmark_cum"],
        label="benchmark",
    )

    return pd.DataFrame([strategy_metrics, benchmark_metrics])


def summarize_by_sample_period(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize strategy and benchmark metrics by sample period.

    Parameters
    ----------
    df : pd.DataFrame
        Backtested dataframe containing 'sample_period'.

    Returns
    -------
    pd.DataFrame
        Multi-indexed table:
        index = (sample_period, portfolio)
    """
    if "sample_period" not in df.columns:
        raise ValueError("Column 'sample_period' not found.")

    frames = []

    for period, period_df in df.groupby("sample_period"):
        summary = summarize_strategy_vs_benchmark(period_df).copy()
        summary["sample_period"] = period
        summary["portfolio"] = summary.index
        frames.append(summary.reset_index(drop=True))

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out = out.set_index(["sample_period", "portfolio"])

    return out


def compare_multiple_strategies(
    strategy_specs: Iterable[tuple[str, pd.Series, pd.Series]]
) -> pd.DataFrame:
    """
    Compare multiple strategies from explicit (name, returns, cumulative) tuples.

    Parameters
    ----------
    strategy_specs : iterable of tuples
        Each tuple is:
        (label, returns_series, cumulative_series)

    Returns
    -------
    pd.DataFrame
        Metric table with one row per strategy.
    """
    rows: List[pd.Series] = []

    for label, returns, cumulative in strategy_specs:
        rows.append(
            compute_performance_metrics(
                returns=returns,
                cumulative=cumulative,
                label=label,
            )
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)