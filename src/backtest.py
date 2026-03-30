"""
src/backtest.py

Backtesting logic for the Market Regimes and Trading Signals project.

Design principles:
- no look-ahead bias
- explicit execution lag
- transaction costs based on turnover
- benchmark comparison against buy-and-hold SPY
- easy support for in-sample / out-of-sample evaluation
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import pandas as pd

from src.config import (
    INITIAL_CAPITAL,
    TRAIN_END,
    TRANSACTION_COST_BPS,
    VALIDATION_END,
)


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def validate_backtest_inputs(df: pd.DataFrame) -> None:
    """
    Validate required inputs for backtesting.

    Required columns:
    - spy_ret_1d
    - target_exposure

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Raises
    ------
    TypeError
        If input is not a DataFrame.
    ValueError
        If dataframe is empty or required columns are missing.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Input dataframe is empty.")

    required = ["spy_ret_1d", "target_exposure"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required backtest columns: {missing}")


# ---------------------------------------------------------------------
# Sample splitting
# ---------------------------------------------------------------------

def add_sample_labels(
    df: pd.DataFrame,
    train_end: Optional[str] = None,
    validation_end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Label each row as train, validation, or test based on index date.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with DatetimeIndex.
    train_end : Optional[str]
        End date for training sample.
    validation_end : Optional[str]
        End date for validation sample.

    Returns
    -------
    pd.DataFrame
        Dataframe with 'sample_period' column added.
    """
    out = df.copy()
    out.index = pd.to_datetime(out.index)
    out.sort_index(inplace=True)

    train_end = TRAIN_END if train_end is None else train_end
    validation_end = VALIDATION_END if validation_end is None else validation_end

    out["sample_period"] = "test"

    if train_end is not None:
        out.loc[out.index <= pd.Timestamp(train_end), "sample_period"] = "train"

    if validation_end is not None:
        validation_mask = (
            (out.index > pd.Timestamp(train_end))
            & (out.index <= pd.Timestamp(validation_end))
        )
        out.loc[validation_mask, "sample_period"] = "validation"

    return out


def get_sample_subset(df: pd.DataFrame, sample_period: str) -> pd.DataFrame:
    """
    Return a subset for a given sample period.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing 'sample_period'.
    sample_period : str
        One of {'train', 'validation', 'test'}.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe.
    """
    if "sample_period" not in df.columns:
        raise ValueError("Column 'sample_period' not found.")

    return df.loc[df["sample_period"] == sample_period].copy()


# ---------------------------------------------------------------------
# Core backtest mechanics
# ---------------------------------------------------------------------

def apply_execution_lag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply one-day lag to target exposure to avoid look-ahead bias.

    Interpretation:
    - target_exposure is determined using information available at day t
    - actual position is held on day t+1

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing 'target_exposure'.

    Returns
    -------
    pd.DataFrame
        Dataframe with 'position' column added.
    """
    if "target_exposure" not in df.columns:
        raise ValueError("Column 'target_exposure' not found.")

    out = df.copy()
    out["position"] = out["target_exposure"].shift(1).fillna(0.0)
    return out


def compute_turnover(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily turnover based on changes in actual position.

    Turnover is defined as the absolute change in position from one day to the next.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing 'position'.

    Returns
    -------
    pd.DataFrame
        Dataframe with 'position_prev' and 'turnover' added.
    """
    if "position" not in df.columns:
        raise ValueError("Column 'position' not found.")

    out = df.copy()
    out["position_prev"] = out["position"].shift(1).fillna(0.0)
    out["turnover"] = (out["position"] - out["position_prev"]).abs()
    return out


def apply_transaction_costs(
    df: pd.DataFrame,
    transaction_cost_bps: Optional[float] = None,
) -> pd.DataFrame:
    """
    Apply transaction costs based on turnover.

    Cost formula:
        transaction_cost = turnover * cost_rate

    where:
        cost_rate = bps / 10,000

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing 'turnover'.
    transaction_cost_bps : Optional[float]
        Cost in basis points per unit turnover. Defaults to config value.

    Returns
    -------
    pd.DataFrame
        Dataframe with transaction cost columns added.
    """
    if "turnover" not in df.columns:
        raise ValueError("Column 'turnover' not found.")

    out = df.copy()
    transaction_cost_bps = (
        TRANSACTION_COST_BPS if transaction_cost_bps is None else transaction_cost_bps
    )

    cost_rate = transaction_cost_bps / 10_000.0
    out["transaction_cost_rate"] = cost_rate
    out["transaction_cost"] = out["turnover"] * cost_rate

    return out


def compute_return_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute gross and net strategy returns plus benchmark returns.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing:
        - spy_ret_1d
        - position
        - transaction_cost

    Returns
    -------
    pd.DataFrame
        Dataframe with return series added.
    """
    required = ["spy_ret_1d", "position", "transaction_cost"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for return calculation: {missing}")

    out = df.copy()

    out["strategy_ret_gross"] = out["position"] * out["spy_ret_1d"]
    out["strategy_ret_net"] = out["strategy_ret_gross"] - out["transaction_cost"]
    out["benchmark_ret"] = out["spy_ret_1d"]

    return out


def compute_cumulative_performance(
    df: pd.DataFrame,
    initial_capital: Optional[float] = None,
) -> pd.DataFrame:
    """
    Compute cumulative value series for strategy and benchmark.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing:
        - strategy_ret_gross
        - strategy_ret_net
        - benchmark_ret
    initial_capital : Optional[float]
        Starting portfolio value.

    Returns
    -------
    pd.DataFrame
        Dataframe with cumulative performance columns added.
    """
    required = ["strategy_ret_gross", "strategy_ret_net", "benchmark_ret"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for cumulative performance: {missing}")

    out = df.copy()
    initial_capital = INITIAL_CAPITAL if initial_capital is None else initial_capital

    out["strategy_cum_gross"] = initial_capital * (1.0 + out["strategy_ret_gross"]).cumprod()
    out["strategy_cum_net"] = initial_capital * (1.0 + out["strategy_ret_net"]).cumprod()
    out["benchmark_cum"] = initial_capital * (1.0 + out["benchmark_ret"]).cumprod()

    return out


# ---------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------

def run_backtest(
    df: pd.DataFrame,
    transaction_cost_bps: Optional[float] = None,
    initial_capital: Optional[float] = None,
    train_end: Optional[str] = None,
    validation_end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run the full backtest pipeline.

    Steps:
    1. validate inputs
    2. add sample labels
    3. apply one-day execution lag
    4. compute turnover
    5. apply transaction costs
    6. compute daily return series
    7. compute cumulative performance

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with at least:
        - spy_ret_1d
        - target_exposure
    transaction_cost_bps : Optional[float]
        Transaction cost in bps per unit turnover.
    initial_capital : Optional[float]
        Starting capital for cumulative series.
    train_end : Optional[str]
        Train sample end date.
    validation_end : Optional[str]
        Validation sample end date.

    Returns
    -------
    pd.DataFrame
        Fully backtested dataframe.
    """
    validate_backtest_inputs(df)

    out = df.copy()
    out.index = pd.to_datetime(out.index)
    out.sort_index(inplace=True)

    out = add_sample_labels(out, train_end=train_end, validation_end=validation_end)
    out = apply_execution_lag(out)
    out = compute_turnover(out)
    out = apply_transaction_costs(out, transaction_cost_bps=transaction_cost_bps)
    out = compute_return_series(out)
    out = compute_cumulative_performance(out, initial_capital=initial_capital)

    return out


# ---------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------

def summarize_backtest(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize basic backtest characteristics for the full sample.

    Parameters
    ----------
    df : pd.DataFrame
        Backtested dataframe.

    Returns
    -------
    pd.DataFrame
        One-row summary table.
    """
    required = [
        "strategy_ret_net",
        "benchmark_ret",
        "turnover",
        "rebalance_flag",
        "strategy_cum_net",
        "benchmark_cum",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required summary columns: {missing}")

    summary = pd.DataFrame({
        "n_obs": [len(df)],
        "avg_daily_strategy_ret_net": [df["strategy_ret_net"].mean()],
        "avg_daily_benchmark_ret": [df["benchmark_ret"].mean()],
        "avg_daily_turnover": [df["turnover"].mean()],
        "rebalance_rate": [df["rebalance_flag"].mean()] if "rebalance_flag" in df.columns else [pd.NA],
        "final_strategy_cum_net": [df["strategy_cum_net"].iloc[-1]],
        "final_benchmark_cum": [df["benchmark_cum"].iloc[-1]],
    })

    return summary


def summarize_backtest_by_sample(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize backtest output separately for train / validation / test.

    Parameters
    ----------
    df : pd.DataFrame
        Backtested dataframe containing 'sample_period'.

    Returns
    -------
    pd.DataFrame
        Summary table grouped by sample period.
    """
    if "sample_period" not in df.columns:
        raise ValueError("Column 'sample_period' not found.")

    grouped = df.groupby("sample_period")

    summary = grouped.agg(
        n_obs=("strategy_ret_net", "count"),
        avg_daily_strategy_ret_net=("strategy_ret_net", "mean"),
        avg_daily_benchmark_ret=("benchmark_ret", "mean"),
        avg_daily_turnover=("turnover", "mean"),
        rebalance_rate=("rebalance_flag", "mean"),
    )

    return summary


def get_backtest_columns() -> Tuple[str, ...]:
    """
    Return key columns created by the backtest pipeline.
    """
    return (
        "sample_period",
        "position",
        "position_prev",
        "turnover",
        "transaction_cost_rate",
        "transaction_cost",
        "strategy_ret_gross",
        "strategy_ret_net",
        "benchmark_ret",
        "strategy_cum_gross",
        "strategy_cum_net",
        "benchmark_cum",
    )