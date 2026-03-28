"""
src/features.py

Feature engineering for the Market Regimes and Trading Signals project.

This module creates a compact set of interpretable daily features designed to
characterize broad market conditions in U.S. equities.

Current feature groups:
- returns
- moving-average trend
- realized volatility
- drawdown
- macro / risk proxy changes

Design principles:
- transparent and easy to explain
- deterministic and reproducible
- no hidden look-ahead logic
- suitable for notebook exploration and later regime classification
"""

from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd

from src.config import (
    DRAWDOWN_WINDOW,
    MA_LONG,
    MA_SHORT,
    TNX_CHANGE_WINDOW,
    TRADING_DAYS_PER_YEAR,
    VIX_CHANGE_WINDOW,
    VOL_WINDOW_LONG,
    VOL_WINDOW_SHORT,
)


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------

def _get_spy_price_column(df: pd.DataFrame) -> str:
    """
    Return the preferred SPY price column.

    Preference order:
    1. spy_adj_close
    2. spy_close

    Returns
    -------
    str
        Name of SPY price column.

    Raises
    ------
    ValueError
        If neither SPY price column is present.
    """
    if "spy_adj_close" in df.columns:
        return "spy_adj_close"
    if "spy_close" in df.columns:
        return "spy_close"
    raise ValueError("Expected either 'spy_adj_close' or 'spy_close' in dataframe.")


def _validate_dataframe(df: pd.DataFrame) -> None:
    """
    Basic validation for input dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Raises
    ------
    TypeError
        If df is not a DataFrame.
    ValueError
        If df is empty.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("Input dataframe is empty.")


def _ensure_sorted_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy with a sorted DatetimeIndex.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with sorted DatetimeIndex.
    """
    out = df.copy()
    out.index = pd.to_datetime(out.index)
    out.sort_index(inplace=True)
    return out


# ---------------------------------------------------------------------
# Feature column registry
# ---------------------------------------------------------------------

def get_feature_columns() -> List[str]:
    """
    Return the main engineered feature columns expected in Version 1.

    Returns
    -------
    List[str]
        List of engineered feature column names.
    """
    return [
        "spy_ret_1d",
        f"spy_ma_{MA_SHORT}",
        f"spy_ma_{MA_LONG}",
        "spy_trend_signal",
        f"spy_rv_{VOL_WINDOW_SHORT}",
        f"spy_rv_{VOL_WINDOW_LONG}",
        f"spy_drawdown_{DRAWDOWN_WINDOW}",
        "vix_level",
        f"vix_change_{VIX_CHANGE_WINDOW}",
        f"tnx_change_{TNX_CHANGE_WINDOW}",
    ]


def get_required_base_columns() -> List[str]:
    """
    Return the base columns that may be used by this module.

    Returns
    -------
    List[str]
        Base input columns.
    """
    return [
        "spy_adj_close",
        "spy_close",
        "vix_close",
        "tnx_close",
    ]


# ---------------------------------------------------------------------
# Feature engineering steps
# ---------------------------------------------------------------------

def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute simple daily SPY returns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing SPY prices.

    Returns
    -------
    pd.DataFrame
        Dataframe with 'spy_ret_1d' added.
    """
    _validate_dataframe(df)
    out = _ensure_sorted_datetime_index(df)

    price_col = _get_spy_price_column(out)
    out["spy_ret_1d"] = out[price_col].pct_change()

    return out


def compute_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute short and long moving averages for SPY, plus a simple trend signal.

    Trend signal definition:
    - 1 if MA_SHORT > MA_LONG
    - 0 otherwise

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing SPY prices.

    Returns
    -------
    pd.DataFrame
        Dataframe with moving average features added.
    """
    _validate_dataframe(df)
    out = _ensure_sorted_datetime_index(df)

    price_col = _get_spy_price_column(out)

    short_col = f"spy_ma_{MA_SHORT}"
    long_col = f"spy_ma_{MA_LONG}"

    out[short_col] = out[price_col].rolling(window=MA_SHORT, min_periods=MA_SHORT).mean()
    out[long_col] = out[price_col].rolling(window=MA_LONG, min_periods=MA_LONG).mean()

    out["spy_trend_signal"] = np.where(
        out[short_col] > out[long_col],
        1,
        0,
    )

    # Preserve NaN logic before both moving averages exist
    out.loc[out[short_col].isna() | out[long_col].isna(), "spy_trend_signal"] = np.nan

    return out


def compute_realized_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling annualized realized volatility from SPY daily returns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing 'spy_ret_1d'.

    Returns
    -------
    pd.DataFrame
        Dataframe with realized volatility features added.

    Raises
    ------
    ValueError
        If 'spy_ret_1d' is missing.
    """
    _validate_dataframe(df)
    out = _ensure_sorted_datetime_index(df)

    if "spy_ret_1d" not in out.columns:
        raise ValueError("Column 'spy_ret_1d' not found. Run compute_returns first.")

    short_vol_col = f"spy_rv_{VOL_WINDOW_SHORT}"
    long_vol_col = f"spy_rv_{VOL_WINDOW_LONG}"

    annualization = np.sqrt(TRADING_DAYS_PER_YEAR)

    out[short_vol_col] = (
        out["spy_ret_1d"]
        .rolling(window=VOL_WINDOW_SHORT, min_periods=VOL_WINDOW_SHORT)
        .std()
        * annualization
    )

    out[long_vol_col] = (
        out["spy_ret_1d"]
        .rolling(window=VOL_WINDOW_LONG, min_periods=VOL_WINDOW_LONG)
        .std()
        * annualization
    )

    return out


def compute_drawdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute trailing drawdown for SPY relative to its rolling peak.

    Drawdown is defined as:
        current_price / rolling_peak - 1

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing SPY prices.

    Returns
    -------
    pd.DataFrame
        Dataframe with trailing drawdown added.
    """
    _validate_dataframe(df)
    out = _ensure_sorted_datetime_index(df)

    price_col = _get_spy_price_column(out)
    drawdown_col = f"spy_drawdown_{DRAWDOWN_WINDOW}"

    rolling_peak = out[price_col].rolling(
        window=DRAWDOWN_WINDOW,
        min_periods=1,
    ).max()

    out[drawdown_col] = (out[price_col] / rolling_peak) - 1.0

    return out


def compute_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute macro / risk proxy features based on VIX and Treasury yield proxy.

    Features:
    - vix_level
    - vix_change_{window}: percentage change in VIX over the configured window
    - tnx_change_{window}: absolute change in Treasury yield proxy over the configured window

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing 'vix_close' and/or 'tnx_close'.

    Returns
    -------
    pd.DataFrame
        Dataframe with macro feature columns added where possible.
    """
    _validate_dataframe(df)
    out = _ensure_sorted_datetime_index(df)

    if "vix_close" in out.columns:
        out["vix_level"] = out["vix_close"]
        out[f"vix_change_{VIX_CHANGE_WINDOW}"] = out["vix_close"].pct_change(VIX_CHANGE_WINDOW)

    if "tnx_close" in out.columns:
        out[f"tnx_change_{TNX_CHANGE_WINDOW}"] = out["tnx_close"].diff(TNX_CHANGE_WINDOW)

    return out


# ---------------------------------------------------------------------
# Pipeline / utilities
# ---------------------------------------------------------------------

def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full Version 1 feature engineering pipeline.

    Pipeline order:
    1. returns
    2. moving averages / trend
    3. realized volatility
    4. drawdown
    5. macro features

    Parameters
    ----------
    df : pd.DataFrame
        Base market dataset.

    Returns
    -------
    pd.DataFrame
        Dataframe with all features added.
    """
    _validate_dataframe(df)

    out = df.copy()
    out = compute_returns(out)
    out = compute_moving_averages(out)
    out = compute_realized_volatility(out)
    out = compute_drawdown(out)
    out = compute_macro_features(out)

    return out


def get_available_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Return engineered feature columns that are present in the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    List[str]
        Available engineered feature columns.
    """
    expected = get_feature_columns()
    return [col for col in expected if col in df.columns]


def drop_feature_warmup_rows(
    df: pd.DataFrame,
    required_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    Drop initial rows where rolling features are not yet available.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing engineered features.
    required_columns : Iterable[str] | None
        Columns that must be non-null. If None, defaults to the main rolling features.

    Returns
    -------
    pd.DataFrame
        Trimmed dataframe after warm-up.
    """
    _validate_dataframe(df)
    out = _ensure_sorted_datetime_index(df)

    if required_columns is None:
        required_columns = [
            f"spy_ma_{MA_SHORT}",
            f"spy_ma_{MA_LONG}",
            f"spy_rv_{VOL_WINDOW_SHORT}",
            f"spy_rv_{VOL_WINDOW_LONG}",
            f"spy_drawdown_{DRAWDOWN_WINDOW}",
        ]

    required_columns = [col for col in required_columns if col in out.columns]

    if not required_columns:
        return out.copy()

    return out.dropna(subset=required_columns).copy()


def summarize_feature_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize feature availability and missingness.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing features.

    Returns
    -------
    pd.DataFrame
        Summary table with missing counts and missing percentages.
    """
    _validate_dataframe(df)

    cols = get_available_feature_columns(df)
    if not cols:
        return pd.DataFrame(columns=["missing_count", "missing_pct"])

    summary = pd.DataFrame({
        "missing_count": df[cols].isna().sum(),
        "missing_pct": df[cols].isna().mean(),
    })

    return summary.sort_values("missing_pct", ascending=False)