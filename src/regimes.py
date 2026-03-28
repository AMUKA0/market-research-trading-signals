"""
src/regimes.py

Rule-based market regime classification for the Market Regimes and Trading Signals project.

This module classifies each day into one of four interpretable regimes:

- risk_on
- recovery
- stress
- risk_off

Design principles:
- simple and explainable
- no look-ahead bias
- thresholds controlled through config where appropriate
- robust enough for a first-pass research project
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from src.config import (
    RECOVERY_VIX_MIN_THRESHOLD,
    RISK_OFF_DRAWDOWN_THRESHOLD,
    STRESS_DRAWDOWN_THRESHOLD,
    VIX_HIGH_THRESHOLD,
    VIX_LOW_THRESHOLD,
    VOL_HIGH_QUANTILE,
    VOL_LOW_QUANTILE,
    VOL_REGIME_LOOKBACK,
)


# ---------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------

def get_required_regime_columns() -> List[str]:
    """
    Return the minimum columns required for regime classification.
    """
    return [
        "spy_trend_signal",
        "spy_rv_20",
        "spy_rv_60",
        "spy_drawdown_252",
        "vix_level",
        "vix_change_20",
    ]


def validate_regime_inputs(df: pd.DataFrame) -> None:
    """
    Validate that the dataframe contains all required columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with engineered features.

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

    missing = [col for col in get_required_regime_columns() if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns for regime classification: {missing}"
        )


# ---------------------------------------------------------------------
# Volatility context helpers
# ---------------------------------------------------------------------

def add_volatility_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling realized-volatility percentile thresholds.

    This allows regime classification to judge whether current realized volatility
    is relatively low or high compared with recent history, rather than relying
    only on fixed absolute levels.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing 'spy_rv_20'.

    Returns
    -------
    pd.DataFrame
        Dataframe with rolling volatility threshold columns added.
    """
    validate_regime_inputs(df)

    out = df.copy()
    out.sort_index(inplace=True)

    out["rv20_low_threshold"] = (
        out["spy_rv_20"]
        .rolling(window=VOL_REGIME_LOOKBACK, min_periods=VOL_REGIME_LOOKBACK)
        .quantile(VOL_LOW_QUANTILE)
    )

    out["rv20_high_threshold"] = (
        out["spy_rv_20"]
        .rolling(window=VOL_REGIME_LOOKBACK, min_periods=VOL_REGIME_LOOKBACK)
        .quantile(VOL_HIGH_QUANTILE)
    )

    return out


# ---------------------------------------------------------------------
# Single-row classification
# ---------------------------------------------------------------------

def classify_regime_row(row: pd.Series) -> str:
    """
    Classify a single row into a market regime.

    Regime intuition:
    - risk_on: positive trend, calm implied vol, subdued realized vol
    - recovery: positive trend, elevated but improving volatility environment
    - risk_off: negative trend with severe stress
    - stress: negative trend / unstable transition but not full risk_off

    Parameters
    ----------
    row : pd.Series
        Row containing engineered features and volatility threshold helpers.

    Returns
    -------
    str
        Regime label.
    """
    trend = row.get("spy_trend_signal")
    rv20 = row.get("spy_rv_20")
    drawdown = row.get("spy_drawdown_252")
    vix = row.get("vix_level")
    vix_change = row.get("vix_change_20")

    rv20_low_threshold = row.get("rv20_low_threshold")
    rv20_high_threshold = row.get("rv20_high_threshold")

    # If core data missing, classification is not reliable yet
    if any(pd.isna(x) for x in [trend, rv20, drawdown, vix, vix_change]):
        return "unknown"

    # -----------------------------------------------------------------
    # Risk-off: severe stress environment
    # -----------------------------------------------------------------
    if (
        trend == 0
        and (
            vix >= VIX_HIGH_THRESHOLD
            or drawdown <= RISK_OFF_DRAWDOWN_THRESHOLD
            or (
                pd.notna(rv20_high_threshold)
                and rv20 >= rv20_high_threshold
                and drawdown <= STRESS_DRAWDOWN_THRESHOLD
            )
        )
    ):
        return "risk_off"

    # -----------------------------------------------------------------
    # Risk-on: positive trend + calm / stable volatility backdrop
    # -----------------------------------------------------------------
    if (
        trend == 1
        and vix < VIX_LOW_THRESHOLD
        and (
            pd.isna(rv20_low_threshold)
            or rv20 <= rv20_low_threshold
        )
    ):
        return "risk_on"

    # -----------------------------------------------------------------
    # Recovery: trend has turned positive, but volatility is still elevated
    # or falling from a stressed level
    # -----------------------------------------------------------------
    if (
        trend == 1
        and vix >= RECOVERY_VIX_MIN_THRESHOLD
        and vix_change < 0
    ):
        return "recovery"

    # -----------------------------------------------------------------
    # Stress: negative trend or elevated instability without full capitulation
    # -----------------------------------------------------------------
    if trend == 0:
        return "stress"

    # Fallback for ambiguous positive-trend environments
    return "recovery"


# ---------------------------------------------------------------------
# Dataset-level API
# ---------------------------------------------------------------------

def assign_regimes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign a regime label to each row in the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with engineered features.

    Returns
    -------
    pd.DataFrame
        Dataframe with helper columns and a final 'regime' column added.
    """
    validate_regime_inputs(df)

    out = add_volatility_regime_features(df)
    out["regime"] = out.apply(classify_regime_row, axis=1)
    return out


def get_regime_counts(df: pd.DataFrame) -> pd.Series:
    """
    Return raw counts of each regime.
    """
    if "regime" not in df.columns:
        raise ValueError("Column 'regime' not found.")
    return df["regime"].value_counts(dropna=False)


def get_regime_distribution(df: pd.DataFrame) -> pd.Series:
    """
    Return normalized distribution of each regime.
    """
    if "regime" not in df.columns:
        raise ValueError("Column 'regime' not found.")
    return df["regime"].value_counts(normalize=True, dropna=False)


def summarize_regime_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize average and volatility of returns by regime.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing 'regime' and 'spy_ret_1d'.

    Returns
    -------
    pd.DataFrame
        Summary table by regime.
    """
    required = ["regime", "spy_ret_1d"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    summary = (
        df.groupby("regime")["spy_ret_1d"]
        .agg(["count", "mean", "std", "min", "max"])
        .sort_index()
    )

    return summary


def get_known_regime_subset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return only rows with known regimes.

    Useful for downstream analysis if initial warm-up rows remain.
    """
    if "regime" not in df.columns:
        raise ValueError("Column 'regime' not found.")

    return df.loc[df["regime"] != "unknown"].copy()