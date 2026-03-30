"""
src/signals.py

Signal construction: map regimes → portfolio exposure.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd

from src.config import REGIME_TO_EXPOSURE


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def validate_signal_inputs(df: pd.DataFrame) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Input dataframe is empty.")

    if "regime" not in df.columns:
        raise ValueError("Column 'regime' not found.")


# ---------------------------------------------------------------------
# Core mapping
# ---------------------------------------------------------------------

def map_regime_to_exposure(
    df: pd.DataFrame,
    mapping: Dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Map regime labels to target exposure.
    """
    validate_signal_inputs(df)

    out = df.copy()
    mapping = REGIME_TO_EXPOSURE if mapping is None else mapping

    out["target_exposure"] = out["regime"].map(mapping)

    if out["target_exposure"].isna().any():
        missing = out.loc[out["target_exposure"].isna(), "regime"].unique()
        raise ValueError(f"Missing mapping for regimes: {missing}")

    return out


# ---------------------------------------------------------------------
# Position change tracking
# ---------------------------------------------------------------------

def add_position_change_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rebalance indicators.
    """
    if "target_exposure" not in df.columns:
        raise ValueError("target_exposure not found")

    out = df.copy()

    out["target_exposure_prev"] = out["target_exposure"].shift(1)
    out["exposure_change"] = out["target_exposure"] - out["target_exposure_prev"]
    out["rebalance_flag"] = out["exposure_change"].fillna(0) != 0

    return out


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

def summarize_signal_distribution(df: pd.DataFrame) -> pd.DataFrame:
    if "target_exposure" not in df.columns:
        raise ValueError("target_exposure not found")

    return pd.DataFrame({
        "count": df["target_exposure"].value_counts(),
        "proportion": df["target_exposure"].value_counts(normalize=True),
    })

# ---------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------

def build_signals(df: pd.DataFrame, mapping: Dict[str, float] | None = None) -> pd.DataFrame:
    """
    Full signal-construction pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing regime labels.
    mapping : dict[str, float] | None
        Optional custom regime-to-exposure mapping. If None, uses the default
        mapping from config.

    Returns
    -------
    pd.DataFrame
        Dataframe with target exposure and rebalance helper columns.
    """
    out = df.copy()
    out = map_regime_to_exposure(out, mapping=mapping)
    out = add_position_change_flags(out)
    return out