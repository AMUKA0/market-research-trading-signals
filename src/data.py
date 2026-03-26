"""
src/data.py

Utilities for downloading, cleaning, saving, and loading market data
for the Market Regimes and Trading Signals project.

Design goals:
- deterministic and easy to debug
- clear column naming
- minimal hidden behavior
- suitable for research workflows
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import yfinance as yf


# ---------------------------------------------------------------------
# Configuration containers
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class DataConfig:
    """Configuration for market data download and storage."""

    tickers: Dict[str, str]
    start_date: str
    end_date: Optional[str]
    raw_dir: Path
    processed_dir: Path
    processed_filename: str = "market_data_daily.csv"


# ---------------------------------------------------------------------
# Core download functions
# ---------------------------------------------------------------------
def download_single_ticker(
    ticker: str,
    start_date: str,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Download OHLCV data for a single ticker from Yahoo Finance.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol.
    start_date : str
        Start date in YYYY-MM-DD format.
    end_date : Optional[str]
        End date in YYYY-MM-DD format. If None, downloads up to latest available.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by date with flat OHLCV-style columns.

    Raises
    ------
    ValueError
        If no data is returned.
    """
    df = yf.download(
        tickers=ticker,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'.")

    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    # Flatten MultiIndex columns if yfinance returns them
    if isinstance(df.columns, pd.MultiIndex):
        if len(df.columns.levels) == 2:
            # Usually looks like ('Close', 'SPY')
            df.columns = df.columns.get_level_values(0)
        else:
            df.columns = [
                "_".join(str(part) for part in col if part != "")
                for col in df.columns.to_flat_index()
            ]

    return df

def standardize_ohlcv_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Rename OHLCV columns using a consistent prefix.
    """
    out = df.copy()

    # Defensive cleanup in case columns are not plain strings
    out.columns = [str(col) for col in out.columns]

    rename_map = {
        "Open": f"{prefix}_open",
        "High": f"{prefix}_high",
        "Low": f"{prefix}_low",
        "Close": f"{prefix}_close",
        "Adj Close": f"{prefix}_adj_close",
        "Volume": f"{prefix}_volume",
    }

    available_map = {k: v for k, v in rename_map.items() if k in out.columns}
    out = out.rename(columns=available_map)

    out.index.name = "date"
    return out

def download_market_data(config: DataConfig) -> Dict[str, pd.DataFrame]:
    """
    Download raw data for all configured tickers.

    Parameters
    ----------
    config : DataConfig
        Data configuration.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary keyed by internal asset name, e.g. {'spy': df, 'vix': df}.
    """
    raw_data: Dict[str, pd.DataFrame] = {}

    for asset_name, ticker in config.tickers.items():
        df = download_single_ticker(
            ticker=ticker,
            start_date=config.start_date,
            end_date=config.end_date,
        )
        df = standardize_ohlcv_columns(df, prefix=asset_name)
        raw_data[asset_name] = df

    return raw_data


# ---------------------------------------------------------------------
# Cleaning / alignment
# ---------------------------------------------------------------------

def merge_market_data(raw_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Outer-join all asset data on the date index.

    Parameters
    ----------
    raw_data : Dict[str, pd.DataFrame]
        Dictionary of standardized per-asset DataFrames.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame indexed by date.
    """
    if not raw_data:
        raise ValueError("raw_data is empty. Nothing to merge.")

    merged: Optional[pd.DataFrame] = None

    for _, df in raw_data.items():
        if merged is None:
            merged = df.copy()
        else:
            merged = merged.join(df, how="outer")

    if merged is None:
        raise ValueError("Failed to merge market data.")

    merged.sort_index(inplace=True)
    merged.index.name = "date"
    return merged


def clean_market_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply basic cleaning rules suitable for daily market data research.

    Cleaning approach:
    - sort by date
    - drop duplicate dates
    - forward-fill missing values for non-trading alignment issues
    - drop rows where core SPY adjusted close is still missing
    """
    out = df.copy()

    out.index = pd.to_datetime(out.index)
    out = out[~out.index.duplicated(keep="first")]
    out.sort_index(inplace=True)

    # Forward-fill for alignment across instruments / calendars
    out.ffill(inplace=True)

    # Drop rows before main asset begins
    core_col = "spy_adj_close" if "spy_adj_close" in out.columns else "spy_close"
    if core_col in out.columns:
        out = out.loc[out[core_col].notna()].copy()

    return out


def select_research_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the columns most useful for downstream research.
    """
    out = df.copy()

    # Ensure flat string columns
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [
            "_".join(str(part) for part in col if part != "")
            for col in out.columns.to_flat_index()
        ]
    else:
        out.columns = [str(col) for col in out.columns]

    preferred_columns = [
        "spy_close",
        "spy_adj_close",
        "spy_volume",
        "vix_close",
        "vix_adj_close",
        "tnx_close",
        "tnx_adj_close",
    ]

    existing = [col for col in preferred_columns if col in out.columns]

    if not existing:
        raise ValueError(
            f"None of the preferred research columns are present. "
            f"Available columns: {list(out.columns)}"
        )

    out = out.loc[:, existing].copy()
    out.index.name = "date"
    return out

# ---------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------

def ensure_directories(config: DataConfig) -> None:
    """Create data directories if they do not already exist."""
    config.raw_dir.mkdir(parents=True, exist_ok=True)
    config.processed_dir.mkdir(parents=True, exist_ok=True)


def save_processed_data(df: pd.DataFrame, config: DataConfig) -> Path:
    """
    Save processed market data to CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Processed dataset.
    config : DataConfig
        Data configuration.

    Returns
    -------
    Path
        Path to saved CSV file.
    """
    ensure_directories(config)
    output_path = config.processed_dir / config.processed_filename
    df.to_csv(output_path, index=True)
    return output_path


def load_processed_data(config: DataConfig) -> pd.DataFrame:
    """
    Load processed market data from CSV.

    Parameters
    ----------
    config : DataConfig
        Data configuration.

    Returns
    -------
    pd.DataFrame
        Loaded processed dataset.
    """
    input_path = config.processed_dir / config.processed_filename

    if not input_path.exists():
        raise FileNotFoundError(
            f"Processed data file not found: {input_path}. "
            "Run the data pipeline first."
        )

    df = pd.read_csv(input_path, parse_dates=["date"], index_col="date")
    df.sort_index(inplace=True)
    return df


# ---------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------

def build_market_dataset(config: DataConfig) -> pd.DataFrame:
    """
    End-to-end pipeline:
    1. download raw per-ticker data
    2. merge on date
    3. clean alignment / missing values
    4. select key research columns

    Parameters
    ----------
    config : DataConfig
        Data configuration.

    Returns
    -------
    pd.DataFrame
        Final processed research dataset.
    """
    raw_data = download_market_data(config)
    merged = merge_market_data(raw_data)
    cleaned = clean_market_data(merged)
    research_df = select_research_columns(cleaned)
    return research_df


def build_and_save_market_dataset(config: DataConfig) -> pd.DataFrame:
    """
    Build the final dataset and save it to disk.

    Parameters
    ----------
    config : DataConfig
        Data configuration.

    Returns
    -------
    pd.DataFrame
        Final processed dataset.
    """
    df = build_market_dataset(config)
    save_processed_data(df, config)
    return df