from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TICKERS = {
    "spy": "SPY",
    "vix": "^VIX",
    "tnx": "^TNX",   # CBOE 10-year Treasury yield index proxy
}

START_DATE = "2012-01-01"
END_DATE = None  # Use None for up-to-date data