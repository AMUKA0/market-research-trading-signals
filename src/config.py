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

# ---------------------------------------------------------------------
# Calendar / scaling
# ---------------------------------------------------------------------

TRADING_DAYS_PER_YEAR = 252

# ---------------------------------------------------------------------
# Feature parameters
# ---------------------------------------------------------------------

MA_SHORT = 50
MA_LONG = 200

VOL_WINDOW_SHORT = 20
VOL_WINDOW_LONG = 60

VIX_CHANGE_WINDOW = 20
TNX_CHANGE_WINDOW = 20

DRAWDOWN_WINDOW = 252  