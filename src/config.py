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

# ---------------------------------------------------------------------
# Regime thresholds (Version 2)
# ---------------------------------------------------------------------

VIX_LOW_THRESHOLD = 18.0
VIX_HIGH_THRESHOLD = 25.0

STRESS_DRAWDOWN_THRESHOLD = -0.10
RISK_OFF_DRAWDOWN_THRESHOLD = -0.15

RECOVERY_VIX_MIN_THRESHOLD = 18.0

VOL_REGIME_LOOKBACK = 252
VOL_LOW_QUANTILE = 0.50
VOL_HIGH_QUANTILE = 0.75

# ---------------------------------------------------------------------
# Signal mapping
# ---------------------------------------------------------------------

REGIME_TO_EXPOSURE = {
    "risk_on": 1.00,
    "recovery": 0.75,
    "stress": 0.25,
    "risk_off": 0.00,
    "unknown": 0.00,
}

# ---------------------------------------------------------------------
# Backtest assumptions
# ---------------------------------------------------------------------

TRANSACTION_COST_BPS = 5.0
INITIAL_CAPITAL = 1.0

# ---------------------------------------------------------------------
# Sample splits
# ---------------------------------------------------------------------

TRAIN_END = "2018-12-31"
VALIDATION_END = "2021-12-31"
TEST_END = None