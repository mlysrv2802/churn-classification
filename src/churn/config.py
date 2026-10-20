import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Defaults work for a local checkout; Docker overrides them with environment variables.
DATA_PATH = Path(os.environ.get("CHURN_DATA_PATH", PROJECT_ROOT / "data" / "telco_churn.csv"))
# models/ is git-ignored: trained artifacts are not source code
MODELS_DIR = Path(os.environ.get("CHURN_MODELS_DIR", PROJECT_ROOT / "models"))

TARGET = "Churn"
ID_COLUMN = "customerID"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# PLACEHOLDER business costs, in arbitrary units. Replace with real numbers.
# Missing a customer who leaves costs more than a retention offer sent by mistake.
COST_FALSE_NEGATIVE = 5.0  # a churner we failed to flag (lost customer)
COST_FALSE_POSITIVE = 1.0  # a stayer we flagged (wasted offer)
