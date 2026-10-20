from pathlib import Path

import pandas as pd

from churn.config import DATA_PATH, ID_COLUMN, TARGET


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    """Read the CSV exactly as shipped, with no cleaning."""
    return pd.read_csv(path)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with numeric TotalCharges and no customerID column.

    This is the first step of every model pipeline, so a saved model accepts raw
    customer records. It is safe to run twice (already-clean data passes through).

    Blank TotalCharges only occur for customers with tenure == 0 (no bill yet),
    so filling them with 0 is semantically correct.
    """
    out = df.copy()
    if "TotalCharges" in out.columns:  # absent in single-column experiments
        out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce").fillna(0.0)
    return out.drop(columns=[ID_COLUMN], errors="ignore")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the training data: prepared features + 0/1 target."""
    out = prepare_features(df)
    out[TARGET] = (out[TARGET] == "Yes").astype(int)
    return out
