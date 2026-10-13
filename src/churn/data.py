from pathlib import Path

import pandas as pd

from churn.config import DATA_PATH, ID_COLUMN, TARGET


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    """Read the CSV exactly as shipped, with no cleaning."""
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy: numeric TotalCharges, binary target, no ID column.

    Blank TotalCharges only occur for customers with tenure == 0 (no bill yet),
    so filling them with 0 is semantically correct.
    """
    out = df.copy()
    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce").fillna(0.0)
    out[TARGET] = (out[TARGET] == "Yes").astype(int)
    return out.drop(columns=[ID_COLUMN])
