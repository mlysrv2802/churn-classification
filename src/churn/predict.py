from pathlib import Path

import pandas as pd

from churn.config import MODELS_DIR
from churn.registry import ModelRegistry


class ChurnPredictor:
    """Loads a saved model once and scores raw customer records.

    By default it loads the production version; pass `version` to load another.
    Load it once at startup and reuse it, like a Spring singleton bean.
    """

    def __init__(self, model_dir: Path = MODELS_DIR, version: str | None = None):
        self.model, self.metadata = ModelRegistry(model_dir).load(version)

    def predict(self, records: pd.DataFrame) -> pd.DataFrame:
        """Score raw records (TotalCharges may be text; customerID is optional)."""
        required = self.metadata["feature_columns"]
        missing = [c for c in required if c not in records.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        proba = self.model.predict_proba(records[required])[:, 1]
        return pd.DataFrame(
            {
                "churn_probability": proba,
                "flagged": proba >= self.metadata["threshold"],
            },
            index=records.index,
        )
