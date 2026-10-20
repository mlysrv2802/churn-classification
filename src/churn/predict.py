import json
from pathlib import Path

import joblib
import pandas as pd

from churn.config import MODELS_DIR
from churn.train import METADATA_FILE, MODEL_FILE


class ChurnPredictor:
    """Loads the saved model once and scores raw customer records.

    Load it once at startup and reuse it, like a Spring singleton bean.
    """

    def __init__(self, model_dir: Path = MODELS_DIR):
        self.model = joblib.load(model_dir / MODEL_FILE)
        self.metadata = json.loads((model_dir / METADATA_FILE).read_text(encoding="utf-8"))

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
