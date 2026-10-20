import platform
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import sklearn

from churn.config import MODELS_DIR
from churn.data import clean, load_raw
from churn.evaluate import evaluate
from churn.features import split
from churn.models import build_logistic_regression
from churn.registry import ModelRegistry
from churn.thresholds import out_of_fold_proba, pick_threshold, threshold_table


def train_and_save(model_dir: Path = MODELS_DIR, by: str = "f1") -> dict:
    """Train the final model, pick its cutoff, and save both as a new registry version.

    The new version becomes production only if it passes the registry's
    "not clearly worse" check (see ModelRegistry.maybe_promote).

    Final model: plain logistic regression. Every candidate scored within
    noise of the best (CV ROC-AUC about 0.845 to 0.849, std about 0.012), so
    the simplest one wins: cheap, explainable, nothing extra to maintain.
    The cutoff is chosen on training data only (out-of-fold predictions).
    """
    X_train, X_test, y_train, y_test = split(clean(load_raw()))
    model = build_logistic_regression()

    train_table = threshold_table(y_train, out_of_fold_proba(model, X_train, y_train))
    threshold = pick_threshold(train_table, by=by)

    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    at_cutoff = threshold_table(y_test, proba, thresholds=[threshold]).iloc[0]

    metadata = {
        "model": "LogisticRegression (plain)",
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "threshold": threshold,
        "threshold_chosen_by": by,
        "train_rows": len(X_train),
        "feature_columns": list(X_train.columns),
        "test_metrics": {
            "roc_auc": float(evaluate(model, X_test, y_test)["roc_auc"]),
            "precision_at_threshold": float(at_cutoff["precision"]),
            "recall_at_threshold": float(at_cutoff["recall"]),
            "f1_at_threshold": float(at_cutoff["f1"]),
        },
        "versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
        },
    }

    registry = ModelRegistry(model_dir)
    version = registry.save(model, metadata)
    promoted, reason = registry.maybe_promote(version)
    return {**metadata, "version": version, "promoted": promoted, "promotion_reason": reason}
