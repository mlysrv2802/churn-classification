import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from churn.config import COST_FALSE_NEGATIVE, COST_FALSE_POSITIVE, RANDOM_STATE

DEFAULT_THRESHOLDS = np.round(np.arange(0.10, 0.91, 0.05), 2)


def out_of_fold_proba(model: BaseEstimator, X, y, folds: int = 5) -> np.ndarray:
    """Churn probability for every training row, each predicted by a model
    that never saw that row. Safe to use for picking a threshold, because
    none of these predictions come from a model trained on the row itself.
    """
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    return cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]


def threshold_table(y_true, proba, thresholds=DEFAULT_THRESHOLDS) -> pd.DataFrame:
    """What happens at each cutoff: how many we flag, how many we catch,
    how many false alarms, and the total cost under the placeholder costs.
    """
    y = np.asarray(y_true)
    p = np.asarray(proba)
    rows = []
    for t in thresholds:
        flagged = p >= t
        tp = int((flagged & (y == 1)).sum())
        fp = int((flagged & (y == 0)).sum())
        fn = int((~flagged & (y == 1)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "threshold": float(t),
                "flagged": int(flagged.sum()),
                "caught": tp,
                "false_alarms": fp,
                "missed": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "cost": fn * COST_FALSE_NEGATIVE + fp * COST_FALSE_POSITIVE,
            }
        )
    return pd.DataFrame(rows).set_index("threshold")


def pick_threshold(table: pd.DataFrame, by: str = "f1") -> float:
    """Cutoff with the best F1 (highest) or the lowest cost."""
    if by == "f1":
        return float(table["f1"].idxmax())
    if by == "cost":
        return float(table["cost"].idxmin())
    raise ValueError("by must be 'f1' or 'cost'")
