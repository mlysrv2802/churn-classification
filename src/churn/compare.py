import pandas as pd
from sklearn.metrics import make_scorer, precision_score
from sklearn.model_selection import StratifiedKFold, cross_validate

from churn.config import RANDOM_STATE

SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]

# A model that never predicts churn has undefined precision; score it as 0.
_SCORERS = {
    m: make_scorer(precision_score, zero_division=0) if m == "precision" else m
    for m in SCORING
}


def cross_validate_models(models: dict, X, y, folds: int = 5) -> pd.DataFrame:
    """Same folds for every model, so differences come from the model alone.

    Returns mean and std of each metric per model. A std larger than the gap
    between two models means the gap is probably noise.
    """
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    rows = {}
    for name, model in models.items():
        res = cross_validate(model, X, y, cv=cv, scoring=_SCORERS, n_jobs=-1)
        rows[name] = {
            f"{m}_{stat}": getattr(res[f"test_{m}"], stat)()
            for m in SCORING
            for stat in ("mean", "std")
        }
    return pd.DataFrame(rows).T
