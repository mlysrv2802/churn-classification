import pandas as pd
from scipy.stats import loguniform
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from churn.config import RANDOM_STATE
from churn.models import build_logistic_regression

DEGREE_PARAM = "prep__num__poly__degree"
C_PARAM = "model__C"
DEGREES = [1, 2, 3, 4, 5]


def tune_logistic_regression(
    X, y, n_iter: int = 40, folds: int = 5, scoring: str = "roc_auc"
) -> RandomizedSearchCV:
    """Search polynomial degree and regularisation strength together.

    They must be tuned jointly: a high degree creates many columns, and only a
    small C (strong regularisation) stops the model memorising noise. Tuning
    degree alone, with C fixed, would punish high degrees unfairly.
    """
    search = RandomizedSearchCV(
        build_logistic_regression(),
        param_distributions={DEGREE_PARAM: DEGREES, C_PARAM: loguniform(1e-3, 1e2)},
        n_iter=n_iter,
        scoring=scoring,
        cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )
    return search.fit(X, y)


def best_per_degree(search: RandomizedSearchCV) -> pd.DataFrame:
    """Best CV score reached by each degree, with the C that achieved it."""
    r = pd.DataFrame(search.cv_results_)
    r["degree"] = r[f"param_{DEGREE_PARAM}"].astype(int)
    r["C"] = r[f"param_{C_PARAM}"].astype(float)
    best = r.loc[r.groupby("degree")["mean_test_score"].idxmax()]
    return best.set_index("degree")[["mean_test_score", "std_test_score", "C"]].sort_index()
