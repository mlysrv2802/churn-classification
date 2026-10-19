import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from churn.config import RANDOM_STATE

MIN_GAIN = 0.002  # ignore log-loss improvements smaller than this


def curved_columns(report: pd.DataFrame, min_gain: float = 0.005) -> list[str]:
    """Columns worth giving a curve: flagged non-linear AND a meaningful gain.

    The report must be computed on TRAINING data only; picking columns by
    looking at the test set would leak test information into the model design.
    """
    keep = report["non_linear"] & (report["gain"] >= min_gain)
    return report.index[keep].tolist()


def numeric_candidates(X: pd.DataFrame) -> list[str]:
    """Numeric columns with more than two distinct values.

    A 0/1 column (SeniorCitizen) can only have two points, and a line through
    two points always fits perfectly, so it cannot be non-linear.
    """
    numeric = X.select_dtypes("number")
    return [c for c in numeric.columns if numeric[c].nunique() > 2]


def linearity_report(X: pd.DataFrame, y, folds: int = 5) -> pd.DataFrame:
    """Rank numeric columns by how much a flexible curve beats a straight line.

    For each column, using that column ALONE, compare on identical CV folds:
      linear:   logistic regression on x            (a straight-line effect)
      flexible: logistic regression on cubic splines of x (can bend freely)
    A column is flagged non-linear when the flexible model wins on the held-out
    folds by more than two standard errors and more than MIN_GAIN.

    Scored with log-loss (lower is better), NOT ROC-AUC: AUC only looks at how
    customers are ranked, so it cannot see a curve that is always going the
    same direction (like churn falling with tenure). Log-loss judges the
    predicted probabilities themselves, so it can.
    """
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for col in numeric_candidates(X):
        linear = Pipeline([("s", StandardScaler()), ("m", LogisticRegression())])
        flexible = Pipeline(
            [
                ("s", SplineTransformer(n_knots=6, degree=3, knots="quantile")),
                ("m", LogisticRegression(max_iter=2000)),
            ]
        )
        args = dict(X=X[[col]], y=y, cv=cv, scoring="neg_log_loss")
        lin = -cross_val_score(linear, **args)
        flex = -cross_val_score(flexible, **args)
        diff = lin - flex  # positive = the curve predicts better than the line
        gain, stderr = diff.mean(), diff.std(ddof=1) / np.sqrt(folds)
        rows.append(
            {
                "column": col,
                "linear_logloss": lin.mean(),
                "flexible_logloss": flex.mean(),
                "gain": gain,
                "stderr": stderr,
                "non_linear": bool(gain > max(2 * stderr, MIN_GAIN)),
            }
        )
    return pd.DataFrame(rows).set_index("column").sort_values("gain", ascending=False)
