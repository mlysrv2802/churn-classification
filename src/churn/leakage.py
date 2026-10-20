import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from churn.config import RANDOM_STATE
from churn.linearity import curved_columns, linearity_report
from churn.models import build_logistic_regression


def _cv(folds: int) -> StratifiedKFold:
    return StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)


def single_feature_scores(X: pd.DataFrame, y, folds: int = 5) -> pd.Series:
    """CV ROC-AUC of a model that sees ONE column at a time.

    A column that almost perfectly predicts churn on its own (AUC near 1.0) is
    suspicious: it is probably derived from the answer, or recorded after the
    customer already left. Honest signals in churn data sit well below 0.9.
    """
    scores = {
        col: cross_val_score(
            build_logistic_regression(), X[[col]], y, cv=_cv(folds), scoring="roc_auc"
        ).mean()
        for col in X.columns
    }
    return pd.Series(scores, name="roc_auc").sort_values(ascending=False)


def train_test_overlap(X_train, y_train, X_test, y_test) -> dict[str, float]:
    """How many test rows have an identical feature row in the training set?

    Identical rows let a model 'remember' instead of generalise. We also report
    how often copying the matching training label would have been right.
    """
    cols = list(X_train.columns)
    train = (
        X_train.assign(_y=y_train.to_numpy())
        .groupby(cols, dropna=False)["_y"]
        .mean()
        .rename("train_rate")
        .reset_index()
    )
    merged = X_test.assign(_y=y_test.to_numpy()).merge(train, on=cols, how="inner")
    agreement = (
        float(((merged["train_rate"] >= 0.5).astype(int) == merged["_y"]).mean())
        if len(merged)
        else float("nan")
    )
    return {
        "test_rows": len(X_test),
        "matching_rows": len(merged),
        "label_agreement": agreement,
    }


def shuffled_label_auc(
    model: BaseEstimator, X: pd.DataFrame, y, folds: int = 5, repeats: int = 3
) -> float:
    """CV ROC-AUC after destroying the link between features and labels.

    The labels are randomly shuffled, so there is nothing real to learn and a
    leak-free pipeline must score about 0.5 (a coin flip). A score clearly
    above 0.5 means information is leaking from somewhere it should not.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    scores = []
    for _ in range(repeats):
        shuffled = pd.Series(rng.permutation(np.asarray(y)), index=X.index)
        scores.append(
            cross_val_score(model, X, shuffled, cv=_cv(folds), scoring="roc_auc").mean()
        )
    return float(np.mean(scores))


def selection_bias_check(X: pd.DataFrame, y, folds: int = 5) -> dict[str, float]:
    """Does choosing the curved columns on ALL training rows inflate CV scores?

    outside: columns chosen once using every row, then cross-validated
             (what compare_curve_treatments.py does)
    inside:  columns re-chosen inside each fold using only that fold's training
             rows, so the held-out fold never influences the choice (honest)
    A large gap between them would mean the earlier numbers were optimistic.
    """
    chosen = curved_columns(linearity_report(X, y))
    outside = cross_val_score(
        build_logistic_regression(spline_columns=chosen),
        X, y, cv=_cv(folds), scoring="roc_auc",
    ).mean()

    inside = []
    for train_idx, val_idx in _cv(folds).split(X, y):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
        cols = curved_columns(linearity_report(X_tr, y_tr))
        model = build_logistic_regression(spline_columns=cols).fit(X_tr, y_tr)
        inside.append(roc_auc_score(y_va, model.predict_proba(X_va)[:, 1]))
    return {"outside": float(outside), "inside": float(np.mean(inside))}
