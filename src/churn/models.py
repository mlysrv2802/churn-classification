from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from churn.config import RANDOM_STATE
from churn.data import prepare_features
from churn.features import build_preprocessor


def build_dummy() -> BaseEstimator:
    """Always predicts the majority class: the floor any real model must beat."""
    return DummyClassifier(strategy="most_frequent")


def _pipeline(
    estimator: BaseEstimator,
    poly_degree: int = 1,
    spline_columns: list[str] | None = None,
) -> Pipeline:
    """Preprocessing + model in ONE pipeline, so train and serve cannot drift."""
    prep = build_preprocessor(poly_degree, spline_columns)
    return Pipeline(
        [
            ("clean", FunctionTransformer(prepare_features)),
            ("prep", prep),
            ("model", estimator),
        ]
    )


def build_logistic_regression(
    poly_degree: int = 1, spline_columns: list[str] | None = None
) -> Pipeline:
    return _pipeline(
        LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        poly_degree,
        spline_columns,
    )


def build_random_forest() -> Pipeline:
    return _pipeline(
        RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=RANDOM_STATE
        )
    )


def build_gradient_boosting() -> Pipeline:
    return _pipeline(HistGradientBoostingClassifier(random_state=RANDOM_STATE))


def candidates() -> dict[str, BaseEstimator]:
    """Every model that takes part in the comparison."""
    return {
        "Dummy": build_dummy(),
        "LogisticRegression": build_logistic_regression(),
        "LogisticRegression+Poly2": build_logistic_regression(poly_degree=2),
        "RandomForest": build_random_forest(),
        "GradientBoosting": build_gradient_boosting(),
    }
