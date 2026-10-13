from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from churn.config import RANDOM_STATE
from churn.features import build_preprocessor


def build_dummy() -> BaseEstimator:
    """Always predicts the majority class: the floor any real model must beat."""
    return DummyClassifier(strategy="most_frequent")


def build_logistic_regression() -> Pipeline:
    """Preprocessing + model in ONE pipeline, so train and serve cannot drift."""
    return Pipeline(
        [
            ("prep", build_preprocessor()),
            ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )
