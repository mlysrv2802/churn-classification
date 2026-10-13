import pytest

from churn.data import clean, load_raw
from churn.features import split
from churn.models import build_dummy, build_logistic_regression


@pytest.fixture(scope="module")
def data():
    return split(clean(load_raw()))


def test_split_is_stratified(data):
    _, _, y_train, y_test = data
    assert abs(y_train.mean() - y_test.mean()) < 0.01
    assert len(y_train) + len(y_test) == 7043


def test_split_has_no_target_in_features(data):
    X_train, *_ = data
    assert "Churn" not in X_train.columns


def test_dummy_never_predicts_churn(data):
    X_train, X_test, y_train, _ = data
    pred = build_dummy().fit(X_train, y_train).predict(X_test)
    assert pred.sum() == 0


def test_logreg_pipeline_beats_zero_recall(data):
    X_train, X_test, y_train, y_test = data
    pred = build_logistic_regression().fit(X_train, y_train).predict(X_test)
    assert pred.sum() > 0
    assert (pred == y_test).mean() > 0.75
