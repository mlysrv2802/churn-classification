from churn.data import clean, load_raw
from churn.features import split
from churn.tuning import DEGREE_PARAM, DEGREES, best_per_degree, tune_logistic_regression


def test_search_explores_degree_and_returns_per_degree_table():
    X_train, _, y_train, _ = split(clean(load_raw()))
    search = tune_logistic_regression(X_train[:1000], y_train[:1000], n_iter=8, folds=2)
    table = best_per_degree(search)
    assert set(table.index) <= set(DEGREES)
    assert search.best_params_[DEGREE_PARAM] in DEGREES
    assert table["mean_test_score"].max() == search.best_score_
