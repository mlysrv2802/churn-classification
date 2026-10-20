import numpy as np
import pandas as pd

from churn.leakage import shuffled_label_auc, single_feature_scores, train_test_overlap
from churn.models import build_logistic_regression


def _frame(n: int = 1500):
    rng = np.random.default_rng(1)
    y = pd.Series(rng.integers(0, 2, size=n))
    X = pd.DataFrame(
        {
            "honest": rng.normal(size=n) + 0.3 * y,   # weak real signal
            "leaky": y + rng.normal(scale=0.05, size=n),  # basically the label
            "noise": rng.normal(size=n),
        }
    )
    return X, y


def test_single_feature_scores_flag_the_leaky_column():
    X, y = _frame()
    scores = single_feature_scores(X, y)
    assert scores.index[0] == "leaky"
    assert scores["leaky"] > 0.95
    assert scores["noise"] < 0.6


def test_shuffled_labels_score_like_a_coin_flip():
    X, y = _frame()
    auc = shuffled_label_auc(build_logistic_regression(), X, y, repeats=2)
    assert abs(auc - 0.5) < 0.05


def test_train_test_overlap_counts_matching_rows_and_labels():
    X_train = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    y_train = pd.Series([1, 0, 1])
    X_test = pd.DataFrame({"a": [1, 9], "b": ["x", "y"]})  # only row 0 matches
    y_test = pd.Series([1, 0])
    out = train_test_overlap(X_train, y_train, X_test, y_test)
    assert out == {"test_rows": 2, "matching_rows": 1, "label_agreement": 1.0}
