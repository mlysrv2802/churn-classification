import numpy as np
import pytest

from churn.evaluate import confusion, evaluate


class FakeModel:
    """Stub with a fixed output, so metric maths can be checked by hand."""

    def predict(self, X):
        return np.array([1, 1, 0, 0, 0, 0])

    def predict_proba(self, X):
        p = np.array([0.9, 0.8, 0.3, 0.2, 0.6, 0.1])
        return np.column_stack([1 - p, p])


Y = np.array([1, 0, 1, 0, 0, 0])


def test_confusion_counts():
    assert confusion(FakeModel(), None, Y) == {"tn": 3, "fp": 1, "fn": 1, "tp": 1}


def test_metrics_by_hand():
    s = evaluate(FakeModel(), None, Y)
    assert s["accuracy"] == pytest.approx(4 / 6)
    assert s["precision"] == pytest.approx(1 / 2)
    assert s["recall"] == pytest.approx(1 / 2)
    assert s["f1"] == pytest.approx(1 / 2)
    assert 0.5 < s["roc_auc"] <= 1.0
