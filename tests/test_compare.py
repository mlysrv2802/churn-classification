from churn.compare import SCORING, cross_validate_models
from churn.data import clean, load_raw
from churn.features import split
from churn.models import candidates


def test_candidates_include_all_models():
    assert set(candidates()) == {
        "Dummy",
        "LogisticRegression",
        "LogisticRegression+Poly2",
        "RandomForest",
        "GradientBoosting",
    }


def test_cross_validation_table_shape_and_ranges():
    X_train, _, y_train, _ = split(clean(load_raw()))
    models = {k: v for k, v in candidates().items() if k in {"Dummy", "LogisticRegression"}}
    table = cross_validate_models(models, X_train, y_train, folds=3)
    assert list(table.index) == ["Dummy", "LogisticRegression"]
    assert len(table.columns) == len(SCORING) * 2
    assert table.loc["Dummy", "roc_auc_mean"] == 0.5
    assert table.loc["LogisticRegression", "roc_auc_mean"] > 0.8
