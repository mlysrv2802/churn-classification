import json

import pandas as pd
import pytest

from churn.data import clean, load_raw, prepare_features
from churn.predict import ChurnPredictor
from churn.train import METADATA_FILE, MODEL_FILE, train_and_save


@pytest.fixture(scope="module")
def saved_dir(tmp_path_factory):
    model_dir = tmp_path_factory.mktemp("models")
    train_and_save(model_dir=model_dir)
    return model_dir


def test_prepare_features_is_safe_to_run_twice_and_without_id():
    raw = load_raw()
    once = prepare_features(raw)
    twice = prepare_features(once)  # no customerID column, TotalCharges already numeric
    pd.testing.assert_frame_equal(once, twice)
    assert "customerID" not in once.columns


def test_training_saves_model_and_metadata(saved_dir):
    assert (saved_dir / MODEL_FILE).exists()
    meta = json.loads((saved_dir / METADATA_FILE).read_text(encoding="utf-8"))
    assert 0.1 <= meta["threshold"] <= 0.9
    assert meta["test_metrics"]["roc_auc"] > 0.8
    assert "Churn" not in meta["feature_columns"]
    assert "customerID" not in meta["feature_columns"]


def test_raw_and_cleaned_records_get_identical_predictions(saved_dir):
    """The train/serve skew fix: the model must handle raw text TotalCharges itself."""
    predictor = ChurnPredictor(saved_dir)
    raw = load_raw()
    blank = raw["TotalCharges"].str.strip() == ""
    sample = pd.concat([raw.head(20), raw[blank].head(3)])  # includes blank TotalCharges

    from_raw = predictor.predict(sample)
    from_clean = predictor.predict(clean(sample))
    pd.testing.assert_frame_equal(from_raw, from_clean)
    assert from_raw["churn_probability"].between(0, 1).all()


def test_flag_follows_saved_threshold(saved_dir):
    predictor = ChurnPredictor(saved_dir)
    out = predictor.predict(load_raw().head(50))
    expected = out["churn_probability"] >= predictor.metadata["threshold"]
    assert (out["flagged"] == expected).all()


def test_missing_columns_give_a_clear_error(saved_dir):
    predictor = ChurnPredictor(saved_dir)
    with pytest.raises(ValueError, match="Missing required columns"):
        predictor.predict(load_raw().head(2).drop(columns=["tenure", "Contract"]))
