import pandas as pd
import pytest

from churn.data import clean, load_raw, prepare_features
from churn.predict import ChurnPredictor
from churn.registry import MODEL_FILE, ModelRegistry
from churn.train import train_and_save


def test_prepare_features_is_safe_to_run_twice_and_without_id():
    raw = load_raw()
    once = prepare_features(raw)
    twice = prepare_features(once)  # no customerID column, TotalCharges already numeric
    pd.testing.assert_frame_equal(once, twice)
    assert "customerID" not in once.columns


def test_training_saves_a_production_version_with_metadata(saved_dir):
    registry = ModelRegistry(saved_dir)
    version = registry.production_version()
    assert version is not None
    assert (registry.versions_dir / version / MODEL_FILE).exists()
    meta = registry.metadata(version)
    assert meta["version"] == version
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


def test_training_twice_keeps_both_versions_and_serves_the_newest(tmp_path):
    first = train_and_save(model_dir=tmp_path)
    second = train_and_save(model_dir=tmp_path)  # same data, so same score: allowed to replace
    assert first["version"] != second["version"]
    assert second["promoted"] is True
    registry = ModelRegistry(tmp_path)
    assert len(registry.list_versions()) == 2
    assert registry.production_version() == second["version"]
    assert ChurnPredictor(tmp_path).metadata["version"] == second["version"]


def test_flag_follows_saved_threshold(saved_dir):
    predictor = ChurnPredictor(saved_dir)
    out = predictor.predict(load_raw().head(50))
    expected = out["churn_probability"] >= predictor.metadata["threshold"]
    assert (out["flagged"] == expected).all()


def test_missing_columns_give_a_clear_error(saved_dir):
    predictor = ChurnPredictor(saved_dir)
    with pytest.raises(ValueError, match="Missing required columns"):
        predictor.predict(load_raw().head(2).drop(columns=["tenure", "Contract"]))
