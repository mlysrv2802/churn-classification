import pandas as pd

from churn.data import clean, load_raw


def _raw_sample() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customerID": ["a", "b"],
            "tenure": [0, 5],
            "MonthlyCharges": [50.0, 20.0],
            "TotalCharges": [" ", "100.5"],
            "Churn": ["No", "Yes"],
        }
    )


def test_clean_converts_blank_total_charges_to_zero():
    out = clean(_raw_sample())
    assert out["TotalCharges"].tolist() == [0.0, 100.5]


def test_clean_encodes_target_and_drops_id():
    out = clean(_raw_sample())
    assert out["Churn"].tolist() == [0, 1]
    assert "customerID" not in out.columns


def test_clean_does_not_mutate_input():
    raw = _raw_sample()
    clean(raw)
    assert raw["TotalCharges"].tolist() == [" ", "100.5"]


def test_real_dataset_cleans_without_nulls():
    out = clean(load_raw())
    assert out.shape == (7043, 20)
    assert out.isna().sum().sum() == 0
    assert out["Churn"].mean() > 0.2
