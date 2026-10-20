import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from churn.data import clean, load_raw
from churn.eda import churn_rate_by_tenure
from churn.features import build_preprocessor, split
from churn.linearity import curved_columns


def test_poly_adds_numeric_columns():
    X_train, *_ = split(clean(load_raw()))
    plain = build_preprocessor(poly_degree=1).fit_transform(X_train)
    poly = build_preprocessor(poly_degree=2).fit_transform(X_train)
    # 4 numeric columns -> 4 originals + 4 squares + 6 pairwise products = 14
    assert poly.shape[1] - plain.shape[1] == 10
    assert poly.shape[0] == plain.shape[0]


def test_spline_columns_expand_only_the_chosen_column():
    X_train, *_ = split(clean(load_raw()))
    plain = build_preprocessor().fit_transform(X_train)
    targeted = build_preprocessor(spline_columns=["MonthlyCharges"]).fit_transform(X_train)
    # one column -> n_knots + degree - 1 = 8 spline columns, so 7 extra
    assert targeted.shape[1] - plain.shape[1] == 7
    assert targeted.shape[0] == plain.shape[0]


def test_plain_numeric_columns_are_only_scaled():
    plain = dict((name, step) for name, step, _ in build_preprocessor().transformers)
    assert isinstance(plain["num"], StandardScaler)  # no hidden polynomial step

    targeted = build_preprocessor(spline_columns=["tenure"]).transformers
    assert isinstance(dict((n, s) for n, s, _ in targeted)["num"], StandardScaler)


def test_spline_and_poly_cannot_be_combined():
    with pytest.raises(ValueError):
        build_preprocessor(poly_degree=2, spline_columns=["tenure"])


def test_curved_columns_applies_flag_and_gain_threshold():
    report = pd.DataFrame(
        {
            "gain": [0.015, 0.008, 0.0035, 0.02],
            "non_linear": [True, True, True, False],
        },
        index=["a", "b", "c", "d"],
    )
    assert curved_columns(report) == ["a", "b"]


def test_tenure_bands_cover_all_customers():
    out = churn_rate_by_tenure(clean(load_raw()))
    assert out["customers"].sum() == 7043
    assert out["churn_rate"].is_monotonic_decreasing
