import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    PolynomialFeatures,
    SplineTransformer,
    StandardScaler,
)

from churn.config import RANDOM_STATE, TARGET, TEST_SIZE


def split(df: pd.DataFrame):
    """Stratified train/test split so both sets keep the ~26% churn rate."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )


def build_preprocessor(
    poly_degree: int = 1, spline_columns: list[str] | None = None
) -> ColumnTransformer:
    """Scale numeric columns, one-hot encode the rest.

    With poly_degree > 1 the numeric columns also get squares and pairwise
    interactions (e.g. tenure * MonthlyCharges), re-scaled afterwards because
    the new columns have very different ranges. Degree 1 leaves the columns
    unchanged, but the "poly" step always exists so a hyperparameter search
    can tune it as `prep__num__poly__degree`.

    With spline_columns, ONLY those columns are expanded into a flexible curve
    (cubic splines); the remaining numeric columns stay plain straight lines.
    This is the targeted alternative to bending every column with polynomials.

    Column types are detected at fit time, so the transformer is only
    fitted on training data (no leakage) when used inside a Pipeline.
    """
    if spline_columns and poly_degree > 1:
        raise ValueError("Use either poly_degree > 1 or spline_columns, not both")

    numeric = Pipeline(
        [
            ("scale", StandardScaler()),
            ("poly", PolynomialFeatures(degree=poly_degree, include_bias=False)),
            ("rescale", StandardScaler()),
        ]
    )
    if spline_columns:
        curved = list(spline_columns)
        curved_step = (
            "curved",
            Pipeline(
                [
                    ("spline", SplineTransformer(n_knots=6, degree=3, knots="quantile")),
                    ("rescale", StandardScaler()),
                ]
            ),
            curved,
        )

        def straight_numeric(X: pd.DataFrame) -> list[str]:
            return [c for c in X.select_dtypes("number").columns if c not in curved]

        numeric_step = ("num", numeric, straight_numeric)
        leading = [curved_step, numeric_step]
    else:
        leading = [("num", numeric, make_column_selector(dtype_include=np.number))]
    return ColumnTransformer(
        [
            *leading,
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                make_column_selector(dtype_exclude=np.number),
            ),
        ]
    )
