import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn.config import RANDOM_STATE, TARGET, TEST_SIZE


def split(df: pd.DataFrame):
    """Stratified train/test split so both sets keep the ~26% churn rate."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )


def build_preprocessor() -> ColumnTransformer:
    """Scale numeric columns, one-hot encode the rest.

    Column types are detected at fit time, so the transformer is only
    fitted on training data (no leakage) when used inside a Pipeline.
    """
    return ColumnTransformer(
        [
            ("num", StandardScaler(), make_column_selector(dtype_include=np.number)),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                make_column_selector(dtype_exclude=np.number),
            ),
        ]
    )
