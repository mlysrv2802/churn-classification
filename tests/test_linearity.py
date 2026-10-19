import numpy as np
import pandas as pd

from churn.linearity import linearity_report, numeric_candidates


def _synthetic(n: int = 3000) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "linear": rng.normal(size=n),
            "u_shaped": rng.normal(size=n),
            "noise": rng.normal(size=n),
            "binary": rng.integers(0, 2, size=n),
        }
    )
    logit = 1.5 * X["linear"] + 2.0 * (X["u_shaped"] ** 2 - 1)
    y = pd.Series(rng.random(n) < 1 / (1 + np.exp(-logit)), name="y").astype(int)
    return X, y


def test_binary_columns_are_skipped():
    X, _ = _synthetic()
    assert "binary" not in numeric_candidates(X)


def test_report_flags_only_the_curved_column():
    X, y = _synthetic()
    report = linearity_report(X, y)
    assert report.loc["u_shaped", "non_linear"]
    assert not report.loc["linear", "non_linear"]
    assert not report.loc["noise", "non_linear"]
    assert report.index[0] == "u_shaped"
