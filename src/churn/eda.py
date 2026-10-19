import pandas as pd

from churn.config import ID_COLUMN, TARGET


def blank_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Rows of the raw frame where TotalCharges is whitespace-only."""
    blank = df["TotalCharges"].str.strip() == ""
    return df.loc[blank, [ID_COLUMN, "tenure", "MonthlyCharges", "TotalCharges"]]


def churn_rate_by_tenure(clean_df: pd.DataFrame) -> pd.DataFrame:
    """Churn rate per tenure band, on the cleaned frame (Churn is 0/1).

    A straight-line relationship would show a steady drop; a steep drop that
    flattens out is the non-linearity polynomial features can capture.
    """
    bands = pd.cut(
        clean_df["tenure"],
        bins=[-1, 6, 12, 24, 48, 72],
        labels=["0-6", "7-12", "13-24", "25-48", "49-72"],
    )
    grouped = clean_df.groupby(bands, observed=True)[TARGET]
    return pd.DataFrame({"customers": grouped.size(), "churn_rate": grouped.mean().round(3)})


def summarize(df: pd.DataFrame) -> str:
    """Human-readable first look at the raw frame."""
    balance = df[TARGET].value_counts(normalize=True).round(3)
    blanks = blank_total_charges(df)
    return "\n".join(
        [
            f"Shape: {df.shape}",
            f"NaN cells: {int(df.isna().sum().sum())}",
            f"Duplicate {ID_COLUMN}s: {int(df[ID_COLUMN].duplicated().sum())}",
            "",
            "Class balance:",
            balance.to_string(),
            "",
            f"Blank TotalCharges rows: {len(blanks)}",
            blanks.to_string(index=False),
        ]
    )
