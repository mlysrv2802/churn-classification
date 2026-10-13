import pandas as pd

from churn.config import ID_COLUMN, TARGET


def blank_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Rows of the raw frame where TotalCharges is whitespace-only."""
    blank = df["TotalCharges"].str.strip() == ""
    return df.loc[blank, [ID_COLUMN, "tenure", "MonthlyCharges", "TotalCharges"]]


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
