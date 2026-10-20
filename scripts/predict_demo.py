import pandas as pd

from churn.data import load_raw
from churn.predict import ChurnPredictor


def main() -> None:
    raw = load_raw()  # raw rows, exactly as in the CSV (TotalCharges is still text)
    blank = raw["TotalCharges"].str.strip() == ""
    sample = pd.concat([raw.head(3), raw[blank].head(2)])  # 3 normal rows + 2 blank-TotalCharges rows

    result = ChurnPredictor().predict(sample)
    out = sample[["customerID", "tenure", "TotalCharges", "Churn"]].join(result)
    print(out.round(3).to_string())


if __name__ == "__main__":
    main()
