from churn.config import PROJECT_ROOT, TARGET
from churn.data import clean, load_raw
from churn.linearity import linearity_report
from churn.plots import plot_churn_vs_numeric


def main() -> None:
    df = clean(load_raw())
    X, y = df.drop(columns=[TARGET]), df[TARGET]

    report = linearity_report(X, y)
    print("Which numeric columns are non-linear? (single-column models, 5-fold CV)\n")
    print(report.round(4).to_string())

    out = plot_churn_vs_numeric(df, list(report.index), PROJECT_ROOT / "reports" / "churn_vs_numeric.png")
    print(f"\nPlot saved to {out}")


if __name__ == "__main__":
    main()
