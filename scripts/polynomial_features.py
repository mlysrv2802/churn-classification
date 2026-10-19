from churn.compare import SCORING, cross_validate_models
from churn.data import clean, load_raw
from churn.eda import churn_rate_by_tenure
from churn.features import split
from churn.models import build_logistic_regression


def main() -> None:
    df = clean(load_raw())

    print("Churn rate by tenure band (is the relationship a straight line?)\n")
    print(churn_rate_by_tenure(df).to_string())

    X_train, X_test, y_train, y_test = split(df)
    models = {
        f"LogReg degree {d}": build_logistic_regression(poly_degree=d) for d in (1, 2, 3)
    }

    print("\n5-fold cross-validation on the TRAIN set (mean +/- std)\n")
    table = cross_validate_models(models, X_train, y_train)
    print(f"{'model':18}" + "".join(f"{m:>16}" for m in SCORING))
    for name, row in table.iterrows():
        cells = "".join(
            f"{row[f'{m}_mean']:>9.3f}+/-{row[f'{m}_std']:.3f}" for m in SCORING
        )
        print(f"{name:18}{cells}")


if __name__ == "__main__":
    main()
