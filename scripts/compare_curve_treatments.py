from churn.compare import SCORING, cross_validate_models
from churn.data import clean, load_raw
from churn.evaluate import evaluate
from churn.features import split
from churn.linearity import curved_columns, linearity_report
from churn.models import build_logistic_regression, build_random_forest


def main() -> None:
    X_train, X_test, y_train, y_test = split(clean(load_raw()))

    # Decide which columns get a curve using TRAIN data only (no test peeking).
    curved = curved_columns(linearity_report(X_train, y_train))
    print(f"Columns given a curve (from the train-set report): {curved}\n")

    models = {
        "Plain lines": build_logistic_regression(),
        "Polynomial degree 2 (all numeric)": build_logistic_regression(poly_degree=2),
        "Splines on curved columns only": build_logistic_regression(spline_columns=curved),
        "RandomForest (finds curves itself)": build_random_forest(),
    }

    print("5-fold cross-validation on the TRAIN set (mean +/- std)\n")
    table = cross_validate_models(models, X_train, y_train)
    print(f"{'model':36}" + "".join(f"{m:>16}" for m in SCORING))
    for name, row in table.iterrows():
        cells = "".join(f"{row[f'{m}_mean']:>9.3f}+/-{row[f'{m}_std']:.3f}" for m in SCORING)
        print(f"{name:36}{cells}")

    print("\nTEST set, scored once\n")
    print(f"{'model':36}" + "".join(f"{m:>11}" for m in SCORING))
    for name, model in models.items():
        s = evaluate(model.fit(X_train, y_train), X_test, y_test)
        print(f"{name:36}" + "".join(f"{s[m]:11.3f}" for m in SCORING))


if __name__ == "__main__":
    main()
