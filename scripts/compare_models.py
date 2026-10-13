from churn.compare import SCORING, cross_validate_models
from churn.data import clean, load_raw
from churn.evaluate import evaluate
from churn.features import split
from churn.models import candidates


def main() -> None:
    X_train, X_test, y_train, y_test = split(clean(load_raw()))
    models = candidates()

    print("5-fold cross-validation on the TRAIN set (mean +/- std)\n")
    table = cross_validate_models(models, X_train, y_train)
    print(f"{'model':20}" + "".join(f"{m:>16}" for m in SCORING))
    for name, row in table.iterrows():
        cells = "".join(
            f"{row[f'{m}_mean']:>9.3f}+/-{row[f'{m}_std']:.3f}" for m in SCORING
        )
        print(f"{name:20}{cells}")

    print("\nFinal check on the untouched TEST set\n")
    print(f"{'model':20}" + "".join(f"{m:>11}" for m in SCORING))
    for name, model in models.items():
        model.fit(X_train, y_train)
        s = evaluate(model, X_test, y_test)
        print(f"{name:20}" + "".join(f"{s[m]:11.3f}" for m in SCORING))


if __name__ == "__main__":
    main()
