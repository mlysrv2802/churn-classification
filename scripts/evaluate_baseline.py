from churn.data import clean, load_raw
from churn.evaluate import confusion, evaluate
from churn.features import split
from churn.models import build_dummy, build_logistic_regression

METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def main() -> None:
    X_train, X_test, y_train, y_test = split(clean(load_raw()))

    models = {
        "Dummy": build_dummy(),
        "LogisticRegression": build_logistic_regression(),
    }

    print(f"{'model':20}" + "".join(f"{m:>11}" for m in METRICS))
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted[name] = model
        scores = evaluate(model, X_test, y_test)
        print(f"{name:20}" + "".join(f"{scores[m]:11.3f}" for m in METRICS))

    cm = confusion(fitted["LogisticRegression"], X_test, y_test)
    print("\nLogisticRegression confusion matrix (test set):")
    print(f"                 pred stays  pred churns")
    print(f"actual stays     {cm['tn']:10d}  {cm['fp']:11d}")
    print(f"actual churns    {cm['fn']:10d}  {cm['tp']:11d}")


if __name__ == "__main__":
    main()
