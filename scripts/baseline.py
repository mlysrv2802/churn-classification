from sklearn.metrics import accuracy_score, recall_score

from churn.data import clean, load_raw
from churn.features import split
from churn.models import build_dummy, build_logistic_regression


def main() -> None:
    X_train, X_test, y_train, y_test = split(clean(load_raw()))

    models = {
        "Dummy (always 'stays')": build_dummy(),
        "LogisticRegression": build_logistic_regression(),
    }

    print(f"{'model':28} {'accuracy':>9} {'churn recall':>13}")
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        print(
            f"{name:28} {accuracy_score(y_test, pred):9.3f} "
            f"{recall_score(y_test, pred):13.3f}"
        )


if __name__ == "__main__":
    main()
