from churn.data import clean, load_raw
from churn.evaluate import evaluate
from churn.features import split
from churn.models import build_logistic_regression
from churn.tuning import C_PARAM, DEGREE_PARAM, best_per_degree, tune_logistic_regression

METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def main() -> None:
    X_train, X_test, y_train, y_test = split(clean(load_raw()))

    search = tune_logistic_regression(X_train, y_train)

    print("Best CV ROC-AUC reached by each polynomial degree (C tuned alongside)\n")
    print(best_per_degree(search).round(4).to_string())
    print(
        f"\nWinner: degree {search.best_params_[DEGREE_PARAM]}, "
        f"C={search.best_params_[C_PARAM]:.4f}, CV ROC-AUC {search.best_score_:.4f}"
    )

    untuned = build_logistic_regression().fit(X_train, y_train)
    print("\nTEST set, scored once:\n")
    print(f"{'model':22}" + "".join(f"{m:>11}" for m in METRICS))
    for name, model in {"untuned (deg 1, C=1)": untuned, "tuned": search.best_estimator_}.items():
        s = evaluate(model, X_test, y_test)
        print(f"{name:22}" + "".join(f"{s[m]:11.3f}" for m in METRICS))


if __name__ == "__main__":
    main()
