from churn.data import clean, load_raw
from churn.features import split
from churn.leakage import (
    selection_bias_check,
    shuffled_label_auc,
    single_feature_scores,
    train_test_overlap,
)
from churn.models import build_logistic_regression, build_random_forest


def main() -> None:
    X_train, X_test, y_train, y_test = split(clean(load_raw()))

    print("1. Single-column predictive power (CV ROC-AUC, train set)")
    print("   Look for anything near 1.0: that column may give away the answer.\n")
    print(single_feature_scores(X_train, y_train).round(3).to_string())

    print("\n2. Identical rows shared by train and test")
    for key, value in train_test_overlap(X_train, y_train, X_test, y_test).items():
        print(f"   {key}: {value:.3f}" if isinstance(value, float) else f"   {key}: {value}")

    print("\n3. Shuffled-label test (must be about 0.5 if the pipeline is clean)")
    for name, model in {
        "LogisticRegression": build_logistic_regression(),
        "RandomForest": build_random_forest(),
    }.items():
        print(f"   {name:20} {shuffled_label_auc(model, X_train, y_train):.3f}")

    print("\n4. Column-selection bias (ROC-AUC of the spline model)")
    result = selection_bias_check(X_train, y_train)
    print(f"   columns chosen on all train rows : {result['outside']:.4f}")
    print(f"   columns re-chosen inside each fold: {result['inside']:.4f}")
    print(f"   difference                        : {result['outside'] - result['inside']:+.4f}")


if __name__ == "__main__":
    main()
