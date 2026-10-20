from churn.data import clean, load_raw
from churn.features import split
from churn.models import build_logistic_regression
from churn.thresholds import out_of_fold_proba, pick_threshold, threshold_table

SHOW = ["flagged", "caught", "false_alarms", "missed", "precision", "recall", "f1", "cost"]


def main() -> None:
    X_train, X_test, y_train, y_test = split(clean(load_raw()))
    model = build_logistic_regression()

    # 1) Choose the cutoff using TRAIN data only (predictions from unseen folds).
    train_table = threshold_table(y_train, out_of_fold_proba(model, X_train, y_train))
    print("What each cutoff does on the TRAIN set (honest, out-of-fold predictions)\n")
    print(train_table[SHOW].round(3).to_string())

    best_f1 = pick_threshold(train_table, by="f1")
    best_cost = pick_threshold(train_table, by="cost")
    print(f"\nBest cutoff by F1  : {best_f1}")
    print(f"Best cutoff by cost: {best_cost}  (uses PLACEHOLDER costs from config.py)")

    # 2) Check the chosen cutoffs ONCE on the untouched test set.
    model.fit(X_train, y_train)
    test_table = threshold_table(
        y_test, model.predict_proba(X_test)[:, 1], thresholds=[0.5, best_f1, best_cost]
    )
    test_table.index = ["default 0.5", f"best F1 ({best_f1})", f"best cost ({best_cost})"]
    print("\nTEST set, scored once\n")
    print(test_table[SHOW].round(3).to_string())


if __name__ == "__main__":
    main()
