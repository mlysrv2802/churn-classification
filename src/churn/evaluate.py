from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate(model, X, y) -> dict[str, float]:
    """Score a fitted classifier. Metrics are for the positive class (churn)."""
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, proba),
    }


def confusion(model, X, y) -> dict[str, int]:
    """Confusion matrix as named counts: tn, fp, fn, tp."""
    tn, fp, fn, tp = confusion_matrix(y, model.predict(X)).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
