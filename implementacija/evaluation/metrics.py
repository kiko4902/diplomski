import numpy as np
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    fbeta_score,
    confusion_matrix,
)


def compute_metrics(y_true, y_pred, y_proba=None):
    result = {}
    result["f1"] = f1_score(y_true, y_pred)
    result["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    result["mcc"] = matthews_corrcoef(y_true, y_pred)
    result["f2"] = fbeta_score(y_true, y_pred, beta=2)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    result["g_mean"] = np.sqrt(sensitivity * specificity)

    if y_proba is not None:
        result["auc_roc"] = roc_auc_score(y_true, y_proba)
        result["auc_pr"] = average_precision_score(y_true, y_proba)
    else:
        result["auc_roc"] = float("nan")
        result["auc_pr"] = float("nan")

    return result
