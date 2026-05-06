import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    roc_auc_score,
    cohen_kappa_score
)


def _extract_confusion_matrix_values(
        y_true: np.ndarray,
        y_pred: np.ndarray
) -> tuple[float, float, float, float]:
    """
    Compute confusion matrix and extract TN, FP, FN, TP.
    Return tuple (TN, FP, FN, TP).
    """
    cm = confusion_matrix(y_true, y_pred)
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TP = cm[1, 1]

    return TN, FP, FN, TP


def compute_sensitivity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute sensitivity (recall) — fraction of true FoG episodes correctly detected.
    Formula: TP / (TP + FN)
    Return 0.0 if there are no positive samples.
    """
    TN, FP, FN, TP = _extract_confusion_matrix_values(y_true, y_pred)
    if TP + FN == 0:
        return 0.0
    else:
        return TP / (TP + FN)


def compute_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute specificity — fraction of normal episodes correctly identified.
    Formula: TN / (TN + FP)
    Return 0.0 if there are no negative samples.
    """
    TN, FP, FN, TP = _extract_confusion_matrix_values(y_true, y_pred)
    if TN + FP == 0:
        return 0.0
    else:
        return TN / (TN + FP)


def compute_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute F1 score using sklearn's f1_score().
    Use zero_division=0 to handle edge cases.
    """
    return f1_score(y_true, y_pred, zero_division=0)


def compute_auc_roc(
        y_true: np.ndarray,
        y_proba: np.ndarray
) -> float:
    """
    Compute AUC-ROC score using sklearn's roc_auc_score().
    y_proba should be the probability of the positive class (column 1).
    Return 0.0 if only one class is present in y_true.
    """
    if len(np.unique(y_true)) < 2:
        return 0.0

    return roc_auc_score(y_true, y_proba[:, 1])


def compute_cohen_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Cohen's Kappa score using sklearn's cohen_kappa_score().
    Measures agreement between predictions and true labels
    accounting for chance.
    Return 0.0 if there is only one class in y_true.
    """
    if len(np.unique(y_true)) < 2:
        return 0.0

    return cohen_kappa_score(y_true, y_pred)


def compute_all_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray
) -> dict[str, float]:
    """
    Compute all metrics and return them in a single dictionary.
    Call every function above and collect results.
    Return dict with keys:
    'sensitivity', 'specificity', 'f1', 'auc_roc', 'cohen_kappa'
    """
    return {
        "sensitivity": compute_sensitivity(y_true, y_pred),
        "specificity": compute_specificity(y_true, y_pred),
        "f1": compute_f1(y_true, y_pred),
        "auc_roc": compute_auc_roc(y_true, y_proba),
        "cohen_kappa": compute_cohen_kappa(y_true, y_pred)
    }


def aggregate_loso_metrics(fold_results: list[dict]) -> dict[str, float]:
    """
    Compute mean and std of each metric across all LOSO folds.
    For each metric key compute mean and std across folds.
    Return dict with keys like:
    'sensitivity_mean', 'sensitivity_std', 'f1_mean', 'f1_std' etc.
    """
    aggregated = {}
    metric_keys = fold_results[0].keys()

    for key in metric_keys:
        values = [fold[key] for fold in fold_results]
        aggregated[f"{key}_mean"] = float(np.mean(values))
        aggregated[f"{key}_std"] = float(np.std(values))

    return aggregated
