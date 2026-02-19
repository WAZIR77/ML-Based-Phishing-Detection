"""
Model evaluation: accuracy, precision, recall, F1, confusion matrix.
Feature importance for explainability (SOC-level).
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
from typing import List, Optional, Any


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """Compute metrics. Labels: 1 = phishing, 0 = legitimate."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Print sklearn classification report and confusion matrix."""
    print(classification_report(y_true, y_pred, target_names=["Legitimate", "Phishing"], digits=4))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_true, y_pred))


def get_feature_importance(
    model: Any,
    feature_names: List[str],
    top_k: int = 15,
) -> List[tuple]:
    """
    Extract feature importance for tree-based models (e.g. Random Forest).
    Returns list of (feature_name, importance) sorted by importance descending.
    """
    if not hasattr(model, "feature_importances_"):
        return []
    imp = model.feature_importances_
    pairs = list(zip(feature_names, imp))
    pairs.sort(key=lambda x: -x[1])
    return pairs[:top_k]
