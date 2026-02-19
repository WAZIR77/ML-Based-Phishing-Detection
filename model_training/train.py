"""
Train multiple models (Logistic Regression baseline, Random Forest primary),
with cross-validation and hyperparameter tuning. Prioritize recall.
Persist best model and artifacts (joblib).
"""
import json
import joblib
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, recall_score, precision_score, f1_score, accuracy_score

try:
    from config import (
        MODELS_DIR, MODEL_PATH, RANDOM_STATE, CV_FOLDS,
        RECALL_WEIGHT, MODEL_FILENAME, FEATURE_NAMES_PATH,
    )
except ImportError:
    MODELS_DIR = Path("models")
    MODEL_PATH = MODELS_DIR / "phishing_classifier.joblib"
    RANDOM_STATE = 42
    CV_FOLDS = 5
    RECALL_WEIGHT = 2.0
    FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.joblib"


def _recall_weighted_scorer(y_true, y_pred):
    """Custom scorer: emphasize recall (minimize false negatives)."""
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred, zero_division=0)
    return 0.4 * acc + 0.6 * rec  # recall weighted higher


def train_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list,
    scale: bool = True,
    save_dir: Path = None,
) -> dict:
    """
    Train Logistic Regression and Random Forest; run CV; tune RF.
    Returns dict with metrics and the chosen model (RF preferred; fallback LR).
    Saves: scaler (if scale), model, feature_names.
    """
    save_dir = save_dir or MODELS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    recall_scorer = make_scorer(recall_score, zero_division=0)
    custom_scorer = make_scorer(_recall_weighted_scorer)

    # Scale features for LR
    scaler = StandardScaler() if scale else None
    if scaler is not None:
        X_tr = scaler.fit_transform(X_train)
    else:
        X_tr = X_train

    results = {}

    # --- Logistic Regression (baseline) ---
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced")
    lr.fit(X_tr, y_train)
    lr_cv = cross_validate(lr, X_tr, y_train, cv=cv, scoring={
        "accuracy": "accuracy",
        "recall": recall_scorer,
        "precision": make_scorer(precision_score, zero_division=0),
        "f1": "f1",
    })
    results["logistic_regression"] = {
        "cv_accuracy_mean": float(np.mean(lr_cv["test_accuracy"])),
        "cv_recall_mean": float(np.mean(lr_cv["test_recall"])),
        "cv_precision_mean": float(np.mean(lr_cv["test_precision"])),
        "cv_f1_mean": float(np.mean(lr_cv["test_f1"])),
    }

    # --- Random Forest (primary) with hyperparameter tuning ---
    # Grid over key params; optimize for recall
    best_rf = None
    best_score = -1
    best_params = None
    for n_est in [100, 200]:
        for max_d in [10, 15, 20]:
            for min_samples_leaf in [2, 5]:
                rf = RandomForestClassifier(
                    n_estimators=n_est,
                    max_depth=max_d,
                    min_samples_leaf=min_samples_leaf,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                    n_jobs=-1,
                )
                scores = cross_validate(rf, X_train, y_train, cv=cv, scoring={"recall": recall_scorer})
                mean_recall = np.mean(scores["test_recall"])
                if mean_recall > best_score:
                    best_score = mean_recall
                    best_rf = rf
                    best_params = {"n_estimators": n_est, "max_depth": max_d, "min_samples_leaf": min_samples_leaf}
    if best_rf is None:
        best_rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1)
        best_params = {}
    best_rf.fit(X_train, y_train)

    rf_cv = cross_validate(best_rf, X_train, y_train, cv=cv, scoring={
        "accuracy": "accuracy",
        "recall": recall_scorer,
        "precision": make_scorer(precision_score, zero_division=0),
        "f1": "f1",
    })
    results["random_forest"] = {
        "cv_accuracy_mean": float(np.mean(rf_cv["test_accuracy"])),
        "cv_recall_mean": float(np.mean(rf_cv["test_recall"])),
        "cv_precision_mean": float(np.mean(rf_cv["test_precision"])),
        "cv_f1_mean": float(np.mean(rf_cv["test_f1"])),
        "best_params": best_params,
    }

    # Choose model: prefer RF (better recall typically)
    results["chosen_model"] = "random_forest"
    chosen = best_rf
    # Persist: we need to store scaler only if we use LR at inference; for RF we don't scale. Store RF and optionally LR for comparison.
    joblib.dump(best_rf, save_dir / MODEL_FILENAME)
    joblib.dump(feature_names, FEATURE_NAMES_PATH)
    if scaler is not None:
        joblib.dump(scaler, save_dir / "scaler.joblib")
    joblib.dump(lr, save_dir / "logistic_regression.joblib")
    with open(save_dir / "training_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


def load_trained_pipeline(save_dir: Path = None):
    """Load persisted model, feature names, and optional scaler."""
    save_dir = save_dir or MODELS_DIR
    model = joblib.load(save_dir / MODEL_FILENAME)
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    scaler = None
    scaler_path = save_dir / "scaler.joblib"
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
    return model, feature_names, scaler
