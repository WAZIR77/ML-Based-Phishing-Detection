"""
Real-time prediction: load model, extract features from URL, return classification,
risk score (0-100), and top contributing features for explainability.
"""
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Project root on path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from config import MODELS_DIR
except ImportError:
    MODELS_DIR = PROJECT_ROOT / "models"

import pandas as pd
import joblib
from feature_extraction import extract_all_features
from feature_extraction import get_feature_names


def _ensure_feature_order(feats: Dict[str, Any], feature_names: List[str]):
    """Build 1-row DataFrame in same order as training; impute missing/None with 0."""
    arr = []
    for n in feature_names:
        v = feats.get(n, 0)
        if v is None:
            v = 0
        try:
            arr.append(float(v))
        except (TypeError, ValueError):
            arr.append(0.0)
    return pd.DataFrame([arr], columns=feature_names)


def load_artifacts():
    """Load model and feature names from disk."""
    model_path = MODELS_DIR / "phishing_classifier.joblib"
    fn_path = MODELS_DIR / "feature_names.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run run_training.py first.")
    model = joblib.load(model_path)
    feature_names = joblib.load(fn_path) if fn_path.exists() else get_feature_names()
    scaler_path = MODELS_DIR / "scaler.joblib"
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    return model, feature_names, scaler


# Lazy load
_model, _feature_names, _scaler = None, None, None


def _get_artifacts():
    global _model, _feature_names, _scaler
    if _model is None:
        _model, _feature_names, _scaler = load_artifacts()
    return _model, _feature_names, _scaler


def predict(
    url: str,
    fetch_content: bool = False,
) -> Tuple[str, float, List[Tuple[str, float]]]:
    """
    Classify URL. Returns:
      - class_label: "Phishing" or "Legitimate"
      - risk_score: 0-100 (100 = highest risk)
      - top_features: list of (feature_name, contribution) for explainability
    """
    model, feature_names, scaler = _get_artifacts()
    feats = extract_all_features(url, fetch_content=fetch_content)
    for k, v in feats.items():
        if v is None:
            feats[k] = 0
    X = _ensure_feature_order(feats, feature_names)
    if scaler is not None and not hasattr(model, "feature_importances_"):
        X = scaler.transform(X)
    proba = getattr(model, "predict_proba", None)
    if proba is not None:
        p = proba(X)[0]
        # index 1 = phishing in sklearn (class 1)
        phishing_prob = float(p[1]) if len(p) > 1 else float(p[0])
    else:
        pred = model.predict(X)[0]
        phishing_prob = 1.0 if pred == 1 else 0.0
    risk_score = round(phishing_prob * 100, 1)
    label = "Phishing" if phishing_prob >= 0.5 else "Legitimate"

    # Feature importance: for RF, use feature_importances_ * (1 if value non-zero else 0) as proxy for "contribution" for this sample
    top_features: List[Tuple[str, float]] = []
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        vals = X.values.flatten()
        # Contribution proxy: importance * normalized feature value for this sample
        contrib = imp * (np.abs(vals) + 1e-6)
        order = np.argsort(-contrib)
        for i in order[:10]:
            if contrib[i] > 0:
                top_features.append((feature_names[i], round(float(contrib[i]), 4)))
    return label, risk_score, top_features


def predict_dict(url: str, fetch_content: bool = False) -> Dict[str, Any]:
    """Return full response dict for API."""
    try:
        label, risk_score, top_features = predict(url, fetch_content=fetch_content)
        return {
            "url": url,
            "classification": label,
            "risk_score": risk_score,
            "top_contributing_features": [{"name": n, "contribution": c} for n, c in top_features],
            "error": None,
        }
    except Exception as e:
        return {
            "url": url,
            "classification": None,
            "risk_score": None,
            "top_contributing_features": [],
            "error": str(e),
        }
