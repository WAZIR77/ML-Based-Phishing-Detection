"""
Feature matrix building and consistent feature order for train/predict.
"""
import numpy as np
import pandas as pd
from typing import List, Tuple

from feature_extraction import get_feature_names


# Exclude label from feature columns
LABEL_COL = "label"


def get_feature_names_ordered() -> List[str]:
    """Ordered list of feature names (no label)."""
    return get_feature_names()


def build_feature_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build X (features), y (labels), and feature_names from a DataFrame produced
    by extract_features_batch (includes 'label' and all extracted columns).
    Imputes missing/NaN with 0. Feature order is fixed for model consistency.
    """
    names = get_feature_names_ordered()
    # Ensure all columns exist; missing -> 0
    for n in names:
        if n not in df.columns:
            df = df.copy()
            df[n] = 0
    X = df[names].fillna(0).astype(np.float64)
    y = df[LABEL_COL].values.astype(np.int32) if LABEL_COL in df.columns else np.array([])
    return X, y, names
