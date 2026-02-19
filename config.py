"""
Configuration for the Phishing Detection ML pipeline.
Centralizes paths, model params, and feature lists.
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
for d in (RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Dataset: expect CSV with 'url' (or 'URL') and 'label' (1=phishing, 0=legitimate)
# Compatible with UCI Phishing Websites, Kaggle URL datasets
DATASET_FILENAME = "phishing_dataset.csv"
DATASET_PATH = RAW_DATA_DIR / DATASET_FILENAME
PROCESSED_FEATURES_PATH = PROCESSED_DATA_DIR / "features_labels.csv"

# ML settings
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
RECALL_WEIGHT = 2.0  # Prioritize recall in scoring (minimize false negatives)

# Model artifact names
MODEL_FILENAME = "phishing_classifier.joblib"
VECTORIZER_FILENAME = "feature_vectorizer.joblib"
FEATURE_NAMES_FILENAME = "feature_names.joblib"
MODEL_PATH = MODELS_DIR / MODEL_FILENAME
VECTORIZER_PATH = MODELS_DIR / VECTORIZER_FILENAME
FEATURE_NAMES_PATH = MODELS_DIR / FEATURE_NAMES_FILENAME

# Content fetch (for content-based features) - safety limits
FETCH_TIMEOUT_SEC = 5
FETCH_MAX_BYTES = 100_000
USER_AGENT = "PhishingDetectionBot/1.0 (Security Research)"

# Suspicious URL keywords (phishing-related)
SUSPICIOUS_KEYWORDS = [
    "login", "signin", "verify", "update", "secure", "account",
    "banking", "paypal", "amazon", "apple", "microsoft", "confirm",
    "suspend", "restore", "password", "credential", "urgent", "click"
]

# URL shortening domains (subset)
URL_SHORTENER_DOMAINS = [
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd",
    "buff.ly", "adf.ly", "bit.do", "lnkd.in", "db.tt", "qr.ae"
]
