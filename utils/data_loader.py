"""
Dataset ingestion for phishing detection.
Supports UCI Phishing Websites, Kaggle-style CSV (url/label), and PhishTank-style exports.
Label: 1 = phishing, 0 = legitimate.
"""
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional

try:
    from config import RAW_DATA_DIR, DATASET_FILENAME, DATASET_PATH, RANDOM_STATE, TEST_SIZE
except ImportError:
    RAW_DATA_DIR = Path("data/raw")
    DATASET_PATH = RAW_DATA_DIR / "phishing_dataset.csv"
    RANDOM_STATE = 42
    TEST_SIZE = 0.2


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize to 'url' and 'label' (1=phishing, 0=legitimate)."""
    df = df.copy()
    # Common column names
    url_cols = [c for c in df.columns if c.lower() in ("url", "website", "link", "domain")]
    label_cols = [c for c in df.columns if c.lower() in ("label", "result", "phishing", "class", "target")]
    if url_cols:
        df["url"] = df[url_cols[0]].astype(str).str.strip()
    else:
        # UCI dataset sometimes has no URL column but has feature columns; we need at least url
        if "url" not in df.columns and len(df.columns) > 0:
            df["url"] = "https://example.com"  # placeholder if only features provided
    if label_cols:
        raw = df[label_cols[0]]
        # Map common encodings to 0/1
        if raw.dtype in ("bool", "int", "int64"):
            df["label"] = (raw.astype(int) != 0).astype(int)
        else:
            raw = raw.astype(str).str.lower()
            df["label"] = raw.replace({"phishing": 1, "legitimate": 0, "bad": 1, "good": 0, "1": 1, "0": 0}).astype(int)
    if "url" not in df.columns:
        df["url"] = ""
    if "label" not in df.columns:
        df["label"] = 0
    return df[["url", "label"]]


def load_phishing_dataset(
    path: Optional[Path] = None,
    sample_frac: Optional[float] = None,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Load CSV from path (or config DATASET_PATH). Expects URL and label columns.
    Returns DataFrame with columns 'url', 'label' (1=phishing, 0=legitimate).
    """
    path = path or DATASET_PATH
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Please add a CSV with 'url' and 'label' (1=phishing, 0=legitimate). "
            "See README for UCI/Kaggle dataset instructions."
        )
    df = pd.read_csv(path)  # load full dataset
    df = _normalize_columns(df)
    # Drop rows with empty URL or invalid label
    df = df[df["url"].notna() & (df["url"].str.strip() != "")]
    df = df[df["label"].isin([0, 1])]
    if sample_frac and sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=random_state)
    return df.reset_index(drop=True)


def prepare_train_test(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split by label."""
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df["label"]
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)
