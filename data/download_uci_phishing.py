"""
Download UCI PhiUSIIL Phishing URL Dataset and save in project format.
Source: https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset
- 235,795 URLs (134,850 legitimate, 100,945 phishing)
- UCI labels: 1 = legitimate, 0 = phishing
- We save: label 1 = phishing, 0 = legitimate (project convention)
"""
import sys
import zipfile
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import requests

try:
    from config import RAW_DATA_DIR, DATASET_PATH
except ImportError:
    RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
    DATASET_PATH = RAW_DATA_DIR / "phishing_dataset.csv"

UCI_DATASET_URL = "https://archive.ics.uci.edu/static/public/967/phiusiil+phishing+url+dataset.zip"
ZIP_PATH = RAW_DATA_DIR / "phiusiil_dataset.zip"
CSV_NAME_IN_ZIP = "PhiUSIIL_Phishing_URL_Dataset.csv"
# Cap rows for memory/speed (set to None to use full dataset)
MAX_ROWS = None  # Full dataset (~235k rows)


def download_and_extract():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading from {UCI_DATASET_URL} ...")
    r = requests.get(UCI_DATASET_URL, timeout=120, stream=True)
    r.raise_for_status()
    with open(ZIP_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Saved to {ZIP_PATH}, extracting...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        names = z.namelist()
        csv_name = next((n for n in names if n.endswith(".csv")), None)
        if not csv_name:
            raise ValueError("No CSV found in zip")
        with z.open(csv_name) as src:
            df = pd.read_csv(src, nrows=MAX_ROWS, encoding="utf-8", on_bad_lines="skip")
    return df, csv_name


def normalize_uci_to_project(df: pd.DataFrame) -> pd.DataFrame:
    """UCI: URL column, Label 1=legitimate 0=phishing -> project: url, label 1=phishing 0=legitimate."""
    df = df.copy()
    # Find URL column (case insensitive)
    url_col = None
    for c in df.columns:
        if c.strip().lower() == "url":
            url_col = c
            break
    if url_col is None:
        raise ValueError("No 'URL' column found. Columns: " + str(list(df.columns)))
    df["url"] = df[url_col].astype(str).str.strip()
    # Label: UCI 0=phishing, 1=legitimate -> we want 1=phishing, 0=legitimate
    label_col = None
    for c in df.columns:
        if c.strip().lower() in ("label", "class", "target"):
            label_col = c
            break
    if label_col is None:
        raise ValueError("No label column found. Columns: " + str(list(df.columns)))
    uci_label = df[label_col].astype(int)
    # our label: 1 = phishing, 0 = legitimate
    df["label"] = (1 - uci_label).clip(0, 1)  # UCI 0 -> 1 (phishing), UCI 1 -> 0 (legitimate)
    out = df[["url", "label"]]
    out = out[out["url"].notna() & (out["url"].str.strip() != "")]
    out = out[out["label"].isin([0, 1])]
    return out.reset_index(drop=True)


def main():
    df, csv_name = download_and_extract()
    print(f"Loaded {len(df)} rows from {csv_name}")
    out = normalize_uci_to_project(df)
    print(f"Normalized: {len(out)} rows. Label counts: {out['label'].value_counts().to_dict()}")
    out.to_csv(DATASET_PATH, index=False)
    print(f"Saved to {DATASET_PATH}")
    # Optional: remove zip to save space
    # ZIP_PATH.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
