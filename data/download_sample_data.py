"""
Generate a small sample dataset for testing the pipeline when no external dataset is present.
For production training, use a real dataset (see README): UCI, Kaggle, or PhishTank.
"""
import csv
import random
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import RAW_DATA_DIR, DATASET_PATH

SAMPLE_URLS_LEGIT = [
    "https://www.google.com/search?q=python",
    "https://github.com/scikit-learn/scikit-learn",
    "https://www.wikipedia.org/wiki/Machine_learning",
    "https://stackoverflow.com/questions",
    "https://www.bbc.com/news",
    "https://www.nytimes.com",
    "https://www.microsoft.com/en-us",
    "https://www.apple.com",
    "https://www.amazon.com/gp/help",
    "https://www.python.org/doc/",
]

SAMPLE_URLS_PHISH = [
    "https://secure-login-verify.account-update.com/confirm",
    "http://192.168.1.1.login.verify.secure.com",
    "https://paypal-verify-urgent.secure-account.com",
    "https://amazon-account-suspend.click-here-now.com",
    "https://bit.ly/2xYz123",
    "http://login.microsoft-verify.secure-site.com",
]


def generate_sample_dataset(num_rows: int = 200, output_path: Path = None) -> Path:
    output_path = output_path or DATASET_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for _ in range(num_rows // 2):
        url = random.choice(SAMPLE_URLS_LEGIT)
        rows.append((url, 0))
    for _ in range(num_rows - len(rows)):
        url = random.choice(SAMPLE_URLS_PHISH)
        rows.append((url, 1))
    random.shuffle(rows)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["url", "label"])
        w.writerows(rows)
    return output_path


if __name__ == "__main__":
    p = generate_sample_dataset(200)
    print(f"Sample dataset written to {p}")
