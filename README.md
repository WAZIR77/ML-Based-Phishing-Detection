# Phishing URL Detection System

Machine learning–based phishing detection that classifies URLs as **Phishing** or **Legitimate**, with a risk score and explainable features. Built with Python, scikit-learn, and Flask.

---

## Features

- **URL features**: length, dots, subdomains, HTTPS, suspicious keywords, URL shorteners, entropy
- **Domain features**: WHOIS age, DNS records, abnormal patterns (optional; batch processing skips slow lookups)
- **Content features** (optional): HTML forms, iframes, redirects, urgency language
- **Models**: Logistic Regression (baseline) and Random Forest (primary), tuned for high recall
- **API**: Flask web UI and REST API with classification, risk score (0–100), and top contributing features

---

## Project Structure

```
├── config.py
├── run_training.py
├── requirements.txt
├── data/
│   ├── raw/                    # CSV dataset (url, label)
│   ├── processed/              # Extracted features
│   ├── download_sample_data.py
│   └── download_uci_phishing.py
├── feature_extraction/
│   ├── url_features.py
│   ├── domain_features.py
│   ├── content_features.py
│   └── extractor.py
├── model_training/
│   ├── pipeline.py
│   └── train.py
├── evaluation/
│   └── metrics.py
├── utils/
│   ├── safe_url.py
│   └── data_loader.py
├── deployment/
│   ├── predictor.py
│   └── app.py
└── models/                     # Saved model artifacts
```

---

## Setup

```bash
cd "Phishing Detection"
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

---

## Dataset

Use a CSV with columns **`url`** and **`label`** (1 = phishing, 0 = legitimate). Place it at `data/raw/phishing_dataset.csv`.

**Download UCI PhiUSIIL dataset:**

```bash
python data/download_uci_phishing.py
```

This fetches the dataset from the UCI repository and saves it in the correct format. Then run training.

---

## Training

From the project root (set `PYTHONPATH` so imports work):

**Windows (PowerShell):**
```powershell
$env:PYTHONPATH = (Get-Location).Path
python run_training.py
```

**Windows (CMD):**
```cmd
set PYTHONPATH=%CD%
python run_training.py
```

**Linux/macOS:**
```bash
export PYTHONPATH=.
python run_training.py
```

Training will load the dataset, extract features, train Logistic Regression and Random Forest with cross-validation, and save the best model to `models/`.

---

## Prediction API & Web UI

Start the Flask app:

```bash
python deployment/app.py
```

- **Web UI:** http://127.0.0.1:5000/ — enter a URL to get classification, risk score, and top features.
- **REST API:**
  - `GET /api/predict?url=https://example.com`
  - `POST /api/predict` with body `{"url": "https://example.com"}`

Response includes `classification`, `risk_score` (0–100), and `top_contributing_features`.

---

## Programmatic Use

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
from deployment.predictor import predict_dict

result = predict_dict("https://example.com")
# result["classification"], result["risk_score"], result["top_contributing_features"]
```

---

## Ethical Use

This tool is for **defensive and educational use only** (e.g. SOC workflows, internal security, learning). Do not use it to create or host phishing sites or to target systems without authorization. See `ETHICS_AND_USE.md` for details.

---

## License

Use at your own risk. Not a replacement for professional security products. Comply with your organization’s policies and applicable laws.
