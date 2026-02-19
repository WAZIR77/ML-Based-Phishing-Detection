"""
End-to-end training entry point.
Usage: from project root, run:
  set PYTHONPATH=. && python run_training.py
  or: python -m run_training
"""
import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DATASET_PATH, PROCESSED_FEATURES_PATH
from utils.data_loader import load_phishing_dataset, prepare_train_test
from feature_extraction import extract_features_batch
from model_training.pipeline import build_feature_matrix, get_feature_names_ordered
from model_training.train import train_models, load_trained_pipeline
from evaluation.metrics import evaluate_model, print_classification_report, get_feature_importance
import numpy as np


def main():
    print("Loading dataset...")
    if not DATASET_PATH.exists():
        print(f"Dataset not found at {DATASET_PATH}. Generating sample data...")
        from data.download_sample_data import generate_sample_dataset
        generate_sample_dataset(200, DATASET_PATH)
    df = load_phishing_dataset()
    print(f"Loaded {len(df)} rows. Labels: {df['label'].value_counts().to_dict()}")

    print("Extracting features (URL + domain; content fetch disabled for speed)...")
    feat_df = extract_features_batch(df["url"].tolist(), labels=df["label"].tolist(), fetch_content=False)
    feat_df.to_csv(PROCESSED_FEATURES_PATH, index=False)
    print(f"Features saved to {PROCESSED_FEATURES_PATH}")

    X, y, feature_names = build_feature_matrix(feat_df)
    print(f"Feature matrix: {X.shape}")

    train_df, test_df = prepare_train_test(feat_df)
    X_train, y_train, _ = build_feature_matrix(train_df)
    X_test, y_test, _ = build_feature_matrix(test_df)

    print("Training models (Logistic Regression + Random Forest with CV)...")
    results = train_models(X_train, y_train, feature_names, scale=True)
    for name, metrics in results.items():
        if isinstance(metrics, dict) and "cv_recall_mean" in metrics:
            print(f"  {name}: CV recall = {metrics['cv_recall_mean']:.4f}, F1 = {metrics['cv_f1_mean']:.4f}")

    print("Evaluating on test set...")
    model, _, scaler = load_trained_pipeline()
    if scaler is not None and hasattr(model, "feature_importances_"):
        # RF was saved; no scaling for RF at inference
        X_test_scaled = X_test  # RF not scaled
    else:
        X_test_scaled = scaler.transform(X_test) if scaler is not None else X_test
    y_pred = model.predict(X_test_scaled)
    print_classification_report(y_test, y_pred)
    ev = evaluate_model(y_test, y_pred)
    print(f"Test accuracy: {ev['accuracy']:.4f}, recall: {ev['recall']:.4f}")

    importance = get_feature_importance(model, feature_names, top_k=10)
    if importance:
        print("Top 10 feature importances (explainability):")
        for name, imp in importance:
            print(f"  {name}: {imp:.4f}")
    print("Training complete. Model saved to models/.")


if __name__ == "__main__":
    main()
