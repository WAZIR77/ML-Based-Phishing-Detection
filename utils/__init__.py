"""Utilities for safe URL handling and data loading."""
from .safe_url import normalize_url, is_safe_url, SafeURLException
from .data_loader import load_phishing_dataset, prepare_train_test

__all__ = [
    "normalize_url",
    "is_safe_url",
    "SafeURLException",
    "load_phishing_dataset",
    "prepare_train_test",
]
