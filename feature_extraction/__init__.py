"""
Modular feature extraction for phishing detection.
URL, domain, and content-based features; clearly separated from model logic.
"""
from .url_features import extract_url_features
from .domain_features import extract_domain_features
from .content_features import extract_content_features
from .extractor import extract_all_features, get_feature_names, extract_features_batch

__all__ = [
    "extract_url_features",
    "extract_domain_features",
    "extract_content_features",
    "extract_all_features",
    "get_feature_names",
    "extract_features_batch",
]
