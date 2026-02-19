"""
Unified feature extraction: URL + domain + content.
Produces a flat dict suitable for vectorization; optional content fetch.
"""
import pandas as pd
from typing import Dict, Any, List, Optional

from .url_features import extract_url_features
from .domain_features import extract_domain_features
from .content_features import extract_content_features


def extract_all_features(
    url: str,
    html: Optional[str] = None,
    fetch_content: bool = False,
    skip_external_lookups: bool = False,
) -> Dict[str, Any]:
    """
    Extract all URL, domain, and (optionally) content features.
    Returns a single flat dict. None values from domain (e.g. WHOIS) should be
    imputed in the training/prediction pipeline.
    Set skip_external_lookups=True to skip WHOIS/DNS (faster for batch).
    """
    url_f = extract_url_features(url)
    domain_f = extract_domain_features(url, skip_external_lookups=skip_external_lookups)
    content_f = extract_content_features(url, html=html, fetch=fetch_content)

    out = {**url_f, **domain_f, **content_f}
    return out


def get_feature_names() -> List[str]:
    """Ordered list of feature names used by the model (for imputation and vectorization)."""
    url_f = extract_url_features("https://example.com")
    domain_f = extract_domain_features("https://example.com")
    content_f = extract_content_features("https://example.com")
    return list(url_f.keys()) + list(domain_f.keys()) + list(content_f.keys())


def extract_features_batch(
    urls: List[str],
    labels: Optional[List[int]] = None,
    fetch_content: bool = False,
) -> pd.DataFrame:
    """
    Batch extraction for training. Optional labels; content fetch disabled by default
    (slow and may hit rate limits). Imputes None domain features with 0.
    """
    rows = []
    for i, url in enumerate(urls):
        try:
            feats = extract_all_features(url, fetch_content=fetch_content, skip_external_lookups=True)
            # Impute None for missing WHOIS/DNS
            for k, v in feats.items():
                if v is None:
                    feats[k] = 0
            if labels is not None and i < len(labels):
                feats["label"] = int(labels[i])
            rows.append(feats)
        except Exception:
            # Skip malformed URLs or use defaults
            feats = extract_all_features("https://example.com", skip_external_lookups=True)
            for k, v in feats.items():
                if v is None:
                    feats[k] = 0
            if labels is not None and i < len(labels):
                feats["label"] = int(labels[i])
            rows.append(feats)
    return pd.DataFrame(rows)
