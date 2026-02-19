"""
URL-based security features for phishing detection.
Pure functions; no model logic.
"""
import re
import math
from urllib.parse import urlparse
from typing import Dict, Any

try:
    from config import SUSPICIOUS_KEYWORDS, URL_SHORTENER_DOMAINS
except ImportError:
    SUSPICIOUS_KEYWORDS = [
        "login", "signin", "verify", "update", "secure", "account",
        "banking", "paypal", "amazon", "apple", "microsoft", "confirm",
        "suspend", "restore", "password", "credential", "urgent", "click"
    ]
    URL_SHORTENER_DOMAINS = [
        "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd",
        "buff.ly", "adf.ly", "bit.do", "lnkd.in", "db.tt", "qr.ae"
    ]


def _entropy(s: str) -> float:
    """Shannon entropy of string (randomness indicator)."""
    if not s:
        return 0.0
    from collections import Counter
    cnt = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in cnt.values() if c > 0)


def _is_ip_host(host: str) -> bool:
    """Check if host is an IP address (v4 or v6 pattern)."""
    if not host:
        return False
    # IPv4
    ipv4 = re.match(r"^(\d{1,3}\.){3}\d{1,3}$", host)
    if ipv4:
        return True
    # IPv6 simplified
    if ":" in host and "[" not in host:
        return True
    if host.startswith("[") and "]" in host:
        return True
    return False


def extract_url_features(url: str) -> Dict[str, Any]:
    """
    Extract comprehensive URL-based security features.
    Handles malformed URLs by returning safe defaults (e.g. 0, False).
    """
    features: Dict[str, Any] = {}
    url = (url or "").strip()
    if not url:
        return _url_defaults()

    try:
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        path = parsed.path or ""
        query = parsed.query or ""
        full_path = path + ("?" + query if query else "")
    except Exception:
        return _url_defaults()

    # Length
    features["url_length"] = min(len(url), 500)
    features["path_length"] = min(len(parsed.path or ""), 300)
    features["query_length"] = min(len(query), 200)

    # Dots and structure
    features["num_dots_url"] = url.count(".")
    features["num_dots_domain"] = host.count(".")
    features["num_subdomains"] = max(0, host.count(".") - 1) if host else 0

    # Special characters
    features["has_at_symbol"] = 1 if "@" in url else 0
    features["num_hyphens"] = url.count("-")
    features["num_underscores"] = url.count("_")
    features["num_special_chars"] = sum(1 for c in url if c in "@-_?=&%#")

    # IP in URL
    features["has_ip_in_url"] = 1 if _is_ip_host(host) else 0

    # HTTPS
    features["uses_https"] = 1 if parsed.scheme == "https" else 0

    # Suspicious keywords (count in path + query, case-insensitive)
    lower_path = full_path.lower()
    features["suspicious_keyword_count"] = sum(1 for k in SUSPICIOUS_KEYWORDS if k in lower_path)
    features["has_suspicious_keyword"] = 1 if features["suspicious_keyword_count"] > 0 else 0

    # URL shortener
    features["is_url_shortener"] = 1 if any(
        host.endswith(d) or host == d for d in URL_SHORTENER_DOMAINS
    ) else 0

    # Entropy (path + query as proxy for randomness)
    features["url_entropy"] = round(_entropy(full_path or url), 4)
    features["domain_entropy"] = round(_entropy(host), 4)

    # Ratio of digits in URL
    digit_count = sum(1 for c in url if c.isdigit())
    features["digit_ratio"] = round(digit_count / max(len(url), 1), 4)

    return features


def _url_defaults() -> Dict[str, Any]:
    """Return default feature dict for invalid/missing URL."""
    return {
        "url_length": 0,
        "path_length": 0,
        "query_length": 0,
        "num_dots_url": 0,
        "num_dots_domain": 0,
        "num_subdomains": 0,
        "has_at_symbol": 0,
        "num_hyphens": 0,
        "num_underscores": 0,
        "num_special_chars": 0,
        "has_ip_in_url": 0,
        "uses_https": 0,
        "suspicious_keyword_count": 0,
        "has_suspicious_keyword": 0,
        "is_url_shortener": 0,
        "url_entropy": 0.0,
        "domain_entropy": 0.0,
        "digit_ratio": 0.0,
    }
