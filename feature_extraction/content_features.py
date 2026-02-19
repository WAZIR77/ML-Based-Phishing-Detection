"""
Content-based features from HTML: forms, iframes, redirects, pop-ups, urgency language.
Fetches page with strict timeout and size limit; safe for malformed URLs (no fetch).
"""
import re
from typing import Dict, Any, Optional
from urllib.parse import urlparse

try:
    import requests
    from config import FETCH_TIMEOUT_SEC, FETCH_MAX_BYTES, USER_AGENT
except ImportError:
    requests = None
    FETCH_TIMEOUT_SEC = 5
    FETCH_MAX_BYTES = 100_000
    USER_AGENT = "PhishingDetectionBot/1.0 (Security Research)"

from utils.safe_url import is_safe_url


# Urgency / phishing language patterns
URGENCY_PATTERNS = [
    r"\b(urgent|immediately|asap|verify\s+now|confirm\s+now|act\s+now)\b",
    r"\b(suspend|suspended|restore\s+account|locked\s+account)\b",
    r"\b(warning|attention\s+required|action\s+required)\b",
    r"\b(click\s+here|verify\s+your\s+identity|confirm\s+your\s+identity)\b",
]


def _fetch_html_safe(url: str) -> Optional[str]:
    """Fetch HTML with timeout and size limit; only for safe URLs."""
    if not requests:
        return None
    ok, reason = is_safe_url(url)
    if not ok:
        return None
    try:
        r = requests.get(
            url,
            timeout=FETCH_TIMEOUT_SEC,
            headers={"User-Agent": USER_AGENT},
            stream=True,
        )
        r.raise_for_status()
        content = b""
        for chunk in r.iter_content(chunk_size=8192):
            content += chunk
            if len(content) >= FETCH_MAX_BYTES:
                break
        return content.decode("utf-8", errors="ignore")
    except Exception:
        return None


def extract_content_features(url: str, html: Optional[str] = None, fetch: bool = False) -> Dict[str, Any]:
    """
    Extract content-based features. If html is None and fetch=True, attempts safe fetch.
    Otherwise uses provided html or returns defaults (so pipeline works without content).
    """
    features: Dict[str, Any] = {
        "has_html_form": 0,
        "form_action_mismatch": 0,
        "num_forms": 0,
        "has_iframe": 0,
        "num_iframes": 0,
        "has_js_redirect": 0,
        "has_popup": 0,
        "urgency_language_score": 0,
        "has_password_input": 0,
    }

    if html is None and fetch:
        html = _fetch_html_safe(url)
    if not html:
        return features

    html_lower = html.lower()

    # Forms
    form_matches = re.findall(r"<form[^>]*>", html_lower, re.IGNORECASE)
    features["num_forms"] = min(len(form_matches), 10)
    features["has_html_form"] = 1 if form_matches else 0

    # Form action URL vs page URL (mismatch can indicate phishing)
    try:
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        page_domain = urlparse(url).netloc.lower()
        for m in re.finditer(r'<form[^>]*action\s*=\s*["\']([^"\']+)["\']', html, re.IGNORECASE):
            action = m.group(1).strip()
            if not action or action.startswith("#") or action.startswith("javascript:"):
                continue
            if action.startswith("//"):
                action_domain = urlparse("https:" + action).netloc.lower()
            elif action.startswith("/"):
                action_domain = page_domain
            else:
                action_domain = urlparse(action).netloc.lower()
            if action_domain and action_domain != page_domain:
                features["form_action_mismatch"] = 1
                break
    except Exception:
        pass

    # Password input
    if re.search(r'<input[^>]*type\s*=\s*["\']password["\']', html_lower):
        features["has_password_input"] = 1

    # Iframes
    iframe_matches = re.findall(r"<iframe", html_lower)
    features["num_iframes"] = min(len(iframe_matches), 10)
    features["has_iframe"] = 1 if iframe_matches else 0

    # JavaScript redirect
    if re.search(r"window\.location\s*=|location\.href\s*=|location\.replace\s*\(", html_lower):
        features["has_js_redirect"] = 1

    # Pop-up (window.open, alert, confirm)
    if re.search(r"window\.open\s*\(|alert\s*\(|confirm\s*\(", html_lower):
        features["has_popup"] = 1

    # Urgency / phishing language
    score = 0
    for pat in URGENCY_PATTERNS:
        if re.search(pat, html_lower):
            score += 1
    features["urgency_language_score"] = min(score, 5)

    return features
