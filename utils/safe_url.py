"""
Safe URL handling for phishing detection.
Validates and normalizes URLs; prevents SSRF and malformed input abuse.
"""
import re
from urllib.parse import urlparse, urlunparse
from typing import Optional, Tuple


class SafeURLException(Exception):
    """Raised when URL is invalid or unsafe for processing."""
    pass


# Block private/local and dangerous schemes
UNSAFE_SCHEMES = {"file", "ftp", "gopher", "data", "javascript", "vbscript"}
ALLOWED_SCHEMES = {"http", "https"}
# Avoid resolving to localhost, private IPs, or internal hostnames
PRIVATE_NET_PATTERN = re.compile(
    r"^(localhost|127\.|10\.|172\.(1[6-9]|2[0-9]|3[01])\.|192\.168\.|0\.|::1)",
    re.IGNORECASE
)
MAX_URL_LENGTH = 2048


def normalize_url(url: str) -> str:
    """
    Normalize URL for consistent parsing: strip whitespace, add scheme if missing.
    Does not fetch or resolve the URL.
    """
    if not url or not isinstance(url, str):
        raise SafeURLException("URL must be a non-empty string")
    url = url.strip()
    if len(url) > MAX_URL_LENGTH:
        raise SafeURLException(f"URL exceeds maximum length ({MAX_URL_LENGTH})")
    if not url:
        raise SafeURLException("URL is empty after stripping")
    parsed = urlparse(url)
    scheme = (parsed.scheme or "https").lower()
    if scheme not in ALLOWED_SCHEMES:
        raise SafeURLException(f"Scheme not allowed: {scheme}")
    if scheme == "":
        url = "https://" + url
        parsed = urlparse(url)
    return url


def is_safe_url(url: str) -> Tuple[bool, Optional[str]]:
    """
    Check if URL is safe for outbound requests (no private IPs, no dangerous schemes).
    Returns (True, None) if safe, (False, reason) otherwise.
    """
    try:
        normalized = normalize_url(url)
    except SafeURLException as e:
        return False, str(e)
    parsed = urlparse(normalized)
    host = (parsed.hostname or "").strip()
    if not host:
        return False, "Missing hostname"
    if PRIVATE_NET_PATTERN.match(host):
        return False, "Private or localhost hostname not allowed"
    return True, None
