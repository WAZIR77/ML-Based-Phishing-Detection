"""
Domain-based security features: WHOIS (age, registration length), DNS, abnormal patterns.
Optional dependencies: python-whois, dnspython. Graceful fallback if unavailable.
"""
import re
import socket
from urllib.parse import urlparse
from typing import Dict, Any, Optional

# Optional WHOIS
try:
    import whois
    HAS_WHOIS = True
except ImportError:
    HAS_WHOIS = False

# Optional DNS
try:
    import dns.resolver
    HAS_DNS = True
except ImportError:
    HAS_DNS = False


def _get_domain_from_url(url: str) -> Optional[str]:
    try:
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        return urlparse(url).hostname
    except Exception:
        return None


def _domain_age_days(domain: str, skip_lookups: bool = False) -> Optional[float]:
    """Return domain age in days if WHOIS available; else None."""
    if skip_lookups or not HAS_WHOIS or not domain:
        return None
    try:
        w = whois.whois(domain)
        if not w.creation_date:
            return None
        created = w.creation_date
        if isinstance(created, list):
            created = min(created)
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        delta = now - created
        return max(0, delta.total_seconds() / 86400)
    except Exception:
        return None


def _registration_length_years(domain: str, skip_lookups: bool = False) -> Optional[float]:
    """Approximate registration length in years from expiry - creation if available."""
    if skip_lookups or not HAS_WHOIS or not domain:
        return None
    try:
        w = whois.whois(domain)
        if not w.creation_date or not w.expiration_date:
            return None
        created = w.creation_date
        expiry = w.expiration_date
        if isinstance(created, list):
            created = min(created)
        if isinstance(expiry, list):
            expiry = max(expiry)
        delta = expiry - created
        return max(0, delta.days / 365.25)
    except Exception:
        return None


def _dns_record_exists(domain: str, skip_lookups: bool = False) -> Optional[int]:
    """1 if A/AAAA record exists, 0 if not, None if DNS check unavailable."""
    if skip_lookups or not HAS_DNS or not domain:
        return None
    try:
        dns.resolver.resolve(domain, "A")
        return 1
    except Exception:
        try:
            dns.resolver.resolve(domain, "AAAA")
            return 1
        except Exception:
            return 0


def _abnormal_domain_pattern(domain: str) -> int:
    """
    Heuristic: very long domain, many subdomains, digits in domain, hyphen-heavy.
    """
    if not domain:
        return 0
    score = 0
    if len(domain) > 40:
        score += 1
    if domain.count(".") > 2:
        score += 1
    if sum(1 for c in domain if c.isdigit()) > 3:
        score += 1
    if domain.count("-") >= 2:
        score += 1
    return min(score, 1)  # binary: 0 or 1


def extract_domain_features(url: str, skip_external_lookups: bool = False) -> Dict[str, Any]:
    """
    Extract domain-based features. WHOIS/DNS features may be None if libs missing
    or lookup fails; downstream pipeline should impute (e.g. median or 0).
    Set skip_external_lookups=True to avoid WHOIS/DNS (faster batch processing).
    """
    domain = _get_domain_from_url(url)
    features: Dict[str, Any] = {}

    features["domain_age_days"] = _domain_age_days(domain, skip_lookups=skip_external_lookups) if domain else None
    features["registration_length_years"] = _registration_length_years(domain, skip_lookups=skip_external_lookups) if domain else None
    features["dns_record_exists"] = _dns_record_exists(domain, skip_lookups=skip_external_lookups) if domain else None
    features["abnormal_domain_pattern"] = _abnormal_domain_pattern(domain or "")

    # Binary: domain age known and very new (< 30 days) often associated with phishing
    age = features.get("domain_age_days")
    features["domain_very_new"] = 1 if age is not None and age < 30 else (0 if age is not None else None)

    return features
