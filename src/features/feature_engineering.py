"""
Feature engineering utilities for URL classification.

This module provides:
- extract_features(url): Extracts all features for a single URL (used in inference)
- transform_dataframe(df): Applies feature extraction to a full dataframe (used in training)

All feature functions are pure and reusable.
"""

from urllib.parse import urlparse
import re
from tld import get_tld
import numpy as np
import pandas as pd

# -----------------------------
# IP Address Checker
# -----------------------------
def having_ip_address(url: str) -> int:
    ipv4_pattern = (
        r"(([01]?\d\d?|2[0-4]\d|25[0-5])\.){3}"
        r"([01]?\d\d?|2[0-4]\d|25[0-5])"
    )
    ipv6_pattern = r"([a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}"

    if re.search(ipv4_pattern, url) or re.search(ipv6_pattern, url):
        return 1
    return 0

# -----------------------------
# Abnormal URL
# -----------------------------
def abnormal_url(url: str) -> int:
    hostname = str(urlparse(url).hostname)
    return 0 if hostname and hostname in url else 1

def shortening_service(url: str) -> int:
    """
    Detect if url uses a known URL-shortener domain.
    Returns 1 if yes, 0 otherwise.
    """
    # exhaustive list (case-insensitive)
    pattern = (
        r'bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
        r'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
        r'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
        r'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|lnkd\.in|'
        r'db\.tt|qr\.ae|adf\.ly|bitly\.com|cur\.lv|tinyurl\.com|ity\.im|'
        r'q\.gs|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
        r'prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|'
        r'tr\.im|link\.zip\.net'
    )
    return 1 if re.search(pattern, url, re.IGNORECASE) else 0

# -----------------------------
# Count Letters
# -----------------------------
def letter_count(url: str) -> int:
    return sum(c.isalpha() for c in url)

# -----------------------------
# Count Digits
# -----------------------------
def digit_count(url: str) -> int:
    return sum(c.isnumeric() for c in url)

# -----------------------------
# HTTPS Detection
# -----------------------------
def http_secure(url: str) -> int:
    return 1 if urlparse(url).scheme == 'https' else 0

# -----------------------------
# Suspicious Keyword Detection
# -----------------------------
def suspicious_words(url: str) -> int:
    pattern = (
        r"PayPal|login|signin|bank|account|update|free|bonus|service|"
        r"ebayisapi|webscr|lucky|secure|verification|confirm"
    )
    return 1 if re.search(pattern, url, re.IGNORECASE) else 0

# -----------------------------
# Number of Directories
# -----------------------------
def count_directories(url: str) -> int:
    return urlparse(url).path.count('/')

# -----------------------------
# Number of Embedded Domains
# -----------------------------
def count_embedded_domains(url: str) -> int:
    urldir = urlparse(url).path
    return urldir.count('//')

# -----------------------------
# First Directory Length
# -----------------------------
def first_dir_length(url: str) -> int:
    try:
        return len(urlparse(url).path.split('/')[1])
    except:
        return 0
    
# -----------------------------
# Top-Level Domain Length
# -----------------------------
def tld_length(url: str) -> int:
    try:
        tld = get_tld(url, fail_silently=True)
        return len(tld) if tld else 0
    except:
        return 0


# Predefined special characters for lexical analysis
SPECIAL_CHARS = ['@', '?', '-', '=', '.', '#', '%', '+', '$', '!', '*', ',', '//']

def count_special_chars(url: str) -> dict:
    """Return a dictionary of special-char counts."""
    counts = {}
    for ch in SPECIAL_CHARS:
        key = "count_slashes" if ch == "//" else f"count_{ch}"
        counts[key] = url.count(ch)
    return counts

# -----------------------------------------
#  SINGLE URL Feature Extraction
# -----------------------------------------

def extract_features(url: str) -> dict:
    """
    Extract all engineered features for a single URL.
    Used in API inference or test cases.
    """
    parsed = urlparse(url)

    features = {
        "url_length": len(url),
        "hostname_length": len(parsed.netloc),
        "count_letters": letter_count(url),
        "count_digits": digit_count(url),
        "count_www": url.count("www"),

        # domain / security-related
        "has_ip": having_ip_address(url),
        "abnormal_url": abnormal_url(url),
        "short_url": shortening_service(url),
        "https": http_secure(url),

        # structure
        "count_dir": count_directories(url),
        "count_embed_domain": count_embedded_domains(url),
        "fd_length": first_dir_length(url),
        "tld_length": tld_length(url),

        # suspicious content
        "suspicious": suspicious_words(url),
    }

    # add special character counts
    features.update(count_special_chars(url))

    return features


# -----------------------------------------
#  DATAFRAME-LEVEL Transformation
# -----------------------------------------

def transform_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature extraction to an entire dataframe.
    Expects a column named 'url'.
    """
    feature_rows = df["url"].apply(extract_features)
    feature_df = pd.DataFrame(list(feature_rows))

    return feature_df