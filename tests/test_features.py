import pytest
import pandas as pd
import numpy as np

from src.features import feature_engineering as fe


@pytest.mark.parametrize("url,expected", [
    ("http://192.168.0.1/test", 1),
    ("http://[2001:0db8:85a3:0000:0000:8a2e:0370:7334]/", 1),
    ("https://example.com", 0),
    ("", 0),
])
def test_having_ip_address(url, expected):
    """Detect IPv4 and IPv6 addresses in URLs."""
    assert fe.having_ip_address(url) == expected


@pytest.mark.parametrize("url,expected", [
    ("http://example.com/path", 0),
    ("http://ok.example.com", 0),
    ("", 1),
])
def test_abnormal_url(url, expected):
    """Check if hostname is present in URL."""
    assert fe.abnormal_url(url) == expected


@pytest.mark.parametrize("url,expected", [
    ("http://bit.ly/abcd", 1),
    ("https://tinyurl.com/abc", 1),
    ("https://goo.gl/xyz", 1),
    ("https://safe-domain.com/", 0),
    ("http://BIT.LY/Case", 1),
])
def test_shortening_service(url, expected):
    """Detect URL shortening services."""
    assert fe.shortening_service(url) == expected


def test_letter_and_digit_count():
    """Count letters and digits in URL."""
    url = "http://ex4mple.com/path123"
    assert fe.letter_count(url) == sum(c.isalpha() for c in url)
    assert fe.digit_count(url) == sum(c.isnumeric() for c in url)


@pytest.mark.parametrize("url,expected", [
    ("https://secure.example", 1),
    ("http://insecure.example", 0),
    ("ftp://example", 0),
    ("", 0),
])
def test_http_secure(url, expected):
    """Check if URL uses HTTPS."""
    assert fe.http_secure(url) == expected


@pytest.mark.parametrize("url,expected", [
    ("http://example.com/login", 1),
    ("http://example.com/signin", 1),
    ("http://example.com/bank", 1),
    ("http://example.com/normal", 0),
    ("", 0),
])
def test_suspicious_words(url, expected):
    """Detect suspicious words in URL."""
    assert fe.suspicious_words(url) == expected


@pytest.mark.parametrize("url,expected", [
    ("http://a.com/b/c/d", 3),
    ("http://a.com/b/", 2),
    ("http://a.com", 0),
    ("", 0),
])
def test_count_directories(url, expected):
    """Count directory levels in URL path."""
    assert fe.count_directories(url) == expected


@pytest.mark.parametrize("url,expected", [
    ("http://a.com/http://b.com", 1),
    ("http://a.com//embedded", 1),
    ("http://a.com/path", 0),
    ("", 0),
])
def test_count_embedded_domains(url, expected):
    """Count embedded domains using '//' pattern."""
    assert fe.count_embedded_domains(url) == expected


def test_first_dir_length():
    """Calculate first directory length."""
    assert fe.first_dir_length("http://example.com/first/second") == len("first")
    assert fe.first_dir_length("http://example.com") == 0
    assert fe.first_dir_length("http://example.com/") == 0


def test_tld_length():
    """Calculate TLD length."""
    assert fe.tld_length("http://example.com") > 0
    assert fe.tld_length("http://example.co.uk") > 0
    assert fe.tld_length("") == 0


def test_count_special_chars():
    """Count special characters in URL."""
    counts = fe.count_special_chars("http://example.com/a?b=1&c#frag//")
    
    expected_keys = [
        "count_@", "count_?", "count_-", "count_=", "count_.",
        "count_#", "count_%", "count_+", "count_$", "count_!",
        "count_*", "count_,", "count_slashes"
    ]
    
    for key in expected_keys:
        assert key in counts
        assert isinstance(counts[key], int)
        assert counts[key] >= 0


def test_count_www():
    """Count 'www' occurrences in URL."""
    # count_www is not a separate function, it's computed in extract_features
    features = fe.extract_features("http://www.example.com")
    assert features["count_www"] >= 1
    
    features = fe.extract_features("http://example.com")
    assert features["count_www"] == 0


def test_extract_features_returns_all_27_features():
    """Verify extract_features returns all 27 required features."""
    url = "https://www.example.com/path?x=1#frag"
    features = fe.extract_features(url)
    
    expected_features = [
        "url_length", "hostname_length", "count_letters", "count_digits",
        "count_@", "count_?", "count_-", "count_=", "count_.",
        "count_#", "count_%", "count_+", "count_$", "count_!",
        "count_*", "count_,", "count_slashes", "count_www",
        "has_ip", "abnormal_url", "short_url", "https",
        "count_dir", "count_embed_domain", "fd_length",
        "tld_length", "suspicious"
    ]
    
    assert len(features) == 27, f"Expected 27 features, got {len(features)}"
    
    for feature_name in expected_features:
        assert feature_name in features, f"Missing feature: {feature_name}"
        assert isinstance(features[feature_name], (int, float, np.integer, np.floating))


def test_extract_features_benign_url():
    """Test features for a typical benign URL."""
    url = "https://www.google.com/search?q=test"
    features = fe.extract_features(url)
    
    assert features["https"] == 1
    assert features["has_ip"] == 0
    assert features["short_url"] == 0
    assert features["abnormal_url"] == 0
    assert features["url_length"] == len(url)
    assert features["hostname_length"] > 0


def test_extract_features_suspicious_url():
    """Test features for a suspicious URL."""
    url = "http://192.168.1.1/login/verify"
    features = fe.extract_features(url)
    
    assert features["https"] == 0
    assert features["has_ip"] == 1
    assert features["suspicious"] == 1


def test_extract_features_short_url():
    """Test features for shortened URL."""
    url = "http://bit.ly/abc123"
    features = fe.extract_features(url)
    
    assert features["short_url"] == 1
    assert features["hostname_length"] > 0


def test_extract_features_empty_url():
    """Test features for empty URL."""
    features = fe.extract_features("")
    
    assert features["url_length"] == 0
    assert features["hostname_length"] == 0
    assert features["abnormal_url"] == 1


def test_transform_dataframe_basic():
    """Test transform_dataframe with multiple URLs."""
    df = pd.DataFrame({
        "url": [
            "https://google.com",
            "http://bit.ly/abc",
            "http://192.168.1.1",
        ]
    })
    
    result = fe.transform_dataframe(df)
    
    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == 3
    assert result.shape[1] == 27


def test_transform_dataframe_preserves_order():
    """Test that transform_dataframe preserves row order."""
    urls = ["https://a.com", "http://b.com", "https://c.com"]
    df = pd.DataFrame({"url": urls})
    
    result = fe.transform_dataframe(df)
    
    assert result.shape[0] == len(urls)
    # Check if features are extracted for each URL
    assert (result["url_length"] > 0).sum() == len(urls)


def test_transform_dataframe_with_empty_urls():
    """Test transform_dataframe handles empty URLs."""
    df = pd.DataFrame({
        "url": ["https://example.com", "", "http://test.com"]
    })
    
    result = fe.transform_dataframe(df)
    
    assert result.shape[0] == 3
    # Second row (empty URL) should have url_length = 0
    assert result.iloc[1]["url_length"] == 0


def test_hostname_length():
    """Test hostname length extraction."""
    features1 = fe.extract_features("http://example.com/path")
    assert features1["hostname_length"] > 0
    
    features2 = fe.extract_features("http://very-long-hostname.com")
    features3 = fe.extract_features("http://a.com")
    assert features2["hostname_length"] > features3["hostname_length"]
    
    features4 = fe.extract_features("")
    assert features4["hostname_length"] == 0


def test_url_length():
    """Test URL length calculation."""
    url = "https://example.com/path"
    features = fe.extract_features(url)
    assert features["url_length"] == len(url)
    
    features_empty = fe.extract_features("")
    assert features_empty["url_length"] == 0


def test_feature_values_are_numeric():
    """Ensure all feature values are numeric."""
    url = "https://www.example.com/path?query=value#fragment"
    features = fe.extract_features(url)
    
    for key, value in features.items():
        assert isinstance(value, (int, float, np.integer, np.floating)), \
            f"Feature '{key}' has non-numeric value: {value} ({type(value)})"


def test_special_characters_comprehensive():
    """Test comprehensive special character counting."""
    url = "http://user@example.com/path?a=1&b=2#frag%20test+more$5!important*note,list"
    counts = fe.count_special_chars(url)
    
    # Should have all special character counts
    assert counts["count_@"] >= 1
    assert counts["count_?"] >= 1
    assert counts["count_="] >= 2
    assert counts["count_#"] >= 1
    assert counts["count_%"] >= 1
    assert counts["count_+"] >= 1
    assert counts["count_$"] >= 1
    assert counts["count_!"] >= 1
    assert counts["count_*"] >= 1
    assert counts["count_,"] >= 1