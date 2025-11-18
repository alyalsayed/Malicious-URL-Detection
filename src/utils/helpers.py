"""
Helper functions and utilities for the malicious URL detector.

Includes logging setup, data validation, and other common utilities.
"""

import logging
from pathlib import Path

import pandas as pd

from src.config import LOG_FORMAT


def setup_logging(log_level: str = "INFO") -> None:
    """
    Setup logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=LOG_FORMAT,
        handlers=[
            logging.StreamHandler(),
            # Uncomment for file logging
            # logging.FileHandler("logs/app.log")
        ],
    )


def validate_data(df: pd.DataFrame, required_columns: list = None) -> bool:
    """
    Validate input data.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names

    Returns:
        True if data is valid, False otherwise
    """
    if required_columns is None:
        required_columns = ["url", "label"]

    # Check if DataFrame is empty
    if df.empty:
        logging.error("DataFrame is empty")
        return False

    # Check required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logging.error(f"Missing required columns: {missing_columns}")
        return False

    # Check for null values
    if df[required_columns].isnull().any().any():
        logging.error("Found null values in required columns")
        return False

    logging.info(f"Data validation passed: {len(df)} samples")
    return True


def ensure_directory(path: Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Path to the directory

    Returns:
        The path (now guaranteed to exist)
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_valid_url(url: str) -> bool:
    """
    Basic URL validation.

    Args:
        url: URL string to validate

    Returns:
        True if URL appears valid, False otherwise
    """
    return isinstance(url, str) and len(url) > 0 and ("." in url)


def truncate_string(s: str, max_length: int = 100) -> str:
    """
    Truncate string to maximum length.

    Args:
        s: String to truncate
        max_length: Maximum length

    Returns:
        Truncated string
    """
    if len(s) > max_length:
        return s[: max_length - 3] + "..."
    return s
