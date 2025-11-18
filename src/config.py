"""
Configuration and constants for the Malicious URL Detection project.
"""

import os
from pathlib import Path

# -----------------------------
# Project Paths
# -----------------------------
PROJECT_ROOT = Path(__file__).parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "best_model.pkl"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
# Ensure directories exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# -----------------------------
# Feature Columns
# (Must match feature_engineering.transform())
# -----------------------------
FEATURE_COLUMNS = [
    "url_length",
    "hostname_length",
    "count_letters",
    "count_digits",
    "count_@",
    "count_?",
    "count_-",
    "count_=",
    "count_.",
    "count_#",
    "count_%",
    "count_+",
    "count_$",
    "count_!",
    "count_*",
    "count_,",
    "count_slashes",
    "count_www",
    "has_ip",
    "abnormal_url",
    "short_url",
    "https",
    "count_dir",
    "count_embed_domain",
    "fd_length",
    "tld_length",
    "suspicious",
]

# -----------------------------
# Label mapping
# -----------------------------
LABEL_MAP = {
    0: "Benign",
    1: "Defacement",
    2: "Malware",
    3: "Phishing"
}


# -----------------------------
# API Configuration
# -----------------------------
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_WORKERS = int(os.getenv("API_WORKERS", "4"))
API_RELOAD = os.getenv("API_RELOAD", "True") == "True"

# -----------------------------
# Logging Settings
# -----------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# -----------------------------
# Model
# -----------------------------
MODEL_TYPE = "RandomForest"

MODEL_PARAMS = {
    "max_depth": 20,
    "min_samples_leaf": 1,
    "min_samples_split": 2,
    "n_estimators": 160
}

# -----------------------------
# Training Settings
# -----------------------------
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1
RANDOM_STATE = 42
