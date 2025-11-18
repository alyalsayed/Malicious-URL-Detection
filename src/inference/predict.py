"""
Pure inference module.

Loads model + label encoder once.
Exposes a clean `predict_url()` function for use by FastAPI or batch scripts.
"""

import pickle
from typing import Dict, Any, List

from src.config import MODEL_PATH, LABEL_ENCODER_PATH, LABEL_MAP
from src.features.feature_engineering import extract_features


# --------------------------------------------------------
# Load model & encoder at import time (only once)
# --------------------------------------------------------
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


model = load_pickle(MODEL_PATH)
label_encoder = load_pickle(LABEL_ENCODER_PATH)

label_map = {i: label for i, label in enumerate(label_encoder.classes_)} \
            if hasattr(label_encoder, "classes_") else LABEL_MAP


# --------------------------------------------------------
# Inference Function (Pure Logic)
# --------------------------------------------------------
def predict_url(url: str) -> Dict[str, Any]:
    """Runs feature extraction + model prediction."""

    # 1. Extract features
    features_dict = extract_features(url)
    X: List[List[float]] = [list(features_dict.values())]

    # 2. Predict class
    class_id: int = int(model.predict(X)[0])
    predicted_label: str = label_map.get(class_id, "unknown")

    # 3. Predict probabilities (if available)
    try:
        proba = model.predict_proba(X)[0]
        proba_dict = {label_map[i]: float(prob) for i, prob in enumerate(proba)}
    except Exception:
        proba_dict = "Model does not support probability output"

    return {
        "input_url": url,
        "predicted_class": predicted_label,
        "class_id": class_id,
        "probabilities": proba_dict,
    }
