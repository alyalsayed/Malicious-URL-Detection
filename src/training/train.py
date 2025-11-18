"""
Training pipeline for Malicious URL Detection.

- Loads data from data/raw/
- Applies feature engineering
- Loads or trains LabelEncoder
- Splits into train/test
- Trains model OR loads existing one
- Evaluates the model
- Saves model + metadata JSON
"""

import json
import pickle
from datetime import datetime
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.preprocessing import LabelEncoder

from src.config import (
    RAW_DATA_DIR,
    MODEL_PATH,
    MODEL_METADATA_PATH,
    MODEL_PARAMS,
    FEATURE_COLUMNS,
    RANDOM_STATE,
    TEST_SIZE,
    LABEL_ENCODER_PATH,
)
from src.features.feature_engineering import transform_dataframe

# ----------------------------------------------------
# Logging Setup
# ----------------------------------------------------
logging.basicConfig(level="INFO")
logger = logging.getLogger("TRAINING")


# ----------------------------------------------------
# Load Raw Dataset
# ----------------------------------------------------
def load_data():
    csv_path = RAW_DATA_DIR / "malicious_phish.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    logger.info(f"Loading dataset → {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Dataset shape: {df.shape}")

    return df


# ----------------------------------------------------
# Label Encoder Handling
# ----------------------------------------------------
def load_or_create_label_encoder(df_type_series):
    """
    Load label_encoder.pkl if exists.
    Otherwise train a new LabelEncoder and save it.
    """

    if LABEL_ENCODER_PATH.exists():
        logger.info("📌 Loading existing label_encoder.pkl ...")
        with open(LABEL_ENCODER_PATH, "rb") as f:
            le = pickle.load(f)

        label_map = {i: cls for i, cls in enumerate(le.classes_)}
        logger.info(f"Loaded label_map: {label_map}")
        return le, label_map

    # Create a new encoder
    logger.info("🔧 label_encoder.pkl not found — training LabelEncoder...")
    le = LabelEncoder()
    le.fit(df_type_series)

    # Save encoder
    with open(LABEL_ENCODER_PATH, "wb") as f:
        pickle.dump(le, f)

    label_map = {i: cls for i, cls in enumerate(le.classes_)}
    logger.info(f"Saved new label_encoder.pkl with label_map: {label_map}")

    return le, label_map


# ----------------------------------------------------
# Apply Feature Engineering
# ----------------------------------------------------
def preprocess(df, label_encoder):
    """
    Converts URL → features and type → numeric labels
    using a consistent saved LabelEncoder.
    """
    logger.info("Applying feature engineering...")

    df_features = transform_dataframe(df)
    df_final = pd.concat([df_features, df["type"]], axis=1)

    df_final["label"] = label_encoder.transform(df_final["type"])

    return df_final.drop(columns=["type"])


# ----------------------------------------------------
# Evaluate Model
# ----------------------------------------------------
def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, average="macro"),
        "recall": recall_score(y_test, preds, average="macro"),
        "f1_score": f1_score(y_test, preds, average="macro"),
    }


# ----------------------------------------------------
# Save Model + Metadata
# ----------------------------------------------------
def save_metadata(metrics, label_map):
    metadata = {
        "version": "0.1.0",
        "status": "trained",
        "message": "Model trained successfully, use best_model.pkl for predictions.",
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "model_type": "RandomForestClassifier",
        "hyperparameters": MODEL_PARAMS,
        "feature_names": FEATURE_COLUMNS,
        "metrics": metrics,
        "label_map": label_map,
    }

    with open(MODEL_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=4)

    logger.info(f"📄 Model metadata saved → {MODEL_METADATA_PATH}")


# ----------------------------------------------------
# Main Pipeline
# ----------------------------------------------------
def train():
    logger.info("🚀 Starting training pipeline...")

    df = load_data()

    # Load or create LabelEncoder
    label_encoder, label_map = load_or_create_label_encoder(df["type"])

    # ------------------------------------------------
    # CASE 1: Model already exists → skip training
    # ------------------------------------------------
    if MODEL_PATH.exists():
        logger.info("⚠ best_model.pkl found — skipping training.")
        logger.info("Loading existing model...")

        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)

        df_prepared = preprocess(df, label_encoder)

        X = df_prepared[FEATURE_COLUMNS]
        y = df_prepared["label"]

        _, X_test, _, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )

        metrics = evaluate(model, X_test, y_test)
        logger.info(f"📊 Loaded Model Metrics: {metrics}")

        save_metadata(metrics, label_map)
        return

    # ------------------------------------------------
    # CASE 2: Model does NOT exist → Train model
    # ------------------------------------------------
    logger.info("🛠 Model not found — starting training...")

    df_prepared = preprocess(df, label_encoder)

    X = df_prepared[FEATURE_COLUMNS]
    y = df_prepared["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    logger.info("Training RandomForestClassifier...")
    model = RandomForestClassifier(**MODEL_PARAMS)
    model.fit(X_train, y_train)

    # Save model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"💾 Model saved → {MODEL_PATH}")

    metrics = evaluate(model, X_test, y_test)
    logger.info(f"📊 Training Metrics: {metrics}")

    save_metadata(metrics, label_map)


if __name__ == "__main__":
    train()
    logger.info("✅ Training pipeline completed!")
