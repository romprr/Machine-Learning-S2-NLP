import os
from pathlib import Path

# Base paths
MODEL3_DIR = Path(__file__).parent.resolve()
BASE_DIR = MODEL3_DIR.parent

# Input data paths (referencing existing data)
RAW_DATA_PATH = BASE_DIR / "handcrafted_features" / "stack-overflow-data.csv"
TRAIN_SPLIT_PATH = BASE_DIR / "model_benchmarking" / "outputs" / "pipeline" / "splits" / "train_data.csv"
TEST_SPLIT_PATH = BASE_DIR / "model_benchmarking" / "outputs" / "pipeline" / "splits" / "test_data.csv"

# Output paths for model3
OUTPUT_DIR = MODEL3_DIR / "outputs"
HANDCRAFTED_FEATURES_PATH = OUTPUT_DIR / "handcrafted_features.csv"
VECTORIZER_PATH = OUTPUT_DIR / "vectorizer.joblib"
SCALER_PATH = OUTPUT_DIR / "scaler.joblib"
LABEL_ENCODER_PATH = OUTPUT_DIR / "label_encoder.joblib"
MODEL_PATH = OUTPUT_DIR / "logistic_regression_model.joblib"
TUNING_RESULTS_PATH = OUTPUT_DIR / "tuning_results.json"
DISCOVERED_FEATURES_PATH = BASE_DIR / "handcrafted_features" / "discovered_candidates.json"

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Shared constants
TOKEN_PATTERN = (
    r"(?u)\[[^\]\s]+\]|-?\\[a-z]+\+?|"
    r"\b[a-z0-9_]+(?:\[\])?(?:[.#+\-][a-z0-9_]+(?:\[\])?)*\b|c\+\+|c#|\.net"
)
