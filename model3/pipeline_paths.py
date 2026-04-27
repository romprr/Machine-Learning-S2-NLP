from pathlib import Path

MODEL3_ROOT = Path(__file__).resolve().parent
BASE_DIR = MODEL3_ROOT.parent

# Input data paths (referencing global project data)
RAW_DATA_PATH = BASE_DIR / "handcrafted_features" / "stack-overflow-data.csv"

# Model3 Pipeline outputs
PIPELINE_OUTPUT_DIR = MODEL3_ROOT / "outputs"
PREPROCESSED_DATA_PATH = PIPELINE_OUTPUT_DIR / "preprocessed_data.csv"
PREPROCESSING_SUMMARY_PATH = PIPELINE_OUTPUT_DIR / "preprocessing_summary.json"

SPLIT_OUTPUT_DIR = PIPELINE_OUTPUT_DIR / "splits"
TRAIN_SPLIT_PATH = SPLIT_OUTPUT_DIR / "train_data.csv"
TEST_SPLIT_PATH = SPLIT_OUTPUT_DIR / "test_data.csv"
SPLIT_SUMMARY_PATH = SPLIT_OUTPUT_DIR / "split_summary.json"

VECTORIZATION_OUTPUT_DIR = PIPELINE_OUTPUT_DIR / "vectorized"
MODELS_OUTPUT_DIR = PIPELINE_OUTPUT_DIR / "models"
MODEL_FINETUNING_OUTPUT_DIR = PIPELINE_OUTPUT_DIR / "model_finetuning"

# Feature metadata
DISCOVERED_FEATURES_PATH = BASE_DIR / "handcrafted_features" / "discovered_candidates.json"

def ensure_pipeline_directories() -> None:
    PIPELINE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SPLIT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VECTORIZATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_FINETUNING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
