from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA_PATH = PROJECT_ROOT.parent / "model_1" / "stack-overflow-data.csv"

PIPELINE_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pipeline"
PREPROCESSED_DATA_PATH = PIPELINE_OUTPUT_DIR / "model_2_preprocessed_stack_overflow_data.csv"
PREPROCESSING_SUMMARY_PATH = PIPELINE_OUTPUT_DIR / "model_2_preprocessing_summary.json"

SPLIT_OUTPUT_DIR = PIPELINE_OUTPUT_DIR / "model_2_splits"
TRAIN_SPLIT_PATH = SPLIT_OUTPUT_DIR / "train_data.csv"
TEST_SPLIT_PATH = SPLIT_OUTPUT_DIR / "test_data.csv"
SPLIT_SUMMARY_PATH = SPLIT_OUTPUT_DIR / "split_summary.json"

VECTORIZATION_OUTPUT_DIR = PIPELINE_OUTPUT_DIR / "model_2_vectorized"
MODELS_OUTPUT_DIR = PIPELINE_OUTPUT_DIR / "model_2_models"
MODEL_FINETUNING_OUTPUT_DIR = PIPELINE_OUTPUT_DIR / "model_2_finetuning"


def ensure_pipeline_directories() -> None:
    PIPELINE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SPLIT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VECTORIZATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_FINETUNING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
