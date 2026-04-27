import json
import pandas as pd
import joblib
import scipy.sparse as sp
from pathlib import Path
from scipy.sparse import save_npz

try:
    from .pipeline_paths import (
        RAW_DATA_PATH,
        TRAIN_SPLIT_PATH,
        TEST_SPLIT_PATH,
        VECTORIZATION_OUTPUT_DIR,
        ensure_pipeline_directories,
    )
    from .feature_pipeline import FeaturePipeline
except ImportError:
    from pipeline_paths import (
        RAW_DATA_PATH,
        TRAIN_SPLIT_PATH,
        TEST_SPLIT_PATH,
        VECTORIZATION_OUTPUT_DIR,
        ensure_pipeline_directories,
    )
    from feature_pipeline import FeaturePipeline

def generate_hybrid_features():
    ensure_pipeline_directories()
    
    print("Loading data...")
    train_df = pd.read_csv(TRAIN_SPLIT_PATH)
    test_df = pd.read_csv(TEST_SPLIT_PATH)
    raw_df = pd.read_csv(RAW_DATA_PATH)
    
    raw_train_df = raw_df.iloc[train_df["row_id"].values].reset_index(drop=True)
    raw_test_df = raw_df.iloc[test_df["row_id"].values].reset_index(drop=True)
    
    pipeline = FeaturePipeline()
    
    print("Fitting and transforming hybrid features...")
    X_train, y_train_encoded = pipeline.fit_transform(train_df, raw_train_df)
    X_test, y_test_encoded = pipeline.transform(test_df, raw_test_df)
    
    output_dir = VECTORIZATION_OUTPUT_DIR / "hybrid"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving artifacts to {output_dir}...")
    save_npz(output_dir / "X_train.npz", X_train)
    save_npz(output_dir / "X_test.npz", X_test)
    
    # Save labels for params_search compatibility
    train_labels = pd.DataFrame({"row_id": train_df["row_id"], "tags": train_df["tags"]})
    test_labels = pd.DataFrame({"row_id": test_df["row_id"], "tags": test_df["tags"]})
    
    train_labels.to_csv(output_dir / "y_train.csv", index=False)
    test_labels.to_csv(output_dir / "y_test.csv", index=False)
    
    pipeline.save()
    
    metadata = {
        "vectorization": "hybrid",
        "train_rows": X_train.shape[0],
        "test_rows": X_test.shape[0],
        "feature_count": X_train.shape[1],
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print("Done.")

if __name__ == "__main__":
    generate_hybrid_features()
