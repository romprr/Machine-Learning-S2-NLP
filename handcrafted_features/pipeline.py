import pandas as pd
import json
import os
import time

from .config import RAW_DATA_PATH, FEATURES_OUTPUT_PATH, METADATA_OUTPUT_PATH
from .extractors_code import RegexFeatureExtractor
from .extractors_text import TextStatsExtractor

def build_features(input_path: str = RAW_DATA_PATH, output_path: str = FEATURES_OUTPUT_PATH):
    print(f"Loading data from {input_path} ...")
    start = time.time()
    df = pd.read_csv(input_path)
    
    if 'post' not in df.columns:
        raise ValueError(f"Expected to find 'post' column but found {df.columns.tolist()}")

    print("Running TextStatsExtractor...")
    text_feat_extractor = TextStatsExtractor()
    text_feats = text_feat_extractor.fit_transform(df['post'])
    
    print("Running RegexFeatureExtractor...")
    regex_extractor = RegexFeatureExtractor()
    code_feats = regex_extractor.fit_transform(df['post'])
    
    features_df = pd.concat([text_feats, code_feats], axis=1)

    print(f"Generated {features_df.shape[1]} features for {features_df.shape[0]} samples.")
    
    print(f"Saving features to {output_path} ...")
    features_df.to_csv(output_path, index=False)
    
    metadata = {
        "num_samples": features_df.shape[0],
        "num_features": features_df.shape[1],
        "feature_names": list(features_df.columns),
        "execution_time_seconds": round(time.time() - start, 2)
    }
    
    with open(METADATA_OUTPUT_PATH, "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Metadata saved to {METADATA_OUTPUT_PATH}")

if __name__ == "__main__":
    if os.path.exists(RAW_DATA_PATH):
        build_features()
    else:
        print(f"File not found: {RAW_DATA_PATH}. Make sure you're in the right directory.")
