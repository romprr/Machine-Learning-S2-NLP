import argparse
import pandas as pd
from pathlib import Path

from .config import RAW_DATA_PATH, TRAIN_SPLIT_PATH, TEST_SPLIT_PATH, OUTPUT_DIR
from .feature_pipeline import FeaturePipeline
from .tuning import run_tuning
from .trainer import Model3Trainer

def load_all_data():
    print("Loading data splits and raw data...")
    train_df = pd.read_csv(TRAIN_SPLIT_PATH)
    test_df = pd.read_csv(TEST_SPLIT_PATH)
    raw_df = pd.read_csv(RAW_DATA_PATH)
    
    # Map back to raw data to get 'post' column for handcrafted features
    raw_train_df = raw_df.iloc[train_df["row_id"].values].reset_index(drop=True)
    raw_test_df = raw_df.iloc[test_df["row_id"].values].reset_index(drop=True)
    
    return train_df, test_df, raw_train_df, raw_test_df

def main():
    parser = argparse.ArgumentParser(description="Run Model 3 Pipeline with Handcrafted Features Only")
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning")
    args = parser.parse_args()

    # Define Handcrafted Output Directory
    HC_OUTPUT_DIR = OUTPUT_DIR / "handcrafted_only"
    HC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    train_df, test_df, raw_train_df, raw_test_df = load_all_data()

    # 2. Feature Pipeline with Handcrafted features ONLY
    print("Initializing pipeline (handcrafted features only)...")
    
    # use_text_features=False removes TF-IDF or CountVectorizer text features completely
    pipeline = FeaturePipeline(use_text_features=False)
    X_train, y_train = pipeline.fit_transform(train_df, raw_train_df)
    X_test, y_test = pipeline.transform(test_df, raw_test_df)
    
    pipeline.save(
        scaler_path=HC_OUTPUT_DIR / "scaler.joblib",
        label_encoder_path=HC_OUTPUT_DIR / "label_encoder.joblib"
    )

    # 3. Tuning (Optional)
    params = None
    if args.tune:
        params = run_tuning(X_train, y_train)
    
    # 4. Training
    trainer = Model3Trainer(params=params)
    trainer.train(X_train, y_train)
    
    # 5. Evaluation
    trainer.evaluate(
        X_test, 
        y_test, 
        label_encoder=pipeline.label_encoder, 
        vectorization_name="handcrafted_only", 
        output_dir=HC_OUTPUT_DIR
    )
    
    # 6. Save Model
    trainer.save(path=HC_OUTPUT_DIR / "logistic_regression_model.joblib")

if __name__ == "__main__":
    main()
