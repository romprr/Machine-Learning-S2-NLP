import argparse
import pandas as pd
from pathlib import Path

from .config import RAW_DATA_PATH, TRAIN_SPLIT_PATH, TEST_SPLIT_PATH
from .feature_pipeline import FeaturePipeline
from .tuning import run_tuning
from .trainer import Model3Trainer

def load_all_data():
    print("Loading data splits and raw data...")
    train_df = pd.read_csv(TRAIN_SPLIT_PATH)
    test_df = pd.read_csv(TEST_SPLIT_PATH)
    raw_df = pd.read_csv(RAW_DATA_PATH)
    
    # Map back to raw data to get 'post' column for handcrafted features
    # Assuming row_id matches index in raw_df or we use it to filter
    raw_train_df = raw_df.iloc[train_df["row_id"].values].reset_index(drop=True)
    raw_test_df = raw_df.iloc[test_df["row_id"].values].reset_index(drop=True)
    
    return train_df, test_df, raw_train_df, raw_test_df

def main():
    parser = argparse.ArgumentParser(description="Run Model 3 Pipeline")
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning")
    args = parser.parse_args()

    # 1. Load data
    train_df, test_df, raw_train_df, raw_test_df = load_all_data()

    # 2. Feature Pipeline
    pipeline = FeaturePipeline()
    X_train, y_train = pipeline.fit_transform(train_df, raw_train_df)
    X_test, y_test = pipeline.transform(test_df, raw_test_df)
    
    pipeline.save()

    # 3. Tuning (Optional)
    params = None
    if args.tune:
        params = run_tuning(X_train, y_train)
    
    # 4. Training
    trainer = Model3Trainer(params=params)
    trainer.train(X_train, y_train)
    
    # 5. Evaluation
    trainer.evaluate(X_test, y_test, label_encoder=pipeline.label_encoder)
    
    # 6. Save Model
    trainer.save()

if __name__ == "__main__":
    main()
