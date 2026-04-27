import json
import pandas as pd
from pathlib import Path
from model3.feature_pipeline import FeaturePipeline
from model3.trainer import Model3Trainer
from model3.config import RAW_DATA_PATH, TRAIN_SPLIT_PATH, TEST_SPLIT_PATH, TUNING_RESULTS_PATH

def load_data():
    print("Loading data...")
    train_df = pd.read_csv(TRAIN_SPLIT_PATH)
    test_df = pd.read_csv(TEST_SPLIT_PATH)
    raw_df = pd.read_csv(RAW_DATA_PATH)
    
    raw_train_df = raw_df.iloc[train_df["row_id"].values].reset_index(drop=True)
    raw_test_df = raw_df.iloc[test_df["row_id"].values].reset_index(drop=True)
    
    return train_df, test_df, raw_train_df, raw_test_df

def main():
    # Load optimized params from research/tuning
    with open(TUNING_RESULTS_PATH, "r") as f:
        best_params = json.load(f)
    
    # Model defaults
    full_params = {
        "solver": "saga",
        "max_iter": 200,
        "tol": 1e-2,
        "random_state": 42
    }
    full_params.update(best_params)

    # Process features
    train_df, test_df, raw_train_df, raw_test_df = load_data()
    
    pipeline = FeaturePipeline()
    X_train, y_train = pipeline.fit_transform(train_df, raw_train_df)
    X_test, y_test = pipeline.transform(test_df, raw_test_df)
    
    # Train final model
    trainer = Model3Trainer(params=full_params)
    trainer.train(X_train, y_train)
    
    # Evaluate and compare against metrics_halving.json
    current_f1 = trainer.evaluate(X_test, y_test, label_encoder=pipeline.label_encoder)
    
    metrics_path = Path(__file__).parent / "outputs" / "metrics_halving.json"
    with open(metrics_path, "r") as f:
        original_metrics = json.load(f)
    
    original_f1 = original_metrics["test_macro_f1"]
    
    print("\n--- Verification Results ---")
    print(f"Original Test Macro F1: {original_f1:.4f}")
    print(f"Current Test Macro F1:  {current_f1:.4f}")
    print(f"Difference:            {abs(current_f1 - original_f1):.6f}")
    
    if abs(current_f1 - original_f1) < 1e-4:
        print("\nSUCCESS: The results match!")
    else:
        print("\nWARNING: There is a slight difference.")

    # Save the final model
    trainer.save()
    pipeline.save()

if __name__ == "__main__":
    main()
