import argparse
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import RAW_DATA_PATH, TRAIN_SPLIT_PATH, TEST_SPLIT_PATH, OUTPUT_DIR, TOKEN_PATTERN
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
    parser = argparse.ArgumentParser(description="Run Model 3 Pipeline with TF-IDF and Rule-based Handcrafted features")
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning")
    args = parser.parse_args()

    # Define TF-IDF Output Directory
    TFIDF_OUTPUT_DIR = OUTPUT_DIR / "tfidf"
    TFIDF_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    train_df, test_df, raw_train_df, raw_test_df = load_all_data()

    # 2. Feature Pipeline with TF-IDF Vectorizer and ONLY rule-based handcrafted features
    print("Initializing hybrid TF-IDF pipeline (excluding generic length/word count stats)...")
    tfidf_vectorizer = TfidfVectorizer(
        lowercase=False,
        preprocessor=None,
        tokenizer=None,
        token_pattern=TOKEN_PATTERN,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    
    # include_generic_text_stats=False removes post length and word count
    pipeline = FeaturePipeline(vectorizer=tfidf_vectorizer, include_generic_text_stats=False)
    X_train, y_train = pipeline.fit_transform(train_df, raw_train_df)
    X_test, y_test = pipeline.transform(test_df, raw_test_df)
    
    pipeline.save(
        vectorizer_path=TFIDF_OUTPUT_DIR / "vectorizer.joblib",
        scaler_path=TFIDF_OUTPUT_DIR / "scaler.joblib",
        label_encoder_path=TFIDF_OUTPUT_DIR / "label_encoder.joblib"
    )

    # 3. Tuning (Optional)
    params = {
        "solver": "saga",
        "max_iter": 200,
        "tol": 1e-2,
        "random_state": 42,
        "C": 4.475040191350905, # Best parameters found during research
        "penalty": "l2"
    }
    
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
        vectorization_name="hybrid_tfidf_rules_only", 
        output_dir=TFIDF_OUTPUT_DIR
    )
    
    # 6. Save Model
    trainer.save(path=TFIDF_OUTPUT_DIR / "logistic_regression_model.joblib")

if __name__ == "__main__":
    main()
