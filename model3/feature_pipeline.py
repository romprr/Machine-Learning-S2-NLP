import pandas as pd
import numpy as np
import scipy.sparse as sp
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder

from .extractors_code import RegexFeatureExtractor
from .extractors_text import TextStatsExtractor
from .config import (
    RAW_DATA_PATH, HANDCRAFTED_FEATURES_PATH, 
    TOKEN_PATTERN, VECTORIZER_PATH, SCALER_PATH, LABEL_ENCODER_PATH
)

class FeaturePipeline:
    def __init__(self, vectorizer=None, scaler=None, label_encoder=None):
        self.vectorizer = vectorizer or CountVectorizer(
            lowercase=False,
            preprocessor=None,
            tokenizer=None,
            token_pattern=TOKEN_PATTERN,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            binary=True,
        )
        self.scaler = scaler or StandardScaler()
        self.label_encoder = label_encoder or LabelEncoder()
        self.text_extractor = TextStatsExtractor()
        self.code_extractor = RegexFeatureExtractor()

    def extract_handcrafted(self, df):
        """Extract handcrafted features from raw posts."""
        print("Extracting handcrafted features...")
        text_feats = self.text_extractor.fit_transform(df['post'])
        code_feats = self.code_extractor.fit_transform(df['post'])
        return pd.concat([text_feats, code_feats], axis=1)

    def fit_transform(self, train_df, raw_train_df):
        """Fit and transform training data."""
        # Extraction & scaling for handcrafted features
        hc_features = self.extract_handcrafted(raw_train_df)
        X_hc_scaled = self.scaler.fit_transform(hc_features)
        
        # Binary vectorization for text
        print("Vectorizing text...")
        X_vec = self.vectorizer.fit_transform(train_df["preprocessed_post"].fillna("").astype(str))
        
        # Combine text and scaled handcrafted features
        X_combined = sp.hstack([X_vec, sp.csr_matrix(X_hc_scaled)]).tocsr()
        
        # Encode targets
        y = self.label_encoder.fit_transform(train_df["tags"].astype(str))
        
        return X_combined, y

    def transform(self, test_df, raw_test_df):
        """Transform test data using fitted components."""
        # Process handcrafted features
        hc_features = self.extract_handcrafted(raw_test_df)
        X_hc_scaled = self.scaler.transform(hc_features)
        
        # Process text features
        X_vec = self.vectorizer.transform(test_df["preprocessed_post"].fillna("").astype(str))
        
        # Combine
        X_combined = sp.hstack([X_vec, sp.csr_matrix(X_hc_scaled)]).tocsr()
        
        # Encode labels
        y = self.label_encoder.transform(test_df["tags"].astype(str))
        
        return X_combined, y

    def save(self):
        """Save the fitted components."""
        joblib.dump(self.vectorizer, VECTORIZER_PATH)
        joblib.dump(self.scaler, SCALER_PATH)
        joblib.dump(self.label_encoder, LABEL_ENCODER_PATH)
        print(f"Feature pipeline components saved to {VECTORIZER_PATH}, {SCALER_PATH}, {LABEL_ENCODER_PATH}")

    @classmethod
    def load(cls):
        """Load the fitted components."""
        vectorizer = joblib.load(VECTORIZER_PATH)
        scaler = joblib.load(SCALER_PATH)
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
        return cls(vectorizer=vectorizer, scaler=scaler, label_encoder=label_encoder)
