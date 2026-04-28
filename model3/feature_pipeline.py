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
    def __init__(self, vectorizer=None, scaler=None, label_encoder=None, include_generic_text_stats=True, use_text_features=True):
        self.use_text_features = use_text_features
        
        if self.use_text_features:
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
        else:
            self.vectorizer = None
            
        self.scaler = scaler or StandardScaler()
        self.label_encoder = label_encoder or LabelEncoder()
        self.text_extractor = TextStatsExtractor(include_generic=include_generic_text_stats)
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
        
        if self.use_text_features:
            # Binary vectorization for text
            print("Vectorizing text...")
            X_vec = self.vectorizer.fit_transform(train_df["preprocessed_post"].fillna("").astype(str))
            
            # Combine text and scaled handcrafted features
            X_combined = sp.hstack([X_vec, sp.csr_matrix(X_hc_scaled)]).tocsr()
        else:
            X_combined = sp.csr_matrix(X_hc_scaled)
        
        # Encode targets
        y = self.label_encoder.fit_transform(train_df["tags"].astype(str))
        
        return X_combined, y

    def transform(self, test_df, raw_test_df):
        """Transform test data using fitted components."""
        # Process handcrafted features
        hc_features = self.extract_handcrafted(raw_test_df)
        X_hc_scaled = self.scaler.transform(hc_features)
        
        if self.use_text_features:
            # Process text features
            X_vec = self.vectorizer.transform(test_df["preprocessed_post"].fillna("").astype(str))
            
            # Combine
            X_combined = sp.hstack([X_vec, sp.csr_matrix(X_hc_scaled)]).tocsr()
        else:
            X_combined = sp.csr_matrix(X_hc_scaled)
        
        # Encode labels
        y = self.label_encoder.transform(test_df["tags"].astype(str))
        
        return X_combined, y

    def save(self, vectorizer_path=VECTORIZER_PATH, scaler_path=SCALER_PATH, label_encoder_path=LABEL_ENCODER_PATH):
        """Save the fitted components."""
        if self.use_text_features:
            joblib.dump(self.vectorizer, vectorizer_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.label_encoder, label_encoder_path)
        print(f"Feature pipeline components saved to {scaler_path}, {label_encoder_path}" + (f", {vectorizer_path}" if self.use_text_features else ""))

    @classmethod
    def load(cls, use_text_features=True):
        """Load the fitted components."""
        if use_text_features:
            vectorizer = joblib.load(VECTORIZER_PATH)
        else:
            vectorizer = None
        scaler = joblib.load(SCALER_PATH)
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
        return cls(vectorizer=vectorizer, scaler=scaler, label_encoder=label_encoder, use_text_features=use_text_features)
