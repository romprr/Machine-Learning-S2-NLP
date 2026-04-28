import pandas as pd
import numpy as np
import json
import os
from sklearn.feature_extraction.text import CountVectorizer

try:
    from .config import RAW_DATA_PATH, HANDCRAFTED_DIR, DISCOVERED_FEATURES_PATH
except ImportError:
    from config import RAW_DATA_PATH, HANDCRAFTED_DIR, DISCOVERED_FEATURES_PATH

def discover_best_ngrams(df, min_coverage=200, min_precision=0.6):
    print(f"Starting automatic feature discovery...")
    print(f"- Minimum Coverage: {min_coverage}")
    print(f"- Minimum Precision: {min_precision}")
    
    token_pattern = r'(?u)\b\w+\b|[\$\_\#\@]?[a-zA-Z0-9_]+|[<>\.\{\}\[\]\(\)]+'
    
    vectorizer = CountVectorizer(
        ngram_range=(1, 3),
        min_df=min_coverage,
        max_df=0.4,
        token_pattern=token_pattern,
        lowercase=True
    )
    
    print("Vectorizing text... this might take a minute.")
    X = vectorizer.fit_transform(df['post'])
    vocab = np.array(vectorizer.get_feature_names_out())
    
    X_bool = (X > 0).astype(int)
    
    global_df = np.array(X_bool.sum(axis=0)).flatten()
    
    candidates = {}
    
    tags = df['tags'].unique()
    for tag in tags:
        print(f"Analyzing candidates for `{tag}`...")
        mask = (df['tags'] == tag).values
        
        X_tag = X_bool[mask]
        
        tag_df = np.array(X_tag.sum(axis=0)).flatten()
        
        precision = tag_df / (global_df + 1e-9)
        
        valid_indices = np.where((tag_df >= min_coverage) & (precision >= min_precision))[0]
        
        tag_candidates = []
        for idx in valid_indices:
            token = vocab[idx]
            cov = int(tag_df[idx])
            prec = float(precision[idx])
            tag_candidates.append({
                "token": token,
                "coverage": cov,
                "precision": round(prec, 4)
            })
            
        tag_candidates.sort(key=lambda x: (x['precision'], x['coverage']), reverse=True)
        candidates[tag] = tag_candidates
    
    return candidates

def main():
    if not os.path.exists(RAW_DATA_PATH):
        print(f"Data not found at {RAW_DATA_PATH}")
        return
        
    df = pd.read_csv(RAW_DATA_PATH)
    
    candidates = discover_best_ngrams(df, min_coverage=200, min_precision=0.6)
    
    total_found = sum(len(c) for c in candidates.values())
    print(f"\nDiscovery complete! Found {total_found} strong candidate features across all tags.")
    
    print(f"Saving raw candidates to {DISCOVERED_FEATURES_PATH}")
    with open(DISCOVERED_FEATURES_PATH, "w", encoding="utf-8") as f:
        json.dump(candidates, f, indent=4)
        
    print("\n--- TOP 3 DISCOVERIES PER TAG ---")
    for tag, feats in candidates.items():
        print(f"[{tag}]")
        for f in feats[:3]:
            print(f"  - '{f['token']}' (Coverage: {f['coverage']}, Precision: {f['precision']})")

if __name__ == "__main__":
    main()
