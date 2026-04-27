import json
import pandas as pd
import scipy.stats as stats
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.linear_model import LogisticRegression
from .config import TUNING_RESULTS_PATH

def run_tuning(X_train, y_train):
    print("Starting hyperparameter tuning...")
    
    estimator = LogisticRegression(
        solver="saga",
        max_iter=200,
        tol=1e-2,
        random_state=42
    )
    
    param_distributions = {
        "C": stats.loguniform(1e-4, 1e2),
        "penalty": ["l1", "l2"]
    }

    search = HalvingRandomSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_candidates=40,
        factor=3,
        min_resources=1500,
        cv=3,
        scoring="f1_macro",
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    search.fit(X_train, y_train)
    
    print("Best params:", search.best_params_)
    print("Best CV score:", search.best_score_)
    
    # Prepare best params for saving
    best_params = {k: (v.item() if hasattr(v, 'item') else v) for k, v in search.best_params_.items()}
    
    results = {
        "best_params": best_params,
        "best_score": float(search.best_score_)
    }
    
    with open(TUNING_RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Tuning results saved to {TUNING_RESULTS_PATH}")
    
    return best_params
