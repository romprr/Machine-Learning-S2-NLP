import argparse
import json
import re
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, StratifiedKFold

try:
    from .pipeline_paths import MODELS_OUTPUT_DIR, VECTORIZATION_OUTPUT_DIR, ensure_pipeline_directories
except ImportError:
    from pipeline_paths import MODELS_OUTPUT_DIR, VECTORIZATION_OUTPUT_DIR, ensure_pipeline_directories

RANDOM_STATE = 42
BASE_MODELS = ("logistic_regression", "xgb_classifier")
DEFAULT_SCORING = "f1_macro"
DEFAULT_CV_FOLDS = 3

def is_base_model(model_name: str) -> bool:
    return model_name in BASE_MODELS

def build_logistic_regression():
    return LogisticRegression(
        max_iter=200,
        solver="saga",
        random_state=RANDOM_STATE,
        tol=1e-2,
        C=4.475040191350905,
        penalty="l2"
    )

def build_model(model_name: str):
    if model_name == "logistic_regression":
        return build_logistic_regression()
    
    raise ValueError(f"Unsupported model: {model_name}")

def evaluate_predictions(y_true: pd.Series, y_pred: pd.Series, labels: list[str]) -> dict[str, float]:
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="micro", zero_division=0)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
    }

def save_cv_results(cv_results_df: pd.DataFrame | None, output_path: Path) -> None:
    if cv_results_df is None: return
    cv_results_df.to_csv(output_path, index=False)

def evaluate_and_save_estimator(
    *,
    model_name: str,
    estimator,
    model_output_dir: Path,
    x_train,
    y_train: pd.Series,
    x_test,
    y_test: pd.Series,
    labels: list[str],
    vectorization_name: str,
    tune_hyperparameters: bool,
    cv_folds: int,
    scoring: str,
    best_cv_score: float | None,
    best_params: dict[str, object] | None,
    cv_results_df: pd.DataFrame | None,
    prefit_estimator: bool = False,
    prefit_time_seconds: float | None = None,
) -> dict[str, object]:
    model_output_dir.mkdir(parents=True, exist_ok=True)
    if not prefit_estimator:
        estimator.fit(x_train, y_train)
    
    predictions = estimator.predict(x_test)
    metrics = evaluate_predictions(y_test, predictions, labels)
    
    save_cv_results(cv_results_df, model_output_dir / "cv_results.csv")
    joblib.dump(estimator, model_output_dir / "model.joblib")
    
    metrics_payload = {
        "vectorization": vectorization_name,
        "model": model_name,
        "best_cv_score": float(best_cv_score) if best_cv_score is not None else None,
        "best_params": best_params or {},
        "fit_time_seconds": round(prefit_time_seconds, 4) if prefit_time_seconds else 0,
        **metrics,
    }
    (model_output_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2))
    return metrics_payload
