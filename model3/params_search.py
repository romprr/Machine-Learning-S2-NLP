import argparse
import json
from numbers import Real
from pathlib import Path
from time import perf_counter

import pandas as pd
import scipy.stats as stats
from scipy.sparse import load_npz
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold, HalvingRandomSearchCV

try:
    from .pipeline_paths import (
        MODEL_FINETUNING_OUTPUT_DIR,
        VECTORIZATION_OUTPUT_DIR,
        ensure_pipeline_directories,
    )
    from .train_models import (
        BASE_MODELS,
        DEFAULT_SCORING,
        RANDOM_STATE,
        build_model,
        evaluate_and_save_estimator,
        is_base_model,
        save_cv_results,
    )
except ImportError:
    from pipeline_paths import (
        MODEL_FINETUNING_OUTPUT_DIR,
        VECTORIZATION_OUTPUT_DIR,
        ensure_pipeline_directories,
    )
    from train_models import (
        BASE_MODELS,
        DEFAULT_SCORING,
        RANDOM_STATE,
        build_model,
        evaluate_and_save_estimator,
        is_base_model,
        save_cv_results,
    )


DEFAULT_VECTORIZATION = "hybrid"
DEFAULT_SEARCH_PROFILE = "halving"
DEFAULT_SEARCH_CV_FOLDS = 3
DEFAULT_SEARCH_VERBOSE = 2

SEARCH_PROFILES = ("standard", "deep", "halving")

STANDARD_MODEL_PARAM_GRIDS = {
    "logistic_regression": {
        "C": [0.5, 1.0, 2.0, 4.0],
        "penalty": ["l2"],
    },
}

DEEP_MODEL_PARAM_GRIDS = {
    "logistic_regression": {
        "C": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        "penalty": ["l1", "l2"],
    },
}

HALVING_MODEL_PARAM_GRIDS = {
    "logistic_regression": {
        "C": stats.loguniform(1e-4, 1e2),
        "penalty": ["l1", "l2"]
    },
    "xgb_classifier": {
        "n_estimators": stats.randint(50, 500),
        "max_depth": stats.randint(3, 15),
        "learning_rate": stats.loguniform(0.01, 0.3),
        "subsample": stats.uniform(0.5, 0.5),
        "colsample_bytree": stats.uniform(0.5, 0.5),
        "min_child_weight": stats.randint(1, 10),
        "reg_lambda": stats.loguniform(1e-3, 10.0),
        "gamma": stats.uniform(0, 5.0),
    }
}

def get_param_grid(model_name: str, search_profile: str):
    if search_profile == "deep":
        grid_source = DEEP_MODEL_PARAM_GRIDS
    elif search_profile == "standard":
        grid_source = STANDARD_MODEL_PARAM_GRIDS
    elif search_profile == "halving":
        grid_source = HALVING_MODEL_PARAM_GRIDS
    else:
        raise ValueError(f"Unsupported search profile: {search_profile}")

    return grid_source[model_name]

def run_search(
    *,
    model_name: str,
    param_grid: dict,
    x_train,
    y_train: pd.Series,
    cv_folds: int,
    scoring: str,
    n_jobs: int,
    search_verbose: int,
    search_profile: str
):
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    
    if search_profile == "halving":
        search = HalvingRandomSearchCV(
            estimator=build_model(model_name),
            param_distributions=param_grid,
            n_candidates=40,
            factor=3,
            min_resources=1500,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            random_state=RANDOM_STATE,
            verbose=search_verbose,
        )
    else:
        search = GridSearchCV(
            estimator=build_model(model_name),
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            refit=True,
            verbose=search_verbose,
        )
        
    started_at = perf_counter()
    search.fit(x_train, y_train)
    fit_time_seconds = perf_counter() - started_at

    cv_results_df = pd.DataFrame(search.cv_results_)
    return (
        search.best_estimator_,
        fit_time_seconds,
        search.best_score_,
        search.best_params_,
        cv_results_df,
    )

def tune_models(
    vectorization_name: str = DEFAULT_VECTORIZATION,
    model_names: tuple[str, ...] = ("logistic_regression",),
    cv_folds: int = DEFAULT_SEARCH_CV_FOLDS,
    scoring: str = DEFAULT_SCORING,
    n_jobs: int = -1,
    search_profile: str = DEFAULT_SEARCH_PROFILE,
):
    ensure_pipeline_directories()
    vectorization_dir = VECTORIZATION_OUTPUT_DIR / vectorization_name
    
    x_train = load_npz(vectorization_dir / "X_train.npz")
    x_test = load_npz(vectorization_dir / "X_test.npz")
    y_train = pd.read_csv(vectorization_dir / "y_train.csv")["tags"].astype(str)
    y_test = pd.read_csv(vectorization_dir / "y_test.csv")["tags"].astype(str)
    labels = sorted(y_train.unique())

    for model_name in model_names:
        print(f"Tuning {model_name}...")
        param_grid = get_param_grid(model_name, search_profile)
        
        model, fit_time, best_score, best_params, cv_results = run_search(
            model_name=model_name,
            param_grid=param_grid,
            x_train=x_train,
            y_train=y_train,
            cv_folds=cv_folds,
            scoring=scoring,
            n_jobs=n_jobs,
            search_verbose=DEFAULT_SEARCH_VERBOSE,
            search_profile=search_profile
        )
        
        model_output_dir = MODEL_FINETUNING_OUTPUT_DIR / vectorization_name / model_name
        evaluate_and_save_estimator(
            model_name=model_name,
            estimator=model,
            model_output_dir=model_output_dir,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            labels=labels,
            vectorization_name=vectorization_name,
            tune_hyperparameters=True,
            cv_folds=cv_folds,
            scoring=scoring,
            best_cv_score=best_score,
            best_params=best_params,
            cv_results_df=cv_results,
            prefit_estimator=True,
            prefit_time_seconds=fit_time
        )
        print(f"Finished {model_name}. Best score: {best_score}")

if __name__ == "__main__":
    tune_models()
