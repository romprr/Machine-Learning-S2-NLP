import argparse
import json
import re
from itertools import combinations
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, RidgeClassifier, SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from .pipeline_paths import MODELS_OUTPUT_DIR, VECTORIZATION_OUTPUT_DIR, ensure_pipeline_directories
except ImportError:
    from pipeline_paths import MODELS_OUTPUT_DIR, VECTORIZATION_OUTPUT_DIR, ensure_pipeline_directories


RANDOM_STATE = 42
BASE_MODELS = (
    "multinomial_nb",
    "complement_nb",
    "ridge_classifier",
    "logistic_regression",
    "linear_svc",
    "sgd_classifier",
    "passive_aggressive",
    "decision_tree",
    "knn",
    "xgb_classifier",
)
ENSEMBLE_MODELS = ("hard_voting_ensemble",)
SUPPORTED_MODELS = BASE_MODELS + ENSEMBLE_MODELS
DEFAULT_MODEL_NAMES = BASE_MODELS
DEFAULT_SCORING = "f1_macro"
DEFAULT_CV_FOLDS = 3
DEFAULT_ENSEMBLE_COMPONENTS = ("multinomial_nb", "logistic_regression", "linear_svc")
DEFAULT_ENSEMBLE_SIZE_MIN = 3
DEFAULT_ENSEMBLE_SIZE_MAX = 3

MODEL_PARAM_GRIDS = {
    "multinomial_nb": {
        "alpha": [0.1, 0.3, 0.5, 0.8, 1.0],
        "fit_prior": [True, False],
    },
    "complement_nb": {
        "alpha": [0.1, 0.3, 0.5, 0.8, 1.0],
        "norm": [False, True],
    },
    "ridge_classifier": {
        "alpha": [0.1, 1.0, 2.0, 5.0],
        "class_weight": [None, "balanced"],
    },
    "logistic_regression": {
        "logisticregression__C": [0.5, 1.0, 2.0, 4.0],
        "logisticregression__class_weight": [None, "balanced"],
    },
    "linear_svc": {
        "C": [0.25, 0.5, 1.0, 2.0, 4.0],
        "class_weight": [None, "balanced"],
    },
    "sgd_classifier": {
        "loss": ["hinge", "log_loss"],
        "alpha": [1e-5, 1e-4, 1e-3],
        "penalty": ["l2", "elasticnet"],
        "class_weight": [None, "balanced"],
    },
    "passive_aggressive": {
        "C": [0.25, 0.5, 1.0, 2.0],
        "loss": ["hinge", "squared_hinge"],
        "class_weight": [None, "balanced"],
    },
    "decision_tree": {
        "max_depth": [20, 40, None],
        "min_samples_split": [2, 10, 20],
        "min_samples_leaf": [1, 5, 10],
        "criterion": ["gini", "entropy"],
    },
    "knn": {
        "n_neighbors": [5, 11, 21],
        "weights": ["uniform", "distance"],
        "metric": ["cosine", "euclidean"],
    },
    "xgb_classifier": {
        "n_estimators": [100, 200],
        "max_depth": [4, 6],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.6, 0.8],
    },
    "hard_voting_ensemble": {
        "nb__alpha": [0.3, 0.8],
        "lr__logisticregression__C": [1.0, 2.0],
        "svc__C": [0.5, 1.0],
    },
}


class TextXGBClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        min_child_weight: float = 1.0,
        reg_lambda: float = 1.0,
        gamma: float = 0.0,
        tree_method: str = "hist",
        max_bin: int = 256,
        n_jobs: int = 1,
        random_state: int = RANDOM_STATE,
        verbosity: int = 0,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.tree_method = tree_method
        self.max_bin = max_bin
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbosity = verbosity

    def fit(self, x_train, y_train):
        if XGBClassifier is None:
            raise ImportError(
                "xgboost is not installed in the current environment. Install it before using xgb_classifier."
            )

        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(np.asarray(y_train))
        self.classes_ = self.label_encoder_.classes_
        x_train = x_train.astype(np.float32, copy=False)
        self.model_ = XGBClassifier(
            objective="multi:softprob",
            num_class=len(self.classes_),
            eval_metric="mlogloss",
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            reg_lambda=self.reg_lambda,
            gamma=self.gamma,
            tree_method=self.tree_method,
            max_bin=self.max_bin,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbosity=self.verbosity,
        )
        self.model_.fit(x_train, y_encoded)
        return self

    def predict(self, x_test):
        x_test = x_test.astype(np.float32, copy=False)
        predictions = self.model_.predict(x_test)
        predictions = np.asarray(predictions, dtype=int)
        return self.label_encoder_.inverse_transform(predictions)

    def predict_proba(self, x_test):
        x_test = x_test.astype(np.float32, copy=False)
        return self.model_.predict_proba(x_test)


def is_base_model(model_name: str) -> bool:
    return model_name in BASE_MODELS


def sanitize_filename(value: str) -> str:
    lowered = value.lower()
    normalized = re.sub(r"[^a-z0-9._-]+", "_", lowered).strip("._")
    return normalized or "model"


def build_multinomial_nb():
    return MultinomialNB()


def build_complement_nb():
    return ComplementNB()


def build_ridge_classifier():
    return RidgeClassifier(
        random_state=RANDOM_STATE,
    )


def build_logistic_regression():
    return make_pipeline(
        MaxAbsScaler(),
        LogisticRegression(
            max_iter=2000,
            solver="saga",
            random_state=RANDOM_STATE,
        ),
    )


def build_linear_svc():
    return LinearSVC(
        C=1.0,
        max_iter=20000,
        random_state=RANDOM_STATE,
    )


def build_sgd_classifier():
    return SGDClassifier(
        max_iter=2000,
        tol=1e-3,
        random_state=RANDOM_STATE,
    )


def build_passive_aggressive():
    return PassiveAggressiveClassifier(
        max_iter=2000,
        tol=1e-3,
        random_state=RANDOM_STATE,
    )


def build_decision_tree():
    return DecisionTreeClassifier(
        random_state=RANDOM_STATE,
    )


def build_knn():
    return KNeighborsClassifier(
        n_neighbors=11,
        weights="distance",
        metric="cosine",
        algorithm="brute",
        n_jobs=1,
    )


def build_xgb_classifier():
    return TextXGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=1.0,
        colsample_bytree=0.8,
        tree_method="hist",
        max_bin=256,
        n_jobs=1,
        random_state=RANDOM_STATE,
        verbosity=0,
    )


def build_hard_voting_ensemble():
    return make_hard_voting_ensemble(DEFAULT_ENSEMBLE_COMPONENTS)


def build_model(model_name: str):
    if model_name == "multinomial_nb":
        return build_multinomial_nb()

    if model_name == "complement_nb":
        return build_complement_nb()

    if model_name == "ridge_classifier":
        return build_ridge_classifier()

    if model_name == "logistic_regression":
        return build_logistic_regression()

    if model_name == "linear_svc":
        return build_linear_svc()

    if model_name == "sgd_classifier":
        return build_sgd_classifier()

    if model_name == "passive_aggressive":
        return build_passive_aggressive()

    if model_name == "decision_tree":
        return build_decision_tree()

    if model_name == "knn":
        return build_knn()

    if model_name == "xgb_classifier":
        return build_xgb_classifier()

    if model_name == "hard_voting_ensemble":
        return build_hard_voting_ensemble()

    supported = ", ".join(SUPPORTED_MODELS)
    raise ValueError(f"Unsupported model: {model_name}. Expected one of: {supported}")


def build_model_with_params(model_name: str, params: dict[str, object] | None = None):
    model = build_model(model_name)
    if params:
        model.set_params(**params)
    return model


def make_hard_voting_ensemble(
    component_model_names: tuple[str, ...],
    tuned_params_by_model: dict[str, dict[str, object]] | None = None,
) -> VotingClassifier:
    estimators = []
    for model_name in component_model_names:
        params = {}
        if tuned_params_by_model and model_name in tuned_params_by_model:
            params = tuned_params_by_model[model_name]
        estimators.append((model_name, build_model_with_params(model_name, params)))

    return VotingClassifier(
        estimators=estimators,
        voting="hard",
        n_jobs=None,
    )


def get_param_grid(model_name: str) -> dict[str, list[object]]:
    try:
        return MODEL_PARAM_GRIDS[model_name]
    except KeyError as error:
        supported = ", ".join(SUPPORTED_MODELS)
        raise ValueError(f"Unsupported model: {model_name}. Expected one of: {supported}") from error


def evaluate_predictions(y_true: pd.Series, y_pred: pd.Series, labels: list[str]) -> dict[str, float]:
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="micro",
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "micro_precision": float(micro_precision),
        "micro_recall": float(micro_recall),
        "micro_f1": float(micro_f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "label_count": len(labels),
    }


def build_search(
    model_name: str,
    cv_folds: int,
    scoring: str,
    n_jobs: int,
) -> GridSearchCV:
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    return GridSearchCV(
        estimator=build_model(model_name),
        param_grid=get_param_grid(model_name),
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
        return_train_score=False,
    )


def fit_model(
    model_name: str,
    x_train,
    y_train: pd.Series,
    tune_hyperparameters: bool,
    cv_folds: int,
    scoring: str,
    n_jobs: int,
):
    if not tune_hyperparameters:
        started_at = perf_counter()
        model = build_model(model_name)
        model.fit(x_train, y_train)
        fit_time_seconds = perf_counter() - started_at
        return model, fit_time_seconds, None, {}, None

    search = build_search(
        model_name=model_name,
        cv_folds=cv_folds,
        scoring=scoring,
        n_jobs=n_jobs,
    )
    started_at = perf_counter()
    search.fit(x_train, y_train)
    fit_time_seconds = perf_counter() - started_at

    cv_results_df = pd.DataFrame(search.cv_results_).sort_values(
        ["rank_test_score", "mean_test_score"],
        ascending=[True, False],
    )
    return search.best_estimator_, fit_time_seconds, search.best_score_, search.best_params_, cv_results_df


def save_cv_results(cv_results_df: pd.DataFrame | None, output_path: Path) -> None:
    if cv_results_df is None:
        return

    columns_to_keep = [
        "rank_test_score",
        "mean_test_score",
        "std_test_score",
        "mean_train_score",
        "std_train_score",
        "mean_fit_time",
        "std_fit_time",
        "mean_score_time",
        "std_score_time",
        "params",
    ]
    available_columns = [column for column in columns_to_keep if column in cv_results_df.columns]
    trimmed = cv_results_df.loc[:, available_columns].copy()
    trimmed["params"] = trimmed["params"].apply(lambda value: json.dumps(value, sort_keys=True))
    trimmed.to_csv(output_path, index=False)


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
    component_models: tuple[str, ...] | None = None,
    prefit_estimator: bool = False,
    prefit_time_seconds: float | None = None,
) -> dict[str, object]:
    model_output_dir.mkdir(parents=True, exist_ok=True)

    if prefit_estimator:
        fit_time_seconds = prefit_time_seconds if prefit_time_seconds is not None else 0.0
    else:
        fit_started_at = perf_counter()
        estimator.fit(x_train, y_train)
        fit_time_seconds = perf_counter() - fit_started_at

    prediction_started_at = perf_counter()
    predictions = pd.Series(estimator.predict(x_test))
    predict_time_seconds = perf_counter() - prediction_started_at

    metrics = evaluate_predictions(y_test, predictions, labels)
    confusion = confusion_matrix(y_test, predictions, labels=labels)
    confusion_df = pd.DataFrame(confusion, index=labels, columns=labels)
    confusion_df.to_csv(model_output_dir / "confusion_matrix.csv")

    save_cv_results(cv_results_df, model_output_dir / "cv_results.csv")
    joblib.dump(estimator, model_output_dir / "model.joblib")

    metrics_payload = {
        "vectorization": vectorization_name,
        "model": model_name,
        "component_models": list(component_models) if component_models else None,
        "tune_hyperparameters": tune_hyperparameters,
        "cv_folds": cv_folds if tune_hyperparameters else None,
        "scoring": scoring if tune_hyperparameters else None,
        "best_cv_score": float(best_cv_score) if best_cv_score is not None else None,
        "best_params": best_params or {},
        "fit_time_seconds": round(fit_time_seconds, 4),
        "predict_time_seconds": round(predict_time_seconds, 4),
        **metrics,
    }
    (model_output_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    return {
        **metrics_payload,
        "best_params": json.dumps(best_params or {}, sort_keys=True),
    }


def benchmark_generated_ensembles(
    *,
    vectorization_name: str,
    models_output_dir: Path,
    x_train,
    y_train: pd.Series,
    x_test,
    y_test: pd.Series,
    labels: list[str],
    candidate_base_models: tuple[str, ...],
    tuned_params_by_model: dict[str, dict[str, object]],
    ensemble_size_min: int,
    ensemble_size_max: int,
) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    if len(candidate_base_models) < 2:
        return summaries

    valid_min = max(2, ensemble_size_min)
    valid_max = min(ensemble_size_max, len(candidate_base_models))
    if valid_min > valid_max:
        return summaries

    for ensemble_size in range(valid_min, valid_max + 1):
        for component_models in combinations(candidate_base_models, ensemble_size):
            ensemble_name = f"hard_voting__{'__'.join(component_models)}"
            output_dir = models_output_dir / vectorization_name / sanitize_filename(ensemble_name)
            estimator = make_hard_voting_ensemble(
                component_model_names=component_models,
                tuned_params_by_model=tuned_params_by_model,
            )
            summaries.append(
                evaluate_and_save_estimator(
                    model_name=ensemble_name,
                    estimator=estimator,
                    model_output_dir=output_dir,
                    x_train=x_train,
                    y_train=y_train,
                    x_test=x_test,
                    y_test=y_test,
                    labels=labels,
                    vectorization_name=vectorization_name,
                    tune_hyperparameters=False,
                    cv_folds=0,
                    scoring="",
                    best_cv_score=None,
                    best_params={},
                    cv_results_df=None,
                    component_models=component_models,
                )
            )

    return summaries


def train_and_evaluate_vectorization(
    vectorization_dir: Path,
    models_output_dir: Path,
    model_names: tuple[str, ...],
    tune_hyperparameters: bool,
    cv_folds: int,
    scoring: str,
    n_jobs: int,
    benchmark_ensembles: bool,
    ensemble_size_min: int,
    ensemble_size_max: int,
) -> list[dict[str, object]]:
    x_train = load_npz(vectorization_dir / "X_train.npz")
    x_test = load_npz(vectorization_dir / "X_test.npz")
    y_train = pd.read_csv(vectorization_dir / "y_train.csv")["tags"].astype(str)
    y_test = pd.read_csv(vectorization_dir / "y_test.csv")["tags"].astype(str)
    labels = sorted(y_train.unique())

    run_summaries: list[dict[str, object]] = []
    tuned_params_by_model: dict[str, dict[str, object]] = {}
    vectorization_name = vectorization_dir.name

    for model_name in model_names:
        if not is_base_model(model_name):
            continue

        model_output_dir = models_output_dir / vectorization_name / model_name
        model, fit_time_seconds, best_cv_score, best_params, cv_results_df = fit_model(
            model_name=model_name,
            x_train=x_train,
            y_train=y_train,
            tune_hyperparameters=tune_hyperparameters,
            cv_folds=cv_folds,
            scoring=scoring,
            n_jobs=n_jobs,
        )
        tuned_params_by_model[model_name] = best_params or {}
        run_summaries.append(
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
                tune_hyperparameters=tune_hyperparameters,
                cv_folds=cv_folds,
                scoring=scoring,
                best_cv_score=best_cv_score,
                best_params=best_params,
                cv_results_df=cv_results_df,
                prefit_estimator=True,
                prefit_time_seconds=fit_time_seconds,
            )
        )

    if "hard_voting_ensemble" in model_names:
        fixed_ensemble_output_dir = models_output_dir / vectorization_name / "hard_voting_ensemble"
        if tune_hyperparameters:
            fixed_ensemble, fit_time_seconds, best_cv_score, best_params, cv_results_df = fit_model(
                model_name="hard_voting_ensemble",
                x_train=x_train,
                y_train=y_train,
                tune_hyperparameters=True,
                cv_folds=cv_folds,
                scoring=scoring,
                n_jobs=n_jobs,
            )
            run_summaries.append(
                evaluate_and_save_estimator(
                    model_name="hard_voting_ensemble",
                    estimator=fixed_ensemble,
                    model_output_dir=fixed_ensemble_output_dir,
                    x_train=x_train,
                    y_train=y_train,
                    x_test=x_test,
                    y_test=y_test,
                    labels=labels,
                    vectorization_name=vectorization_name,
                    tune_hyperparameters=True,
                    cv_folds=cv_folds,
                    scoring=scoring,
                    best_cv_score=best_cv_score,
                    best_params=best_params,
                    cv_results_df=cv_results_df,
                    component_models=DEFAULT_ENSEMBLE_COMPONENTS,
                    prefit_estimator=True,
                    prefit_time_seconds=fit_time_seconds,
                )
            )
        else:
            fixed_ensemble = make_hard_voting_ensemble(
                component_model_names=DEFAULT_ENSEMBLE_COMPONENTS,
                tuned_params_by_model=tuned_params_by_model,
            )
            run_summaries.append(
                evaluate_and_save_estimator(
                    model_name="hard_voting_ensemble",
                    estimator=fixed_ensemble,
                    model_output_dir=fixed_ensemble_output_dir,
                    x_train=x_train,
                    y_train=y_train,
                    x_test=x_test,
                    y_test=y_test,
                    labels=labels,
                    vectorization_name=vectorization_name,
                    tune_hyperparameters=False,
                    cv_folds=0,
                    scoring="",
                    best_cv_score=None,
                    best_params={},
                    cv_results_df=None,
                    component_models=DEFAULT_ENSEMBLE_COMPONENTS,
                )
            )

    if benchmark_ensembles:
        candidate_base_models = tuple(model_name for model_name in model_names if is_base_model(model_name))
        run_summaries.extend(
            benchmark_generated_ensembles(
                vectorization_name=vectorization_name,
                models_output_dir=models_output_dir,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                labels=labels,
                candidate_base_models=candidate_base_models,
                tuned_params_by_model=tuned_params_by_model,
                ensemble_size_min=ensemble_size_min,
                ensemble_size_max=ensemble_size_max,
            )
        )

    return run_summaries


def train_and_evaluate_all(
    vectorization_output_dir: Path | str = VECTORIZATION_OUTPUT_DIR,
    models_output_dir: Path | str = MODELS_OUTPUT_DIR,
    vectorization_names: tuple[str, ...] | None = None,
    model_names: tuple[str, ...] | None = DEFAULT_MODEL_NAMES,
    tune_hyperparameters: bool = True,
    cv_folds: int = DEFAULT_CV_FOLDS,
    scoring: str = DEFAULT_SCORING,
    n_jobs: int = -1,
    benchmark_ensembles: bool = False,
    ensemble_size_min: int = DEFAULT_ENSEMBLE_SIZE_MIN,
    ensemble_size_max: int = DEFAULT_ENSEMBLE_SIZE_MAX,
) -> Path:
    ensure_pipeline_directories()

    vectorization_output_dir = Path(vectorization_output_dir)
    models_output_dir = Path(models_output_dir)
    models_output_dir.mkdir(parents=True, exist_ok=True)
    selected_model_names = model_names or DEFAULT_MODEL_NAMES

    if vectorization_names is None:
        vectorization_directories = sorted(path for path in vectorization_output_dir.iterdir() if path.is_dir())
    else:
        vectorization_directories = [vectorization_output_dir / name for name in vectorization_names]

    if not vectorization_directories:
        raise FileNotFoundError(f"No vectorization directories found in: {vectorization_output_dir}")

    all_summaries: list[dict[str, object]] = []
    for vectorization_directory in vectorization_directories:
        if not vectorization_directory.exists():
            raise FileNotFoundError(f"Missing vectorization directory: {vectorization_directory}")
        all_summaries.extend(
            train_and_evaluate_vectorization(
                vectorization_dir=vectorization_directory,
                models_output_dir=models_output_dir,
                model_names=selected_model_names,
                tune_hyperparameters=tune_hyperparameters,
                cv_folds=cv_folds,
                scoring=scoring,
                n_jobs=n_jobs,
                benchmark_ensembles=benchmark_ensembles,
                ensemble_size_min=ensemble_size_min,
                ensemble_size_max=ensemble_size_max,
            )
        )

    summary_df = pd.DataFrame(all_summaries).sort_values(
        ["vectorization", "macro_f1", "micro_f1", "accuracy"],
        ascending=[True, False, False, False],
    )
    summary_path = models_output_dir / "metrics_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    return summary_path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark text classifiers from saved vectorization artifacts and optionally search hard-voting ensembles."
    )
    parser.add_argument(
        "--vectorization-dir",
        type=Path,
        default=VECTORIZATION_OUTPUT_DIR,
        help="Directory that contains one subdirectory per vectorization.",
    )
    parser.add_argument(
        "--models-output-dir",
        type=Path,
        default=MODELS_OUTPUT_DIR,
        help="Directory where trained models and benchmark reports will be saved.",
    )
    parser.add_argument(
        "--vectorizations",
        nargs="*",
        help="Optional subset of vectorization directory names to evaluate, for example: tfidf.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODEL_NAMES),
        choices=SUPPORTED_MODELS,
        help="Base model families to benchmark. Add hard_voting_ensemble to also benchmark the fixed default ensemble.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=DEFAULT_CV_FOLDS,
        help="Number of stratified folds to use during hyperparameter tuning.",
    )
    parser.add_argument(
        "--scoring",
        default=DEFAULT_SCORING,
        help="Scikit-learn scoring metric used by GridSearchCV, for example f1_macro or accuracy.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel jobs for GridSearchCV. Use 1 to disable parallel search.",
    )
    parser.add_argument(
        "--no-tuning",
        action="store_true",
        help="Skip GridSearchCV and train each selected base model with its default configuration.",
    )
    parser.add_argument(
        "--benchmark-ensembles",
        action="store_true",
        help="Benchmark generated hard-voting ensembles built from the selected base models.",
    )
    parser.add_argument(
        "--ensemble-size-min",
        type=int,
        default=DEFAULT_ENSEMBLE_SIZE_MIN,
        help="Minimum number of base models per generated ensemble.",
    )
    parser.add_argument(
        "--ensemble-size-max",
        type=int,
        default=DEFAULT_ENSEMBLE_SIZE_MAX,
        help="Maximum number of base models per generated ensemble.",
    )
    return parser


if __name__ == "__main__":
    parser = build_argument_parser()
    arguments = parser.parse_args()

    summary_path = train_and_evaluate_all(
        vectorization_output_dir=arguments.vectorization_dir,
        models_output_dir=arguments.models_output_dir,
        vectorization_names=tuple(arguments.vectorizations) if arguments.vectorizations else None,
        model_names=tuple(arguments.models),
        tune_hyperparameters=not arguments.no_tuning,
        cv_folds=arguments.cv_folds,
        scoring=arguments.scoring,
        n_jobs=arguments.n_jobs,
        benchmark_ensembles=arguments.benchmark_ensembles,
        ensemble_size_min=arguments.ensemble_size_min,
        ensemble_size_max=arguments.ensemble_size_max,
    )
    print(f"Saved metrics summary to: {summary_path.resolve()}")
