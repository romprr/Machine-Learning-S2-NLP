import argparse
import json
from pathlib import Path
from time import perf_counter

import joblib
import pandas as pd
from scipy.sparse import load_npz
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVC

try:
    from .pipeline_paths import MODELS_OUTPUT_DIR, VECTORIZATION_OUTPUT_DIR, ensure_pipeline_directories
except ImportError:
    from pipeline_paths import MODELS_OUTPUT_DIR, VECTORIZATION_OUTPUT_DIR, ensure_pipeline_directories


RANDOM_STATE = 42
SUPPORTED_MODELS = ("multinomial_nb", "logistic_regression", "linear_svc", "hard_voting_ensemble")
DEFAULT_MODEL_NAMES = ("multinomial_nb", "logistic_regression", "linear_svc")
DEFAULT_SCORING = "f1_macro"
DEFAULT_CV_FOLDS = 3

MODEL_PARAM_GRIDS = {
    "multinomial_nb": {
        "alpha": [0.1, 0.3, 0.5, 0.8, 1.0],
        "fit_prior": [True, False],
    },
    "logistic_regression": {
        "logisticregression__C": [0.5, 1.0, 2.0, 4.0],
        "logisticregression__class_weight": [None, "balanced"],
    },
    "linear_svc": {
        "C": [0.25, 0.5, 1.0, 2.0, 4.0],
        "class_weight": [None, "balanced"],
    },
    "hard_voting_ensemble": {
        "nb__alpha": [0.3, 0.8],
        "lr__logisticregression__C": [1.0, 2.0],
        "svc__C": [0.5, 1.0],
    },
}


def build_multinomial_nb():
    return MultinomialNB()


def build_logistic_regression():
    return make_pipeline(
        MaxAbsScaler(),
        LogisticRegression(
            max_iter=2000,
            solver="saga",
            multi_class="multinomial",
            random_state=RANDOM_STATE,
        ),
    )


def build_linear_svc():
    return LinearSVC(
        C=1.0,
        max_iter=20000,
        random_state=RANDOM_STATE,
    )


def build_hard_voting_ensemble():
    return VotingClassifier(
        estimators=[
            ("nb", build_multinomial_nb()),
            ("lr", build_logistic_regression()),
            ("svc", build_linear_svc()),
        ],
        voting="hard",
        n_jobs=None,
    )


def build_model(model_name: str):
    if model_name == "multinomial_nb":
        return build_multinomial_nb()

    if model_name == "logistic_regression":
        return build_logistic_regression()

    if model_name == "linear_svc":
        return build_linear_svc()

    if model_name == "hard_voting_ensemble":
        return build_hard_voting_ensemble()

    supported = ", ".join(SUPPORTED_MODELS)
    raise ValueError(f"Unsupported model: {model_name}. Expected one of: {supported}")


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
        return_train_score=True,
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


def train_and_evaluate_vectorization(
    vectorization_dir: Path,
    models_output_dir: Path,
    model_names: tuple[str, ...],
    tune_hyperparameters: bool,
    cv_folds: int,
    scoring: str,
    n_jobs: int,
) -> list[dict[str, object]]:
    x_train = load_npz(vectorization_dir / "X_train.npz")
    x_test = load_npz(vectorization_dir / "X_test.npz")
    y_train = pd.read_csv(vectorization_dir / "y_train.csv")["tags"].astype(str)
    y_test = pd.read_csv(vectorization_dir / "y_test.csv")["tags"].astype(str)
    labels = sorted(y_train.unique())

    run_summaries: list[dict[str, object]] = []
    vectorization_name = vectorization_dir.name

    for model_name in model_names:
        model_output_dir = models_output_dir / vectorization_name / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)

        model, fit_time_seconds, best_cv_score, best_params, cv_results_df = fit_model(
            model_name=model_name,
            x_train=x_train,
            y_train=y_train,
            tune_hyperparameters=tune_hyperparameters,
            cv_folds=cv_folds,
            scoring=scoring,
            n_jobs=n_jobs,
        )

        prediction_started_at = perf_counter()
        predictions = pd.Series(model.predict(x_test))
        predict_time_seconds = perf_counter() - prediction_started_at

        metrics = evaluate_predictions(y_test, predictions, labels)
        confusion = confusion_matrix(y_test, predictions, labels=labels)
        confusion_df = pd.DataFrame(confusion, index=labels, columns=labels)
        confusion_df.to_csv(model_output_dir / "confusion_matrix.csv")

        save_cv_results(cv_results_df, model_output_dir / "cv_results.csv")
        joblib.dump(model, model_output_dir / "model.joblib")

        metrics_payload = {
            "vectorization": vectorization_name,
            "model": model_name,
            "tune_hyperparameters": tune_hyperparameters,
            "cv_folds": cv_folds if tune_hyperparameters else None,
            "scoring": scoring if tune_hyperparameters else None,
            "best_cv_score": float(best_cv_score) if best_cv_score is not None else None,
            "best_params": best_params,
            "fit_time_seconds": round(fit_time_seconds, 4),
            "predict_time_seconds": round(predict_time_seconds, 4),
            **metrics,
        }
        (model_output_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

        run_summaries.append(
            {
                **metrics_payload,
                "best_params": json.dumps(best_params, sort_keys=True) if best_params else "",
            }
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
        description="Benchmark classifiers from saved vectorization artifacts with optional hyperparameter tuning."
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
        help="Model families to benchmark.",
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
        help="Skip GridSearchCV and train each selected model with its default configuration.",
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
    )
    print(f"Saved metrics summary to: {summary_path.resolve()}")
