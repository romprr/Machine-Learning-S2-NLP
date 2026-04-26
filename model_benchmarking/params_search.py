import argparse
import json
from pathlib import Path

import pandas as pd
from scipy.sparse import load_npz

try:
    from .pipeline_paths import (
        MODEL_FINETUNING_OUTPUT_DIR,
        VECTORIZATION_OUTPUT_DIR,
        ensure_pipeline_directories,
    )
    from .train_models import (
        BASE_MODELS,
        DEFAULT_CV_FOLDS,
        DEFAULT_SCORING,
        evaluate_and_save_estimator,
        fit_model,
        is_base_model,
    )
except ImportError:
    from pipeline_paths import (
        MODEL_FINETUNING_OUTPUT_DIR,
        VECTORIZATION_OUTPUT_DIR,
        ensure_pipeline_directories,
    )
    from train_models import (
        BASE_MODELS,
        DEFAULT_CV_FOLDS,
        DEFAULT_SCORING,
        evaluate_and_save_estimator,
        fit_model,
        is_base_model,
    )


DEFAULT_VECTORIZATION = "tfidf"


def resolve_output_dir(base_output_dir: Path, run_name: str | None) -> Path:
    if not run_name:
        return base_output_dir
    return base_output_dir / run_name


def tune_vectorization_models(
    *,
    vectorization_name: str = DEFAULT_VECTORIZATION,
    vectorization_output_dir: Path | str = VECTORIZATION_OUTPUT_DIR,
    output_dir: Path | str = MODEL_FINETUNING_OUTPUT_DIR,
    model_names: tuple[str, ...] = BASE_MODELS,
    cv_folds: int = DEFAULT_CV_FOLDS,
    scoring: str = DEFAULT_SCORING,
    n_jobs: int = -1,
    run_name: str | None = None,
) -> Path:
    ensure_pipeline_directories()

    vectorization_output_dir = Path(vectorization_output_dir)
    output_dir = resolve_output_dir(Path(output_dir), run_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    vectorization_dir = vectorization_output_dir / vectorization_name
    if not vectorization_dir.exists():
        raise FileNotFoundError(f"Missing vectorization directory: {vectorization_dir}")

    invalid_models = [model_name for model_name in model_names if not is_base_model(model_name)]
    if invalid_models:
        invalid_text = ", ".join(invalid_models)
        raise ValueError(f"params_search only supports base models. Invalid values: {invalid_text}")

    x_train = load_npz(vectorization_dir / "X_train.npz")
    x_test = load_npz(vectorization_dir / "X_test.npz")
    y_train = pd.read_csv(vectorization_dir / "y_train.csv")["tags"].astype(str)
    y_test = pd.read_csv(vectorization_dir / "y_test.csv")["tags"].astype(str)
    labels = sorted(y_train.unique())

    run_summaries: list[dict[str, object]] = []
    for model_name in model_names:
        model_output_dir = output_dir / vectorization_name / model_name
        model, fit_time_seconds, best_cv_score, best_params, cv_results_df = fit_model(
            model_name=model_name,
            x_train=x_train,
            y_train=y_train,
            tune_hyperparameters=True,
            cv_folds=cv_folds,
            scoring=scoring,
            n_jobs=n_jobs,
        )

        summary = evaluate_and_save_estimator(
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
            best_cv_score=best_cv_score,
            best_params=best_params,
            cv_results_df=cv_results_df,
            prefit_estimator=True,
            prefit_time_seconds=fit_time_seconds,
        )
        (model_output_dir / "best_params.json").write_text(
            json.dumps(best_params or {}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        run_summaries.append(summary)

    summary_df = pd.DataFrame(run_summaries).sort_values(
        ["macro_f1", "micro_f1", "accuracy"],
        ascending=[False, False, False],
    )
    summary_path = output_dir / vectorization_name / "metrics_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)

    config_payload = {
        "vectorization": vectorization_name,
        "model_names": list(model_names),
        "cv_folds": cv_folds,
        "scoring": scoring,
        "n_jobs": n_jobs,
        "run_name": run_name,
        "vectorization_dir": str(vectorization_dir.resolve()),
        "output_dir": str((output_dir / vectorization_name).resolve()),
    }
    (output_dir / vectorization_name / "search_config.json").write_text(
        json.dumps(config_payload, indent=2),
        encoding="utf-8",
    )
    return summary_path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run hyperparameter search for selected base models on one vectorization artifact set."
    )
    parser.add_argument(
        "--vectorization",
        default=DEFAULT_VECTORIZATION,
        help="Vectorization directory name to tune against, for example tfidf.",
    )
    parser.add_argument(
        "--vectorization-dir",
        type=Path,
        default=VECTORIZATION_OUTPUT_DIR,
        help="Directory that contains saved vectorization artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MODEL_FINETUNING_OUTPUT_DIR,
        help="Directory where fine-tuning artifacts will be saved.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(BASE_MODELS),
        choices=BASE_MODELS,
        help="Base model families to fine-tune.",
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
        "--run-name",
        help="Optional subfolder name inside model_finetuning to keep multiple tuning runs separate.",
    )
    return parser


if __name__ == "__main__":
    parser = build_argument_parser()
    arguments = parser.parse_args()

    summary_path = tune_vectorization_models(
        vectorization_name=arguments.vectorization,
        vectorization_output_dir=arguments.vectorization_dir,
        output_dir=arguments.output_dir,
        model_names=tuple(arguments.models),
        cv_folds=arguments.cv_folds,
        scoring=arguments.scoring,
        n_jobs=arguments.n_jobs,
        run_name=arguments.run_name,
    )
    print(f"Saved fine-tuning summary to: {summary_path.resolve()}")
