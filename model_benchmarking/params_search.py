import argparse
import json
from numbers import Real
from pathlib import Path
from time import perf_counter

import pandas as pd
from scipy.sparse import load_npz
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold

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


DEFAULT_VECTORIZATION = "tfidf"
DEFAULT_SEARCH_PROFILE = "deep"
DEFAULT_SEARCH_CV_FOLDS = 5
DEFAULT_DEEP_CV_REPEATS = 2
DEFAULT_REFINE_ROUNDS = 2
DEFAULT_SEARCH_VERBOSE = 2

SEARCH_PROFILES = ("standard", "deep")

STANDARD_MODEL_PARAM_GRIDS = {
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
        "logisticregression__penalty": ["l2"],
    },
    "linear_svc": {
        "C": [0.25, 0.5, 1.0, 2.0, 4.0],
        "class_weight": [None, "balanced"],
        "loss": ["squared_hinge"],
    },
    "sgd_classifier": {
        "loss": ["hinge", "log_loss"],
        "alpha": [1e-5, 1e-4, 1e-3],
        "penalty": ["l2"],
        "class_weight": [None, "balanced"],
        "learning_rate": ["optimal"],
    },
    "passive_aggressive": {
        "C": [0.25, 0.5, 1.0, 2.0],
        "loss": ["hinge", "squared_hinge"],
        "class_weight": [None, "balanced"],
    },
}

DEEP_MODEL_PARAM_GRIDS = {
    "multinomial_nb": {
        "alpha": [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0],
        "fit_prior": [True, False],
    },
    "complement_nb": {
        "alpha": [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0],
        "norm": [False, True],
    },
    "ridge_classifier": {
        "alpha": [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
        "class_weight": [None, "balanced"],
    },
    "logistic_regression": [
        {
            "logisticregression__C": [0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0],
            "logisticregression__class_weight": [None, "balanced"],
            "logisticregression__penalty": ["l2"],
        },
        {
            "logisticregression__C": [0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0],
            "logisticregression__class_weight": [None, "balanced"],
            "logisticregression__penalty": ["elasticnet"],
            "logisticregression__l1_ratio": [0.15, 0.3, 0.5, 0.7, 0.85],
        },
    ],
    "linear_svc": {
        "C": [0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0],
        "class_weight": [None, "balanced"],
        "loss": ["hinge", "squared_hinge"],
    },
    "sgd_classifier": [
        {
            "loss": ["hinge", "log_loss"],
            "alpha": [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3],
            "penalty": ["l2"],
            "class_weight": [None, "balanced"],
            "learning_rate": ["optimal", "adaptive"],
            "eta0": [0.001, 0.003, 0.01, 0.03],
            "average": [False, True],
        },
        {
            "loss": ["hinge", "log_loss"],
            "alpha": [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4],
            "penalty": ["elasticnet"],
            "class_weight": [None, "balanced"],
            "learning_rate": ["optimal", "adaptive"],
            "eta0": [0.001, 0.003, 0.01],
            "average": [False, True],
            "l1_ratio": [0.15, 0.3, 0.5, 0.7, 0.85],
        },
    ],
    "passive_aggressive": {
        "C": [0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0],
        "loss": ["hinge", "squared_hinge"],
        "class_weight": [None, "balanced"],
        "average": [False, True],
    },
}


def resolve_output_dir(base_output_dir: Path, run_name: str | None) -> Path:
    if not run_name:
        return base_output_dir
    return base_output_dir / run_name


def normalize_grid_spec(param_grid: dict[str, list[object]] | list[dict[str, list[object]]]) -> list[dict[str, list[object]]]:
    if isinstance(param_grid, list):
        return [{key: list(values) for key, values in grid.items()} for grid in param_grid]
    return [{key: list(values) for key, values in param_grid.items()}]


def estimate_grid_size(param_grid: dict[str, list[object]] | list[dict[str, list[object]]]) -> int:
    total = 0
    for grid in normalize_grid_spec(param_grid):
        branch_total = 1
        for values in grid.values():
            branch_total *= len(values)
        total += branch_total
    return total


def log_progress(message: str) -> None:
    print(f"[params_search] {message}", flush=True)


def get_param_grid(model_name: str, search_profile: str) -> dict[str, list[object]] | list[dict[str, list[object]]]:
    if search_profile == "deep":
        grid_source = DEEP_MODEL_PARAM_GRIDS
    elif search_profile == "standard":
        grid_source = STANDARD_MODEL_PARAM_GRIDS
    else:
        supported = ", ".join(SEARCH_PROFILES)
        raise ValueError(f"Unsupported search profile: {search_profile}. Expected one of: {supported}")

    try:
        selected_grid = grid_source[model_name]
    except KeyError as error:
        supported_models = ", ".join(BASE_MODELS)
        raise ValueError(f"Unsupported model for param search: {model_name}. Expected one of: {supported_models}") from error

    if isinstance(selected_grid, list):
        return [{key: list(values) for key, values in grid.items()} for grid in selected_grid]
    return {key: list(values) for key, values in selected_grid.items()}


def is_numeric_value(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def compute_midpoint(left: Real, right: Real) -> Real | None:
    if left == right:
        return None
    if isinstance(left, int) and isinstance(right, int):
        midpoint = int(round((left + right) / 2))
        if midpoint in (left, right):
            return None
        return midpoint

    if left > 0 and right > 0:
        midpoint = (float(left) * float(right)) ** 0.5
    else:
        midpoint = (float(left) + float(right)) / 2.0

    rounded = round(midpoint, 10)
    if rounded in (left, right):
        return None
    return rounded


def build_refined_numeric_values(values: list[object], best_value: object) -> list[object]:
    ordered = sorted({value for value in values if is_numeric_value(value)})
    if not ordered:
        return [best_value]

    best_numeric = float(best_value)
    best_index = min(range(len(ordered)), key=lambda index: abs(float(ordered[index]) - best_numeric))

    refined: list[object] = [ordered[best_index]]
    if best_index > 0:
        left_value = ordered[best_index - 1]
        refined.append(left_value)
        midpoint = compute_midpoint(left_value, ordered[best_index])
        if midpoint is not None:
            refined.append(midpoint)

    if best_index < len(ordered) - 1:
        right_value = ordered[best_index + 1]
        refined.append(right_value)
        midpoint = compute_midpoint(ordered[best_index], right_value)
        if midpoint is not None:
            refined.append(midpoint)

    return sorted(set(refined), key=float)


def select_best_grid_branch(
    param_grid: dict[str, list[object]] | list[dict[str, list[object]]],
    best_params: dict[str, object],
) -> dict[str, list[object]]:
    candidate_grids = normalize_grid_spec(param_grid)
    best_param_names = set(best_params.keys())
    for grid in candidate_grids:
        if set(grid.keys()) == best_param_names:
            return grid

    best_grid = candidate_grids[0]
    best_overlap = -1
    for grid in candidate_grids:
        overlap = len(best_param_names.intersection(grid.keys()))
        if overlap > best_overlap:
            best_overlap = overlap
            best_grid = grid
    return best_grid


def build_refined_param_grid(
    *,
    base_param_grid: dict[str, list[object]] | list[dict[str, list[object]]],
    best_params: dict[str, object],
) -> dict[str, list[object]]:
    active_grid = select_best_grid_branch(base_param_grid, best_params)
    refined_grid: dict[str, list[object]] = {}
    for param_name, values in active_grid.items():
        best_value = best_params.get(param_name, values[0])
        if all(is_numeric_value(value) for value in values):
            refined_values = build_refined_numeric_values(values, best_value)
            refined_grid[param_name] = refined_values
            continue

        refined_grid[param_name] = [best_value]

    return refined_grid


def has_expandable_values(param_grid: dict[str, list[object]]) -> bool:
    return any(len(values) > 1 for values in param_grid.values())


def build_cv_strategy(cv_folds: int, cv_repeats: int):
    if cv_repeats > 1:
        return RepeatedStratifiedKFold(
            n_splits=cv_folds,
            n_repeats=cv_repeats,
            random_state=RANDOM_STATE,
        )
    return StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)


def run_grid_search(
    *,
    model_name: str,
    param_grid: dict[str, list[object]] | list[dict[str, list[object]]],
    x_train,
    y_train: pd.Series,
    cv_folds: int,
    cv_repeats: int,
    scoring: str,
    n_jobs: int,
    search_verbose: int,
):
    cv = build_cv_strategy(cv_folds, cv_repeats)
    search = GridSearchCV(
        estimator=build_model(model_name),
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
        return_train_score=True,
        verbose=search_verbose,
    )
    started_at = perf_counter()
    search.fit(x_train, y_train)
    fit_time_seconds = perf_counter() - started_at

    cv_results_df = pd.DataFrame(search.cv_results_).sort_values(
        ["rank_test_score", "mean_test_score"],
        ascending=[True, False],
    )
    return search.best_estimator_, fit_time_seconds, search.best_score_, search.best_params_, cv_results_df


def tune_vectorization_models(
    *,
    vectorization_name: str = DEFAULT_VECTORIZATION,
    vectorization_output_dir: Path | str = VECTORIZATION_OUTPUT_DIR,
    output_dir: Path | str = MODEL_FINETUNING_OUTPUT_DIR,
    model_names: tuple[str, ...] = BASE_MODELS,
    cv_folds: int = DEFAULT_SEARCH_CV_FOLDS,
    cv_repeats: int = DEFAULT_DEEP_CV_REPEATS,
    scoring: str = DEFAULT_SCORING,
    n_jobs: int = -1,
    run_name: str | None = None,
    search_profile: str = DEFAULT_SEARCH_PROFILE,
    refine_rounds: int = DEFAULT_REFINE_ROUNDS,
    search_verbose: int = DEFAULT_SEARCH_VERBOSE,
) -> Path:
    ensure_pipeline_directories()
    if refine_rounds < 1:
        raise ValueError("refine_rounds must be at least 1.")

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

    log_progress(
        "Loaded vectorization "
        f"{vectorization_name} with {x_train.shape[0]} train rows, {x_test.shape[0]} test rows, "
        f"and {x_train.shape[1]} features."
    )
    log_progress(
        f"Running {len(model_names)} model(s) with profile={search_profile}, cv_folds={cv_folds}, "
        f"cv_repeats={cv_repeats}, refine_rounds={refine_rounds}, n_jobs={n_jobs}."
    )

    run_summaries: list[dict[str, object]] = []
    total_models = len(model_names)
    for model_index, model_name in enumerate(model_names, start=1):
        model_output_dir = output_dir / vectorization_name / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)
        param_grid = get_param_grid(model_name, search_profile)
        search_history: list[dict[str, object]] = []

        log_progress(
            f"Starting model {model_index}/{total_models}: {model_name}. "
            f"Initial grid size={estimate_grid_size(param_grid)}."
        )

        model = None
        fit_time_seconds = 0.0
        best_cv_score = None
        best_params: dict[str, object] = {}
        cv_results_df = None

        current_grid = param_grid
        for round_index in range(refine_rounds):
            round_name = "round_1" if round_index == 0 else f"round_{round_index + 1}"
            log_progress(
                f"{model_name} {round_name}: searching {estimate_grid_size(current_grid)} combinations "
                f"({estimate_grid_size(current_grid) * cv_folds * cv_repeats} CV fits)."
            )
            model, round_fit_time, best_cv_score, best_params, cv_results_df = run_grid_search(
                model_name=model_name,
                param_grid=current_grid,
                x_train=x_train,
                y_train=y_train,
                cv_folds=cv_folds,
                cv_repeats=cv_repeats,
                scoring=scoring,
                n_jobs=n_jobs,
                search_verbose=search_verbose,
            )
            fit_time_seconds += round_fit_time

            round_output_path = model_output_dir / f"cv_results_{round_name}.csv"
            save_cv_results(cv_results_df, round_output_path)
            search_history.append(
                {
                    "round": round_name,
                    "param_grid": current_grid,
                    "param_grid_size": estimate_grid_size(current_grid),
                    "best_cv_score": float(best_cv_score) if best_cv_score is not None else None,
                    "best_params": best_params,
                    "fit_time_seconds": round(round_fit_time, 4),
                }
            )
            log_progress(
                f"{model_name} {round_name} complete in {round(round_fit_time, 2)}s. "
                f"best_cv_score={best_cv_score:.6f}, best_params={json.dumps(best_params, sort_keys=True)}"
            )

            if round_index >= refine_rounds - 1:
                break

            refined_grid = build_refined_param_grid(
                base_param_grid=current_grid,
                best_params=best_params,
            )
            if not has_expandable_values(refined_grid):
                log_progress(f"{model_name}: refinement converged early after {round_name}.")
                break
            current_grid = refined_grid

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
        (model_output_dir / "search_history.json").write_text(
            json.dumps(search_history, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        log_progress(
            f"Finished {model_name}. test_macro_f1={summary['macro_f1']:.6f}, "
            f"test_accuracy={summary['accuracy']:.6f}."
        )
        run_summaries.append(summary)

    summary_df = pd.DataFrame(run_summaries).sort_values(
        ["macro_f1", "micro_f1", "accuracy"],
        ascending=[False, False, False],
    )
    summary_path = output_dir / vectorization_name / "metrics_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    log_progress(f"Saved tuning summary to: {summary_path.resolve()}")

    config_payload = {
        "vectorization": vectorization_name,
        "model_names": list(model_names),
        "cv_folds": cv_folds,
        "cv_repeats": cv_repeats,
        "scoring": scoring,
        "n_jobs": n_jobs,
        "run_name": run_name,
        "search_profile": search_profile,
        "refine_rounds": refine_rounds,
        "search_verbose": search_verbose,
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
        "--search-profile",
        default=DEFAULT_SEARCH_PROFILE,
        choices=SEARCH_PROFILES,
        help="Parameter grid density. Use deep for a broader first pass before refinement.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=DEFAULT_SEARCH_CV_FOLDS,
        help="Number of stratified folds to use during hyperparameter tuning.",
    )
    parser.add_argument(
        "--cv-repeats",
        type=int,
        default=DEFAULT_DEEP_CV_REPEATS,
        help="How many repeated cross-validation rounds to run for each grid-search round.",
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
        "--refine-rounds",
        type=int,
        default=DEFAULT_REFINE_ROUNDS,
        help="How many grid-search rounds to run. Later rounds refine around the previous best params.",
    )
    parser.add_argument(
        "--search-verbose",
        type=int,
        default=DEFAULT_SEARCH_VERBOSE,
        help="Verbosity level passed to GridSearchCV. Use 0 for quieter output.",
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
        cv_repeats=arguments.cv_repeats,
        scoring=arguments.scoring,
        n_jobs=arguments.n_jobs,
        run_name=arguments.run_name,
        search_profile=arguments.search_profile,
        refine_rounds=arguments.refine_rounds,
        search_verbose=arguments.search_verbose,
    )
    print(f"Saved fine-tuning summary to: {summary_path.resolve()}")
