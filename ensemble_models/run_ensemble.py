import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

try:
    from .pipeline_paths import RUNS_OUTPUT_DIR, ensure_output_directories
except ImportError:
    from pipeline_paths import RUNS_OUTPUT_DIR, ensure_output_directories

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model_1.train_models import build_model_with_params


DEFAULT_MODEL_1_MODEL = "sgd_classifier"
DEFAULT_MODEL_1_VECTORIZATION = "tfidf"
DEFAULT_MODEL_1_RUN = "tfidf_sgd_classifier"

DEFAULT_MODEL_2_MODEL = "ridge_classifier"
DEFAULT_MODEL_2_VECTORIZATION = "char_tfidf"
DEFAULT_MODEL_2_RUN = "char_tfidf_ridge_classifier_fast"

DEFAULT_MODEL_3_MODEL = "logistic_regression"
DEFAULT_MODEL_3_VECTORIZATION = "one_hot"

GENERIC_VARIANTS = (
    "hard_majority",
    "hard_weighted",
    "soft_score_equal",
    "soft_score_weighted",
    "soft_calibrated_equal",
    "soft_calibrated_weighted",
)
LEGACY_TWO_MODEL_VARIANTS = (
    "hard_tiebreak_model_1",
    "hard_tiebreak_model_2",
)
SUPPORTED_VARIANTS = LEGACY_TWO_MODEL_VARIANTS + GENERIC_VARIANTS


@dataclass(frozen=True)
class BranchConfig:
    branch_id: str
    display_name: str
    model_name: str
    vectorization_name: str
    vectorization_dir: Path
    artifact_base_dir: Path
    artifact_run_name: str | None


@dataclass
class BranchArtifacts:
    config: BranchConfig
    x_train: object
    x_test: object
    y_train_df: pd.DataFrame
    y_test_df: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    best_params: dict[str, object]
    metrics: dict[str, object]
    estimator: object | None = None
    calibrated_estimator: object | None = None
    fit_time_seconds: float = 0.0
    calibration_fit_time_seconds: float = 0.0


def load_branch_artifacts(vectorization_dir: Path) -> tuple:
    x_train = load_npz(vectorization_dir / "X_train.npz")
    x_test = load_npz(vectorization_dir / "X_test.npz")
    y_train = pd.read_csv(vectorization_dir / "y_train.csv", keep_default_na=False)
    y_test = pd.read_csv(vectorization_dir / "y_test.csv", keep_default_na=False)
    return x_train, x_test, y_train, y_test


def resolve_artifact_root(base_dir: Path, run_name: str | None) -> Path:
    run_dir = base_dir / run_name if run_name else base_dir
    if not run_dir.exists():
        raise FileNotFoundError(f"Could not find artifact directory: {run_dir}")
    return run_dir


def find_model_file(run_dir: Path, model_name: str, filename: str) -> Path:
    matches = [path for path in run_dir.rglob(filename) if path.parent.name == model_name]
    if not matches:
        raise FileNotFoundError(f"Could not find {filename} for model {model_name} under {run_dir}")
    if len(matches) > 1:
        matches = sorted(matches, key=lambda path: len(path.parts))
    return matches[0]


def load_branch_metadata(run_dir: Path, model_name: str) -> tuple[dict[str, object], dict[str, object]]:
    best_params: dict[str, object] = {}
    try:
        best_params_path = find_model_file(run_dir, model_name, "best_params.json")
    except FileNotFoundError:
        best_params_path = None

    metrics_path = find_model_file(run_dir, model_name, "metrics.json")
    if best_params_path is not None:
        best_params = json.loads(best_params_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return best_params, metrics


def verify_aligned_labels(branches: list[BranchArtifacts]) -> tuple[pd.Series, pd.Series]:
    reference = branches[0]
    reference_train = reference.y_train_df
    reference_test = reference.y_test_df

    for branch in branches[1:]:
        for split_name, left_frame, right_frame in (
            ("train", reference_train, branch.y_train_df),
            ("test", reference_test, branch.y_test_df),
        ):
            if list(left_frame["row_id"]) != list(right_frame["row_id"]):
                raise ValueError(
                    f"Row alignment mismatch between {reference.config.display_name} and "
                    f"{branch.config.display_name} on {split_name} split."
                )
            if list(left_frame["tags"]) != list(right_frame["tags"]):
                raise ValueError(
                    f"Label mismatch between {reference.config.display_name} and "
                    f"{branch.config.display_name} on {split_name} split."
                )

    return reference_train["tags"].astype(str), reference_test["tags"].astype(str)


def evaluate_predictions(y_true: pd.Series, y_pred: pd.Series, labels: list[str]) -> dict[str, float]:
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="micro", zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
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
    }


def fit_estimator(model_name: str, best_params: dict[str, object], x_train, y_train: pd.Series):
    estimator = build_model_with_params(model_name, best_params)
    started_at = perf_counter()
    estimator.fit(x_train, y_train)
    fit_time_seconds = perf_counter() - started_at
    return estimator, fit_time_seconds


def fit_calibrated_estimator(model_name: str, best_params: dict[str, object], x_train, y_train: pd.Series):
    base_estimator = build_model_with_params(model_name, best_params)
    calibrated = CalibratedClassifierCV(estimator=base_estimator, method="sigmoid", cv=3)
    started_at = perf_counter()
    calibrated.fit(x_train, y_train)
    fit_time_seconds = perf_counter() - started_at
    return calibrated, fit_time_seconds


def ensure_2d_scores(scores: np.ndarray) -> np.ndarray:
    if scores.ndim == 1:
        return np.column_stack([-scores, scores])
    return scores


def softmax(scores: np.ndarray) -> np.ndarray:
    stabilized = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(stabilized)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def align_matrix(matrix: np.ndarray, estimator_classes: np.ndarray, label_order: list[str]) -> np.ndarray:
    aligned = np.zeros((matrix.shape[0], len(label_order)), dtype=float)
    class_to_index = {label: index for index, label in enumerate(estimator_classes)}
    for target_index, label in enumerate(label_order):
        aligned[:, target_index] = matrix[:, class_to_index[label]]
    return aligned


def predict_soft_scores(estimator, x_test, label_order: list[str]) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(x_test)
        return align_matrix(probabilities, estimator.classes_, label_order)

    if not hasattr(estimator, "decision_function"):
        raise ValueError(f"Estimator {type(estimator).__name__} does not support soft voting inputs.")

    decision_scores = ensure_2d_scores(np.asarray(estimator.decision_function(x_test)))
    score_probabilities = softmax(decision_scores)
    return align_matrix(score_probabilities, estimator.classes_, label_order)


def normalize_weights(weights: list[float]) -> list[float]:
    total = sum(weights)
    if total <= 0:
        return [1.0 / len(weights)] * len(weights)
    return [weight / total for weight in weights]


def choose_label_from_candidates(
    candidate_indices: list[int],
    label_scores: dict[str, float],
    branch_predictions: list[str],
    branch_weights: list[float],
) -> str:
    candidate_labels = [branch_predictions[index] for index in candidate_indices]
    best_score = max(label_scores[label] for label in candidate_labels)
    best_labels = [label for label in candidate_labels if label_scores[label] == best_score]
    if len(best_labels) == 1:
        return best_labels[0]

    best_branch_index = max(candidate_indices, key=lambda index: branch_weights[index])
    if branch_predictions[best_branch_index] in best_labels:
        return branch_predictions[best_branch_index]

    return best_labels[0]


def vote_hard_majority(predictions_by_branch: list[np.ndarray], branch_weights: list[float]) -> np.ndarray:
    ensembled: list[str] = []
    for sample_predictions in zip(*predictions_by_branch):
        label_counts: dict[str, int] = {}
        label_weight_totals: dict[str, float] = {}
        for index, label in enumerate(sample_predictions):
            label_counts[label] = label_counts.get(label, 0) + 1
            label_weight_totals[label] = label_weight_totals.get(label, 0.0) + branch_weights[index]

        max_votes = max(label_counts.values())
        majority_labels = [label for label, count in label_counts.items() if count == max_votes]
        if len(majority_labels) == 1:
            ensembled.append(majority_labels[0])
            continue

        candidate_indices = [index for index, label in enumerate(sample_predictions) if label in majority_labels]
        ensembled.append(
            choose_label_from_candidates(
                candidate_indices=candidate_indices,
                label_scores=label_weight_totals,
                branch_predictions=list(sample_predictions),
                branch_weights=branch_weights,
            )
        )

    return np.asarray(ensembled)


def vote_hard_weighted(predictions_by_branch: list[np.ndarray], branch_weights: list[float]) -> np.ndarray:
    ensembled: list[str] = []
    for sample_predictions in zip(*predictions_by_branch):
        label_weight_totals: dict[str, float] = {}
        for index, label in enumerate(sample_predictions):
            label_weight_totals[label] = label_weight_totals.get(label, 0.0) + branch_weights[index]

        max_weight = max(label_weight_totals.values())
        weighted_labels = [label for label, weight in label_weight_totals.items() if weight == max_weight]
        if len(weighted_labels) == 1:
            ensembled.append(weighted_labels[0])
            continue

        candidate_indices = [index for index, label in enumerate(sample_predictions) if label in weighted_labels]
        ensembled.append(
            choose_label_from_candidates(
                candidate_indices=candidate_indices,
                label_scores=label_weight_totals,
                branch_predictions=list(sample_predictions),
                branch_weights=branch_weights,
            )
        )

    return np.asarray(ensembled)


def vote_soft(
    score_matrices: list[np.ndarray],
    label_order: list[str],
    branch_weights: list[float],
) -> np.ndarray:
    combined = np.zeros_like(score_matrices[0], dtype=float)
    for weight, matrix in zip(branch_weights, score_matrices):
        combined += matrix * weight
    predicted_indices = combined.argmax(axis=1)
    return np.asarray([label_order[index] for index in predicted_indices])


def save_variant_outputs(
    *,
    output_dir: Path,
    variant_name: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    labels: list[str],
    summary_payload: dict[str, object],
) -> dict[str, object]:
    variant_dir = output_dir / variant_name
    variant_dir.mkdir(parents=True, exist_ok=True)

    predictions_df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    predictions_df.to_csv(variant_dir / "predictions.csv", index=False)

    confusion = confusion_matrix(y_true, y_pred, labels=labels)
    confusion_df = pd.DataFrame(confusion, index=labels, columns=labels)
    confusion_df.to_csv(variant_dir / "confusion_matrix.csv")

    metrics = evaluate_predictions(y_true, pd.Series(y_pred), labels)
    payload = {**summary_payload, **metrics}
    (variant_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def benchmark_ensemble_variants(
    *,
    output_dir: Path,
    variant_names: tuple[str, ...],
    branches: list[BranchArtifacts],
    y_test: pd.Series,
    labels: list[str],
    metadata: dict[str, object],
) -> list[dict[str, object]]:
    normalized_weights = normalize_weights([float(branch.metrics["macro_f1"]) for branch in branches])
    predictions_by_branch = [np.asarray(branch.estimator.predict(branch.x_test)) for branch in branches]
    score_probs_by_branch = [predict_soft_scores(branch.estimator, branch.x_test, labels) for branch in branches]

    has_calibrated_estimators = all(branch.calibrated_estimator is not None for branch in branches)
    calibrated_probs_by_branch = None
    if has_calibrated_estimators:
        calibrated_probs_by_branch = [
            predict_soft_scores(branch.calibrated_estimator, branch.x_test, labels) for branch in branches
        ]

    results: list[dict[str, object]] = []
    for variant_name in variant_names:
        started_at = perf_counter()

        if variant_name == "hard_majority":
            y_pred = vote_hard_majority(predictions_by_branch, normalized_weights)
        elif variant_name == "hard_weighted":
            y_pred = vote_hard_weighted(predictions_by_branch, normalized_weights)
        elif variant_name == "soft_score_equal":
            equal_weights = [1.0 / len(branches)] * len(branches)
            y_pred = vote_soft(score_probs_by_branch, labels, equal_weights)
        elif variant_name == "soft_score_weighted":
            y_pred = vote_soft(score_probs_by_branch, labels, normalized_weights)
        elif variant_name == "soft_calibrated_equal":
            if calibrated_probs_by_branch is None:
                raise ValueError("Calibrated soft voting requested without calibrated estimators.")
            equal_weights = [1.0 / len(branches)] * len(branches)
            y_pred = vote_soft(calibrated_probs_by_branch, labels, equal_weights)
        elif variant_name == "soft_calibrated_weighted":
            if calibrated_probs_by_branch is None:
                raise ValueError("Calibrated soft voting requested without calibrated estimators.")
            y_pred = vote_soft(calibrated_probs_by_branch, labels, normalized_weights)
        elif variant_name == "hard_tiebreak_model_1":
            if len(branches) != 2:
                raise ValueError("hard_tiebreak_model_1 is only supported for two-model ensembles.")
            y_pred = np.asarray([
                left if left != right else left
                for left, right in zip(predictions_by_branch[0], predictions_by_branch[1])
            ])
        elif variant_name == "hard_tiebreak_model_2":
            if len(branches) != 2:
                raise ValueError("hard_tiebreak_model_2 is only supported for two-model ensembles.")
            y_pred = np.asarray([
                right if left != right else left
                for left, right in zip(predictions_by_branch[0], predictions_by_branch[1])
            ])
        else:
            supported = ", ".join(SUPPORTED_VARIANTS)
            raise ValueError(f"Unsupported variant: {variant_name}. Expected one of: {supported}")

        predict_time_seconds = perf_counter() - started_at
        summary_payload = {
            **metadata,
            "variant": variant_name,
            "predict_time_seconds": round(predict_time_seconds, 4),
        }
        results.append(
            save_variant_outputs(
                output_dir=output_dir,
                variant_name=variant_name,
                y_true=y_test,
                y_pred=y_pred,
                labels=labels,
                summary_payload=summary_payload,
            )
        )

    return results


def add_branch_arguments(
    parser: argparse.ArgumentParser,
    *,
    branch_number: int,
    model_name: str,
    vectorization_name: str,
    vectorization_dir: str,
    artifact_dir: str,
    run_name: str | None,
) -> None:
    prefix = f"model-{branch_number}"
    parser.add_argument(f"--{prefix}-vectorization-dir", type=Path, default=Path(vectorization_dir))
    parser.add_argument(f"--{prefix}-artifact-dir", type=Path, default=Path(artifact_dir))
    parser.add_argument(f"--{prefix}-run-name", default=run_name)
    parser.add_argument(f"--{prefix}-name", default=model_name)
    parser.add_argument(f"--{prefix}-vectorization", default=vectorization_name)


def build_branch_config(arguments: argparse.Namespace, branch_number: int) -> BranchConfig:
    prefix = f"model_{branch_number}"
    branch_id = f"model_{branch_number}"
    return BranchConfig(
        branch_id=branch_id,
        display_name=f"Model {branch_number}",
        model_name=getattr(arguments, f"{prefix}_name"),
        vectorization_name=getattr(arguments, f"{prefix}_vectorization"),
        vectorization_dir=getattr(arguments, f"{prefix}_vectorization_dir"),
        artifact_base_dir=getattr(arguments, f"{prefix}_artifact_dir"),
        artifact_run_name=getattr(arguments, f"{prefix}_run_name"),
    )


def load_branch(branch_config: BranchConfig) -> BranchArtifacts:
    x_train, x_test, y_train_df, y_test_df = load_branch_artifacts(branch_config.vectorization_dir)
    artifact_root = resolve_artifact_root(branch_config.artifact_base_dir, branch_config.artifact_run_name)
    best_params, metrics = load_branch_metadata(artifact_root, branch_config.model_name)
    return BranchArtifacts(
        config=branch_config,
        x_train=x_train,
        x_test=x_test,
        y_train_df=y_train_df,
        y_test_df=y_test_df,
        y_train=y_train_df["tags"].astype(str),
        y_test=y_test_df["tags"].astype(str),
        best_params=best_params,
        metrics=metrics,
    )


def default_run_name(branches: list[BranchConfig]) -> str:
    return "_".join(branch.branch_id for branch in branches) + "_ensemble"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark ensemble variants that combine two or three trained model branches."
    )
    parser.add_argument("--output-dir", type=Path, default=RUNS_OUTPUT_DIR, help="Directory where ensemble artifacts will be saved.")
    parser.add_argument("--run-name", help="Optional subdirectory name for this ensemble benchmark run.")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(GENERIC_VARIANTS),
        choices=SUPPORTED_VARIANTS,
        help="Ensemble variants to benchmark.",
    )
    parser.add_argument(
        "--include-model-3",
        action="store_true",
        help="Include the third branch in the ensemble. Defaults to the one_hot logistic_regression baseline.",
    )

    add_branch_arguments(
        parser,
        branch_number=1,
        model_name=DEFAULT_MODEL_1_MODEL,
        vectorization_name=DEFAULT_MODEL_1_VECTORIZATION,
        vectorization_dir="model_1/outputs/pipeline/model_1_vectorized/tfidf",
        artifact_dir="model_1/outputs/pipeline/model_1_finetuning",
        run_name=DEFAULT_MODEL_1_RUN,
    )
    add_branch_arguments(
        parser,
        branch_number=2,
        model_name=DEFAULT_MODEL_2_MODEL,
        vectorization_name=DEFAULT_MODEL_2_VECTORIZATION,
        vectorization_dir="model_2/outputs/pipeline/model_2_vectorized/char_tfidf",
        artifact_dir="model_2/outputs/pipeline/model_2_finetuning",
        run_name=DEFAULT_MODEL_2_RUN,
    )
    add_branch_arguments(
        parser,
        branch_number=3,
        model_name=DEFAULT_MODEL_3_MODEL,
        vectorization_name=DEFAULT_MODEL_3_VECTORIZATION,
        vectorization_dir="model_1/outputs/pipeline/model_1_vectorized/one_hot",
        artifact_dir="model_1/outputs/pipeline/model_1_models/one_hot",
        run_name=None,
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    arguments = parser.parse_args()
    ensure_output_directories()

    selected_configs = [
        build_branch_config(arguments, 1),
        build_branch_config(arguments, 2),
    ]
    if arguments.include_model_3:
        selected_configs.append(build_branch_config(arguments, 3))

    if len(selected_configs) > 2:
        invalid_legacy_variants = [variant for variant in arguments.variants if variant in LEGACY_TWO_MODEL_VARIANTS]
        if invalid_legacy_variants:
            invalid_text = ", ".join(invalid_legacy_variants)
            raise ValueError(f"These variants only work for two-model ensembles: {invalid_text}")

    branches = [load_branch(config) for config in selected_configs]
    y_train, y_test = verify_aligned_labels(branches)
    labels = sorted(y_test.unique().tolist())

    for branch in branches:
        print(
            f"Fitting {branch.config.display_name} "
            f"({branch.config.model_name}, {branch.config.vectorization_name}) "
            f"with params from {resolve_artifact_root(branch.config.artifact_base_dir, branch.config.artifact_run_name)}."
        )
        branch.estimator, branch.fit_time_seconds = fit_estimator(
            branch.config.model_name,
            branch.best_params,
            branch.x_train,
            y_train,
        )

    requested_variants = tuple(arguments.variants)
    needs_calibration = any("soft_calibrated" in variant for variant in requested_variants)
    if needs_calibration:
        for branch in branches:
            print(f"Calibrating {branch.config.display_name} ({branch.config.model_name}) for calibrated soft voting.")
            branch.calibrated_estimator, branch.calibration_fit_time_seconds = fit_calibrated_estimator(
                branch.config.model_name,
                branch.best_params,
                branch.x_train,
                y_train,
            )

    output_dir = arguments.output_dir / (arguments.run_name or default_run_name(selected_configs))
    output_dir.mkdir(parents=True, exist_ok=True)

    branch_metadata = []
    for branch in branches:
        branch_metadata.append(
            {
                "branch_id": branch.config.branch_id,
                "display_name": branch.config.display_name,
                "model_name": branch.config.model_name,
                "vectorization": branch.config.vectorization_name,
                "artifact_run_name": branch.config.artifact_run_name,
                "best_params": branch.best_params,
                "macro_f1": float(branch.metrics["macro_f1"]),
                "fit_time_seconds": round(branch.fit_time_seconds, 4),
                "calibration_fit_time_seconds": round(branch.calibration_fit_time_seconds, 4),
            }
        )

    summary_metadata = {
        "branch_count": len(branches),
        "branches": branch_metadata,
    }
    (output_dir / "run_config.json").write_text(json.dumps(summary_metadata, indent=2), encoding="utf-8")

    results = benchmark_ensemble_variants(
        output_dir=output_dir,
        variant_names=requested_variants,
        branches=branches,
        y_test=y_test,
        labels=labels,
        metadata=summary_metadata,
    )

    summary_df = pd.DataFrame(results).sort_values("macro_f1", ascending=False)
    summary_path = output_dir / "metrics_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved ensemble metrics summary to: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
