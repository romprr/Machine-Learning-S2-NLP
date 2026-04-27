import argparse
import json
import sys
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

DEFAULT_VARIANTS = (
    "hard_tiebreak_model_1",
    "hard_tiebreak_model_2",
    "hard_weighted",
    "soft_score_equal",
    "soft_score_weighted",
    "soft_calibrated_equal",
    "soft_calibrated_weighted",
)


def load_branch_artifacts(vectorization_dir: Path) -> tuple:
    x_train = load_npz(vectorization_dir / "X_train.npz")
    x_test = load_npz(vectorization_dir / "X_test.npz")
    y_train = pd.read_csv(vectorization_dir / "y_train.csv", keep_default_na=False)
    y_test = pd.read_csv(vectorization_dir / "y_test.csv", keep_default_na=False)
    return x_train, x_test, y_train, y_test


def resolve_branch_run_dir(base_dir: Path, run_name: str) -> Path:
    run_dir = base_dir / run_name
    if not run_dir.exists():
        raise FileNotFoundError(f"Could not find run directory: {run_dir}")
    return run_dir


def find_model_file(run_dir: Path, model_name: str, filename: str) -> Path:
    matches = [path for path in run_dir.rglob(filename) if path.parent.name == model_name]
    if not matches:
        raise FileNotFoundError(f"Could not find {filename} for model {model_name} under {run_dir}")
    if len(matches) > 1:
        matches = sorted(matches, key=lambda path: len(path.parts))
    return matches[0]


def load_branch_metadata(run_dir: Path, model_name: str) -> tuple[dict[str, object], dict[str, object]]:
    best_params_path = find_model_file(run_dir, model_name, "best_params.json")
    metrics_path = find_model_file(run_dir, model_name, "metrics.json")
    best_params = json.loads(best_params_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return best_params, metrics


def verify_aligned_labels(
    model_1_y_train: pd.DataFrame,
    model_1_y_test: pd.DataFrame,
    model_2_y_train: pd.DataFrame,
    model_2_y_test: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    for split_name, left_frame, right_frame in (
        ("train", model_1_y_train, model_2_y_train),
        ("test", model_1_y_test, model_2_y_test),
    ):
        if list(left_frame["row_id"]) != list(right_frame["row_id"]):
            raise ValueError(f"Row alignment mismatch between model branches on {split_name} split.")
        if list(left_frame["tags"]) != list(right_frame["tags"]):
            raise ValueError(f"Label mismatch between model branches on {split_name} split.")

    return model_1_y_train["tags"].astype(str), model_1_y_test["tags"].astype(str)


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


def vote_hard(
    predictions_model_1: np.ndarray,
    predictions_model_2: np.ndarray,
    model_1_weight: float,
    model_2_weight: float,
    tiebreak: str,
) -> np.ndarray:
    ensembled: list[str] = []
    for left_label, right_label in zip(predictions_model_1, predictions_model_2):
        if left_label == right_label:
            ensembled.append(left_label)
            continue

        if tiebreak == "model_1":
            ensembled.append(left_label)
            continue

        if tiebreak == "model_2":
            ensembled.append(right_label)
            continue

        if model_1_weight >= model_2_weight:
            ensembled.append(left_label)
        else:
            ensembled.append(right_label)

    return np.asarray(ensembled)


def vote_soft(
    scores_model_1: np.ndarray,
    scores_model_2: np.ndarray,
    label_order: list[str],
    model_1_weight: float,
    model_2_weight: float,
) -> np.ndarray:
    combined = (scores_model_1 * model_1_weight) + (scores_model_2 * model_2_weight)
    predicted_indices = combined.argmax(axis=1)
    return np.asarray([label_order[index] for index in predicted_indices])


def normalize_weights(model_1_weight: float, model_2_weight: float) -> tuple[float, float]:
    total = model_1_weight + model_2_weight
    if total <= 0:
        return 0.5, 0.5
    return model_1_weight / total, model_2_weight / total


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
    model_1_estimator,
    model_2_estimator,
    model_1_calibrated,
    model_2_calibrated,
    x_test_model_1,
    x_test_model_2,
    y_test: pd.Series,
    labels: list[str],
    model_1_weight: float,
    model_2_weight: float,
    metadata: dict[str, object],
) -> list[dict[str, object]]:
    normalized_weight_1, normalized_weight_2 = normalize_weights(model_1_weight, model_2_weight)

    predictions_model_1 = np.asarray(model_1_estimator.predict(x_test_model_1))
    predictions_model_2 = np.asarray(model_2_estimator.predict(x_test_model_2))

    score_probs_model_1 = predict_soft_scores(model_1_estimator, x_test_model_1, labels)
    score_probs_model_2 = predict_soft_scores(model_2_estimator, x_test_model_2, labels)

    calibrated_probs_model_1 = None
    calibrated_probs_model_2 = None
    if model_1_calibrated is not None and model_2_calibrated is not None:
        calibrated_probs_model_1 = predict_soft_scores(model_1_calibrated, x_test_model_1, labels)
        calibrated_probs_model_2 = predict_soft_scores(model_2_calibrated, x_test_model_2, labels)

    results: list[dict[str, object]] = []
    for variant_name in variant_names:
        started_at = perf_counter()

        if variant_name == "hard_tiebreak_model_1":
            y_pred = vote_hard(predictions_model_1, predictions_model_2, normalized_weight_1, normalized_weight_2, "model_1")
        elif variant_name == "hard_tiebreak_model_2":
            y_pred = vote_hard(predictions_model_1, predictions_model_2, normalized_weight_1, normalized_weight_2, "model_2")
        elif variant_name == "hard_weighted":
            y_pred = vote_hard(predictions_model_1, predictions_model_2, normalized_weight_1, normalized_weight_2, "weighted")
        elif variant_name == "soft_score_equal":
            y_pred = vote_soft(score_probs_model_1, score_probs_model_2, labels, 0.5, 0.5)
        elif variant_name == "soft_score_weighted":
            y_pred = vote_soft(
                score_probs_model_1,
                score_probs_model_2,
                labels,
                normalized_weight_1,
                normalized_weight_2,
            )
        elif variant_name == "soft_calibrated_equal":
            if calibrated_probs_model_1 is None or calibrated_probs_model_2 is None:
                raise ValueError("Calibrated soft voting requested without calibrated estimators.")
            y_pred = vote_soft(calibrated_probs_model_1, calibrated_probs_model_2, labels, 0.5, 0.5)
        elif variant_name == "soft_calibrated_weighted":
            if calibrated_probs_model_1 is None or calibrated_probs_model_2 is None:
                raise ValueError("Calibrated soft voting requested without calibrated estimators.")
            y_pred = vote_soft(
                calibrated_probs_model_1,
                calibrated_probs_model_2,
                labels,
                normalized_weight_1,
                normalized_weight_2,
            )
        else:
            supported = ", ".join(DEFAULT_VARIANTS)
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


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark ensemble variants that combine the tuned Model 1 and Model 2 branches."
    )
    parser.add_argument("--output-dir", type=Path, default=RUNS_OUTPUT_DIR, help="Directory where ensemble artifacts will be saved.")
    parser.add_argument("--run-name", default="model_1_model_2_ensemble", help="Subdirectory name for this ensemble benchmark run.")
    parser.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS), choices=DEFAULT_VARIANTS, help="Ensemble variants to benchmark.")

    parser.add_argument("--model-1-vectorization-dir", type=Path, default=Path("model_1/outputs/pipeline/model_1_vectorized/tfidf"))
    parser.add_argument("--model-1-run-dir", type=Path, default=Path("model_1/outputs/pipeline/model_1_finetuning"))
    parser.add_argument("--model-1-run-name", default=DEFAULT_MODEL_1_RUN)
    parser.add_argument("--model-1-name", default=DEFAULT_MODEL_1_MODEL)
    parser.add_argument("--model-1-vectorization", default=DEFAULT_MODEL_1_VECTORIZATION)

    parser.add_argument("--model-2-vectorization-dir", type=Path, default=Path("model_2/outputs/pipeline/model_2_vectorized/char_tfidf"))
    parser.add_argument("--model-2-run-dir", type=Path, default=Path("model_2/outputs/pipeline/model_2_finetuning"))
    parser.add_argument("--model-2-run-name", default=DEFAULT_MODEL_2_RUN)
    parser.add_argument("--model-2-name", default=DEFAULT_MODEL_2_MODEL)
    parser.add_argument("--model-2-vectorization", default=DEFAULT_MODEL_2_VECTORIZATION)
    return parser


def main() -> None:
    parser = build_argument_parser()
    arguments = parser.parse_args()
    ensure_output_directories()

    model_1_x_train, model_1_x_test, model_1_y_train_df, model_1_y_test_df = load_branch_artifacts(arguments.model_1_vectorization_dir)
    model_2_x_train, model_2_x_test, model_2_y_train_df, model_2_y_test_df = load_branch_artifacts(arguments.model_2_vectorization_dir)
    y_train, y_test = verify_aligned_labels(model_1_y_train_df, model_1_y_test_df, model_2_y_train_df, model_2_y_test_df)
    labels = sorted(y_test.unique().tolist())

    model_1_run_dir = resolve_branch_run_dir(arguments.model_1_run_dir, arguments.model_1_run_name)
    model_2_run_dir = resolve_branch_run_dir(arguments.model_2_run_dir, arguments.model_2_run_name)
    model_1_best_params, model_1_metrics = load_branch_metadata(model_1_run_dir, arguments.model_1_name)
    model_2_best_params, model_2_metrics = load_branch_metadata(model_2_run_dir, arguments.model_2_name)

    print(
        f"Fitting model 1 ({arguments.model_1_name}, {arguments.model_1_vectorization}) "
        f"with best params from {model_1_run_dir}."
    )
    model_1_estimator, model_1_fit_time = fit_estimator(arguments.model_1_name, model_1_best_params, model_1_x_train, y_train)

    print(
        f"Fitting model 2 ({arguments.model_2_name}, {arguments.model_2_vectorization}) "
        f"with best params from {model_2_run_dir}."
    )
    model_2_estimator, model_2_fit_time = fit_estimator(arguments.model_2_name, model_2_best_params, model_2_x_train, y_train)

    requested_variants = tuple(arguments.variants)
    needs_calibration = any("soft_calibrated" in variant for variant in requested_variants)
    model_1_calibrated = None
    model_2_calibrated = None
    calibration_fit_time_1 = 0.0
    calibration_fit_time_2 = 0.0

    if needs_calibration:
        print(f"Calibrating model 1 ({arguments.model_1_name}) for calibrated soft voting.")
        model_1_calibrated, calibration_fit_time_1 = fit_calibrated_estimator(
            arguments.model_1_name, model_1_best_params, model_1_x_train, y_train
        )
        print(f"Calibrating model 2 ({arguments.model_2_name}) for calibrated soft voting.")
        model_2_calibrated, calibration_fit_time_2 = fit_calibrated_estimator(
            arguments.model_2_name, model_2_best_params, model_2_x_train, y_train
        )

    output_dir = arguments.output_dir / arguments.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_metadata = {
        "model_1_name": arguments.model_1_name,
        "model_1_vectorization": arguments.model_1_vectorization,
        "model_1_run_name": arguments.model_1_run_name,
        "model_1_best_params": model_1_best_params,
        "model_1_macro_f1": float(model_1_metrics["macro_f1"]),
        "model_1_fit_time_seconds": round(model_1_fit_time, 4),
        "model_1_calibration_fit_time_seconds": round(calibration_fit_time_1, 4),
        "model_2_name": arguments.model_2_name,
        "model_2_vectorization": arguments.model_2_vectorization,
        "model_2_run_name": arguments.model_2_run_name,
        "model_2_best_params": model_2_best_params,
        "model_2_macro_f1": float(model_2_metrics["macro_f1"]),
        "model_2_fit_time_seconds": round(model_2_fit_time, 4),
        "model_2_calibration_fit_time_seconds": round(calibration_fit_time_2, 4),
    }
    (output_dir / "run_config.json").write_text(json.dumps(summary_metadata, indent=2), encoding="utf-8")

    results = benchmark_ensemble_variants(
        output_dir=output_dir,
        variant_names=requested_variants,
        model_1_estimator=model_1_estimator,
        model_2_estimator=model_2_estimator,
        model_1_calibrated=model_1_calibrated,
        model_2_calibrated=model_2_calibrated,
        x_test_model_1=model_1_x_test,
        x_test_model_2=model_2_x_test,
        y_test=y_test,
        labels=labels,
        model_1_weight=float(model_1_metrics["macro_f1"]),
        model_2_weight=float(model_2_metrics["macro_f1"]),
        metadata=summary_metadata,
    )

    summary_df = pd.DataFrame(results).sort_values("macro_f1", ascending=False)
    summary_path = output_dir / "metrics_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved ensemble metrics summary to: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
