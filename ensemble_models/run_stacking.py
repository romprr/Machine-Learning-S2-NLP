import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import joblib
import pandas as pd
from scipy.sparse import csr_matrix, hstack, load_npz
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

try:
    from .pipeline_paths import RUNS_OUTPUT_DIR, ensure_output_directories
except ImportError:
    from pipeline_paths import RUNS_OUTPUT_DIR, ensure_output_directories

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model_1.train_models import RANDOM_STATE, build_model_with_params


DEFAULT_MODEL_1_MODEL = "sgd_classifier"
DEFAULT_MODEL_1_VECTORIZATION = "tfidf"
DEFAULT_MODEL_1_RUN = "tfidf_sgd_classifier"

DEFAULT_MODEL_2_MODEL = "ridge_classifier"
DEFAULT_MODEL_2_VECTORIZATION = "char_tfidf"
DEFAULT_MODEL_2_RUN = "char_tfidf_ridge_classifier_fast"

DEFAULT_MODEL_3_MODEL = "logistic_regression"
DEFAULT_MODEL_3_VECTORIZATION = "one_hot"


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
    x_train: csr_matrix
    x_test: csr_matrix
    y_train_df: pd.DataFrame
    y_test_df: pd.DataFrame
    best_params: dict[str, object]
    metrics: dict[str, object]


class SparseColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return x[:, self.start : self.end]


def load_branch_artifacts(vectorization_dir: Path) -> tuple[csr_matrix, csr_matrix, pd.DataFrame, pd.DataFrame]:
    x_train = load_npz(vectorization_dir / "X_train.npz").tocsr()
    x_test = load_npz(vectorization_dir / "X_test.npz").tocsr()
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
    reference_train = branches[0].y_train_df
    reference_test = branches[0].y_test_df

    for branch in branches[1:]:
        for split_name, left_frame, right_frame in (
            ("train", reference_train, branch.y_train_df),
            ("test", reference_test, branch.y_test_df),
        ):
            if list(left_frame["row_id"]) != list(right_frame["row_id"]):
                raise ValueError(
                    f"Row alignment mismatch between {branches[0].config.display_name} and "
                    f"{branch.config.display_name} on {split_name} split."
                )
            if list(left_frame["tags"]) != list(right_frame["tags"]):
                raise ValueError(
                    f"Label mismatch between {branches[0].config.display_name} and "
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


def make_branch_config(arguments: argparse.Namespace, branch_number: int) -> BranchConfig:
    prefix = f"model_{branch_number}"
    return BranchConfig(
        branch_id=f"model_{branch_number}",
        display_name=f"Model {branch_number}",
        model_name=getattr(arguments, f"{prefix}_name"),
        vectorization_name=getattr(arguments, f"{prefix}_vectorization"),
        vectorization_dir=getattr(arguments, f"{prefix}_vectorization_dir"),
        artifact_base_dir=getattr(arguments, f"{prefix}_artifact_dir"),
        artifact_run_name=getattr(arguments, f"{prefix}_run_name"),
    )


def load_branch(config: BranchConfig) -> BranchArtifacts:
    x_train, x_test, y_train_df, y_test_df = load_branch_artifacts(config.vectorization_dir)
    artifact_root = resolve_artifact_root(config.artifact_base_dir, config.artifact_run_name)
    best_params, metrics = load_branch_metadata(artifact_root, config.model_name)
    return BranchArtifacts(
        config=config,
        x_train=x_train,
        x_test=x_test,
        y_train_df=y_train_df,
        y_test_df=y_test_df,
        best_params=best_params,
        metrics=metrics,
    )


def build_branch_estimator(model_name: str, best_params: dict[str, object], start: int, end: int, calibration_cv: int):
    base_estimator = build_model_with_params(model_name, best_params)
    estimator = base_estimator
    if not hasattr(base_estimator, "predict_proba"):
        estimator = CalibratedClassifierCV(estimator=base_estimator, method="sigmoid", cv=calibration_cv)

    return Pipeline(
        steps=[
            ("select_features", SparseColumnSelector(start=start, end=end)),
            ("classifier", estimator),
        ]
    )


def build_combined_matrices(branches: list[BranchArtifacts]) -> tuple[csr_matrix, csr_matrix, dict[str, tuple[int, int]]]:
    x_train = hstack([branch.x_train for branch in branches], format="csr")
    x_test = hstack([branch.x_test for branch in branches], format="csr")

    spans: dict[str, tuple[int, int]] = {}
    cursor = 0
    for branch in branches:
        width = branch.x_train.shape[1]
        spans[branch.config.branch_id] = (cursor, cursor + width)
        cursor += width

    return x_train, x_test, spans


def build_stacking_estimators(
    branches: list[BranchArtifacts],
    spans: dict[str, tuple[int, int]],
    calibration_cv: int,
) -> list[tuple[str, Pipeline]]:
    estimators: list[tuple[str, Pipeline]] = []
    for branch in branches:
        start, end = spans[branch.config.branch_id]
        estimators.append(
            (
                branch.config.branch_id,
                build_branch_estimator(
                    model_name=branch.config.model_name,
                    best_params=branch.best_params,
                    start=start,
                    end=end,
                    calibration_cv=calibration_cv,
                ),
            )
        )
    return estimators


def build_stacking_models(
    base_estimators: list[tuple[str, Pipeline]],
    cv_folds: int,
) -> dict[str, StackingClassifier]:
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    final_estimator = LogisticRegression(
        max_iter=4000,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )

    return {
        "stacking_classifier": StackingClassifier(
            estimators=base_estimators,
            final_estimator=final_estimator,
            stack_method="predict_proba",
            cv=cv,
            n_jobs=-1,
        ),
        "stacking_classifier_passthrough": StackingClassifier(
            estimators=base_estimators,
            final_estimator=final_estimator,
            stack_method="predict_proba",
            passthrough=True,
            cv=cv,
            n_jobs=-1,
        ),
    }


def save_variant_outputs(
    *,
    output_dir: Path,
    variant_name: str,
    estimator,
    y_true: pd.Series,
    y_pred: pd.Series,
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

    joblib.dump(estimator, variant_dir / "model.joblib")

    metrics = evaluate_predictions(y_true, y_pred, labels)
    payload = {**summary_payload, **metrics}
    (variant_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def benchmark_stacking_models(
    *,
    output_dir: Path,
    models: dict[str, StackingClassifier],
    x_train: csr_matrix,
    x_test: csr_matrix,
    y_train: pd.Series,
    y_test: pd.Series,
    labels: list[str],
    metadata: dict[str, object],
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for variant_name, estimator in models.items():
        print(f"Fitting {variant_name}...")
        fit_started_at = perf_counter()
        estimator.fit(x_train, y_train)
        fit_time_seconds = perf_counter() - fit_started_at

        predict_started_at = perf_counter()
        predictions = pd.Series(estimator.predict(x_test))
        predict_time_seconds = perf_counter() - predict_started_at

        summary_payload = {
            **metadata,
            "variant": variant_name,
            "fit_time_seconds": round(fit_time_seconds, 4),
            "predict_time_seconds": round(predict_time_seconds, 4),
        }
        results.append(
            save_variant_outputs(
                output_dir=output_dir,
                variant_name=variant_name,
                estimator=estimator,
                y_true=y_test,
                y_pred=predictions,
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


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark StackingClassifier variants on the three model branches."
    )
    parser.add_argument("--output-dir", type=Path, default=RUNS_OUTPUT_DIR, help="Directory where stacking artifacts will be saved.")
    parser.add_argument("--run-name", default="model_1_model_2_model_3_stacking", help="Subdirectory name for this stacking benchmark run.")
    parser.add_argument("--cv-folds", type=int, default=3, help="Number of stratified folds used by StackingClassifier.")
    parser.add_argument(
        "--calibration-cv",
        type=int,
        default=3,
        help="Calibration folds for base estimators that do not expose predict_proba.",
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

    branches = [load_branch(make_branch_config(arguments, index)) for index in (1, 2, 3)]
    y_train, y_test = verify_aligned_labels(branches)
    labels = sorted(y_test.unique().tolist())

    x_train, x_test, spans = build_combined_matrices(branches)
    base_estimators = build_stacking_estimators(
        branches=branches,
        spans=spans,
        calibration_cv=arguments.calibration_cv,
    )
    models = build_stacking_models(base_estimators=base_estimators, cv_folds=arguments.cv_folds)

    output_dir = arguments.output_dir / arguments.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    branch_metadata = []
    for branch in branches:
        start, end = spans[branch.config.branch_id]
        branch_metadata.append(
            {
                "branch_id": branch.config.branch_id,
                "display_name": branch.config.display_name,
                "model_name": branch.config.model_name,
                "vectorization": branch.config.vectorization_name,
                "artifact_run_name": branch.config.artifact_run_name,
                "best_params": branch.best_params,
                "macro_f1": float(branch.metrics["macro_f1"]),
                "feature_span": [start, end],
                "feature_count": end - start,
            }
        )

    metadata = {
        "branch_count": len(branches),
        "combined_train_shape": [int(x_train.shape[0]), int(x_train.shape[1])],
        "combined_test_shape": [int(x_test.shape[0]), int(x_test.shape[1])],
        "cv_folds": arguments.cv_folds,
        "calibration_cv": arguments.calibration_cv,
        "stack_method": "predict_proba",
        "final_estimator": {
            "type": "LogisticRegression",
            "max_iter": 4000,
            "random_state": RANDOM_STATE,
            "class_weight": "balanced",
        },
        "branches": branch_metadata,
    }
    (output_dir / "run_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    results = benchmark_stacking_models(
        output_dir=output_dir,
        models=models,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        labels=labels,
        metadata=metadata,
    )

    summary_df = pd.DataFrame(results).sort_values("macro_f1", ascending=False)
    summary_path = output_dir / "metrics_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved stacking metrics summary to: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
