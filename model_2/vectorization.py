import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from scipy.sparse import save_npz

try:
    from .pipeline_paths import (
        TRAIN_SPLIT_PATH,
        TEST_SPLIT_PATH,
        VECTORIZATION_OUTPUT_DIR,
        ensure_pipeline_directories,
    )
    from .preprocessing import build_char_tfidf_vectorizer
except ImportError:
    from pipeline_paths import (
        TRAIN_SPLIT_PATH,
        TEST_SPLIT_PATH,
        VECTORIZATION_OUTPUT_DIR,
        ensure_pipeline_directories,
    )
    from preprocessing import build_char_tfidf_vectorizer


SUPPORTED_VECTORIZATIONS = ("char_tfidf",)


def build_vectorizer(vectorization_name: str):
    if vectorization_name == "char_tfidf":
        return build_char_tfidf_vectorizer()

    supported = ", ".join(SUPPORTED_VECTORIZATIONS)
    raise ValueError(f"Unsupported vectorization: {vectorization_name}. Expected one of: {supported}")


def save_labels(dataframe: pd.DataFrame, output_path: Path) -> None:
    dataframe.loc[:, ["row_id", "tags"]].to_csv(output_path, index=False)


def vectorize_split(
    train_path: Path | str = TRAIN_SPLIT_PATH,
    test_path: Path | str = TEST_SPLIT_PATH,
    output_dir: Path | str = VECTORIZATION_OUTPUT_DIR,
    vectorization_names: tuple[str, ...] = SUPPORTED_VECTORIZATIONS,
) -> list[Path]:
    ensure_pipeline_directories()

    train_path = Path(train_path)
    test_path = Path(test_path)
    output_dir = Path(output_dir)

    if not train_path.exists():
        raise FileNotFoundError(f"Could not find train split: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Could not find test split: {test_path}")

    train_df = pd.read_csv(train_path, keep_default_na=False)
    test_df = pd.read_csv(test_path, keep_default_na=False)

    required_columns = {"row_id", "preprocessed_post", "tags"}
    for frame_name, dataframe in (("train", train_df), ("test", test_df)):
        missing_columns = required_columns.difference(dataframe.columns)
        if missing_columns:
            missing_text = ", ".join(sorted(missing_columns))
            raise ValueError(f"Missing required columns in {frame_name} split: {missing_text}")

    saved_directories: list[Path] = []
    for vectorization_name in vectorization_names:
        vectorizer = build_vectorizer(vectorization_name)
        vector_output_dir = output_dir / vectorization_name
        vector_output_dir.mkdir(parents=True, exist_ok=True)

        x_train = vectorizer.fit_transform(train_df["preprocessed_post"].astype(str))
        x_test = vectorizer.transform(test_df["preprocessed_post"].astype(str))

        save_npz(vector_output_dir / "X_train.npz", x_train)
        save_npz(vector_output_dir / "X_test.npz", x_test)
        joblib.dump(vectorizer, vector_output_dir / "vectorizer.joblib")
        save_labels(train_df, vector_output_dir / "y_train.csv")
        save_labels(test_df, vector_output_dir / "y_test.csv")

        metadata = {
            "vectorization": vectorization_name,
            "train_rows": int(x_train.shape[0]),
            "test_rows": int(x_test.shape[0]),
            "feature_count": int(x_train.shape[1]),
            "train_path": str(train_path.resolve()),
            "test_path": str(test_path.resolve()),
        }
        (vector_output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        saved_directories.append(vector_output_dir)

    return saved_directories


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit vectorizers on the training split and save train/test sparse matrices."
    )
    parser.add_argument("--train-input", type=Path, default=TRAIN_SPLIT_PATH, help="Path to the train CSV split.")
    parser.add_argument("--test-input", type=Path, default=TEST_SPLIT_PATH, help="Path to the test CSV split.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=VECTORIZATION_OUTPUT_DIR,
        help="Base directory where vectorization artifacts will be saved.",
    )
    parser.add_argument(
        "--vectorizations",
        nargs="+",
        default=list(SUPPORTED_VECTORIZATIONS),
        choices=SUPPORTED_VECTORIZATIONS,
        help="Vectorization variants to build.",
    )
    return parser


if __name__ == "__main__":
    parser = build_argument_parser()
    arguments = parser.parse_args()

    output_directories = vectorize_split(
        train_path=arguments.train_input,
        test_path=arguments.test_input,
        output_dir=arguments.output_dir,
        vectorization_names=tuple(arguments.vectorizations),
    )
    for output_directory in output_directories:
        print(f"Saved vectorization artifacts to: {output_directory.resolve()}")
