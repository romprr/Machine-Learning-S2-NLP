import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from .pipeline_paths import (
        PREPROCESSED_DATA_PATH,
        SPLIT_SUMMARY_PATH,
        TEST_SPLIT_PATH,
        TRAIN_SPLIT_PATH,
        ensure_pipeline_directories,
    )
except ImportError:
    from pipeline_paths import (
        PREPROCESSED_DATA_PATH,
        SPLIT_SUMMARY_PATH,
        TEST_SPLIT_PATH,
        TRAIN_SPLIT_PATH,
        ensure_pipeline_directories,
    )


def build_balanced_split(
    df: pd.DataFrame,
    train_per_class: int = 1500,
    test_per_class: int = 500,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_columns = {"row_id", "preprocessed_post", "tags"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing_text}")

    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    for tag, tag_frame in df.groupby("tags", sort=True):
        shuffled = tag_frame.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        required_rows = train_per_class + test_per_class
        if len(shuffled) < required_rows:
            raise ValueError(
                f"Tag {tag!r} has only {len(shuffled)} rows, but {required_rows} rows are required "
                "for the requested split."
            )

        train_parts.append(shuffled.iloc[:train_per_class].copy())
        test_parts.append(shuffled.iloc[train_per_class : train_per_class + test_per_class].copy())

    train_df = pd.concat(train_parts, ignore_index=True).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    test_df = pd.concat(test_parts, ignore_index=True).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return train_df, test_df


def save_split_summary(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    summary_path: Path,
    source_path: Path,
    train_per_class: int,
    test_per_class: int,
    random_state: int,
) -> None:
    summary = {
        "source_path": str(source_path.resolve()),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_per_class": train_per_class,
        "test_per_class": test_per_class,
        "random_state": random_state,
        "train_distribution": train_df["tags"].value_counts().sort_index().to_dict(),
        "test_distribution": test_df["tags"].value_counts().sort_index().to_dict(),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def create_split_artifacts(
    input_path: Path | str = PREPROCESSED_DATA_PATH,
    train_output_path: Path | str = TRAIN_SPLIT_PATH,
    test_output_path: Path | str = TEST_SPLIT_PATH,
    summary_output_path: Path | str = SPLIT_SUMMARY_PATH,
    train_per_class: int = 1500,
    test_per_class: int = 500,
    random_state: int = 42,
) -> tuple[Path, Path, Path]:
    ensure_pipeline_directories()

    input_path = Path(input_path)
    train_output_path = Path(train_output_path)
    test_output_path = Path(test_output_path)
    summary_output_path = Path(summary_output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Could not find preprocessed dataset: {input_path}")

    dataframe = pd.read_csv(input_path, keep_default_na=False)
    train_df, test_df = build_balanced_split(
        dataframe,
        train_per_class=train_per_class,
        test_per_class=test_per_class,
        random_state=random_state,
    )

    train_output_path.parent.mkdir(parents=True, exist_ok=True)
    test_output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(train_output_path, index=False)
    test_df.to_csv(test_output_path, index=False)
    save_split_summary(
        train_df=train_df,
        test_df=test_df,
        summary_path=summary_output_path,
        source_path=input_path,
        train_per_class=train_per_class,
        test_per_class=test_per_class,
        random_state=random_state,
    )
    return train_output_path, test_output_path, summary_output_path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create balanced train/test CSV splits from the preprocessed Stack Overflow dataset."
    )
    parser.add_argument("--input", type=Path, default=PREPROCESSED_DATA_PATH, help="Path to the preprocessed CSV.")
    parser.add_argument("--train-output", type=Path, default=TRAIN_SPLIT_PATH, help="Output path for the train CSV.")
    parser.add_argument("--test-output", type=Path, default=TEST_SPLIT_PATH, help="Output path for the test CSV.")
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=SPLIT_SUMMARY_PATH,
        help="Output path for the split summary JSON.",
    )
    parser.add_argument("--train-per-class", type=int, default=1500, help="Number of training rows per tag.")
    parser.add_argument("--test-per-class", type=int, default=500, help="Number of test rows per tag.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed used for shuffling.")
    return parser


if __name__ == "__main__":
    parser = build_argument_parser()
    arguments = parser.parse_args()

    train_path, test_path, summary_path = create_split_artifacts(
        input_path=arguments.input,
        train_output_path=arguments.train_output,
        test_output_path=arguments.test_output,
        summary_output_path=arguments.summary_output,
        train_per_class=arguments.train_per_class,
        test_per_class=arguments.test_per_class,
        random_state=arguments.random_state,
    )
    print(f"Saved train split to: {train_path.resolve()}")
    print(f"Saved test split to: {test_path.resolve()}")
    print(f"Saved split summary to: {summary_path.resolve()}")
