from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
RUNS_OUTPUT_DIR = OUTPUT_DIR / "runs"


def ensure_output_directories() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
