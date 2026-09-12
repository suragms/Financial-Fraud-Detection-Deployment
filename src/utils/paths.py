"""Project-relative path helpers.

Paths are resolved from this file's location so the project works regardless
of the current working directory or the user's Windows username.
"""

from pathlib import Path

_MARKERS = ("run_validation.py", "src", "data")


def find_project_root(start: Path | None = None) -> Path:
    """Walk upward from ``start`` until the project root is found."""
    current = (start or Path(__file__)).resolve()
    if current.is_file():
        current = current.parent

    for candidate in [current, *current.parents]:
        if all((candidate / marker).exists() for marker in _MARKERS):
            return candidate

    raise FileNotFoundError(
        "Could not locate the project root. Expected a directory containing "
        "run_validation.py, src/, and data/."
    )


PROJECT_ROOT = find_project_root()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
BASELINE_MODELS_DIR = MODELS_DIR / "baseline"
REPORTS_DIR = PROJECT_ROOT / "reports"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
EDA_ARTIFACTS_DIR = ARTIFACTS_DIR / "eda"
EVALUATION_ARTIFACTS_DIR = ARTIFACTS_DIR / "model_evaluation"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
TESTS_DIR = PROJECT_ROOT / "tests"

DATASET_FILENAME = "financial_fraud_detection_dataset.csv"
RAW_DATASET_PATH = RAW_DATA_DIR / DATASET_FILENAME


def require_path(path: Path, *, kind: str = "path") -> Path:
    """Return ``path`` if it exists; otherwise raise a clear error."""
    resolved = path.resolve()
    if not resolved.exists():
        try:
            relative = path.relative_to(PROJECT_ROOT)
        except ValueError:
            relative = path
        raise FileNotFoundError(
            f"Expected {kind} was not found: {relative.as_posix()} "
            f"(resolved from the project root)."
        )
    return resolved
