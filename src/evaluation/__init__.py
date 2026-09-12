"""Model evaluation on the untouched test set."""

from src.evaluation.compare import run_evaluation
from src.evaluation.metrics import compute_binary_metrics

__all__ = ["compute_binary_metrics", "run_evaluation"]
