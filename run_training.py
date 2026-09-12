"""Train Phase 5 candidate pipelines from the project root."""

from __future__ import annotations

import sys

from src.models.pipeline import train_and_save


def main() -> int:
    result = train_and_save()
    print("PHASE 5 TRAINING COMPLETE")
    print()
    print("Models:", ", ".join(result["model_names"]))
    split = result["split"]
    print(f"Train rows: {split['train_row_count']}  fraud={split['train_fraud_count']}  rate={split['train_fraud_rate'] * 100:.2f}%")
    print(f"Test rows:  {split['test_row_count']}  fraud={split['test_fraud_count']}  rate={split['test_fraud_rate'] * 100:.2f}%")
    print()
    print("Artifacts:")
    for item in result["artifacts"]:
        print(" -", item)
    print("Metadata:", result["metadata_path"])
    print("Log:", result["log_path"])
    print()
    print("No model is declared best. Comparison belongs to Phase 6.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
