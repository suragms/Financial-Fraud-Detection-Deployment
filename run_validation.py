"""Run schema and data-quality validation for the primary fraud dataset."""

from __future__ import annotations

import sys

from src.data.loader import (
    format_validation_report,
    load_fraud_dataset,
    validate_fraud_dataset,
)
from src.utils.paths import RAW_DATASET_PATH


def main() -> int:
    try:
        frame = load_fraud_dataset()
    except FileNotFoundError as exc:
        print("DATA VALIDATION FAILED")
        print()
        print(exc)
        return 1
    except ValueError as exc:
        print("DATA VALIDATION FAILED")
        print()
        print(exc)
        return 1

    report = validate_fraud_dataset(frame, source_path=RAW_DATASET_PATH)
    print(format_validation_report(report))
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
