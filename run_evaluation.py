"""Phase 6 fair comparison of saved Phase 5 candidate pipelines.

Loads artifacts. Recreates the untouched test fold. Does not retrain.
"""

from __future__ import annotations

import sys

from src.evaluation.compare import DISPLAY_COLS, run_evaluation


def _print_table(comparison) -> None:
    trained = comparison[comparison["Strategy"] != "Dummy"][DISPLAY_COLS].copy()
    print(trained.to_string(index=False, float_format=lambda value: f"{value:0.4f}"))
    print()
    dummy = comparison[comparison["Strategy"] == "Dummy"].iloc[0]
    print("Dummy Always Legitimate (not a trained model)")
    print(
        f"Accuracy={dummy['Accuracy']:.4f}  Precision={dummy['Precision']:.4f}  "
        f"Recall={dummy['Recall']:.4f}  F1={dummy['F1']:.4f}  "
        f"ROC_AUC={dummy['ROC_AUC']:.4f}  PR_AUC={dummy['PR_AUC']:.4f}"
    )


def main() -> int:
    result = run_evaluation()
    split = result["split_info"]
    selected = result["selected"]
    print("PHASE 6 MODEL COMPARISON")
    print()
    print(f"Dataset hash: {result['dataset_hash']}")
    print(f"Test rows: {split['test_rows']}")
    print(f"Test fraud: {split['test_fraud_count']} ({split['test_fraud_rate'] * 100:.2f}%)")
    print(f"Test legitimate: {split['test_legitimate_count']}")
    print()
    _print_table(result["comparison"])
    print()
    print("Selected candidate:", selected["Model"], "/", selected["Strategy"])
    print("Artifact:", f"models/baseline/{selected['key']}.joblib")
    print(selected["selection_reason"])
    print()
    print("Wrote:")
    print(" -", result["csv_path"])
    print(" -", result["md_path"])
    print(" -", result["json_path"])
    for name, path in result["charts"].items():
        print(" -", path)
    print()
    print("This is a selected candidate, not a production model.")
    print("PHASE 6 COMPLETE — MODEL COMPARISON VALIDATED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
