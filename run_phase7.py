"""Phase 7: freeze calibration, threshold, and the production scoring pipeline."""

from __future__ import annotations

import sys

from src.models.phase7 import run_phase7


def main() -> int:
    result = run_phase7()
    metrics = result["test_metrics"]
    print("PHASE 7 FINAL MODEL AND RISK SCORING")
    print()
    print(f"Calibration method: {result['calibration']['method']}")
    print(f"Selected threshold: {result['threshold']:.2f}")
    print("Final model: Logistic Regression + Class Weight")
    print()
    print("Untouched test evaluation (one shot, not used for tuning)")
    print(f"  Accuracy  = {metrics['Accuracy']:.4f}")
    print(f"  Precision = {metrics['Precision']:.4f}")
    print(f"  Recall    = {metrics['Recall']:.4f}")
    print(f"  F1        = {metrics['F1']:.4f}")
    print(f"  ROC-AUC   = {metrics['ROC_AUC']:.4f}")
    print(f"  PR-AUC    = {metrics['PR_AUC']:.4f}")
    print(f"  TN={metrics['TN']}  FP={metrics['FP']}  FN={metrics['FN']}  TP={metrics['TP']}")
    print(f"  FPR={metrics['False_Positive_Rate']:.4f}  FNR={metrics['False_Negative_Rate']:.4f}")
    print(
        f"  Flagged: {metrics['flagged']} ({metrics['flagged_rate'] * 100:.2f}%)  "
        f"Legitimate: {metrics['legitimate_predicted']} ({metrics['legitimate_predicted_rate'] * 100:.2f}%)"
    )
    print()
    print("Risk bands (test set)")
    print(result["band_summary"].to_string(index=False))
    print()
    print("Example (Transaction_ID, risk_score, risk_band, predicted_label)")
    print(result["example"].to_string(index=False))
    print()
    print("Hashes")
    for name, digest in result["hashes"].items():
        print(f"  {name}: {digest}")
    print()
    print("Wrote:")
    print(" -", result["production_path"])
    print(" -", result["metadata_path"])
    print(" - reports/phase7_model_selection.md")
    print(" - reports/threshold_analysis.csv")
    print(" - reports/final_model_metrics.csv")
    print(" - reports/risk_band_summary.csv")
    print(" - artifacts/model_selection/")
    print()
    print(result["conclusion"])
    print()
    print("============================================================")
    print("PHASE 7 COMPLETE — FINAL MODEL & RISK SCORING VALIDATED")
    print("============================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
