"""
compare_balanced_vs_baseline.py

ADDITIONAL, NON-FROZEN COMPARISON SCRIPT.

Builds a single comparison table across:
  - the ORIGINAL, FROZEN classifier baselines (xlmr_gold, muril_gold)
  - the NEW, non-frozen prior-calibrated experiment
    (xlmr_gold_calibrated, muril_gold_calibrated)
  - the NEW, non-frozen class-weighted experiment
    (xlmr_gold_balanced, muril_gold_balanced)

This script only READS existing prediction files; it does not retrain,
does not touch gold_test.csv, and does not modify any frozen output.
It writes a single new file:

    outputs/analysis/classifier/comparison/balanced_vs_baseline_comparison.csv

It does not overwrite any existing comparison CSV produced by
scripts/ml/compare_classifiers.py.
"""

import os
import pandas as pd

from sklearn.metrics import classification_report


LABELS = [
    "COMPLIANCE",
    "NON_COMPLIANCE",
    "REFUSAL"
]

OUTPUT_DIR = "outputs/analysis/classifier/comparison"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "balanced_vs_baseline_comparison.csv"
)

EXPERIMENTS = [
    {
        "model": "XLM-R",
        "experiment": "Original Frozen Baseline",
        "predictions_file": (
            "outputs/analysis/classifier/xlmr_gold/"
            "xlmr_gold_predictions.csv"
        )
    },
    {
        "model": "MuRIL",
        "experiment": "Original Frozen Baseline",
        "predictions_file": (
            "outputs/analysis/classifier/muril_gold/"
            "muril_gold_predictions.csv"
        )
    },
    {
        "model": "XLM-R",
        "experiment": "Inference-Time Prior Calibration (non-frozen)",
        "predictions_file": (
            "outputs/analysis/classifier/xlmr_gold_calibrated/"
            "predictions.csv"
        )
    },
    {
        "model": "MuRIL",
        "experiment": "Inference-Time Prior Calibration (non-frozen)",
        "predictions_file": (
            "outputs/analysis/classifier/muril_gold_calibrated/"
            "predictions.csv"
        )
    },
    {
        "model": "XLM-R",
        "experiment": "Class-Weighted Retraining (non-frozen)",
        "predictions_file": (
            "outputs/analysis/classifier/xlmr_gold_balanced/"
            "predictions.csv"
        )
    },
    {
        "model": "MuRIL",
        "experiment": "Class-Weighted Retraining (non-frozen)",
        "predictions_file": (
            "outputs/analysis/classifier/muril_gold_balanced/"
            "predictions.csv"
        )
    }
]

MAJORITY_CLASS_COUNT = 33
GOLD_TOTAL = 63


def evaluate(predictions_file):

    df = pd.read_csv(
        predictions_file,
        keep_default_na=False
    )

    true_labels = df["label"]
    predicted_labels = df["predicted_label"]

    report = classification_report(
        true_labels,
        predicted_labels,
        labels=LABELS,
        target_names=LABELS,
        output_dict=True,
        zero_division=0
    )

    return report


def build_row(model, experiment, report):

    row = {
        "Model": model,
        "Experiment": experiment,
        "Accuracy": report["accuracy"]
    }

    for label in LABELS:
        row[f"{label} Precision"] = report[label]["precision"]
        row[f"{label} Recall"] = report[label]["recall"]
        row[f"{label} F1"] = report[label]["f1-score"]

    return row


def main():

    print("=" * 70)
    print("BALANCED VS BASELINE CLASSIFIER COMPARISON (NON-FROZEN)")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    rows = []

    for cfg in EXPERIMENTS:

        if not os.path.exists(cfg["predictions_file"]):
            raise FileNotFoundError(
                f"Missing predictions file for "
                f"{cfg['model']} / {cfg['experiment']}: "
                f"{cfg['predictions_file']}. Run the corresponding "
                f"experiment script first."
            )

        report = evaluate(cfg["predictions_file"])

        rows.append(
            build_row(
                cfg["model"],
                cfg["experiment"],
                report
            )
        )

    # ------------------------------------------------------------
    # Reference row: majority-class (always predict COMPLIANCE)
    # baseline on the 63-case GOLD set. Included only as a fixed
    # reference point; it is not a trained model and does not use
    # any experiment output.
    # ------------------------------------------------------------

    majority_row = {
        "Model": "N/A",
        "Experiment": (
            "Reference: Majority-Class Baseline "
            "(always predict COMPLIANCE)"
        ),
        "Accuracy": MAJORITY_CLASS_COUNT / GOLD_TOTAL,
        "COMPLIANCE Precision": MAJORITY_CLASS_COUNT / GOLD_TOTAL,
        "COMPLIANCE Recall": 1.0,
        "COMPLIANCE F1": (
            2 * (MAJORITY_CLASS_COUNT / GOLD_TOTAL) * 1.0
            / ((MAJORITY_CLASS_COUNT / GOLD_TOTAL) + 1.0)
        ),
        "NON_COMPLIANCE Precision": 0.0,
        "NON_COMPLIANCE Recall": 0.0,
        "NON_COMPLIANCE F1": 0.0,
        "REFUSAL Precision": 0.0,
        "REFUSAL Recall": 0.0,
        "REFUSAL F1": 0.0
    }

    rows.append(majority_row)

    comparison_df = pd.DataFrame(rows)

    column_order = [
        "Model",
        "Experiment",
        "Accuracy",
        "COMPLIANCE Precision",
        "COMPLIANCE Recall",
        "COMPLIANCE F1",
        "NON_COMPLIANCE Precision",
        "NON_COMPLIANCE Recall",
        "NON_COMPLIANCE F1",
        "REFUSAL Precision",
        "REFUSAL Recall",
        "REFUSAL F1"
    ]

    comparison_df = comparison_df[column_order]

    comparison_df.to_csv(OUTPUT_FILE, index=False)

    print()
    print(
        comparison_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print()
    print(f"Comparison saved: {OUTPUT_FILE}")

    print()
    print("=" * 70)
    print("COMPARISON COMPLETE")
    print(
        "This table combines the original frozen baseline with "
        "new, non-frozen calibration and class-weighting experiments. "
        "It does not replace the original frozen comparison CSVs."
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
