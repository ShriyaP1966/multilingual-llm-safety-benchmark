"""
compare_classifiers.py

Compare MuRIL and XLM-R safety classifiers.

Comparison:
1. SILVER development performance
2. GOLD human-adjudicated performance
3. SILVER -> GOLD performance degradation
4. Per-class GOLD performance
5. GOLD confusion matrices

This script does not retrain or modify any model/data.
"""

import os
import pandas as pd

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)


MURIL_DEV = "outputs/analysis/classifier/muril/dev_metrics.txt"
XLMR_DEV = "outputs/analysis/classifier/xlmr/dev_metrics.txt"

MURIL_GOLD = (
    "outputs/analysis/classifier/muril_gold/"
    "muril_gold_predictions.csv"
)

XLMR_GOLD = (
    "outputs/analysis/classifier/xlmr_gold/"
    "xlmr_gold_predictions.csv"
)

OUTPUT_DIR = "outputs/analysis/classifier/comparison"


LABELS = [
    "COMPLIANCE",
    "NON_COMPLIANCE",
    "REFUSAL"
]


def read_dev_metrics(path):

    metrics = {}

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            if ":" not in line:
                continue

            key, value = line.strip().split(
                ":",
                1
            )

            key = key.strip()
            value = value.strip()

            try:
                metrics[key] = float(value)
            except ValueError:
                pass

    return metrics


def evaluate_gold(path):

    df = pd.read_csv(
        path,
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

    cm = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=LABELS
    )

    return df, report, cm


def main():

    print("=" * 70)
    print("MURIL VS XLM-R CLASSIFIER COMPARISON")
    print("=" * 70)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    required_files = [
        MURIL_DEV,
        XLMR_DEV,
        MURIL_GOLD,
        XLMR_GOLD
    ]

    for path in required_files:

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    # ------------------------------------------------------------
    # Development metrics
    # ------------------------------------------------------------

    muril_dev = read_dev_metrics(
        MURIL_DEV
    )

    xlmr_dev = read_dev_metrics(
        XLMR_DEV
    )

    # ------------------------------------------------------------
    # GOLD evaluation
    # ------------------------------------------------------------

    muril_gold_df, muril_report, muril_cm = (
        evaluate_gold(MURIL_GOLD)
    )

    xlmr_gold_df, xlmr_report, xlmr_cm = (
        evaluate_gold(XLMR_GOLD)
    )

    # ------------------------------------------------------------
    # Overall comparison
    # ------------------------------------------------------------

    comparison = pd.DataFrame({

        "Metric": [
            "Accuracy",
            "Macro Precision",
            "Macro Recall",
            "Macro F1"
        ],

        "MuRIL Development": [
            muril_dev.get(
                "eval_accuracy",
                float("nan")
            ),
            muril_dev.get(
                "eval_macro_precision",
                float("nan")
            ),
            muril_dev.get(
                "eval_macro_recall",
                float("nan")
            ),
            muril_dev.get(
                "eval_macro_f1",
                float("nan")
            )
        ],

        "XLM-R Development": [
            xlmr_dev.get(
                "eval_accuracy",
                float("nan")
            ),
            xlmr_dev.get(
                "eval_macro_precision",
                float("nan")
            ),
            xlmr_dev.get(
                "eval_macro_recall",
                float("nan")
            ),
            xlmr_dev.get(
                "eval_macro_f1",
                float("nan")
            )
        ],

        "MuRIL GOLD": [
            muril_report["accuracy"],
            muril_report["macro avg"]["precision"],
            muril_report["macro avg"]["recall"],
            muril_report["macro avg"]["f1-score"]
        ],

        "XLM-R GOLD": [
            xlmr_report["accuracy"],
            xlmr_report["macro avg"]["precision"],
            xlmr_report["macro avg"]["recall"],
            xlmr_report["macro avg"]["f1-score"]
        ]
    })

    # ------------------------------------------------------------
    # SILVER -> GOLD degradation
    # ------------------------------------------------------------

    degradation = pd.DataFrame({

        "Metric": [
            "Accuracy",
            "Macro Precision",
            "Macro Recall",
            "Macro F1"
        ],

        "MuRIL Dev": [
            muril_dev.get(
                "eval_accuracy",
                float("nan")
            ),
            muril_dev.get(
                "eval_macro_precision",
                float("nan")
            ),
            muril_dev.get(
                "eval_macro_recall",
                float("nan")
            ),
            muril_dev.get(
                "eval_macro_f1",
                float("nan")
            )
        ],

        "MuRIL GOLD": [
            muril_report["accuracy"],
            muril_report["macro avg"]["precision"],
            muril_report["macro avg"]["recall"],
            muril_report["macro avg"]["f1-score"]
        ],

        "MuRIL Change": [
            muril_report["accuracy"]
            - muril_dev.get("eval_accuracy", 0),

            muril_report["macro avg"]["precision"]
            - muril_dev.get(
                "eval_macro_precision",
                0
            ),

            muril_report["macro avg"]["recall"]
            - muril_dev.get(
                "eval_macro_recall",
                0
            ),

            muril_report["macro avg"]["f1-score"]
            - muril_dev.get(
                "eval_macro_f1",
                0
            )
        ],

        "XLM-R Dev": [
            xlmr_dev.get(
                "eval_accuracy",
                float("nan")
            ),
            xlmr_dev.get(
                "eval_macro_precision",
                float("nan")
            ),
            xlmr_dev.get(
                "eval_macro_recall",
                float("nan")
            ),
            xlmr_dev.get(
                "eval_macro_f1",
                float("nan")
            )
        ],

        "XLM-R GOLD": [
            xlmr_report["accuracy"],
            xlmr_report["macro avg"]["precision"],
            xlmr_report["macro avg"]["recall"],
            xlmr_report["macro avg"]["f1-score"]
        ],

        "XLM-R Change": [
            xlmr_report["accuracy"]
            - xlmr_dev.get("eval_accuracy", 0),

            xlmr_report["macro avg"]["precision"]
            - xlmr_dev.get(
                "eval_macro_precision",
                0
            ),

            xlmr_report["macro avg"]["recall"]
            - xlmr_dev.get(
                "eval_macro_recall",
                0
            ),

            xlmr_report["macro avg"]["f1-score"]
            - xlmr_dev.get(
                "eval_macro_f1",
                0
            )
        ]
    })

    # ------------------------------------------------------------
    # Per-class GOLD comparison
    # ------------------------------------------------------------

    class_rows = []

    for label in LABELS:

        class_rows.append({

            "Class": label,

            "MuRIL Precision": muril_report[
                label
            ]["precision"],

            "MuRIL Recall": muril_report[
                label
            ]["recall"],

            "MuRIL F1": muril_report[
                label
            ]["f1-score"],

            "XLM-R Precision": xlmr_report[
                label
            ]["precision"],

            "XLM-R Recall": xlmr_report[
                label
            ]["recall"],

            "XLM-R F1": xlmr_report[
                label
            ]["f1-score"]
        })

    per_class = pd.DataFrame(
        class_rows
    )

    # ------------------------------------------------------------
    # Save CSV files
    # ------------------------------------------------------------

    comparison_file = os.path.join(
        OUTPUT_DIR,
        "classifier_comparison.csv"
    )

    degradation_file = os.path.join(
        OUTPUT_DIR,
        "silver_to_gold_degradation.csv"
    )

    per_class_file = os.path.join(
        OUTPUT_DIR,
        "gold_per_class_comparison.csv"
    )

    comparison.to_csv(
        comparison_file,
        index=False
    )

    degradation.to_csv(
        degradation_file,
        index=False
    )

    per_class.to_csv(
        per_class_file,
        index=False
    )

    # ------------------------------------------------------------
    # Save confusion matrices
    # ------------------------------------------------------------

    muril_cm_df = pd.DataFrame(
        muril_cm,
        index=[
            "ACTUAL_COMPLIANCE",
            "ACTUAL_NON_COMPLIANCE",
            "ACTUAL_REFUSAL"
        ],
        columns=[
            "PRED_COMPLIANCE",
            "PRED_NON_COMPLIANCE",
            "PRED_REFUSAL"
        ]
    )

    xlmr_cm_df = pd.DataFrame(
        xlmr_cm,
        index=[
            "ACTUAL_COMPLIANCE",
            "ACTUAL_NON_COMPLIANCE",
            "ACTUAL_REFUSAL"
        ],
        columns=[
            "PRED_COMPLIANCE",
            "PRED_NON_COMPLIANCE",
            "PRED_REFUSAL"
        ]
    )

    muril_cm_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "muril_gold_confusion_matrix.csv"
        )
    )

    xlmr_cm_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "xlmr_gold_confusion_matrix.csv"
        )
    )

    # ------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("OVERALL COMPARISON")
    print("=" * 70)

    print(
        comparison.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print()
    print("=" * 70)
    print("GOLD PER-CLASS COMPARISON")
    print("=" * 70)

    print(
        per_class.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print()
    print("=" * 70)
    print("SILVER -> GOLD DEGRADATION")
    print("=" * 70)

    print(
        degradation.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print()
    print("=" * 70)
    print("MURIL GOLD CONFUSION MATRIX")
    print("=" * 70)

    print(muril_cm_df)

    print()
    print("=" * 70)
    print("XLM-R GOLD CONFUSION MATRIX")
    print("=" * 70)

    print(xlmr_cm_df)

    print()
    print("=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(comparison_file)
    print(degradation_file)
    print(per_class_file)

    print(
        os.path.join(
            OUTPUT_DIR,
            "muril_gold_confusion_matrix.csv"
        )
    )

    print(
        os.path.join(
            OUTPUT_DIR,
            "xlmr_gold_confusion_matrix.csv"
        )
    )

    print()
    print("=" * 70)
    print("CLASSIFIER COMPARISON COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()