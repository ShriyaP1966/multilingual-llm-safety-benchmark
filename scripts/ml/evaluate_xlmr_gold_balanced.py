"""
evaluate_xlmr_gold_balanced.py

ADDITIONAL, NON-FROZEN EXPERIMENT.

Evaluates the class-weighted XLM-R checkpoint produced by
scripts/ml/train_xlmr_balanced.py
(outputs/analysis/classifier/xlmr_balanced) on the untouched
human-adjudicated GOLD test set. Mirrors the structure of the
original, frozen scripts/ml/evaluate_xlmr_gold.py, which this script
does not modify and does not call.

Input:
    Response text only

Model:
    Class-weighted fine-tuned xlm-roberta-base
    (outputs/analysis/classifier/xlmr_balanced -- NOT the original
    outputs/analysis/classifier/xlmr checkpoint)

Test data:
    63 GOLD human-adjudicated responses (never used for training or
    weight computation).
"""

import os
import numpy as np
import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)


MODEL_DIR = "outputs/analysis/classifier/xlmr_balanced"
GOLD_FILE = "outputs/analysis/classifier/gold_test.csv"
OUTPUT_DIR = "outputs/analysis/classifier/xlmr_gold_balanced"

MAX_LENGTH = 512

LABEL2ID = {
    "COMPLIANCE": 0,
    "NON_COMPLIANCE": 1,
    "REFUSAL": 2
}

ID2LABEL = {
    0: "COMPLIANCE",
    1: "NON_COMPLIANCE",
    2: "REFUSAL"
}

LABELS_IN_ORDER = ["COMPLIANCE", "NON_COMPLIANCE", "REFUSAL"]


def main():

    print("=" * 70)
    print("XLM-R BALANCED GOLD TEST EVALUATION")
    print("=" * 70)

    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(
            f"Balanced XLM-R model not found: {MODEL_DIR}. "
            f"Run scripts/ml/train_xlmr_balanced.py first."
        )

    if not os.path.exists(GOLD_FILE):
        raise FileNotFoundError(
            f"Gold test file not found: {GOLD_FILE}"
        )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    gold_df = pd.read_csv(
        GOLD_FILE,
        keep_default_na=False
    )

    print(
        f"Gold test rows: {len(gold_df)}"
    )

    if len(gold_df) != 63:
        raise ValueError(
            f"Expected exactly 63 GOLD rows, "
            f"found {len(gold_df)}."
        )

    if set(gold_df["label_source"]) != {"GOLD"}:
        raise ValueError(
            "Gold test file contains non-GOLD rows."
        )

    allowed_labels = set(LABEL2ID.keys())

    if not set(gold_df["label"]).issubset(
        allowed_labels
    ):
        raise ValueError(
            "Unexpected labels found in GOLD test data."
        )

    print()
    print("GOLD LABEL DISTRIBUTION")
    print(
        gold_df["label"].value_counts()
    )

    print()
    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR
    )

    print("Loading balanced XLM-R model...")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model.to(device)
    model.eval()

    print(
        f"Device: {device}"
    )

    texts = gold_df["response"].astype(str).tolist()

    true_labels = [
        LABEL2ID[label]
        for label in gold_df["label"]
    ]

    predictions = []

    print()
    print("Generating GOLD predictions...")

    with torch.no_grad():

        for text in texts:

            encoded = tokenizer(
                text,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt"
            )

            encoded = {
                key: value.to(device)
                for key, value in encoded.items()
            }

            outputs = model(
                **encoded
            )

            predicted_id = torch.argmax(
                outputs.logits,
                dim=-1
            ).item()

            predictions.append(
                predicted_id
            )

    accuracy = accuracy_score(
        true_labels,
        predictions
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels,
        predictions,
        labels=[0, 1, 2],
        zero_division=0
    )

    macro_precision = float(np.mean(precision))
    macro_recall = float(np.mean(recall))
    macro_f1 = float(np.mean(f1))

    print()
    print("=" * 70)
    print("GOLD TEST RESULTS (CLASS-WEIGHTED XLM-R)")
    print("=" * 70)

    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Macro Precision:   {macro_precision:.4f}")
    print(f"Macro Recall:      {macro_recall:.4f}")
    print(f"Macro F1:          {macro_f1:.4f}")

    report = classification_report(
        true_labels,
        predictions,
        labels=[0, 1, 2],
        target_names=LABELS_IN_ORDER,
        zero_division=0
    )

    print()
    print("CLASSIFICATION REPORT")
    print(report)

    cm = confusion_matrix(
        true_labels,
        predictions,
        labels=[0, 1, 2]
    )

    print("CONFUSION MATRIX")
    print("                    PREDICTED")
    print("                 C       NC       R")
    print(
        f"ACTUAL C         {cm[0,0]:<8}{cm[0,1]:<9}{cm[0,2]}"
    )
    print(
        f"ACTUAL NC        {cm[1,0]:<8}{cm[1,1]:<9}{cm[1,2]}"
    )
    print(
        f"ACTUAL R         {cm[2,0]:<8}{cm[2,1]:<9}{cm[2,2]}"
    )

    results_df = gold_df.copy()

    results_df["predicted_label"] = [
        ID2LABEL[p]
        for p in predictions
    ]

    results_df["correct"] = (
        results_df["label"]
        == results_df["predicted_label"]
    )

    predictions_file = os.path.join(
        OUTPUT_DIR,
        "predictions.csv"
    )

    results_df.to_csv(
        predictions_file,
        index=False,
        encoding="utf-8"
    )

    metrics_file = os.path.join(
        OUTPUT_DIR,
        "metrics.txt"
    )

    with open(
        metrics_file,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "XLM-R GOLD TEST EVALUATION (CLASS-WEIGHTED, BALANCED)\n"
        )

        file.write("=" * 70 + "\n\n")

        file.write(
            "Non-frozen experiment. Evaluates "
            "outputs/analysis/classifier/xlmr_balanced, not the "
            "original outputs/analysis/classifier/xlmr checkpoint.\n\n"
        )

        file.write(
            f"Gold test rows: {len(gold_df)}\n\n"
        )

        file.write(f"Accuracy: {accuracy:.6f}\n")
        file.write(f"Macro Precision: {macro_precision:.6f}\n")
        file.write(f"Macro Recall: {macro_recall:.6f}\n")
        file.write(f"Macro F1: {macro_f1:.6f}\n\n")

        file.write("CLASSIFICATION REPORT\n")
        file.write(report)

        file.write("\nCONFUSION MATRIX\n")
        file.write(np.array2string(cm))

    print()
    print(f"Predictions saved: {predictions_file}")
    print(f"Metrics saved: {metrics_file}")

    print()
    print("=" * 70)
    print("XLM-R BALANCED GOLD EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
