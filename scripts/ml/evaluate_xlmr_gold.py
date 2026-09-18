"""
evaluate_xlmr_gold.py

Evaluate the trained XLM-R safety classifier
on the untouched human-adjudicated GOLD test set.

Input:
    Response text only

Model:
    Fine-tuned xlm-roberta-base

Test data:
    63 GOLD human-adjudicated responses

The GOLD test set is never used during training.
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


MODEL_DIR = "outputs/analysis/classifier/xlmr"
GOLD_FILE = "outputs/analysis/classifier/gold_test.csv"
OUTPUT_DIR = "outputs/analysis/classifier/xlmr_gold"

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


def main():

    print("=" * 70)
    print("XLM-R GOLD TEST EVALUATION")
    print("=" * 70)

    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(
            f"XLM-R model not found: {MODEL_DIR}"
        )

    if not os.path.exists(GOLD_FILE):
        raise FileNotFoundError(
            f"Gold test file not found: {GOLD_FILE}"
        )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # ------------------------------------------------------------
    # Load GOLD test data
    # ------------------------------------------------------------

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

    # ------------------------------------------------------------
    # Load tokenizer and model
    # ------------------------------------------------------------

    print()
    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR
    )

    print("Loading XLM-R model...")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR
    )

    # ------------------------------------------------------------
    # Device
    # ------------------------------------------------------------

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

    if torch.cuda.is_available():
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    # ------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------

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

    # ------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------

    accuracy = accuracy_score(
        true_labels,
        predictions
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            true_labels,
            predictions,
            average="macro",
            zero_division=0
        )
    )

    print()
    print("=" * 70)
    print("GOLD TEST RESULTS")
    print("=" * 70)

    print(
        f"Accuracy:          {accuracy:.4f}"
    )

    print(
        f"Macro Precision:   {precision:.4f}"
    )

    print(
        f"Macro Recall:      {recall:.4f}"
    )

    print(
        f"Macro F1:          {f1:.4f}"
    )

    # ------------------------------------------------------------
    # Classification report
    # ------------------------------------------------------------

    report = classification_report(
        true_labels,
        predictions,
        labels=[0, 1, 2],
        target_names=[
            "COMPLIANCE",
            "NON_COMPLIANCE",
            "REFUSAL"
        ],
        zero_division=0
    )

    print()
    print("CLASSIFICATION REPORT")
    print(report)

    # ------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------

    cm = confusion_matrix(
        true_labels,
        predictions,
        labels=[0, 1, 2]
    )

    print("CONFUSION MATRIX")
    print(
        "                    PREDICTED"
    )

    print(
        "                 C       NC       R"
    )

    print(
        f"ACTUAL C         "
        f"{cm[0,0]:<8}"
        f"{cm[0,1]:<9}"
        f"{cm[0,2]}"
    )

    print(
        f"ACTUAL NC        "
        f"{cm[1,0]:<8}"
        f"{cm[1,1]:<9}"
        f"{cm[1,2]}"
    )

    print(
        f"ACTUAL R         "
        f"{cm[2,0]:<8}"
        f"{cm[2,1]:<9}"
        f"{cm[2,2]}"
    )

    # ------------------------------------------------------------
    # Save row-level predictions
    # ------------------------------------------------------------

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
        "xlmr_gold_predictions.csv"
    )

    results_df.to_csv(
        predictions_file,
        index=False,
        encoding="utf-8"
    )

    # ------------------------------------------------------------
    # Save metrics
    # ------------------------------------------------------------

    metrics_file = os.path.join(
        OUTPUT_DIR,
        "xlmr_gold_metrics.txt"
    )

    with open(
        metrics_file,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "XLM-R GOLD TEST EVALUATION\n"
        )

        file.write(
            "=" * 70 + "\n\n"
        )

        file.write(
            f"Gold test rows: {len(gold_df)}\n\n"
        )

        file.write(
            f"Accuracy: {accuracy:.6f}\n"
        )

        file.write(
            f"Macro Precision: {precision:.6f}\n"
        )

        file.write(
            f"Macro Recall: {recall:.6f}\n"
        )

        file.write(
            f"Macro F1: {f1:.6f}\n\n"
        )

        file.write(
            "CLASSIFICATION REPORT\n"
        )

        file.write(
            report
        )

        file.write(
            "\nCONFUSION MATRIX\n"
        )

        file.write(
            np.array2string(cm)
        )

    print()
    print(
        f"Predictions saved: {predictions_file}"
    )

    print(
        f"Metrics saved: {metrics_file}"
    )

    print()
    print("=" * 70)
    print("XLM-R GOLD EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()