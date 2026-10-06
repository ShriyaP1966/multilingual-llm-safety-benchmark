"""
evaluate_muril_gold_balanced.py

ADDITIONAL, NON-FROZEN EXPERIMENT.

Evaluates the class-weighted MuRIL checkpoint produced by
scripts/ml/train_muril_balanced.py
(outputs/analysis/classifier/muril_balanced) on the untouched
human-adjudicated GOLD test set. Mirrors the structure of the
original, frozen scripts/ml/evaluate_muril_gold.py, which this script
does not modify and does not call.

Input:
    Response text only

Model:
    Class-weighted fine-tuned google/muril-base-cased
    (outputs/analysis/classifier/muril_balanced -- NOT the original
    outputs/analysis/classifier/muril checkpoint)

Test data:
    63 GOLD human-adjudicated responses (never used for training or
    weight computation).
"""

import os
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

MODEL_DIR = "outputs/analysis/classifier/muril_balanced"
GOLD_FILE = "outputs/analysis/classifier/gold_test.csv"
OUTPUT_DIR = "outputs/analysis/classifier/muril_gold_balanced"

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
    print("MuRIL BALANCED GOLD TEST EVALUATION")
    print("=" * 70)

    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(
            f"Balanced MuRIL model not found: {MODEL_DIR}. "
            f"Run scripts/ml/train_muril_balanced.py first."
        )

    if not os.path.exists(GOLD_FILE):
        raise FileNotFoundError(
            f"Gold test file not found: {GOLD_FILE}"
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    gold_df = pd.read_csv(
        GOLD_FILE,
        keep_default_na=False
    )

    print(f"Gold test rows: {len(gold_df)}")

    if len(gold_df) != 63:
        raise ValueError(
            f"Expected 63 gold responses, found {len(gold_df)}"
        )

    if "label_source" not in gold_df.columns:
        raise ValueError(
            "Gold test file is missing label_source column."
        )

    if set(gold_df["label_source"]) != {"GOLD"}:
        raise ValueError(
            "Gold test contains non-GOLD rows."
        )

    allowed_labels = set(LABEL2ID.keys())

    if not set(gold_df["label"]).issubset(allowed_labels):
        raise ValueError("Unexpected labels found in gold test.")

    print()
    print("GOLD LABEL DISTRIBUTION")
    print(gold_df["label"].value_counts())

    gold_data = pd.DataFrame({
        "text": gold_df["response"].astype(str),
        "labels": gold_df["label"].map(LABEL2ID)
    })

    gold_dataset = Dataset.from_pandas(
        gold_data,
        preserve_index=False
    )

    print()
    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR
    )

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LENGTH
        )

    print("Tokenizing gold test set...")

    gold_dataset = gold_dataset.map(
        tokenize_function,
        batched=True
    )

    print()
    print("Loading balanced MuRIL model...")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR
    )

    evaluation_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_eval_batch_size=8,
        report_to="none",
        fp16=torch.cuda.is_available()
    )

    trainer = Trainer(
        model=model,
        args=evaluation_args,
        processing_class=tokenizer
    )

    print()
    print("=" * 70)
    print("RUNNING GOLD TEST PREDICTIONS")
    print("=" * 70)

    predictions = trainer.predict(gold_dataset)

    logits = predictions.predictions
    predicted_ids = np.argmax(logits, axis=-1)

    true_ids = np.array(
        gold_data["labels"].tolist()
    )

    accuracy = accuracy_score(true_ids, predicted_ids)

    precision, recall, f1, support = precision_recall_fscore_support(
        true_ids,
        predicted_ids,
        labels=[0, 1, 2],
        zero_division=0
    )

    macro_precision = float(np.mean(precision))
    macro_recall = float(np.mean(recall))
    macro_f1 = float(np.mean(f1))

    print()
    print("=" * 70)
    print("GOLD TEST RESULTS (CLASS-WEIGHTED MuRIL)")
    print("=" * 70)

    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Macro Precision:   {macro_precision:.4f}")
    print(f"Macro Recall:      {macro_recall:.4f}")
    print(f"Macro F1:          {macro_f1:.4f}")

    report = classification_report(
        true_ids,
        predicted_ids,
        labels=[0, 1, 2],
        target_names=LABELS_IN_ORDER,
        zero_division=0
    )

    print()
    print("CLASSIFICATION REPORT")
    print(report)

    cm = confusion_matrix(
        true_ids,
        predicted_ids,
        labels=[0, 1, 2]
    )

    print("CONFUSION MATRIX")
    print()
    print("                    PREDICTED")
    print("                 C       NC       R")
    print(
        f"ACTUAL C       {cm[0,0]:>3}     {cm[0,1]:>3}     {cm[0,2]:>3}"
    )
    print(
        f"ACTUAL NC      {cm[1,0]:>3}     {cm[1,1]:>3}     {cm[1,2]:>3}"
    )
    print(
        f"ACTUAL R       {cm[2,0]:>3}     {cm[2,1]:>3}     {cm[2,2]:>3}"
    )

    result_df = gold_df.copy()

    result_df["predicted_label"] = [
        ID2LABEL[int(pred)]
        for pred in predicted_ids
    ]

    result_df["correct"] = (
        result_df["label"]
        == result_df["predicted_label"]
    )

    result_file = os.path.join(
        OUTPUT_DIR,
        "predictions.csv"
    )

    result_df.to_csv(
        result_file,
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

        file.write("MuRIL GOLD TEST EVALUATION (CLASS-WEIGHTED, BALANCED)\n")
        file.write("=" * 50 + "\n\n")

        file.write(
            "Non-frozen experiment. Evaluates "
            "outputs/analysis/classifier/muril_balanced, not the "
            "original outputs/analysis/classifier/muril checkpoint.\n\n"
        )

        file.write(f"Gold test size: {len(gold_df)}\n")
        file.write(f"Accuracy: {accuracy:.6f}\n")
        file.write(f"Macro Precision: {macro_precision:.6f}\n")
        file.write(f"Macro Recall: {macro_recall:.6f}\n")
        file.write(f"Macro F1: {macro_f1:.6f}\n\n")

        file.write("CLASSIFICATION REPORT\n")
        file.write(report)
        file.write("\n\n")

        file.write("CONFUSION MATRIX\n")
        file.write(np.array2string(cm))

    print()
    print("=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(f"Predictions: {result_file}")
    print(f"Metrics:     {metrics_file}")

    print()
    print("=" * 70)
    print("MuRIL BALANCED GOLD TEST EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
