"""
evaluate_muril_gold.py

Evaluate the trained MuRIL classifier on the 63 human-adjudicated
GOLD test responses.

IMPORTANT:
- The GOLD test set is never used for training.
- Input is response text only.
- Human labels are treated as the ground truth.
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

MODEL_DIR = "outputs/analysis/classifier/muril"
GOLD_FILE = "outputs/analysis/classifier/gold_test.csv"
OUTPUT_DIR = "outputs/analysis/classifier/muril_gold"

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
    print("MuRIL GOLD TEST EVALUATION")
    print("=" * 70)

    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(
            f"Trained MuRIL model not found: {MODEL_DIR}"
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

    # ------------------------------------------------------------------
    # Prepare dataset
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Load trained MuRIL
    # ------------------------------------------------------------------

    print()
    print("Loading trained MuRIL model...")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR
    )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    evaluation_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_eval_batch_size=8,
        report_to="none",
        # CUDA-conditional so gold evaluation also runs on CPU.
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

    # ------------------------------------------------------------------
    # Overall metrics
    # ------------------------------------------------------------------

    accuracy = accuracy_score(
        true_ids,
        predicted_ids
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        true_ids,
        predicted_ids,
        labels=[0, 1, 2],
        zero_division=0
    )

    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)

    print()
    print("=" * 70)
    print("GOLD TEST RESULTS")
    print("=" * 70)

    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Macro Precision:   {macro_precision:.4f}")
    print(f"Macro Recall:      {macro_recall:.4f}")
    print(f"Macro F1:          {macro_f1:.4f}")

    # ------------------------------------------------------------------
    # Classification report
    # ------------------------------------------------------------------

    report = classification_report(
        true_ids,
        predicted_ids,
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

    # ------------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Save predictions
    # ------------------------------------------------------------------

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
        "muril_gold_predictions.csv"
    )

    result_df.to_csv(
        result_file,
        index=False,
        encoding="utf-8"
    )

    # ------------------------------------------------------------------
    # Save metrics
    # ------------------------------------------------------------------

    metrics_file = os.path.join(
        OUTPUT_DIR,
        "muril_gold_metrics.txt"
    )

    with open(
        metrics_file,
        "w",
        encoding="utf-8"
    ) as file:

        file.write("MuRIL GOLD TEST EVALUATION\n")
        file.write("=" * 50 + "\n\n")

        file.write(f"Gold test size: {len(gold_df)}\n")
        file.write(f"Accuracy: {accuracy:.6f}\n")
        file.write(
            f"Macro Precision: {macro_precision:.6f}\n"
        )
        file.write(
            f"Macro Recall: {macro_recall:.6f}\n"
        )
        file.write(
            f"Macro F1: {macro_f1:.6f}\n\n"
        )

        file.write("CLASSIFICATION REPORT\n")
        file.write(report)
        file.write("\n\n")

        file.write("CONFUSION MATRIX\n")
        file.write(
            np.array2string(cm)
        )

    print()
    print("=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(f"Predictions: {result_file}")
    print(f"Metrics:     {metrics_file}")

    print()
    print("=" * 70)
    print("MuRIL GOLD TEST EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()