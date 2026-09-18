"""
train_muril.py

Fine-tune MuRIL for multilingual LLM safety classification.

Input:
    Response text only

Labels:
    COMPLIANCE
    NON_COMPLIANCE
    REFUSAL

Training data:
    SILVER labels only

Development data:
    SILVER labels only

Gold human-adjudicated data:
    Completely excluded from training/development.
    It will be used later for final human-grounded evaluation.
"""

import os
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MODEL_NAME = "google/muril-base-cased"

TRAIN_FILE = (
    "outputs/analysis/classifier/train_silver.csv"
)

DEV_FILE = (
    "outputs/analysis/classifier/dev_silver.csv"
)

OUTPUT_DIR = (
    "outputs/analysis/classifier/muril"
)

MAX_LENGTH = 512

RANDOM_SEED = 42

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
    print("MuRIL MULTILINGUAL SAFETY CLASSIFIER")
    print("=" * 70)

    set_seed(RANDOM_SEED)

    # -----------------------------------------------------
    # Load datasets
    # -----------------------------------------------------

    if not os.path.exists(TRAIN_FILE):
        raise FileNotFoundError(
            f"Training file not found: {TRAIN_FILE}"
        )

    if not os.path.exists(DEV_FILE):
        raise FileNotFoundError(
            f"Development file not found: {DEV_FILE}"
        )

    train_df = pd.read_csv(
        TRAIN_FILE,
        keep_default_na=False
    )

    dev_df = pd.read_csv(
        DEV_FILE,
        keep_default_na=False
    )

    print(f"Training rows: {len(train_df)}")
    print(f"Development rows: {len(dev_df)}")

    # -----------------------------------------------------
    # Verify SILVER-only training
    # -----------------------------------------------------

    if set(train_df["label_source"]) != {"SILVER"}:
        raise ValueError(
            "Training data contains non-SILVER labels."
        )

    if set(dev_df["label_source"]) != {"SILVER"}:
        raise ValueError(
            "Development data contains non-SILVER labels."
        )

    # -----------------------------------------------------
    # Validate labels
    # -----------------------------------------------------

    allowed_labels = set(LABEL2ID.keys())

    train_labels = set(train_df["label"])
    dev_labels = set(dev_df["label"])

    if not train_labels.issubset(allowed_labels):
        raise ValueError(
            f"Unexpected training labels: "
            f"{train_labels - allowed_labels}"
        )

    if not dev_labels.issubset(allowed_labels):
        raise ValueError(
            f"Unexpected development labels: "
            f"{dev_labels - allowed_labels}"
        )

    # -----------------------------------------------------
    # Keep RESPONSE ONLY
    # -----------------------------------------------------

    train_data = pd.DataFrame({
        "text": train_df["response"].astype(str),
        "labels": train_df["label"].map(LABEL2ID)
    })

    dev_data = pd.DataFrame({
        "text": dev_df["response"].astype(str),
        "labels": dev_df["label"].map(LABEL2ID)
    })

    # -----------------------------------------------------
    # Check for missing values
    # -----------------------------------------------------

    if train_data["text"].str.strip().eq("").any():
        raise ValueError(
            "Empty training responses detected."
        )

    if dev_data["text"].str.strip().eq("").any():
        raise ValueError(
            "Empty development responses detected."
        )

    if train_data["labels"].isna().any():
        raise ValueError(
            "Missing training labels detected."
        )

    if dev_data["labels"].isna().any():
        raise ValueError(
            "Missing development labels detected."
        )

    # -----------------------------------------------------
    # Convert to Hugging Face datasets
    # -----------------------------------------------------

    train_dataset = Dataset.from_pandas(
        train_data,
        preserve_index=False
    )

    dev_dataset = Dataset.from_pandas(
        dev_data,
        preserve_index=False
    )

    # -----------------------------------------------------
    # Load tokenizer
    # -----------------------------------------------------

    print()
    print(f"Loading tokenizer: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    # -----------------------------------------------------
    # Tokenization
    # -----------------------------------------------------

    def tokenize_function(examples):

        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LENGTH
        )

    print("Tokenizing training data...")

    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True
    )

    print("Tokenizing development data...")

    dev_dataset = dev_dataset.map(
        tokenize_function,
        batched=True
    )

    # -----------------------------------------------------
    # Load classification model
    # -----------------------------------------------------

    print()
    print(f"Loading model: {MODEL_NAME}")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )

    # -----------------------------------------------------
    # Data collator
    # -----------------------------------------------------

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer
    )

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    def compute_metrics(eval_pred):

        logits, labels = eval_pred

        predictions = np.argmax(
            logits,
            axis=-1
        )

        accuracy = accuracy_score(
            labels,
            predictions
        )

        precision, recall, f1, _ = (
            precision_recall_fscore_support(
                labels,
                predictions,
                average="macro",
                zero_division=0
            )
        )

        return {
            "accuracy": accuracy,
            "macro_precision": precision,
            "macro_recall": recall,
            "macro_f1": f1
        }

    # -----------------------------------------------------
    # Training arguments
    # -----------------------------------------------------

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,

        num_train_epochs=3,

        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,

        learning_rate=2e-5,
        weight_decay=0.01,

        eval_strategy="epoch",
        save_strategy="epoch",

        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,

        logging_strategy="steps",
        logging_steps=10,

        save_total_limit=2,

        report_to="none",

        # Mixed precision is used only when CUDA is present, and is
        # set identically in train_muril.py and train_xlmr.py.
        # The existing saved checkpoints predate this: XLM-R was
        # trained with fp16 on and MuRIL with it off, so the current
        # MuRIL-vs-XLM-R metrics are not a clean comparison. Retrain
        # both before reporting a head-to-head result.
        fp16=torch.cuda.is_available(),

        seed=RANDOM_SEED
    )

    # -----------------------------------------------------
    # Trainer
    # -----------------------------------------------------

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # -----------------------------------------------------
    # Train
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("STARTING MuRIL TRAINING")
    print("=" * 70)

    trainer.train()

    # -----------------------------------------------------
    # Development evaluation
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("DEVELOPMENT EVALUATION")
    print("=" * 70)

    dev_metrics = trainer.evaluate()

    for key, value in dev_metrics.items():

        if isinstance(value, float):
            print(f"{key}: {value:.4f}")

    # -----------------------------------------------------
    # Save final model
    # -----------------------------------------------------

    print()
    print("Saving model...")

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # -----------------------------------------------------
    # Save metrics
    # -----------------------------------------------------

    metrics_file = os.path.join(
        OUTPUT_DIR,
        "dev_metrics.txt"
    )

    with open(
        metrics_file,
        "w",
        encoding="utf-8"
    ) as file:

        for key, value in dev_metrics.items():

            if isinstance(value, float):
                file.write(
                    f"{key}: {value:.6f}\n"
                )
            else:
                file.write(
                    f"{key}: {value}\n"
                )

    print()
    print(f"Model output: {OUTPUT_DIR}")
    print(f"Metrics output: {metrics_file}")

    print()
    print("=" * 70)
    print("MuRIL TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()