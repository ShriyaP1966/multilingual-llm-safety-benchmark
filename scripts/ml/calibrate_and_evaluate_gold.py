"""
calibrate_and_evaluate_gold.py

ADDITIONAL, NON-FROZEN DIAGNOSTIC EXPERIMENT.

This script does NOT retrain anything and does NOT touch the original
frozen checkpoints or frozen evaluation outputs. It loads the existing,
already-trained XLM-R and MuRIL checkpoints exactly as they are
(outputs/analysis/classifier/xlmr and outputs/analysis/classifier/muril)
and applies an inference-time logit adjustment before argmax, to test
whether correcting for training-set class imbalance improves COMPLIANCE
recall/F1 on the GOLD audit/adjudication subset.

Motivation
----------
outputs/analysis/classifier/train_silver.csv is severely imbalanced
(COMPLIANCE ~3.5% of training rows). A softmax classifier trained on an
imbalanced set learns a decision rule that is implicitly biased toward
the majority classes, because the model's output approximates:

    P_model(y | x) ~= P(x | y) * pi_train(y) / P(x)

where pi_train(y) is the class prior IN THE TRAINING DATA. When the
training prior over-represents REFUSAL/NON_COMPLIANCE and
under-represents COMPLIANCE, argmax over P_model(y|x) is biased against
COMPLIANCE regardless of the true class-conditional likelihoods
P(x|y).

Calibration method: training-prior logit adjustment
-----------------------------------------------------
This is the standard "logit adjustment" / prior-correction technique
(Saerens, Latinne & Decaestecker, 2002; Menon et al., "Long-Tail
Learning via Logit Adjustment", ICLR 2021). For each class y, subtract
the log of its TRAINING prior from the raw logit before softmax:

    adjusted_logit(y | x) = raw_logit(y | x) - log( pi_train(y) )

Equivalently, in probability space:

    P_adjusted(y | x)  proportional-to  P_model(y | x) / pi_train(y)

Dividing out the training prior removes the model's learned bias
toward frequent training classes, which is algebraically equivalent to
assuming a UNIFORM prior at inference time (1/3 per class here). We
deliberately calibrate to a uniform prior rather than the GOLD set's
empirical class distribution, because the GOLD/audit label
distribution must never be used to fit or tune anything (see hard
constraints in outputs/analysis/classifier/BALANCED_EXPERIMENT_NOTES.md).
pi_train(y) is computed ONLY from
outputs/analysis/classifier/train_silver.csv. gold_test.csv is used
only once, at the very end, purely for measuring the calibrated
predictions -- never for fitting, tuning, or choosing this formula.

Inputs (read-only):
    outputs/analysis/classifier/xlmr                  (frozen checkpoint, untouched)
    outputs/analysis/classifier/muril                 (frozen checkpoint, untouched)
    outputs/analysis/classifier/train_silver.csv       (source of training class priors ONLY)
    outputs/analysis/classifier/gold_test.csv          (final held-out evaluation ONLY)

Outputs (new, non-frozen):
    outputs/analysis/classifier/xlmr_gold_calibrated/metrics.txt
    outputs/analysis/classifier/xlmr_gold_calibrated/predictions.csv
    outputs/analysis/classifier/muril_gold_calibrated/metrics.txt
    outputs/analysis/classifier/muril_gold_calibrated/predictions.csv

This is a diagnostic experiment. It does not replace or modify the
original V2 research results or the original frozen classifier
evaluation outputs.
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


TRAIN_FILE = "outputs/analysis/classifier/train_silver.csv"
GOLD_FILE = "outputs/analysis/classifier/gold_test.csv"

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

MODELS = [
    {
        "name": "XLM-R",
        "checkpoint_dir": "outputs/analysis/classifier/xlmr",
        "output_dir": "outputs/analysis/classifier/xlmr_gold_calibrated",
        "prefix": "xlmr_gold_calibrated"
    },
    {
        "name": "MuRIL",
        "checkpoint_dir": "outputs/analysis/classifier/muril",
        "output_dir": "outputs/analysis/classifier/muril_gold_calibrated",
        "prefix": "muril_gold_calibrated"
    }
]


def compute_training_priors(train_file):
    """Compute class priors pi_train(y) from train_silver.csv ONLY."""

    train_df = pd.read_csv(
        train_file,
        keep_default_na=False
    )

    if set(train_df["label_source"]) != {"SILVER"}:
        raise ValueError(
            "Training file contains non-SILVER labels; refusing to "
            "compute priors from it."
        )

    counts = train_df["label"].value_counts()

    total = counts.sum()

    priors = np.array([
        counts.get(label, 0) / total
        for label in LABELS_IN_ORDER
    ])

    if (priors <= 0).any():
        raise ValueError(
            "One or more classes has zero training examples; cannot "
            "take log(0) for logit adjustment."
        )

    return priors


def load_gold(gold_file):

    gold_df = pd.read_csv(
        gold_file,
        keep_default_na=False
    )

    if len(gold_df) != 63:
        raise ValueError(
            f"Expected exactly 63 GOLD rows, found {len(gold_df)}."
        )

    if set(gold_df["label_source"]) != {"GOLD"}:
        raise ValueError(
            "Gold test file contains non-GOLD rows."
        )

    allowed_labels = set(LABEL2ID.keys())

    if not set(gold_df["label"]).issubset(allowed_labels):
        raise ValueError(
            "Unexpected labels found in GOLD test data."
        )

    return gold_df


def get_raw_logits(checkpoint_dir, texts, device):

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)

    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint_dir
    )

    model.to(device)
    model.eval()

    all_logits = []

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

            outputs = model(**encoded)

            all_logits.append(
                outputs.logits.squeeze(0).cpu().numpy()
            )

    return np.stack(all_logits, axis=0)


def evaluate_and_save(
    name,
    output_dir,
    prefix,
    gold_df,
    raw_logits,
    log_priors,
    true_ids
):

    os.makedirs(output_dir, exist_ok=True)

    # ----------------------------------------------------------------
    # Inference-time logit adjustment:
    #   adjusted_logit(y|x) = raw_logit(y|x) - log(pi_train(y))
    # ----------------------------------------------------------------

    adjusted_logits = raw_logits - log_priors[np.newaxis, :]

    predicted_ids = np.argmax(adjusted_logits, axis=-1)

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

    report = classification_report(
        true_ids,
        predicted_ids,
        labels=[0, 1, 2],
        target_names=LABELS_IN_ORDER,
        zero_division=0
    )

    cm = confusion_matrix(
        true_ids,
        predicted_ids,
        labels=[0, 1, 2]
    )

    print()
    print("=" * 70)
    print(f"{name} CALIBRATED GOLD RESULTS")
    print("=" * 70)
    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Macro Precision:   {macro_precision:.4f}")
    print(f"Macro Recall:      {macro_recall:.4f}")
    print(f"Macro F1:          {macro_f1:.4f}")
    print()
    print(report)
    print("CONFUSION MATRIX")
    print(cm)

    # ----------------------------------------------------------------
    # Save predictions.csv
    # ----------------------------------------------------------------

    results_df = gold_df.copy()

    results_df["predicted_label"] = [
        ID2LABEL[int(p)]
        for p in predicted_ids
    ]

    results_df["correct"] = (
        results_df["label"] == results_df["predicted_label"]
    )

    predictions_file = os.path.join(output_dir, "predictions.csv")

    results_df.to_csv(predictions_file, index=False, encoding="utf-8")

    # ----------------------------------------------------------------
    # Save metrics.txt
    # ----------------------------------------------------------------

    metrics_file = os.path.join(output_dir, "metrics.txt")

    with open(metrics_file, "w", encoding="utf-8") as file:

        file.write(f"{name} GOLD TEST EVALUATION (PRIOR-CALIBRATED)\n")
        file.write("=" * 70 + "\n\n")
        file.write(
            "Non-frozen diagnostic experiment. Original checkpoint is "
            "unmodified; only inference-time logits are adjusted.\n\n"
        )
        file.write(
            "Calibration: adjusted_logit(y|x) = raw_logit(y|x) - "
            "log(pi_train(y)), pi_train computed from "
            "train_silver.csv only.\n\n"
        )
        file.write(
            "Training priors (pi_train) by class:\n"
        )

        for label, prior in zip(LABELS_IN_ORDER, np.exp(log_priors)):
            file.write(f"  {label}: {prior:.6f}\n")

        file.write(f"\nGold test rows: {len(gold_df)}\n\n")
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


def main():

    print("=" * 70)
    print("INFERENCE-TIME PRIOR CALIBRATION (NON-FROZEN EXPERIMENT)")
    print("=" * 70)

    for cfg in MODELS:
        if not os.path.exists(cfg["checkpoint_dir"]):
            raise FileNotFoundError(
                f"Original {cfg['name']} checkpoint not found: "
                f"{cfg['checkpoint_dir']}. This script requires the "
                f"existing frozen checkpoint and will not retrain or "
                f"guess a substitute."
            )

    if not os.path.exists(TRAIN_FILE):
        raise FileNotFoundError(f"Training file not found: {TRAIN_FILE}")

    if not os.path.exists(GOLD_FILE):
        raise FileNotFoundError(f"Gold test file not found: {GOLD_FILE}")

    priors = compute_training_priors(TRAIN_FILE)
    log_priors = np.log(priors)

    print()
    print("Training class priors (from train_silver.csv only):")
    for label, prior in zip(LABELS_IN_ORDER, priors):
        print(f"  {label}: {prior:.6f}")

    gold_df = load_gold(GOLD_FILE)

    texts = gold_df["response"].astype(str).tolist()

    true_ids = np.array([
        LABEL2ID[label]
        for label in gold_df["label"]
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print()
    print(f"Device: {device}")

    for cfg in MODELS:

        print()
        print(f"Loading original {cfg['name']} checkpoint from "
              f"{cfg['checkpoint_dir']} (read-only, not modified)...")

        raw_logits = get_raw_logits(
            cfg["checkpoint_dir"],
            texts,
            device
        )

        evaluate_and_save(
            cfg["name"],
            cfg["output_dir"],
            cfg["prefix"],
            gold_df,
            raw_logits,
            log_priors,
            true_ids
        )

    print()
    print("=" * 70)
    print("CALIBRATION EXPERIMENT COMPLETE")
    print(
        "Original checkpoints and frozen evaluation outputs were not "
        "modified."
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
