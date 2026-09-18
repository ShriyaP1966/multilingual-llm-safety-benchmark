"""
train_final_classifier.py

Train and evaluate the final response-safety classifier for
"Cross-Lingual Vulnerability and Prompt Injection in Low-Resource
Languages".

Task
----
    LLM response -> MuRIL / XLM-R -> COMPLIANCE | NON_COMPLIANCE | REFUSAL

Input is the response text only. Metadata is retained in the split
files for analysis but is never given to the model.

What this adds over train_muril.py / train_xlmr.py
--------------------------------------------------
Those scripts are preserved and unmodified, along with their
checkpoints and metrics; they constitute the initial targeted-audit
experiment. This script is the final experiment and differs in:

  - it consumes the grouped, leakage-checked split from
    prepare_final_split.py
  - it applies balanced class weights, since COMPLIANCE is ~4% of
    silver training data
  - it always reports the majority-class baseline beside accuracy
  - it reports per-class precision / recall / F1 and macro F1 on both
    the silver dev set and the human gold set
  - it writes to a separate output directory, leaving the earlier
    experiment intact

Reporting constraints
---------------------
The 63 gold labels come from a targeted audit that oversampled
suspected evaluator errors. Gold metrics are targeted-audit
performance, not population-level accuracy, and the script prints that
alongside the numbers so it cannot be quietly dropped.

Usage
-----
    python scripts/ml/train_final_classifier.py --model xlmr
    python scripts/ml/train_final_classifier.py --model muril
    python scripts/ml/train_final_classifier.py --model xlmr --no-class-weights
"""

from pathlib import Path

import argparse
import json

import numpy as np
import pandas as pd
import torch

from datasets import Dataset

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SPLIT_DIR = (
    PROJECT_ROOT / "outputs" / "analysis" / "classifier_final"
)

MODEL_CHOICES = {
    "xlmr": "xlm-roberta-base",
    "muril": "google/muril-base-cased",
}

LABELS = ["COMPLIANCE", "NON_COMPLIANCE", "REFUSAL"]
LABEL2ID = {label: index for index, label in enumerate(LABELS)}
ID2LABEL = {index: label for label, index in LABEL2ID.items()}

MAX_LENGTH = 512
RANDOM_SEED = 42
EPOCHS = 3
LEARNING_RATE = 2e-5


class WeightedTrainer(Trainer):
    """
    Trainer with a class-weighted cross-entropy loss.

    Needed because REFUSAL outnumbers COMPLIANCE roughly 16:1 in the
    silver training split; unweighted training collapses onto the
    majority classes.
    """

    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        **kwargs,
    ):
        labels = inputs.pop("labels")

        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
        else:
            weight = None

        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, len(LABELS)),
            labels.view(-1),
            weight=weight,
        )

        return (loss, outputs) if return_outputs else loss


def load_split(name):
    """
    Load one split file and validate it.
    """

    path = SPLIT_DIR / name

    if not path.exists():
        raise FileNotFoundError(
            f"Split file not found: {path}\n"
            f"Run scripts/ml/prepare_final_split.py first."
        )

    frame = pd.read_csv(path, keep_default_na=False)

    unexpected = set(frame["label"]) - set(LABELS)

    if unexpected:
        raise ValueError(f"Unexpected labels in {name}: {unexpected}")

    empty = frame["response"].astype(str).str.strip() == ""

    if empty.any():
        raise ValueError(f"Empty responses in {name}: {empty.sum()}")

    return frame


def to_dataset(frame, tokenizer):
    """
    Tokenize response text only.
    """

    data = pd.DataFrame({
        "text": frame["response"].astype(str),
        "labels": frame["label"].map(LABEL2ID),
    })

    if data["labels"].isna().any():
        raise ValueError("Unmapped labels encountered.")

    dataset = Dataset.from_pandas(data, preserve_index=False)

    return dataset.map(
        lambda batch: tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
        ),
        batched=True,
    )


def metrics_block(y_true, y_pred, split_name):
    """
    Accuracy, macro scores, per-class scores and the majority baseline.
    """

    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    counts = pd.Series(y_true).value_counts()

    baseline_label = counts.index[0]
    baseline = counts.iloc[0] / len(y_true)

    per_class = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(LABELS))),
        zero_division=0,
    )

    return {
        "split": split_name,
        "n": int(len(y_true)),
        "accuracy": round(float(accuracy), 4),
        "macro_precision": round(float(precision), 4),
        "macro_recall": round(float(recall), 4),
        "macro_f1": round(float(f1), 4),
        "majority_baseline": round(float(baseline), 4),
        "majority_class": ID2LABEL[int(baseline_label)],
        "beats_baseline": bool(accuracy > baseline),
        "per_class": {
            LABELS[index]: {
                "precision": round(float(per_class[0][index]), 4),
                "recall": round(float(per_class[1][index]), 4),
                "f1": round(float(per_class[2][index]), 4),
                "support": int(per_class[3][index]),
            }
            for index in range(len(LABELS))
        },
    }


def print_metrics(block):
    """
    Print one metrics block with the baseline made unavoidable.
    """

    print()
    print(f"  {block['split']}  (n={block['n']})")
    print(f"    accuracy          : {block['accuracy']:.4f}")
    print(f"    majority baseline : {block['majority_baseline']:.4f} "
          f"(always {block['majority_class']})")
    print(f"    beats baseline    : "
          f"{'YES' if block['beats_baseline'] else 'NO'}")
    print(f"    macro F1          : {block['macro_f1']:.4f}")
    print(f"    macro precision   : {block['macro_precision']:.4f}")
    print(f"    macro recall      : {block['macro_recall']:.4f}")
    print(f"    {'class':18s}{'P':>8s}{'R':>8s}{'F1':>8s}{'n':>7s}")

    for label in LABELS:
        row = block["per_class"][label]
        print(
            f"    {label:18s}{row['precision']:8.3f}"
            f"{row['recall']:8.3f}{row['f1']:8.3f}{row['support']:7d}"
        )


def main():

    parser = argparse.ArgumentParser(
        description="Train the final response-safety classifier."
    )

    parser.add_argument(
        "--model",
        choices=sorted(MODEL_CHOICES),
        required=True,
    )
    parser.add_argument("--outdir", default=None)
    parser.add_argument(
        "--split-dir",
        default=None,
        help=(
            "Directory holding train_silver.csv / dev_silver.csv / "
            "gold_test.csv / class_weights.json. Defaults to "
            "outputs/analysis/classifier_final (the v1-silver split). "
            "Pass a v2 split dir to train on v2 silver labels."
        ),
    )
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable balanced class weights (for ablation).",
    )

    arguments = parser.parse_args()

    global SPLIT_DIR
    if arguments.split_dir:
        SPLIT_DIR = Path(arguments.split_dir)

    checkpoint = MODEL_CHOICES[arguments.model]

    suffix = "" if not arguments.no_class_weights else "_unweighted"

    output_dir = Path(
        arguments.outdir
        or SPLIT_DIR / f"{arguments.model}_final{suffix}"
    )

    print("=" * 70)
    print(f"FINAL CLASSIFIER — {arguments.model.upper()}")
    print("=" * 70)

    set_seed(RANDOM_SEED)

    train_frame = load_split("train_silver.csv")
    dev_frame = load_split("dev_silver.csv")
    gold_frame = load_split("gold_test.csv")

    if set(train_frame["label_source"]) != {"SILVER"}:
        raise ValueError("Training data must be SILVER only.")

    if set(dev_frame["label_source"]) != {"SILVER"}:
        raise ValueError("Dev data must be SILVER only.")

    if set(gold_frame["label_source"]) != {"GOLD"}:
        raise ValueError("Gold data must be GOLD only.")

    print(f"Base model : {checkpoint}")
    print(f"Train      : {len(train_frame)} (silver)")
    print(f"Dev        : {len(dev_frame)} (silver)")
    print(f"Gold test  : {len(gold_frame)} (human-adjudicated)")

    weights_path = SPLIT_DIR / "class_weights.json"

    class_weights = None

    if not arguments.no_class_weights:

        if not weights_path.exists():
            raise FileNotFoundError(
                f"Class weights not found: {weights_path}"
            )

        stored = json.loads(weights_path.read_text(encoding="utf-8"))

        class_weights = torch.tensor(
            [float(stored["weights"][label]) for label in LABELS],
            dtype=torch.float,
        )

        print(f"Class weights: "
              f"{ {l: round(float(w), 3) for l, w in zip(LABELS, class_weights)} }")
    else:
        print("Class weights: DISABLED (ablation)")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    train_dataset = to_dataset(train_frame, tokenizer)
    dev_dataset = to_dataset(dev_frame, tokenizer)
    gold_dataset = to_dataset(gold_frame, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        accuracy = accuracy_score(labels, predictions)

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average="macro", zero_division=0
        )

        return {
            "accuracy": accuracy,
            "macro_precision": precision,
            "macro_recall": recall,
            "macro_f1": f1,
        }

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        # Set identically for both models so the comparison is not
        # confounded by numerical precision.
        fp16=torch.cuda.is_available(),
        seed=RANDOM_SEED,
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    print()
    print("Training...")

    trainer.train()

    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    print()
    print("RESULTS")

    for name, dataset, frame in [
        ("silver dev", dev_dataset, dev_frame),
        ("human gold", gold_dataset, gold_frame),
    ]:

        prediction = trainer.predict(dataset)

        y_pred = np.argmax(prediction.predictions, axis=-1)
        y_true = np.array(
            [LABEL2ID[label] for label in frame["label"]]
        )

        block = metrics_block(y_true, y_pred, name)

        results[name] = block

        print_metrics(block)

        matrix = pd.DataFrame(
            confusion_matrix(
                y_true, y_pred, labels=list(range(len(LABELS)))
            ),
            index=[f"true_{label}" for label in LABELS],
            columns=[f"pred_{label}" for label in LABELS],
        )

        matrix.to_csv(
            output_dir
            / f"{name.replace(' ', '_')}_confusion_matrix.csv",
            encoding="utf-8",
        )

        predictions_frame = frame.copy()
        predictions_frame["predicted_label"] = [
            ID2LABEL[int(value)] for value in y_pred
        ]

        predictions_frame.to_csv(
            output_dir
            / f"{name.replace(' ', '_')}_predictions.csv",
            index=False,
            encoding="utf-8",
        )

        report_text = classification_report(
            y_true,
            y_pred,
            labels=list(range(len(LABELS))),
            target_names=LABELS,
            zero_division=0,
        )

        (
            output_dir
            / f"{name.replace(' ', '_')}_classification_report.txt"
        ).write_text(report_text, encoding="utf-8")

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    summary = {
        "title": (
            "Cross-Lingual Vulnerability and Prompt Injection in "
            "Low-Resource Languages"
        ),
        "experiment": "final response-safety classifier",
        "model_key": arguments.model,
        "base_model": checkpoint,
        "class_weights_applied": not arguments.no_class_weights,
        "class_weights": (
            {
                label: round(float(weight), 4)
                for label, weight in zip(LABELS, class_weights)
            }
            if class_weights is not None else None
        ),
        "split_dir": str(SPLIT_DIR),
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "seed": RANDOM_SEED,
        "fp16": bool(torch.cuda.is_available()),
        "results": results,
        "reporting_constraints": [
            "Gold labels come from a targeted audit that oversampled "
            "suspected evaluator errors; gold metrics are "
            "targeted-audit performance, not population accuracy.",
            "Accuracy must always be reported beside the "
            "majority-class baseline recorded above.",
            "Training labels are evaluator-produced silver labels, "
            "not ground truth.",
            "Class weights address training imbalance and do not "
            "correct train/gold distribution shift.",
        ],
    }

    (output_dir / "final_metrics.json").write_text(
        json.dumps(summary, indent=4),
        encoding="utf-8",
    )

    print()
    print(f"Written to: {output_dir}")
    print()
    print("Gold metrics are TARGETED-AUDIT performance, not")
    print("population-level accuracy. Report beside the baseline.")
    print("=" * 70)


if __name__ == "__main__":
    main()
