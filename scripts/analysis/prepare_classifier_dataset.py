"""
prepare_classifier_dataset.py

Prepare the dataset for multilingual safety classification.

Gold labels:
    Human-adjudicated responses from the targeted audit.

Silver labels:
    Automatically evaluated responses without human adjudication.

The original benchmark evaluation files are never modified.
"""

import os
import pandas as pd


EVALUATION_FILE = "outputs/evaluated/combined_evaluation.csv"
HUMAN_AUDIT_FILE = "outputs/analysis/audit/human_review_queue.csv"

OUTPUT_DIR = "outputs/analysis/classifier"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "classifier_dataset.csv"
)


KEY_COLUMNS = [
    "attack_id",
    "variation_id",
    "language",
    "model",
    "attack_category"
]


def main():

    print("=" * 70)
    print("PREPARING CLASSIFIER DATASET")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load benchmark evaluation data
    # ---------------------------------------------------------

    if not os.path.exists(EVALUATION_FILE):
        raise FileNotFoundError(
            f"Evaluation file not found: {EVALUATION_FILE}"
        )

    if not os.path.exists(HUMAN_AUDIT_FILE):
        raise FileNotFoundError(
            f"Human audit file not found: {HUMAN_AUDIT_FILE}"
        )

    evaluation = pd.read_csv(
        EVALUATION_FILE,
        keep_default_na=False
    )

    human = pd.read_csv(
        HUMAN_AUDIT_FILE,
        keep_default_na=False
    )

    print(f"Benchmark responses loaded: {len(evaluation)}")
    print(f"Human audit rows loaded: {len(human)}")

    # ---------------------------------------------------------
    # Validate required columns
    # ---------------------------------------------------------

    required_evaluation = KEY_COLUMNS + [
        "prompt",
        "response",
        "safety_label"
    ]

    required_human = KEY_COLUMNS + [
        "human_label"
    ]

    for column in required_evaluation:
        if column not in evaluation.columns:
            raise ValueError(
                f"Missing evaluation column: {column}"
            )

    for column in required_human:
        if column not in human.columns:
            raise ValueError(
                f"Missing human-audit column: {column}"
            )

    # ---------------------------------------------------------
    # Keep only human-adjudicated labels
    # ---------------------------------------------------------

    human_labels = human[
        human["human_label"] != ""
    ][KEY_COLUMNS + ["human_label"]].copy()

    print(
        f"Human-adjudicated labels available: "
        f"{len(human_labels)}"
    )

    # ---------------------------------------------------------
    # Check for duplicate audit keys
    # ---------------------------------------------------------

    duplicate_human = human_labels.duplicated(
        subset=KEY_COLUMNS,
        keep=False
    )

    if duplicate_human.any():
        duplicates = human_labels[
            duplicate_human
        ]

        raise ValueError(
            "Duplicate human-audit keys found:\n"
            f"{duplicates.to_string(index=False)}"
        )

    # ---------------------------------------------------------
    # Merge human labels onto benchmark responses
    # ---------------------------------------------------------

    dataset = evaluation.merge(
        human_labels,
        on=KEY_COLUMNS,
        how="left"
    )

    # ---------------------------------------------------------
    # Assign label source
    # ---------------------------------------------------------

    # Start with the original automatic evaluator label.
    # This becomes the SILVER label for unaudited responses.
    dataset["label"] = dataset["safety_label"]

    dataset["label_source"] = "SILVER"

    # Replace the automatic label with the human-adjudicated
    # label wherever a GOLD label exists.
    human_mask = (
        dataset["human_label"].notna()
        & (dataset["human_label"].astype(str).str.strip() != "")
    )
    dataset.loc[
        human_mask,
        "label"
    ] = dataset.loc[
        human_mask,
        "human_label"
    ]

    dataset.loc[
        human_mask,
        "label_source"
    ] = "GOLD"

    # ---------------------------------------------------------
    # Validate label completeness
    # ---------------------------------------------------------

    missing_labels = (
        dataset["label"].isna()
        | (dataset["label"].astype(str).str.strip() == "")
    )

    if missing_labels.any():
        raise ValueError(
            f"Missing classifier labels detected: "
            f"{missing_labels.sum()} rows."
        )

    allowed_labels = {
        "COMPLIANCE",
        "NON_COMPLIANCE",
        "REFUSAL"
    }

    invalid_labels = set(dataset["label"]) - allowed_labels

    if invalid_labels:
        raise ValueError(
            f"Unexpected labels found: {invalid_labels}"
        )

    # ---------------------------------------------------------
    # Select classifier fields
    # ---------------------------------------------------------

    classifier_columns = [
        "attack_id",
        "variation_id",
        "language",
        "model",
        "attack_category",
        "prompt",
        "response",
        "label",
        "label_source"
    ]

    dataset = dataset[
        classifier_columns
    ].copy()

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    dataset.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print()
    print("LABEL DISTRIBUTION")
    print(dataset["label"].value_counts())

    print()
    print("LABEL SOURCE")
    print(dataset["label_source"].value_counts())

    print()
    print("GOLD LABEL DISTRIBUTION")
    print(
        dataset[
            dataset["label_source"] == "GOLD"
        ]["label"].value_counts()
    )

    print()
    print(f"Output: {OUTPUT_FILE}")

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()