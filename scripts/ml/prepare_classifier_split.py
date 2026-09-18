"""
prepare_classifier_split.py

Create a leakage-safe split for multilingual safety classification.

Dataset:
    624 benchmark responses

Labels:
    GOLD   = human-adjudicated labels
    SILVER = automatic evaluator labels

Design:
    - GOLD responses are kept completely out of training.
    - SILVER data is split into train/development sets.
    - Exact duplicate responses are kept in the same split.
    - The classifier input will be RESPONSE ONLY.
    - Metadata is retained for analysis but is NOT model input.
"""

import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split


INPUT_FILE = (
    "outputs/analysis/classifier/classifier_dataset.csv"
)

OUTPUT_DIR = "outputs/analysis/classifier"

TRAIN_FILE = os.path.join(
    OUTPUT_DIR,
    "train_silver.csv"
)

DEV_FILE = os.path.join(
    OUTPUT_DIR,
    "dev_silver.csv"
)

GOLD_FILE = os.path.join(
    OUTPUT_DIR,
    "gold_test.csv"
)


RANDOM_STATE = 42

# Sizes of the recorded 624-response study. These are asserted so an
# unnoticed change in the upstream data is caught, but they stop being
# true as soon as a larger human-validation sample lands, so
# --allow-size-change downgrades the check to a warning.
EXPECTED_GOLD = 63
EXPECTED_SILVER = 561


def report_label_distribution(train, dev, gold):
    """
    Report label distribution across the three splits.

    A classifier trained on one label distribution and tested on a very
    different one produces metrics that describe the split rather than
    the model. The recorded study hit exactly that: the targeted audit
    pulled almost every COMPLIANCE case into the gold set, leaving
    training with 3.5% COMPLIANCE against a gold set that is 52%
    COMPLIANCE, and both classifiers then scored zero on that class and
    landed below the majority baseline.

    This is a diagnostic only. It does not alter the split.
    """

    print()
    print("LABEL DISTRIBUTION ACROSS SPLITS")
    print("-" * 70)

    labels = sorted(
        set(train["label"])
        | set(dev["label"])
        | set(gold["label"])
    )

    shares = {}

    header = "label".ljust(18)

    for name in ["train", "dev", "gold"]:
        header += name.rjust(17)

    print(header)

    for label in labels:

        row = label.ljust(18)

        for name, frame in [
            ("train", train),
            ("dev", dev),
            ("gold", gold),
        ]:

            count = (frame["label"] == label).sum()

            share = (
                100 * count / len(frame)
                if len(frame) else 0.0
            )

            shares[(name, label)] = share

            row += f"{count:6d} ({share:5.1f}%)".rjust(17)

        print(row)

    # --------------------------------------------------------
    # Majority baseline on the gold set.
    #
    # Any reported gold accuracy must be compared against this,
    # not against zero.
    # --------------------------------------------------------

    gold_counts = gold["label"].value_counts()

    baseline_label = gold_counts.index[0]
    baseline = 100 * gold_counts.iloc[0] / len(gold)

    print()
    print(
        f"Gold majority baseline: {baseline:.2f}% "
        f"(always predict {baseline_label})"
    )

    # --------------------------------------------------------
    # Largest train-to-gold divergence.
    # --------------------------------------------------------

    divergences = [
        (
            abs(
                shares[("train", label)]
                - shares[("gold", label)]
            ),
            label,
        )
        for label in labels
    ]

    worst, worst_label = max(divergences)

    print(
        f"Largest train/gold divergence: {worst_label} "
        f"({shares[('train', worst_label)]:.1f}% vs "
        f"{shares[('gold', worst_label)]:.1f}%, "
        f"{worst:.1f} points)"
    )

    if worst >= 20.0:
        print()
        print("WARNING: severe train/gold distribution shift.")
        print(
            "  Gold metrics will largely reflect this shift rather"
        )
        print(
            "  than classifier quality. Draw the test set from a"
        )
        print(
            "  representative sample, not a difficulty-targeted"
        )
        print(
            "  audit queue, before reporting a head-to-head result."
        )

    return worst


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Create a leakage-safe split for multilingual safety "
            "classification."
        )
    )

    parser.add_argument(
        "--allow-size-change",
        action="store_true",
        help=(
            "Warn instead of failing when the GOLD/SILVER counts "
            "differ from the recorded study. Use this once a larger "
            "human-validation sample is available."
        ),
    )

    arguments = parser.parse_args()

    print("=" * 70)
    print("PREPARING LEAKAGE-SAFE CLASSIFIER SPLIT")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Classifier dataset not found: {INPUT_FILE}"
        )

    dataset = pd.read_csv(
        INPUT_FILE,
        keep_default_na=False
    )

    print(f"Total responses loaded: {len(dataset)}")

    # ---------------------------------------------------------
    # Validate required columns
    # ---------------------------------------------------------

    required_columns = [
        "response",
        "label",
        "label_source"
    ]

    for column in required_columns:
        if column not in dataset.columns:
            raise ValueError(
                f"Missing required column: {column}"
            )

    # ---------------------------------------------------------
    # Validate response text
    # ---------------------------------------------------------

    empty_response = (
        dataset["response"]
        .astype(str)
        .str.strip()
        == ""
    )

    if empty_response.any():
        raise ValueError(
            f"Empty responses detected: "
            f"{empty_response.sum()}"
        )

    # ---------------------------------------------------------
    # Separate GOLD and SILVER data
    # ---------------------------------------------------------

    gold = dataset[
        dataset["label_source"] == "GOLD"
    ].copy()

    silver = dataset[
        dataset["label_source"] == "SILVER"
    ].copy()

    print(f"GOLD responses: {len(gold)}")
    print(f"SILVER responses: {len(silver)}")

    for name, actual, expected in [
        ("GOLD", len(gold), EXPECTED_GOLD),
        ("SILVER", len(silver), EXPECTED_SILVER),
    ]:

        if actual == expected:
            continue

        message = (
            f"Expected {expected} {name} responses, found {actual}."
        )

        if arguments.allow_size_change:
            print(f"WARNING: {message}")
        else:
            raise ValueError(
                f"{message} Pass --allow-size-change if this is "
                f"intentional (for example after adding a larger "
                f"human-validation sample)."
            )

    # ---------------------------------------------------------
    # Check exact duplicate responses
    # ---------------------------------------------------------

    silver["response_group"] = (
        silver["response"]
        .astype(str)
        .str.strip()
    )

    duplicate_groups = (
        silver["response_group"]
        .value_counts()
    )

    duplicate_groups = duplicate_groups[
        duplicate_groups > 1
    ]

    print()
    print(
        f"Duplicate SILVER response groups: "
        f"{len(duplicate_groups)}"
    )

    print(
        f"Rows involved in duplicate groups: "
        f"{duplicate_groups.sum()}"
    )

    # ---------------------------------------------------------
    # Create group-level labels
    # ---------------------------------------------------------

    groups = (
        silver[
            [
                "response_group",
                "label"
            ]
        ]
        .drop_duplicates()
        .copy()
    )

    # A response text should not have conflicting labels.
    conflicting_groups = (
        silver.groupby("response_group")["label"]
        .nunique()
    )

    conflicting_groups = conflicting_groups[
        conflicting_groups > 1
    ]

    if len(conflicting_groups) > 0:
        raise ValueError(
            "Conflicting labels found for identical "
            "response texts."
        )

    # ---------------------------------------------------------
    # Split unique response groups
    # ---------------------------------------------------------

    train_groups, dev_groups = train_test_split(
        groups,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=groups["label"]
    )

    train_group_values = set(
        train_groups["response_group"]
    )

    dev_group_values = set(
        dev_groups["response_group"]
    )

    # ---------------------------------------------------------
    # Assign rows to train/dev
    # ---------------------------------------------------------

    train = silver[
        silver["response_group"]
        .isin(train_group_values)
    ].copy()

    dev = silver[
        silver["response_group"]
        .isin(dev_group_values)
    ].copy()

    # ---------------------------------------------------------
    # Remove helper column
    # ---------------------------------------------------------

    train.drop(
        columns=["response_group"],
        inplace=True
    )

    dev.drop(
        columns=["response_group"],
        inplace=True
    )

    # ---------------------------------------------------------
    # Verify no response leakage
    # ---------------------------------------------------------

    train_responses = set(
        train["response"]
        .astype(str)
        .str.strip()
    )

    dev_responses = set(
        dev["response"]
        .astype(str)
        .str.strip()
    )

    overlap = train_responses.intersection(
        dev_responses
    )

    if overlap:
        raise ValueError(
            f"Response leakage detected: "
            f"{len(overlap)} overlapping responses."
        )

    # ---------------------------------------------------------
    # Verify GOLD does not enter training
    # ---------------------------------------------------------

    if "GOLD" in train["label_source"].unique():
        raise ValueError(
            "GOLD data detected in training set."
        )

    if "GOLD" in dev["label_source"].unique():
        raise ValueError(
            "GOLD data detected in development set."
        )

    # ---------------------------------------------------------
    # Save files
    # ---------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    train.to_csv(
        TRAIN_FILE,
        index=False,
        encoding="utf-8"
    )

    dev.to_csv(
        DEV_FILE,
        index=False,
        encoding="utf-8"
    )

    gold.to_csv(
        GOLD_FILE,
        index=False,
        encoding="utf-8"
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print()
    print("SPLIT SUMMARY")
    print("-" * 70)

    print(f"Train SILVER: {len(train)}")
    print(f"Dev SILVER:   {len(dev)}")
    print(f"Gold test:    {len(gold)}")
    print(
        f"Total:        "
        f"{len(train) + len(dev) + len(gold)}"
    )

    print()
    print("TRAIN LABEL DISTRIBUTION")
    print(train["label"].value_counts())

    print()
    print("DEV LABEL DISTRIBUTION")
    print(dev["label"].value_counts())

    print()
    print("GOLD TEST LABEL DISTRIBUTION")
    print(gold["label"].value_counts())

    print()
    print("GOLD TEST SOURCE")
    print(gold["label_source"].value_counts())

    print()
    print(f"Train output: {TRAIN_FILE}")
    print(f"Dev output:   {DEV_FILE}")
    print(f"Gold output:  {GOLD_FILE}")

    shift = report_label_distribution(train, dev, gold)

    print()
    print("=" * 70)

    # The leakage check above covers duplicate response text and GOLD
    # contamination. It says nothing about distribution shift, so state
    # both outcomes rather than the reassuring half.
    print("RESPONSE-OVERLAP AND GOLD-ISOLATION CHECKS PASSED")

    if shift >= 20.0:
        print("DISTRIBUTION CHECK FAILED — see warning above")

    print("=" * 70)


if __name__ == "__main__":
    main()