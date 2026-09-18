"""
prepare_final_split.py

Final leakage-safe classifier split for
"Cross-Lingual Vulnerability and Prompt Injection in Low-Resource
Languages".

Task
----
Input is an LLM RESPONSE; output is its safety class. This is a
response-safety classifier, not a prompt-harmfulness classifier.

    LLM response -> MuRIL / XLM-R -> COMPLIANCE | NON_COMPLIANCE | REFUSAL

What this changes versus prepare_classifier_split.py
----------------------------------------------------
The earlier split is preserved and not modified. It deduplicated on
exact response text, which prevents only verbatim overlap. The real
leakage vector in this benchmark is different: the same prompt set is
answered in three languages by two models, producing six related
responses. Exact-text deduplication lets those land on opposite sides
of the split.

This script groups by the benchmark's actual unit of independence, so
all responses derived from one prompt stay together.

    --grouping prompt_set   (default)  attack_id x variation_id, 104 groups
    --grouping attack_id               13 groups, coarser

prompt_set is the default because it matches the unit of independence
used in the statistical analysis while still leaving enough groups for
a usable dev split. attack_id is available for a stricter check that no
attack category is shared between train and dev.

Gold / test set
---------------
The 63 existing human-adjudicated labels are used unchanged as the
human-grounded test set. They are NOT relabelled, replaced or
resampled.

They come from a TARGETED AUDIT that deliberately oversampled suspected
evaluator errors, so their class distribution is not representative of
the 624-response population. This script quantifies that shift and
prints it rather than hiding it. Gold results must therefore be
reported as targeted-audit performance, never as population-level
classifier accuracy.

Class weights
-------------
Balanced class weights are computed from the training split and written
alongside it, because COMPLIANCE is severely under-represented in
silver training data. Weights alone do not fix the distribution shift
in the gold set, and this is stated in the report.

Outputs
-------
outputs/analysis/classifier_final/
    train_silver.csv
    dev_silver.csv
    gold_test.csv
    class_weights.json
    split_report.md

Usage
-----
    python scripts/ml/prepare_final_split.py
    python scripts/ml/prepare_final_split.py --grouping attack_id
"""

from pathlib import Path

import argparse
import json

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT = (
    PROJECT_ROOT / "outputs" / "analysis" / "classifier"
    / "classifier_dataset.csv"
)

DEFAULT_OUTDIR = (
    PROJECT_ROOT / "outputs" / "analysis" / "classifier_final"
)

LABELS = ["COMPLIANCE", "NON_COMPLIANCE", "REFUSAL"]

DEV_FRACTION = 0.20
RANDOM_STATE = 42


def build_group_key(frame, grouping):
    """
    Return the grouping series used to keep related responses together.
    """

    if grouping == "prompt_set":
        return (
            frame["attack_id"].astype(str)
            + "|"
            + frame["variation_id"].astype(str)
        )

    if grouping == "attack_id":
        return frame["attack_id"].astype(str)

    raise ValueError(f"Unknown grouping: {grouping}")


def distribution(frame):
    """
    Label counts and percentage shares.
    """

    counts = frame["label"].value_counts()

    return {
        label: {
            "count": int(counts.get(label, 0)),
            "pct": round(
                100 * counts.get(label, 0) / len(frame), 2
            ) if len(frame) else 0.0,
        }
        for label in LABELS
    }


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Final leakage-safe classifier split grouped by the "
            "benchmark's unit of independence."
        )
    )

    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument(
        "--grouping",
        choices=["prompt_set", "attack_id"],
        default="prompt_set",
    )

    arguments = parser.parse_args()

    input_path = Path(arguments.input)
    output_dir = Path(arguments.outdir)

    print("=" * 70)
    print("FINAL CLASSIFIER SPLIT")
    print("=" * 70)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Classifier dataset not found: {input_path}"
        )

    dataset = pd.read_csv(input_path, keep_default_na=False)

    required = [
        "attack_id", "variation_id", "language", "model",
        "attack_category", "response", "label", "label_source",
    ]

    missing = [c for c in required if c not in dataset.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    print(f"Input: {input_path}")
    print(f"Rows : {len(dataset)}")
    print(f"Grouping: {arguments.grouping}")

    empty = dataset["response"].astype(str).str.strip() == ""

    if empty.any():
        raise ValueError(
            f"Empty responses detected: {empty.sum()}"
        )

    unexpected = set(dataset["label"]) - set(LABELS)

    if unexpected:
        raise ValueError(f"Unexpected labels: {unexpected}")

    # --------------------------------------------------------
    # GOLD stays exactly as it is.
    # --------------------------------------------------------

    gold = dataset[dataset["label_source"] == "GOLD"].copy()
    silver = dataset[dataset["label_source"] == "SILVER"].copy()

    print(f"GOLD (human-adjudicated, unchanged): {len(gold)}")
    print(f"SILVER (evaluator-labelled)        : {len(silver)}")

    if gold.empty:
        raise ValueError("No GOLD rows found.")

    # --------------------------------------------------------
    # Grouped train/dev split over SILVER only.
    # --------------------------------------------------------

    silver["group_key"] = build_group_key(silver, arguments.grouping)

    groups = silver["group_key"].to_numpy()

    n_groups = silver["group_key"].nunique()

    print(f"Groups available in SILVER: {n_groups}")

    if n_groups < 5:
        raise ValueError(
            f"Only {n_groups} groups; too few for a meaningful "
            f"grouped split. Use --grouping prompt_set."
        )

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=DEV_FRACTION,
        random_state=RANDOM_STATE,
    )

    train_index, dev_index = next(
        splitter.split(silver, groups=groups)
    )

    train = silver.iloc[train_index].copy()
    dev = silver.iloc[dev_index].copy()

    # --------------------------------------------------------
    # Leakage checks
    # --------------------------------------------------------

    train_groups = set(train["group_key"])
    dev_groups = set(dev["group_key"])

    group_overlap = train_groups & dev_groups

    if group_overlap:
        raise ValueError(
            f"Group leakage: {len(group_overlap)} groups in both "
            f"train and dev."
        )

    train_text = set(train["response"].astype(str).str.strip())

    # Two different prompt sets can independently produce an identical
    # response (typically a short generic refusal). Grouping cannot
    # catch that, so drop such rows from dev: keeping them would let
    # the model score on text it trained on verbatim.
    dev_stripped = dev["response"].astype(str).str.strip()

    duplicate_text = dev_stripped.isin(train_text)

    text_overlap_removed = int(duplicate_text.sum())

    if text_overlap_removed:
        dev = dev[~duplicate_text].copy()

    dev_text = set(dev["response"].astype(str).str.strip())

    text_overlap = train_text & dev_text

    if "GOLD" in set(train["label_source"]) | set(dev["label_source"]):
        raise ValueError("GOLD data leaked into train or dev.")

    gold_text = set(gold["response"].astype(str).str.strip())

    gold_in_train = gold_text & train_text
    gold_in_dev = gold_text & dev_text

    print()
    print("LEAKAGE CHECKS")
    print(f"  group overlap train/dev      : {len(group_overlap)}")
    print(f"  exact response overlap       : {len(text_overlap)}")
    print(f"  dev rows dropped as duplicate text: "
          f"{text_overlap_removed}")
    print(f"  gold response text in train  : {len(gold_in_train)}")
    print(f"  gold response text in dev    : {len(gold_in_dev)}")

    # --------------------------------------------------------
    # Class weights from the training split.
    # --------------------------------------------------------

    present = [
        label for label in LABELS
        if (train["label"] == label).any()
    ]

    counts = np.array(
        [(train["label"] == label).sum() for label in present],
        dtype=float,
    )

    weights_values = len(train) / (len(present) * counts)

    class_weights = {
        label: round(float(weight), 4)
        for label, weight in zip(present, weights_values)
    }

    for label in LABELS:
        class_weights.setdefault(label, 0.0)

    # --------------------------------------------------------
    # Distribution shift
    # --------------------------------------------------------

    train_dist = distribution(train)
    dev_dist = distribution(dev)
    gold_dist = distribution(gold)

    shifts = {
        label: round(
            abs(train_dist[label]["pct"] - gold_dist[label]["pct"]), 1
        )
        for label in LABELS
    }

    worst_label = max(shifts, key=shifts.get)
    worst_shift = shifts[worst_label]

    gold_counts = gold["label"].value_counts()
    baseline_label = gold_counts.index[0]
    baseline = round(100 * gold_counts.iloc[0] / len(gold), 2)

    print()
    print("LABEL DISTRIBUTION")
    print(f"  {'label':18s}{'train':>16s}{'dev':>16s}{'gold':>16s}")

    for label in LABELS:
        print(
            f"  {label:18s}"
            f"{train_dist[label]['count']:6d} "
            f"({train_dist[label]['pct']:5.1f}%)"
            f"{dev_dist[label]['count']:7d} "
            f"({dev_dist[label]['pct']:5.1f}%)"
            f"{gold_dist[label]['count']:7d} "
            f"({gold_dist[label]['pct']:5.1f}%)"
        )

    print()
    print(f"  Gold majority baseline: {baseline}% "
          f"(always predict {baseline_label})")
    print(f"  Largest train/gold shift: {worst_label} "
          f"{worst_shift} points")
    print(f"  Class weights: {class_weights}")

    if worst_shift >= 20.0:
        print()
        print("  NOTE: gold remains distribution-shifted relative to")
        print("  training. This is inherent to reusing the targeted")
        print("  audit as the test set. Gold metrics are therefore")
        print("  targeted-audit performance, not population accuracy.")

    # --------------------------------------------------------
    # Write
    # --------------------------------------------------------

    output_dir.mkdir(parents=True, exist_ok=True)

    for frame, name in [
        (train, "train_silver.csv"),
        (dev, "dev_silver.csv"),
        (gold, "gold_test.csv"),
    ]:
        frame.drop(
            columns=["group_key"], errors="ignore"
        ).to_csv(
            output_dir / name, index=False, encoding="utf-8"
        )

    (output_dir / "class_weights.json").write_text(
        json.dumps(
            {
                "grouping": arguments.grouping,
                "computed_from": "train_silver.csv",
                "method": "balanced: n / (n_classes * class_count)",
                "weights": class_weights,
            },
            indent=4,
        ),
        encoding="utf-8",
    )

    lines = []
    lines.append("# Final classifier split")
    lines.append("")
    lines.append(
        "Cross-Lingual Vulnerability and Prompt Injection in "
        "Low-Resource Languages"
    )
    lines.append("")
    lines.append("## Task")
    lines.append("")
    lines.append(
        "Response-safety classification. Input is an LLM response; "
        "output is COMPLIANCE, NON_COMPLIANCE or REFUSAL. This is not "
        "a prompt-harmfulness classifier."
    )
    lines.append("")
    lines.append("## Split design")
    lines.append("")
    lines.append(f"- Grouping: `{arguments.grouping}`")
    lines.append(f"- Groups in silver: {n_groups}")
    lines.append(f"- Dev fraction: {DEV_FRACTION}")
    lines.append(f"- Random state: {RANDOM_STATE}")
    lines.append("")
    lines.append(
        "Responses derived from one prompt set are kept together, "
        "because the same prompt is answered in three languages by two "
        "models and those six responses are closely related. "
        "Deduplicating on exact response text alone would allow them "
        "to be split across train and dev."
    )
    lines.append("")
    lines.append("## Sizes")
    lines.append("")
    lines.append("| split | rows | source |")
    lines.append("|---|---:|---|")
    lines.append(f"| train | {len(train)} | evaluator (silver) |")
    lines.append(f"| dev | {len(dev)} | evaluator (silver) |")
    lines.append(f"| gold test | {len(gold)} | human-adjudicated |")
    lines.append("")
    lines.append("## Label distribution")
    lines.append("")
    lines.append("| label | train | dev | gold |")
    lines.append("|---|---|---|---|")

    for label in LABELS:
        lines.append(
            f"| {label} "
            f"| {train_dist[label]['count']} "
            f"({train_dist[label]['pct']}%) "
            f"| {dev_dist[label]['count']} "
            f"({dev_dist[label]['pct']}%) "
            f"| {gold_dist[label]['count']} "
            f"({gold_dist[label]['pct']}%) |"
        )

    lines.append("")
    lines.append(f"- Gold majority baseline: **{baseline}%** "
                 f"(always predict {baseline_label})")
    lines.append(f"- Largest train/gold divergence: "
                 f"**{worst_label}, {worst_shift} points**")
    lines.append("")
    lines.append("## Leakage checks")
    lines.append("")
    lines.append("| check | result |")
    lines.append("|---|---:|")
    lines.append(f"| groups shared train/dev | {len(group_overlap)} |")
    lines.append(f"| identical response text train/dev "
                 f"| {len(text_overlap)} |")
    lines.append(f"| dev rows dropped as duplicate text "
                 f"| {text_overlap_removed} |")
    lines.append(f"| gold text present in train | {len(gold_in_train)} |")
    lines.append(f"| gold text present in dev | {len(gold_in_dev)} |")
    lines.append("")
    lines.append("## Class weights")
    lines.append("")
    lines.append("| label | weight |")
    lines.append("|---|---:|")

    for label in LABELS:
        lines.append(f"| {label} | {class_weights[label]} |")

    lines.append("")
    lines.append("## Interpretation constraints")
    lines.append("")
    lines.append(
        "- The 63 gold labels are reused unchanged from the targeted "
        "audit. That audit deliberately oversampled suspected "
        "evaluator errors, so its class distribution does not match "
        "the 624-response population."
    )
    lines.append(
        "- Gold metrics are therefore **targeted-audit performance** "
        "and must not be reported as population-level classifier "
        "accuracy."
    )
    lines.append(
        "- Class weights address imbalance in training. They do not "
        "correct the train/gold distribution shift."
    )
    lines.append(
        "- Any gold accuracy must be reported beside the "
        f"{baseline}% majority baseline."
    )
    lines.append(
        "- A representative human-validated sample would be required "
        "to estimate population-level classifier performance. That is "
        "identified as future strengthening, not completed work."
    )

    (output_dir / "split_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    print()
    print(f"Written to: {output_dir}")
    print("Existing split in outputs/analysis/classifier/ untouched.")
    print("=" * 70)


if __name__ == "__main__":
    main()
