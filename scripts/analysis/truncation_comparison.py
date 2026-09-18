"""
truncation_comparison.py

Compare the 512-token and 2048-token benchmark experiments.

Purpose
-------
The original experiment capped generation at 512 output tokens.
Responses hit that ceiling at very different rates across models and
languages, and the safety evaluator keys on refusal language that
often appears late in a response. This script quantifies how much of
the original safety signal was attributable to the cap.

Both experiments are preserved. The 512 run is the baseline; the 2048
run is the follow-up. Neither is overwritten.

Interpretation discipline
-------------------------
This is a comparison of two experiments that differ in one configured
parameter. It licenses statements of the form "raising the output cap
was associated with a different distribution of safety outcomes".

It does NOT on its own establish a script-level or tokenizer-level
cause for the language differences. Several explanations remain
consistent with the data and are not separated here:

    - tokens required per unit of meaning differing by script
    - models being more verbose in some languages than others
    - translation affecting prompt length or response style
    - per-language response conventions

The script therefore reports tokens-per-character by language as
descriptive evidence, explicitly labelled as descriptive, and draws no
causal conclusion.

Outputs
-------
outputs/analysis/truncation/
    truncation_comparison.md
    ceiling_rates_by_factor.csv
    label_shift_512_to_2048.csv
    token_length_comparison.csv
    paired_label_changes.csv

Usage
-----
    python scripts/analysis/truncation_comparison.py
"""

from pathlib import Path

import argparse

import numpy as np
import pandas as pd

from scipy.stats import chi2_contingency


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_512 = {
    "gpt_oss": PROJECT_ROOT / "outputs" / "raw"
    / "openai_gpt_oss_20b_results.csv",
    "qwen": PROJECT_ROOT / "outputs" / "raw"
    / "qwen_qwen3.8_27b_results.csv",
}

RAW_2048 = {
    "gpt_oss": PROJECT_ROOT / "outputs" / "raw_2048"
    / "openai_gpt_oss_20b_results.csv",
    "qwen": PROJECT_ROOT / "outputs" / "raw_2048"
    / "qwen_qwen3.8_27b_results.csv",
}

EVAL_512 = (
    PROJECT_ROOT / "outputs" / "evaluated_final"
    / "combined_final_512.csv"
)

EVAL_2048 = (
    PROJECT_ROOT / "outputs" / "evaluated_final"
    / "gpt_oss_2048_evaluation.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "analysis" / "truncation"

LABELS = ["COMPLIANCE", "NON_COMPLIANCE", "REFUSAL"]
KEY = ["attack_id", "variation_id", "language", "model"]

CEILING_512 = 512
CEILING_2048 = 2048


def load_raw(paths, ceiling, run_label):
    """
    Load and normalise one experiment's raw responses.
    """

    frames = []

    for model_name, path in paths.items():

        if not Path(path).exists():
            raise FileNotFoundError(f"Missing raw file: {path}")

        frame = pd.read_csv(path, keep_default_na=False)
        frame["model"] = model_name
        frame["run"] = run_label

        frame["tokens"] = pd.to_numeric(
            frame["output_tokens"], errors="coerce"
        )

        frame["at_ceiling"] = frame["tokens"] >= ceiling

        frame["response_chars"] = (
            frame["response"].astype(str).str.len()
        )

        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def ceiling_rates(frame, factors):
    """
    Share of responses reaching the configured ceiling, per factor.
    """

    usable = frame.dropna(subset=["tokens"])

    rows = []

    for factor in factors:

        grouped = usable.groupby(factor)["at_ceiling"].agg(
            ["sum", "count"]
        )

        for name, record in grouped.iterrows():
            rows.append({
                "run": frame["run"].iloc[0],
                "factor": factor,
                "level": name,
                "at_ceiling": int(record["sum"]),
                "n": int(record["count"]),
                "pct": round(
                    100 * record["sum"] / record["count"], 1
                ),
            })

    return pd.DataFrame(rows)


def main():

    parser = argparse.ArgumentParser(
        description="Compare the 512-token and 2048-token experiments."
    )

    parser.add_argument(
        "--models",
        default="gpt_oss",
        help=(
            "Comma-separated models to include. Defaults to gpt_oss, "
            "because the Qwen 2048 arm is permanently excluded from "
            "the final scope (provider output-token-rate limit), so "
            "only GPT-OSS has both budgets."
        ),
    )

    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail unless both runs have 312 responses per model.",
    )

    arguments = parser.parse_args()

    print("=" * 70)
    print("512 vs 2048 TRUNCATION COMPARISON")
    print("=" * 70)

    selected = [n.strip() for n in arguments.models.split(",")]

    print(f"Models in scope: {selected}")
    print(
        "Qwen 2048 arm is permanently excluded (provider "
        "output-token-rate limit)."
    )

    raw_512 = load_raw(
        {k: v for k, v in RAW_512.items() if k in selected},
        CEILING_512, "512",
    )

    raw_2048 = load_raw(
        {k: v for k, v in RAW_2048.items() if k in selected},
        CEILING_2048, "2048",
    )

    counts = {
        "512": raw_512.groupby("model").size().to_dict(),
        "2048": raw_2048.groupby("model").size().to_dict(),
    }

    print(f"512  run responses: {counts['512']}")
    print(f"2048 run responses: {counts['2048']}")

    incomplete = [
        f"{run}/{model}={n}"
        for run, per_model in counts.items()
        for model, n in per_model.items()
        if n != 312
    ]

    if incomplete:
        message = (
            f"Runs are not complete at 312 per model: "
            f"{', '.join(incomplete)}"
        )

        if arguments.require_complete:
            raise ValueError(message)

        print()
        print(f"WARNING: {message}")
        print(
            "Comparisons below are restricted to the intersection of "
            "keys present in both runs."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Ceiling rates
    # --------------------------------------------------------

    factors = ["model", "language", "attack_category", "variation_id"]

    rates = pd.concat(
        [
            ceiling_rates(raw_512, factors),
            ceiling_rates(raw_2048, factors),
        ],
        ignore_index=True,
    )

    rates.to_csv(
        OUTPUT_DIR / "ceiling_rates_by_factor.csv",
        index=False,
        encoding="utf-8",
    )

    print()
    print("SHARE OF RESPONSES AT THE CONFIGURED CEILING")

    for factor in ["model", "language"]:
        print()
        print(f"  by {factor}")
        pivot = rates[rates["factor"] == factor].pivot_table(
            index="level", columns="run", values="pct"
        )
        print(pivot.to_string())

    # --------------------------------------------------------
    # Token lengths (descriptive)
    # --------------------------------------------------------

    combined_raw = pd.concat([raw_512, raw_2048], ignore_index=True)

    usable = combined_raw.dropna(subset=["tokens"]).copy()

    usable["tokens_per_char"] = (
        usable["tokens"] / usable["response_chars"].clip(lower=1)
    )

    lengths = usable.groupby(["run", "model", "language"]).agg(
        mean_tokens=("tokens", "mean"),
        median_tokens=("tokens", "median"),
        max_tokens=("tokens", "max"),
        mean_chars=("response_chars", "mean"),
        mean_tokens_per_char=("tokens_per_char", "mean"),
        n=("tokens", "size"),
    ).round(3).reset_index()

    lengths.to_csv(
        OUTPUT_DIR / "token_length_comparison.csv",
        index=False,
        encoding="utf-8",
    )

    print()
    print("RESPONSE LENGTH (descriptive)")
    print(lengths.to_string(index=False))

    # --------------------------------------------------------
    # Label shift, only if the 2048 evaluation exists
    # --------------------------------------------------------

    label_shift = None
    paired = None

    if EVAL_2048.exists() and EVAL_512.exists():

        eval_512 = pd.read_csv(EVAL_512, keep_default_na=False)
        eval_2048 = pd.read_csv(EVAL_2048, keep_default_na=False)

        eval_512 = eval_512[eval_512["model"].isin(selected)]
        eval_2048 = eval_2048[eval_2048["model"].isin(selected)]

        merged = eval_512[KEY + ["safety_label"]].merge(
            eval_2048[KEY + ["safety_label"]],
            on=KEY,
            suffixes=("_512", "_2048"),
            how="inner",
        )

        print()
        print(f"PAIRED LABEL COMPARISON (n={len(merged)})")

        transitions = pd.crosstab(
            merged["safety_label_512"],
            merged["safety_label_2048"],
        ).reindex(index=LABELS, columns=LABELS, fill_value=0)

        print(transitions.to_string())

        changed = int(
            (
                merged["safety_label_512"]
                != merged["safety_label_2048"]
            ).sum()
        )

        print()
        print(
            f"labels differing between runs: {changed}/{len(merged)} "
            f"({100 * changed / len(merged):.1f}%)"
        )

        transitions.to_csv(
            OUTPUT_DIR / "paired_label_changes.csv",
            encoding="utf-8",
        )

        totals = pd.DataFrame({
            "run_512": merged["safety_label_512"].value_counts(),
            "run_2048": merged["safety_label_2048"].value_counts(),
        }).reindex(LABELS).fillna(0).astype(int)

        totals["delta"] = (
            totals["run_2048"] - totals["run_512"]
        ).astype(int)

        totals["pct_512"] = (
            100 * totals["run_512"] / len(merged)
        ).round(1)

        totals["pct_2048"] = (
            100 * totals["run_2048"] / len(merged)
        ).round(1)

        totals.to_csv(
            OUTPUT_DIR / "label_shift_512_to_2048.csv",
            encoding="utf-8",
        )

        print()
        print(totals.to_string())

        label_shift = totals
        paired = (merged, transitions, changed)

    else:
        print()
        print(
            "2048 evaluation not found; label comparison skipped. "
            "Run merge_results.py and evaluate.py on the 2048 run "
            "first."
        )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    lines = []

    lines.append("# 512 vs 2048 output-cap comparison")
    lines.append("")
    lines.append(
        "Cross-Lingual Vulnerability and Prompt Injection in "
        "Low-Resource Languages"
    )
    lines.append("")
    lines.append(
        "Two preserved experiments differing in one configured "
        "parameter: `generation.max_output_tokens`."
    )
    lines.append("")
    lines.append("| run | responses per model | raw output |")
    lines.append("|---|---|---|")
    lines.append(
        f"| 512 (baseline) | {counts['512']} | `outputs/raw/` |"
    )
    lines.append(
        f"| 2048 (follow-up) | {counts['2048']} | `outputs/raw_2048/` |"
    )
    lines.append("")

    lines.append("## Responses reaching the configured ceiling")
    lines.append("")

    for factor in ["model", "language"]:
        pivot = rates[rates["factor"] == factor].pivot_table(
            index="level", columns="run", values="pct"
        )
        lines.append(f"### by {factor} (%)")
        lines.append("")
        lines.append("| level | 512 run | 2048 run |")
        lines.append("|---|---:|---:|")
        for level, row in pivot.iterrows():
            lines.append(
                f"| {level} | {row.get('512', float('nan')):.1f} "
                f"| {row.get('2048', float('nan')):.1f} |"
            )
        lines.append("")

    lines.append("## Response length")
    lines.append("")
    lines.append(
        "| run | model | language | mean tokens | mean chars "
        "| tokens per char |"
    )
    lines.append("|---|---|---|---:|---:|---:|")

    for _, row in lengths.iterrows():
        lines.append(
            f"| {row['run']} | {row['model']} | {row['language']} "
            f"| {row['mean_tokens']:.0f} | {row['mean_chars']:.0f} "
            f"| {row['mean_tokens_per_char']:.3f} |"
        )

    lines.append("")
    lines.append(
        "Tokens per character is **descriptive only**. It is reported "
        "because it is the quantity a script-level explanation would "
        "predict, not because this comparison establishes one."
    )
    lines.append("")

    if label_shift is not None:
        merged, transitions, changed = paired

        lines.append("## Safety-label shift")
        lines.append("")
        lines.append(
            f"Paired on {KEY}; n = {len(merged)}. "
            f"Labels differing between runs: {changed} "
            f"({100 * changed / len(merged):.1f}%)."
        )
        lines.append("")
        lines.append("| label | 512 | 2048 | delta |")
        lines.append("|---|---:|---:|---:|")

        for label in LABELS:
            row = label_shift.loc[label]
            lines.append(
                f"| {label} | {row['run_512']} ({row['pct_512']}%) "
                f"| {row['run_2048']} ({row['pct_2048']}%) "
                f"| {int(row['delta']):+d} |"
            )

        lines.append("")
        lines.append("### Transitions (512 row, 2048 column)")
        lines.append("")
        header = "| 512 \\ 2048 | " + " | ".join(LABELS) + " |"
        lines.append(header)
        lines.append("|---|" + "".join("---:|" for _ in LABELS))

        for label in LABELS:
            values = " | ".join(
                str(int(transitions.loc[label, column]))
                for column in LABELS
            )
            lines.append(f"| {label} | {values} |")

        lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "- Raising the output cap **was associated with** a different "
        "distribution of safety outcomes. That is an association "
        "between a configuration change and measured labels."
    )
    lines.append(
        "- This comparison does **not** establish that script-level "
        "tokenization caused the language differences. Model "
        "verbosity by language, translation effects on prompt length "
        "and style, and per-language response conventions remain "
        "consistent with the data and are not separated here."
    )
    lines.append(
        "- The 512 run is retained as a sensitivity baseline. Neither "
        "run is treated as invalid; they answer the same question "
        "under different generation budgets."
    )
    lines.append(
        "- Safety labels on both runs are evaluator-produced silver "
        "labels and inherit the evaluator's measurement error."
    )

    (OUTPUT_DIR / "truncation_comparison.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print()
    print(f"Written to: {OUTPUT_DIR}")
    print("Both experiments preserved; neither was modified.")
    print("=" * 70)


if __name__ == "__main__":
    main()
