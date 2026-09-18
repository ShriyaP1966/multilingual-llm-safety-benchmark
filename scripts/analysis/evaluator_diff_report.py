"""
evaluator_diff_report.py

Document how the current evaluator differs from a saved evaluation.

Why this exists
---------------
The evaluator was modified after the original 512-token benchmark
results were written. Re-running it on the same preserved responses
does not reproduce outputs/evaluated/combined_evaluation.csv. That gap
has to be measured and documented before any label is regenerated,
otherwise the paper cannot state that code, labels, statistics, figures
and results agree.

This script only reads and reports. It never modifies the saved
evaluation, never touches human labels, and never writes into
outputs/evaluated/.

Outputs
-------
outputs/analysis/evaluator/
    evaluator_diff_summary.md         human-readable record
    evaluator_diff_rows.csv           the individual changed rows
    evaluator_diff_transitions.csv    old -> new label counts
    evaluator_diff_by_<factor>.csv    changes per model/language/
                                      attack_category/variation_id

Usage
-----
    python scripts/analysis/evaluator_diff_report.py
    python scripts/analysis/evaluator_diff_report.py \
        --saved outputs/evaluated/combined_evaluation.csv
"""

from pathlib import Path

import argparse
import importlib.util

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EVALUATOR_PATH = PROJECT_ROOT / "scripts" / "analysis" / "evaluate.py"

DEFAULT_SAVED = (
    PROJECT_ROOT / "outputs" / "evaluated" / "combined_evaluation.csv"
)

DEFAULT_INPUTS = {
    "gpt_oss": PROJECT_ROOT / "data" / "working_dataset_gpt_oss.csv",
    "qwen": PROJECT_ROOT / "data" / "working_dataset_qwen.csv",
}

HUMAN_AUDIT = (
    PROJECT_ROOT / "outputs" / "analysis" / "audit"
    / "human_review_queue.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "analysis" / "evaluator"
)

KEY = ["attack_id", "variation_id", "language", "model"]

LANGUAGE_COLUMNS = {
    "en": "response_en",
    "hi": "response_hi",
    "mr": "response_mr",
}

FACTORS = ["model", "language", "attack_category", "variation_id"]


# ============================================================
# LOAD THE EVALUATOR AS A MODULE
# ============================================================

def load_evaluator():
    """
    Import scripts/analysis/evaluate.py without running main().
    """

    spec = importlib.util.spec_from_file_location(
        "evaluate_current",
        EVALUATOR_PATH
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


# ============================================================
# RECOMPUTE LABELS
# ============================================================

def recompute_labels(evaluator, inputs):
    """
    Classify every preserved response with the current evaluator.

    Reads the same wide datasets the evaluator normally consumes and
    returns one row per (attack_id, variation_id, language, model).
    """

    rows = []

    for model_name, path in inputs.items():

        if not Path(path).exists():
            raise FileNotFoundError(
                f"Input dataset not found: {path}"
            )

        frame = pd.read_csv(path, keep_default_na=False)

        for _, record in frame.iterrows():

            for language, column in LANGUAGE_COLUMNS.items():

                response = record[column]

                override_key = (
                    model_name,
                    record["attack_id"],
                    record["variation_id"],
                    language,
                )

                if override_key in evaluator.MANUAL_OVERRIDES:
                    label = evaluator.MANUAL_OVERRIDES[override_key]
                else:
                    label = evaluator.classify_response(
                        response,
                        language=language
                    )

                rows.append({
                    "attack_id": record["attack_id"],
                    "variation_id": record["variation_id"],
                    "attack_category": record["attack_category"],
                    "language": language,
                    "model": model_name,
                    "current_label": label,
                })

    return pd.DataFrame(rows)


# ============================================================
# HUMAN-AGREEMENT CHECK ON THE CHANGED ROWS
# ============================================================

def human_agreement_on_changes(changed):
    """
    For changed rows that were human-adjudicated, report whether the
    new label agrees with the human more often than the saved one.

    This is the only direct evidence available on whether the change is
    an improvement. It is a small, non-representative subset drawn from
    a targeted audit, so it settles nothing on its own.
    """

    if not HUMAN_AUDIT.exists():
        return None

    human = pd.read_csv(HUMAN_AUDIT, keep_default_na=False)

    human = human[
        human["human_label"].astype(str).str.strip() != ""
    ][KEY + ["human_label"]]

    overlap = changed.merge(human, on=KEY, how="inner")

    if overlap.empty:
        return {
            "overlap": 0,
            "saved_correct": 0,
            "current_correct": 0,
        }

    return {
        "overlap": len(overlap),
        "saved_correct": int(
            (overlap["saved_label"] == overlap["human_label"]).sum()
        ),
        "current_correct": int(
            (overlap["current_label"] == overlap["human_label"]).sum()
        ),
    }


# ============================================================
# REPORT
# ============================================================

def write_report(saved, current, changed, agreement, saved_path):
    """
    Write the markdown summary and supporting CSVs.
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    transitions = (
        pd.crosstab(changed["saved_label"], changed["current_label"])
        if not changed.empty
        else pd.DataFrame()
    )

    totals = pd.DataFrame({
        "saved": saved["safety_label"].value_counts(),
        "current": current["current_label"].value_counts(),
    }).fillna(0).astype(int)

    totals["delta"] = totals["current"] - totals["saved"]

    changed.to_csv(
        OUTPUT_DIR / "evaluator_diff_rows.csv",
        index=False,
        encoding="utf-8"
    )

    if not transitions.empty:
        transitions.to_csv(
            OUTPUT_DIR / "evaluator_diff_transitions.csv",
            encoding="utf-8"
        )

    per_factor = {}

    merged_count = len(saved)

    for factor in FACTORS:

        table = pd.DataFrame({
            "rows": saved.groupby(factor).size(),
            "changed": changed.groupby(factor).size(),
        }).fillna(0)

        table["changed"] = table["changed"].astype(int)

        table["pct"] = (
            100 * table["changed"] / table["rows"]
        ).round(1)

        table = table.sort_values("changed", ascending=False)

        table.to_csv(
            OUTPUT_DIR / f"evaluator_diff_by_{factor}.csv",
            encoding="utf-8"
        )

        per_factor[factor] = table

    lines = []
    lines.append("# Evaluator change record")
    lines.append("")
    lines.append(
        "Cross-Lingual Vulnerability and Prompt Injection in "
        "Low-Resource Languages"
    )
    lines.append("")
    lines.append(
        "Comparison of the current evaluator against a saved "
        "evaluation, on identical preserved responses."
    )
    lines.append("")
    lines.append(f"- Saved evaluation: `{saved_path}`")
    lines.append(f"- Evaluator: `scripts/analysis/evaluate.py`")
    lines.append(f"- Rows compared: {merged_count}")
    lines.append(f"- Labels changed: {len(changed)} "
                 f"({100 * len(changed) / merged_count:.1f}%)")
    lines.append("")

    lines.append("## Reproduces the saved evaluation?")
    lines.append("")
    lines.append(
        "**No.**" if len(changed) else "**Yes — labels are identical.**"
    )
    lines.append("")

    if len(changed):

        lines.append("## Label transitions")
        lines.append("")
        lines.append("| saved | current | count |")
        lines.append("|---|---|---:|")

        for old_label in transitions.index:
            for new_label in transitions.columns:
                count = int(transitions.loc[old_label, new_label])
                if count:
                    lines.append(
                        f"| {old_label} | {new_label} | {count} |"
                    )

        lines.append("")
        lines.append("## Net label totals")
        lines.append("")
        lines.append("| label | saved | current | delta |")
        lines.append("|---|---:|---:|---:|")

        for label in totals.index:
            row = totals.loc[label]
            lines.append(
                f"| {label} | {row['saved']} | {row['current']} "
                f"| {row['delta']:+d} |"
            )

        for factor in FACTORS:

            lines.append("")
            lines.append(f"## Changes by {factor}")
            lines.append("")
            lines.append(f"| {factor} | rows | changed | % |")
            lines.append("|---|---:|---:|---:|")

            for name, row in per_factor[factor].iterrows():
                lines.append(
                    f"| {name} | {int(row['rows'])} "
                    f"| {int(row['changed'])} | {row['pct']} |"
                )

        lines.append("")
        lines.append("## Agreement with existing human labels")
        lines.append("")

        if agreement is None:
            lines.append("Human audit file not found.")
        elif agreement["overlap"] == 0:
            lines.append(
                "None of the changed rows were human-adjudicated, so "
                "there is no direct evidence on whether the change is "
                "an improvement."
            )
        else:
            lines.append(
                f"{agreement['overlap']} of the {len(changed)} changed "
                f"rows are human-adjudicated."
            )
            lines.append("")
            lines.append(
                f"- Saved label agrees with human: "
                f"{agreement['saved_correct']}/{agreement['overlap']}"
            )
            lines.append(
                f"- Current label agrees with human: "
                f"{agreement['current_correct']}/{agreement['overlap']}"
            )
            lines.append("")

            if agreement["current_correct"] > agreement["saved_correct"]:
                verdict = (
                    "The current evaluator agrees with the human labels "
                    "more often on this subset."
                )
            elif agreement["current_correct"] < agreement["saved_correct"]:
                verdict = (
                    "The current evaluator agrees with the human labels "
                    "LESS often on this subset."
                )
            else:
                verdict = (
                    "Agreement is unchanged on this subset: the current "
                    "evaluator is different, not demonstrably better."
                )

            lines.append(verdict)
            lines.append("")
            lines.append(
                "This subset comes from a targeted audit that "
                "deliberately oversampled suspected evaluator errors. "
                "It is not a representative sample and cannot establish "
                "population-level accuracy."
            )

        lines.append("")
        lines.append("## Treatment")
        lines.append("")
        lines.append(
            "The saved evaluation is preserved unchanged. Labels used "
            "for final results are regenerated from the frozen "
            "evaluator on the final benchmark run, so that code, "
            "labels, statistics, figures and reported results all "
            "agree. This record documents the difference between the "
            "two so the earlier numbers remain interpretable."
        )

    (OUTPUT_DIR / "evaluator_diff_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8"
    )

    return transitions, totals, per_factor


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Document how the current evaluator differs from a saved "
            "evaluation. Read-only."
        )
    )

    parser.add_argument(
        "--saved",
        default=str(DEFAULT_SAVED),
        help="Saved evaluation CSV to compare against.",
    )

    arguments = parser.parse_args()

    saved_path = Path(arguments.saved)

    print("=" * 68)
    print("EVALUATOR CHANGE RECORD")
    print("=" * 68)

    if not saved_path.exists():
        raise FileNotFoundError(
            f"Saved evaluation not found: {saved_path}"
        )

    evaluator = load_evaluator()

    saved = pd.read_csv(saved_path, keep_default_na=False)

    print(f"Saved evaluation : {saved_path}  ({len(saved)} rows)")
    print(f"MANUAL_OVERRIDES : {len(evaluator.MANUAL_OVERRIDES)}")

    current = recompute_labels(evaluator, DEFAULT_INPUTS)

    print(f"Recomputed       : {len(current)} rows")

    merged = saved.merge(
        current[KEY + ["current_label"]],
        on=KEY,
        how="outer",
        indicator=True,
    )

    unmatched = (merged["_merge"] != "both").sum()

    if unmatched:
        raise ValueError(
            f"{unmatched} keys did not join; cannot compare safely."
        )

    merged = merged.rename(columns={"safety_label": "saved_label"})

    changed = merged[
        merged["saved_label"] != merged["current_label"]
    ].copy()

    print()
    print(
        f"Labels changed   : {len(changed)} / {len(merged)} "
        f"({100 * len(changed) / len(merged):.1f}%)"
    )

    agreement = human_agreement_on_changes(changed)

    saved_for_groups = merged.rename(
        columns={"saved_label": "safety_label"}
    )

    transitions, totals, per_factor = write_report(
        saved_for_groups,
        current,
        changed,
        agreement,
        saved_path,
    )

    if len(changed):
        print()
        print("Transitions (saved -> current):")
        print(transitions.to_string())
        print()
        print("Net totals:")
        print(totals.to_string())

        if agreement and agreement["overlap"]:
            print()
            print(
                f"Human-adjudicated among changed: "
                f"{agreement['overlap']}"
            )
            print(
                f"  saved agrees  : "
                f"{agreement['saved_correct']}/{agreement['overlap']}"
            )
            print(
                f"  current agrees: "
                f"{agreement['current_correct']}/{agreement['overlap']}"
            )

    print()
    print(f"Report written to: {OUTPUT_DIR}")
    print()
    print("The saved evaluation was NOT modified.")
    print("=" * 68)


if __name__ == "__main__":
    main()
