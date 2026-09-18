"""
final_report_tables.py

Build the paper-ready tables from the frozen outputs, and run the
end-to-end integrity checks for
"Cross-Lingual Vulnerability and Prompt Injection in Low-Resource
Languages".

Reads only. Writes tables into outputs/analysis/final_tables/ and
touches nothing else.

Final experimental scope
------------------------
    GPT-OSS @ 512   included
    Qwen    @ 512   included
    GPT-OSS @ 2048  included (generation-length comparison only)
    Qwen    @ 2048  EXCLUDED - provider output-token-rate limit

Tables produced
---------------
    T1  experimental scope and completion
    T2  model x safety (primary analysis set)
    T3  language x safety (primary analysis set)
    T4  model x language x safety
    T5  attack category x safety
    T6  prompt variation x safety
    T7  exploratory chi-square with assumption diagnostics
    T8  confirmatory tests, Benjamini-Hochberg corrected
    T9  evaluator vs human agreement on the 63 targeted-audit cases
    T10 classifier results with majority baselines
    T11 generation-length comparison, GPT-OSS 512 vs 2048

Usage
-----
    python scripts/analysis/final_report_tables.py
"""

from pathlib import Path

import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FINAL_DIR = PROJECT_ROOT / "outputs" / "evaluated_final"
PRIMARY = FINAL_DIR / "combined_final_512.csv"
GPT_2048 = FINAL_DIR / "gpt_oss_2048_evaluation.csv"
MANIFEST = FINAL_DIR / "FREEZE_MANIFEST.json"

HUMAN_AUDIT = (
    PROJECT_ROOT / "outputs" / "analysis" / "audit"
    / "human_review_queue.csv"
)

CLASSIFIER_DIR = (
    PROJECT_ROOT / "outputs" / "analysis" / "classifier_final"
)

CONFIRMATORY = (
    PROJECT_ROOT / "outputs" / "analysis"
    / "statistics_confirmatory_final"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "analysis" / "final_tables"

LABELS = ["COMPLIANCE", "NON_COMPLIANCE", "REFUSAL"]
KEY = ["attack_id", "variation_id", "language", "model"]

# Artifacts that must remain byte-identical to their recorded state.
PROTECTED = [
    "outputs/raw/openai_gpt_oss_20b_results.csv",
    "outputs/raw/qwen_qwen3.8_27b_results.csv",
    "outputs/evaluated/combined_evaluation.csv",
    "outputs/evaluated/gpt_oss_evaluation.csv",
    "outputs/evaluated/qwen_evaluation.csv",
    "outputs/analysis/audit/human_review_queue.csv",
    "outputs/analysis/classifier/gold_test.csv",
    "data/working_dataset_gpt_oss.csv",
    "data/working_dataset_qwen.csv",
    "validation_sample.csv",
    "manual_ground_truth.csv",
]


def markdown_table(frame, index_label=""):
    """
    Render a DataFrame as markdown without requiring tabulate.
    """

    header = f"| {index_label or (frame.index.name or '')} | " + " | ".join(
        str(column) for column in frame.columns
    ) + " |"

    divider = "|---|" + "".join("---:|" for _ in frame.columns)

    lines = [header, divider]

    for name, row in frame.iterrows():
        rendered = " | ".join(
            f"{value:.4g}" if isinstance(value, float) else str(value)
            for value in row
        )
        lines.append(f"| {name} | {rendered} |")

    return "\n".join(lines)


def counts_table(frame, factor):
    """
    Cross-tabulate a factor against the safety label, with totals.
    """

    table = pd.crosstab(
        frame[factor], frame["safety_label"]
    ).reindex(columns=LABELS, fill_value=0)

    table["TOTAL"] = table.sum(axis=1)

    return table


def verify_protected():
    """
    Confirm the protected artifacts match CHECKPOINT_MANIFEST.sha256.
    """

    manifest_path = PROJECT_ROOT / "CHECKPOINT_MANIFEST.sha256"

    if not manifest_path.exists():
        return None, "manifest missing"

    recorded = {}

    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        digest, _, name = line.partition(" ")
        recorded[name.strip().lstrip("*")] = digest.strip()

    results = []

    for relative in PROTECTED:

        path = PROJECT_ROOT / relative

        if not path.exists():
            results.append((relative, "MISSING", ""))
            continue

        digest = hashlib.sha256()

        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(65536), b""):
                digest.update(block)

        actual = digest.hexdigest()

        expected = recorded.get(relative)

        if expected is None:
            results.append((relative, "not in manifest", actual[:16]))
        elif expected == actual:
            results.append((relative, "OK", actual[:16]))
        else:
            results.append((relative, "CHANGED", actual[:16]))

    failures = [row for row in results if row[1] not in ("OK",)]

    return results, failures


def main():

    parser = argparse.ArgumentParser(
        description="Build final tables and run integrity checks."
    )

    parser.add_argument("--outdir", default=str(OUTPUT_DIR))

    arguments = parser.parse_args()

    output_dir = Path(arguments.outdir)

    print("=" * 70)
    print("FINAL TABLES AND END-TO-END CHECKS")
    print("=" * 70)

    for required in [PRIMARY, GPT_2048, MANIFEST]:
        if not required.exists():
            raise FileNotFoundError(
                f"Missing frozen output: {required}\n"
                f"Run scripts/analysis/finalize_labels.py first."
            )

    output_dir.mkdir(parents=True, exist_ok=True)

    primary = pd.read_csv(PRIMARY, keep_default_na=False)
    gpt2048 = pd.read_csv(GPT_2048, keep_default_na=False)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    sections = []

    sections.append("# Final tables")
    sections.append("")
    sections.append(
        "Cross-Lingual Vulnerability and Prompt Injection in "
        "Low-Resource Languages"
    )
    sections.append("")
    sections.append(
        f"Generated from the frozen label set. Evaluator SHA256 "
        f"`{manifest['evaluator']['sha256'][:16]}...`."
    )
    sections.append("")
    sections.append(
        "All safety labels are evaluator-produced **silver labels**, "
        "not ground truth. Statistical results are **associations**; "
        "no causal reading is licensed."
    )
    sections.append("")

    # ----- T1 scope -----

    scope_rows = []

    for entry in manifest["scope_included"]:
        scope_rows.append({
            "run": entry["run"],
            "model": entry["model"],
            "responses": entry["rows"],
            "status": "included",
        })

    for entry in manifest["scope_excluded"]:
        scope_rows.append({
            "run": entry["run"],
            "model": entry["model"],
            "responses": 0,
            "status": "EXCLUDED",
        })

    scope = pd.DataFrame(scope_rows).set_index("run")

    scope.to_csv(output_dir / "T1_experimental_scope.csv")

    sections.append("## T1 Experimental scope")
    sections.append("")
    sections.append(markdown_table(scope, "run"))
    sections.append("")
    sections.append(
        "The Qwen 2048 arm is permanently excluded: the provider's "
        "output-tokens-per-minute limit (1000) on the on_demand tier "
        "is below the configured 2048 cap, so requests are rejected "
        "before generation. Consequently the primary analysis set is "
        "the 512-token experiment, which is the only balanced "
        "two-model design."
    )
    sections.append("")

    # ----- T2..T6 counts -----

    for code, factor, title in [
        ("T2", "model", "Model x safety label"),
        ("T3", "language", "Language x safety label"),
        ("T5", "attack_category", "Attack category x safety label"),
        ("T6", "variation_id", "Prompt variation x safety label"),
    ]:
        table = counts_table(primary, factor)
        table.to_csv(output_dir / f"{code}_{factor}_safety.csv")

        sections.append(f"## {code} {title}")
        sections.append("")
        sections.append(markdown_table(table, factor))
        sections.append("")

        print(f"{code}: {title}")
        print(table.to_string())
        print()

    model_language = pd.crosstab(
        [primary["model"], primary["language"]],
        primary["safety_label"],
    ).reindex(columns=LABELS, fill_value=0)

    model_language["TOTAL"] = model_language.sum(axis=1)

    model_language.to_csv(output_dir / "T4_model_language_safety.csv")

    sections.append("## T4 Model x language x safety label")
    sections.append("")
    sections.append(
        "| model | language | "
        + " | ".join(LABELS)
        + " | TOTAL |"
    )
    sections.append("|---|---|" + "".join("---:|" for _ in range(4)))

    for (model_name, language), row in model_language.iterrows():
        sections.append(
            f"| {model_name} | {language} | "
            + " | ".join(str(int(row[label])) for label in LABELS)
            + f" | {int(row['TOTAL'])} |"
        )

    sections.append("")

    # ----- T7 / T8 statistics -----

    exploratory_path = CONFIRMATORY / "exploratory_diagnostics.csv"
    confirmatory_path = CONFIRMATORY / "confirmatory_tests.csv"

    if exploratory_path.exists():
        exploratory = pd.read_csv(exploratory_path)

        keep = exploratory[[
            "factor", "chi_square", "df", "p_value_uncorrected",
            "cramers_v", "cells_expected_below_5", "cells",
            "min_expected",
        ]].set_index("factor")

        keep.to_csv(output_dir / "T7_exploratory_diagnostics.csv")

        sections.append("## T7 Exploratory chi-square with diagnostics")
        sections.append("")
        sections.append(markdown_table(keep, "factor"))
        sections.append("")
        sections.append(
            "Retained as exploratory. These tests assume independence, "
            "which this design violates: each prompt set is measured "
            "six times. Two of the four also breach the "
            "expected-frequency guideline."
        )
        sections.append("")

    if confirmatory_path.exists():
        confirmatory = pd.read_csv(confirmatory_path)

        keep = confirmatory[[
            "analysis", "method", "statistic", "df", "p_value",
            "p_value_bh", "significant_bh_05", "n_units",
        ]].set_index("analysis")

        keep.to_csv(output_dir / "T8_confirmatory_tests.csv")

        sections.append("## T8 Confirmatory tests (Benjamini-Hochberg)")
        sections.append("")
        sections.append(markdown_table(
            keep.drop(columns=["method"]), "analysis"
        ))
        sections.append("")
        sections.append(
            "Methods: Stuart-Maxwell marginal homogeneity for the "
            "paired three-class comparisons; GEE logistic with "
            "prompt-set clusters and a pre-declared binary REFUSAL "
            "contrast for the between-prompt factors. McNemar and "
            "Cochran's Q are inapplicable because the outcome has "
            "three unordered categories."
        )
        sections.append("")

        print("T8 confirmatory:")
        print(keep.drop(columns=["method"]).to_string())
        print()

    # ----- T9 evaluator vs human -----

    if HUMAN_AUDIT.exists():

        human = pd.read_csv(HUMAN_AUDIT, keep_default_na=False)

        human = human[
            human["human_label"].astype(str).str.strip() != ""
        ][KEY + ["human_label"]]

        merged = human.merge(
            primary[KEY + ["safety_label"]], on=KEY, how="inner"
        )

        accuracy = float(
            (merged["safety_label"] == merged["human_label"]).mean()
        )

        kappa = float(cohen_kappa_score(
            merged["safety_label"], merged["human_label"], labels=LABELS
        ))

        precision, recall, f1, support = (
            precision_recall_fscore_support(
                merged["human_label"],
                merged["safety_label"],
                labels=LABELS,
                zero_division=0,
            )
        )

        agreement = pd.DataFrame({
            "precision": precision.round(4),
            "recall": recall.round(4),
            "f1": f1.round(4),
            "support": support,
        }, index=LABELS)

        agreement.to_csv(output_dir / "T9_evaluator_vs_human.csv")

        matrix = pd.DataFrame(
            confusion_matrix(
                merged["safety_label"],
                merged["human_label"],
                labels=LABELS,
            ),
            index=[f"evaluator_{label}" for label in LABELS],
            columns=[f"human_{label}" for label in LABELS],
        )

        matrix.to_csv(
            output_dir / "T9b_evaluator_human_confusion.csv"
        )

        sections.append("## T9 Evaluator vs human labels")
        sections.append("")
        sections.append(
            f"On the {len(merged)} human-adjudicated cases: "
            f"raw agreement **{accuracy:.4f}**, "
            f"Cohen's kappa **{kappa:+.4f}**."
        )
        sections.append("")
        sections.append(markdown_table(agreement, "class"))
        sections.append("")
        sections.append(markdown_table(matrix, "evaluator \\\\ human"))
        sections.append("")
        sections.append(
            "**These 63 cases are a targeted audit.** The audit queue "
            "scored actionable-content at +3 and refusal-plus-"
            "actionable at +4, so it deliberately oversampled "
            "suspected evaluator errors. This is a lower bound on a "
            "difficulty-selected subset, **not** population-level "
            "evaluator accuracy. A representative sample would be "
            "required for that and was not collected."
        )
        sections.append("")

        print(f"T9: agreement={accuracy:.4f} kappa={kappa:+.4f} "
              f"(n={len(merged)}, targeted audit)")
        print()

    # ----- T10 classifier -----

    classifier_rows = []

    for model_key in ["xlmr", "muril"]:
        for weighted, suffix in [(True, ""), (False, "_unweighted")]:

            path = (
                CLASSIFIER_DIR
                / f"{model_key}_final{suffix}"
                / "final_metrics.json"
            )

            if not path.exists():
                continue

            data = json.loads(path.read_text(encoding="utf-8"))

            for split, result in data["results"].items():
                classifier_rows.append({
                    "model": model_key,
                    "class_weights": "on" if weighted else "off",
                    "split": split,
                    "n": result["n"],
                    "accuracy": result["accuracy"],
                    "majority_baseline": result["majority_baseline"],
                    "beats_baseline": result["beats_baseline"],
                    "macro_f1": result["macro_f1"],
                    "COMPLIANCE_f1": (
                        result["per_class"]["COMPLIANCE"]["f1"]
                    ),
                })

    if classifier_rows:

        classifier = pd.DataFrame(classifier_rows).set_index("model")

        classifier.to_csv(output_dir / "T10_classifier_results.csv")

        sections.append("## T10 Classifier results")
        sections.append("")
        sections.append(markdown_table(classifier, "model"))
        sections.append("")
        sections.append(
            "COMPLIANCE F1 is 0.0000 in every configuration. With 17 "
            "COMPLIANCE examples in silver training, the class is not "
            "learned, and balanced class weighting does not "
            "compensate. Both models exceed the baseline on silver "
            "dev and fall below it on human gold, consistent with the "
            "distribution shift inherent in reusing a targeted audit "
            "as the test set. The MuRIL / XLM-R ordering reverses "
            "between weighted and unweighted runs, so no ranking "
            "between the two models is reportable."
        )
        sections.append("")

        print("T10 classifier:")
        print(classifier.to_string())
        print()

    # ----- T11 generation length -----

    paired = primary[primary["model"] == "gpt_oss"][
        KEY + ["safety_label"]
    ].merge(
        gpt2048[KEY + ["safety_label"]],
        on=KEY,
        suffixes=("_512", "_2048"),
        how="inner",
    )

    transitions = pd.crosstab(
        paired["safety_label_512"], paired["safety_label_2048"]
    ).reindex(index=LABELS, columns=LABELS, fill_value=0)

    transitions.to_csv(output_dir / "T11_generation_length.csv")

    changed = int(
        (paired["safety_label_512"] != paired["safety_label_2048"]).sum()
    )

    sections.append("## T11 Generation length: GPT-OSS 512 vs 2048")
    sections.append("")
    sections.append(
        f"Paired on prompt set x language; n = {len(paired)}. "
        f"Labels differing between budgets: **{changed}** "
        f"({100 * changed / len(paired):.1f}%)."
    )
    sections.append("")
    sections.append(markdown_table(transitions, "512 \\\\ 2048"))
    sections.append("")
    sections.append(
        "Raising the cap reduced GPT-OSS responses at the ceiling from "
        "25.2% to 2.6%, yet changed only "
        f"{changed} of {len(paired)} labels, with the REFUSAL count "
        "identical at both budgets. For this model, safety-label "
        "sensitivity to the generation budget is therefore modest."
    )
    sections.append("")
    sections.append(
        "This test was only possible for GPT-OSS. Qwen showed a far "
        "higher ceiling rate at 512 (66.2%), and its 2048 arm is "
        "excluded, so its generation-length sensitivity is **not "
        "measured** and remains an open limitation."
    )
    sections.append("")

    print(f"T11: {changed}/{len(paired)} labels differ between budgets")
    print(transitions.to_string())
    print()

    # ----- integrity -----

    results, failures = verify_protected()

    sections.append("## Integrity check")
    sections.append("")

    if results is None:
        sections.append("Checksum manifest not found.")
    else:
        sections.append("| artifact | status |")
        sections.append("|---|---|")
        for relative, status, _ in results:
            sections.append(f"| `{relative}` | {status} |")
        sections.append("")

        if failures:
            sections.append(
                f"**{len(failures)} artifact(s) differ from their "
                f"recorded checksum.**"
            )
        else:
            sections.append(
                "All protected artifacts match their recorded "
                "checksums. The original 512-token results and the 63 "
                "human-adjudicated labels are unmodified."
            )

    sections.append("")

    (output_dir / "FINAL_TABLES.md").write_text(
        "\n".join(sections) + "\n", encoding="utf-8"
    )

    print("=" * 70)
    print("INTEGRITY CHECK")
    print("=" * 70)

    for relative, status, digest in results or []:
        print(f"  {status:16s} {relative}")

    print()

    if failures:
        print(f"FAILURES: {len(failures)}")
        raise SystemExit(1)

    print("All protected artifacts unmodified.")
    print()
    print(f"Tables written to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
