"""
confirmatory_analysis.py

Confirmatory statistical analysis respecting the benchmark's repeated
measures structure.

Why this exists
---------------
The benchmark has repeated structure:

    104 prompt sets x 2 models x 3 languages = 624 observations

Every prompt set is measured six times. The ordinary chi-square tests
in statistical_analysis.py treat all 624 rows as independent, which
they are not, so their p-values are anti-conservative. Those tests are
retained as EXPLORATORY and are not deleted or replaced. This script
adds the confirmatory layer.

Design
------
The outcome has three unordered categories (COMPLIANCE,
NON_COMPLIANCE, REFUSAL), so binary paired tests such as McNemar and
Cochran's Q do not apply directly. Methods are chosen per factor
according to how that factor sits in the design:

  MODEL         Within-prompt factor. Each prompt set x language is
                answered by both models, giving genuinely paired
                3-category observations.
                -> Stuart-Maxwell test of marginal homogeneity on the
                   3x3 paired table.

  LANGUAGE      Within-prompt factor with three levels. Pairwise
                Stuart-Maxwell across the three language pairs, then
                Benjamini-Hochberg across that family.

  ATTACK CAT.   Between-prompt factors: a prompt set belongs to exactly
  VARIATION     one attack category and one variation, so no paired
                multinomial test exists for them. Clustering is instead
                handled by GEE with an exchangeable working correlation
                and prompt-set clusters, on a pre-declared binary
                contrast (REFUSAL vs not REFUSAL). Collapsing to binary
                loses information and is reported as such.

Effect sizes use a cluster bootstrap that resamples prompt sets, so the
intervals inherit the repeated structure rather than assuming
independence.

All p-values within the confirmatory family are corrected with
Benjamini-Hochberg.

Association only. Nothing here licenses a causal claim.

Outputs
-------
outputs/analysis/statistics_confirmatory/
    confirmatory_summary.md
    exploratory_diagnostics.csv
    confirmatory_tests.csv
    paired_model_table.csv
    paired_language_tables.csv
    effect_sizes_bootstrap.csv

Usage
-----
    python scripts/analysis/confirmatory_analysis.py
    python scripts/analysis/confirmatory_analysis.py \
        --input outputs/evaluated_2048/combined_evaluation.csv \
        --outdir outputs/analysis/statistics_confirmatory_2048 \
        --label 2048
"""

from pathlib import Path

import argparse

import numpy as np
import pandas as pd

from scipy.stats import chi2_contingency
from statsmodels.stats.contingency_tables import SquareTable
from statsmodels.stats.multitest import multipletests

import statsmodels.api as sm


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT = (
    PROJECT_ROOT / "outputs" / "evaluated" / "combined_evaluation.csv"
)

DEFAULT_OUTDIR = (
    PROJECT_ROOT / "outputs" / "analysis" / "statistics_confirmatory"
)

LABELS = ["COMPLIANCE", "NON_COMPLIANCE", "REFUSAL"]
LANGUAGES = ["en", "hi", "mr"]
MODELS = ["gpt_oss", "qwen"]

# A prompt set is the unit of independence.
PROMPT_KEY = ["attack_id", "variation_id"]

BOOTSTRAP_DRAWS = 2000
RANDOM_SEED = 42


# ============================================================
# EXPLORATORY DIAGNOSTICS
# ============================================================

def cramers_v(table):
    """
    Cramer's V for a contingency table.
    """

    chi2 = chi2_contingency(table)[0]

    n = np.asarray(table).sum()

    if n == 0:
        return 0.0

    denominator = min(table.shape[0] - 1, table.shape[1] - 1)

    if denominator == 0:
        return 0.0

    return float((chi2 / (n * denominator)) ** 0.5)


def exploratory_diagnostics(frame):
    """
    Re-run the exploratory chi-square tests and attach the assumption
    diagnostics they were previously reported without.

    These are NOT the confirmatory results. They are recorded so the
    paper can state exactly why a confirmatory layer was needed.
    """

    rows = []

    for factor in ["model", "language", "attack_category", "variation_id"]:

        table = pd.crosstab(
            frame[factor],
            frame["safety_label"]
        ).reindex(columns=LABELS, fill_value=0)

        chi2, p_value, degrees, expected = chi2_contingency(table)

        small = int((expected < 5).sum())

        rows.append({
            "factor": factor,
            "test": "Pearson chi-square (independence assumed)",
            "chi_square": round(float(chi2), 4),
            "df": int(degrees),
            "p_value_uncorrected": float(p_value),
            "cramers_v": round(cramers_v(table), 4),
            "cells": int(expected.size),
            "cells_expected_below_5": small,
            "pct_cells_below_5": round(100 * small / expected.size, 1),
            "min_expected": round(float(expected.min()), 2),
            "independence_violated": True,
            "note": (
                "Exploratory only. Treats 624 clustered observations "
                "as independent; each prompt set contributes 6."
            ),
        })

    return pd.DataFrame(rows)


# ============================================================
# PAIRED TABLES
# ============================================================

def paired_model_table(frame):
    """
    Build the 3x3 paired table of gpt_oss label vs qwen label.

    Pair unit: (attack_id, variation_id, language) — the same prompt in
    the same language answered by both models.
    """

    wide = frame.pivot_table(
        index=PROMPT_KEY + ["language"],
        columns="model",
        values="safety_label",
        aggfunc="first",
    )

    wide = wide.dropna(subset=MODELS)

    table = pd.crosstab(
        wide[MODELS[0]],
        wide[MODELS[1]],
    ).reindex(index=LABELS, columns=LABELS, fill_value=0)

    table.index.name = f"{MODELS[0]} (row)"
    table.columns.name = f"{MODELS[1]} (col)"

    return table, len(wide)


def paired_language_table(frame, first, second):
    """
    Build the 3x3 paired table for two languages.

    Pair unit: (attack_id, variation_id, model) — the same prompt set
    and model, answered in two languages.
    """

    wide = frame.pivot_table(
        index=PROMPT_KEY + ["model"],
        columns="language",
        values="safety_label",
        aggfunc="first",
    )

    wide = wide.dropna(subset=[first, second])

    table = pd.crosstab(
        wide[first],
        wide[second],
    ).reindex(index=LABELS, columns=LABELS, fill_value=0)

    table.index.name = f"{first} (row)"
    table.columns.name = f"{second} (col)"

    return table, len(wide)


def stuart_maxwell(table):
    """
    Stuart-Maxwell test of marginal homogeneity on a square table.

    statsmodels exposes this as SquareTable.homogeneity(); symmetry()
    is Bowker's test and answers a different question.
    """

    result = SquareTable(table.to_numpy()).homogeneity()

    return {
        "statistic": round(float(result.statistic), 4),
        "df": int(result.df),
        "p_value": float(result.pvalue),
    }


# ============================================================
# GEE FOR BETWEEN-PROMPT FACTORS
# ============================================================

def gee_refusal(frame, factor):
    """
    GEE on the pre-declared binary contrast REFUSAL vs not REFUSAL,
    with prompt-set clusters and an exchangeable working correlation.

    Used for attack_category and variation_id, which are between-prompt
    factors with no paired multinomial analogue. The binary collapse is
    a deliberate, pre-declared simplification and is reported as one.
    """

    work = frame.copy()

    work["is_refusal"] = (work["safety_label"] == "REFUSAL").astype(int)

    work["cluster"] = (
        work["attack_id"].astype(str)
        + "|"
        + work["variation_id"].astype(str)
    )

    design = pd.get_dummies(
        work[factor].astype(str),
        prefix=factor,
        drop_first=True,
        dtype=float,
    )

    if design.shape[1] == 0:
        return None

    design = sm.add_constant(design, has_constant="add")

    model = sm.GEE(
        work["is_refusal"].to_numpy(dtype=float),
        design.to_numpy(dtype=float),
        groups=work["cluster"].to_numpy(),
        family=sm.families.Binomial(),
        cov_struct=sm.cov_struct.Exchangeable(),
    )

    fitted = model.fit()

    # Wald test that all non-intercept coefficients are zero.
    count = design.shape[1] - 1

    constraint = np.zeros((count, design.shape[1]))
    constraint[:, 1:] = np.eye(count)

    wald = fitted.wald_test(constraint, scalar=True)

    return {
        "statistic": round(float(wald.statistic), 4),
        "df": int(count),
        "p_value": float(wald.pvalue),
        "n_clusters": int(work["cluster"].nunique()),
        "n_observations": int(len(work)),
    }


# ============================================================
# CLUSTER BOOTSTRAP EFFECT SIZES
# ============================================================

def cluster_bootstrap_difference(frame, factor, level_a, level_b):
    """
    Per-class difference in proportion between two levels of a factor,
    with a 95% cluster-bootstrap interval.

    Prompt sets are resampled with replacement, so the interval
    reflects the repeated structure instead of assuming 624
    independent observations.
    """

    work = frame[frame[factor].isin([level_a, level_b])].copy()

    work["cluster"] = (
        work["attack_id"].astype(str)
        + "|"
        + work["variation_id"].astype(str)
    )

    clusters = work["cluster"].unique()

    grouped = {name: part for name, part in work.groupby("cluster")}

    def shares(sample):
        out = {}
        for label in LABELS:
            for level in (level_a, level_b):
                subset = sample[sample[factor] == level]
                out[(level, label)] = (
                    (subset["safety_label"] == label).mean()
                    if len(subset) else np.nan
                )
        return out

    observed = shares(work)

    generator = np.random.default_rng(RANDOM_SEED)

    draws = {label: [] for label in LABELS}

    for _ in range(BOOTSTRAP_DRAWS):

        picked = generator.choice(
            clusters,
            size=len(clusters),
            replace=True,
        )

        sample = pd.concat(
            [grouped[name] for name in picked],
            ignore_index=True,
        )

        stat = shares(sample)

        for label in LABELS:
            draws[label].append(
                stat[(level_a, label)] - stat[(level_b, label)]
            )

    rows = []

    for label in LABELS:

        values = np.array(draws[label], dtype=float)
        values = values[~np.isnan(values)]

        difference = (
            observed[(level_a, label)] - observed[(level_b, label)]
        )

        low, high = (
            np.percentile(values, [2.5, 97.5])
            if values.size else (np.nan, np.nan)
        )

        rows.append({
            "factor": factor,
            "comparison": f"{level_a} - {level_b}",
            "label": label,
            f"share_{level_a}": round(
                100 * observed[(level_a, label)], 2
            ),
            f"share_{level_b}": round(
                100 * observed[(level_b, label)], 2
            ),
            "difference_pp": round(100 * difference, 2),
            "ci95_low_pp": round(100 * low, 2),
            "ci95_high_pp": round(100 * high, 2),
            "excludes_zero": bool(low > 0 or high < 0),
            "n_clusters": int(len(clusters)),
        })

    return pd.DataFrame(rows)


# ============================================================
# REPORT
# ============================================================

def table_to_markdown(table):
    """
    Render a DataFrame as a markdown table.

    Written out rather than using DataFrame.to_markdown(), which
    requires the optional `tabulate` package. Avoids adding a
    dependency for cosmetic output.
    """

    index_name = table.index.name or ""

    header = f"| {index_name} | " + " | ".join(
        str(column) for column in table.columns
    ) + " |"

    divider = "|---|" + "".join("---:|" for _ in table.columns)

    lines = [header, divider]

    for name, row in table.iterrows():
        lines.append(
            f"| {name} | "
            + " | ".join(str(value) for value in row)
            + " |"
        )

    return "\n".join(lines)

def build_report(label, input_path, counts, exploratory,
                 confirmatory, model_table, language_tables,
                 effects):
    """
    Assemble the markdown record.
    """

    lines = []

    lines.append("# Confirmatory statistical analysis")
    lines.append("")
    lines.append(
        "Cross-Lingual Vulnerability and Prompt Injection in "
        "Low-Resource Languages"
    )
    lines.append("")
    lines.append(f"- Run label: **{label}**")
    lines.append(f"- Input: `{input_path}`")
    lines.append(f"- Observations: {counts['rows']}")
    lines.append(f"- Prompt sets (independent units): {counts['prompts']}")
    lines.append(f"- Design: {counts['prompts']} prompt sets "
                 f"x {counts['models']} models "
                 f"x {counts['languages']} languages")
    lines.append("")
    lines.append(
        "Every prompt set is measured six times, so the 624 rows are "
        "clustered rather than independent. Exploratory chi-square "
        "results are retained unchanged; the confirmatory tests below "
        "account for the repeated structure."
    )
    lines.append("")

    lines.append("## Why a confirmatory layer was needed")
    lines.append("")
    lines.append(
        "| factor | chi2 | df | p (uncorrected) | Cramer's V "
        "| cells E<5 | min E |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")

    for _, row in exploratory.iterrows():
        lines.append(
            f"| {row['factor']} | {row['chi_square']} | {row['df']} "
            f"| {row['p_value_uncorrected']:.3g} | {row['cramers_v']} "
            f"| {row['cells_expected_below_5']}/{row['cells']} "
            f"({row['pct_cells_below_5']}%) | {row['min_expected']} |"
        )

    lines.append("")
    lines.append(
        "All four assume independence, which this design violates. "
        "The attack-category and variation tables additionally breach "
        "the expected-frequency guideline."
    )
    lines.append("")

    lines.append("## Confirmatory results")
    lines.append("")
    lines.append(
        "| analysis | method | statistic | df | p | p (BH) "
        "| significant at BH 0.05 |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---|")

    for _, row in confirmatory.iterrows():
        lines.append(
            f"| {row['analysis']} | {row['method']} "
            f"| {row['statistic']} | {row['df']} "
            f"| {row['p_value']:.3g} | {row['p_value_bh']:.3g} "
            f"| {'yes' if row['significant_bh_05'] else 'no'} |"
        )

    lines.append("")
    lines.append(
        f"Benjamini-Hochberg applied across the "
        f"{len(confirmatory)} tests in this family."
    )
    lines.append("")

    lines.append("## Paired model table")
    lines.append("")
    lines.append(
        "Pair unit: the same prompt set in the same language answered "
        "by both models."
    )
    lines.append("")
    lines.append(table_to_markdown(model_table))
    lines.append("")

    lines.append("## Paired language tables")
    lines.append("")
    lines.append(
        "Pair unit: the same prompt set and model answered in two "
        "languages."
    )

    for name, table in language_tables.items():
        lines.append("")
        lines.append(f"### {name}")
        lines.append("")
        lines.append(table_to_markdown(table))

    lines.append("")
    lines.append("## Effect sizes with cluster-bootstrap intervals")
    lines.append("")
    lines.append(
        f"Differences in percentage points. {BOOTSTRAP_DRAWS} draws "
        f"resampling prompt sets, seed {RANDOM_SEED}."
    )
    lines.append("")
    lines.append(
        "| comparison | label | difference (pp) | 95% CI "
        "| excludes zero |"
    )
    lines.append("|---|---|---:|---|---|")

    for _, row in effects.iterrows():
        lines.append(
            f"| {row['comparison']} | {row['label']} "
            f"| {row['difference_pp']:+.2f} "
            f"| [{row['ci95_low_pp']:+.2f}, {row['ci95_high_pp']:+.2f}] "
            f"| {'yes' if row['excludes_zero'] else 'no'} |"
        )

    lines.append("")
    lines.append("## Interpretation rules applied")
    lines.append("")
    lines.append(
        "- These are association tests. No causal statement follows "
        "from them."
    )
    lines.append(
        "- Correct phrasing: a given prompt framing *was associated "
        "with* a different distribution of safety outcomes. Not that "
        "it *caused* unsafe behaviour."
    )
    lines.append(
        "- Attack-category and variation results rest on a binary "
        "REFUSAL contrast, which discards information about the "
        "COMPLIANCE / NON_COMPLIANCE distinction."
    )
    lines.append(
        "- Safety labels are produced by a rule-based evaluator. They "
        "are silver labels, not ground truth, and the confirmatory "
        "results inherit that measurement error."
    )

    return "\n".join(lines) + "\n"


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Confirmatory analysis respecting the repeated-measures "
            "structure. Does not modify existing exploratory outputs."
        )
    )

    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--label", default="512")

    arguments = parser.parse_args()

    input_path = Path(arguments.input)
    output_dir = Path(arguments.outdir)

    print("=" * 68)
    print("CONFIRMATORY STATISTICAL ANALYSIS")
    print("=" * 68)

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    frame = pd.read_csv(input_path, keep_default_na=False)

    required = [
        "attack_id", "variation_id", "attack_category",
        "language", "model", "safety_label",
    ]

    missing = [c for c in required if c not in frame.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    unexpected = set(frame["safety_label"]) - set(LABELS)

    if unexpected:
        print(
            f"WARNING: labels outside the taxonomy present and "
            f"excluded from paired tables: {sorted(unexpected)}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    counts = {
        "rows": len(frame),
        "prompts": frame[PROMPT_KEY].drop_duplicates().shape[0],
        "models": frame["model"].nunique(),
        "languages": frame["language"].nunique(),
    }

    print(f"Input      : {input_path}")
    print(f"Rows       : {counts['rows']}")
    print(f"Prompt sets: {counts['prompts']}")
    print()

    # --------------------------------------------------------
    # Exploratory diagnostics
    # --------------------------------------------------------

    exploratory = exploratory_diagnostics(frame)

    exploratory.to_csv(
        output_dir / "exploratory_diagnostics.csv",
        index=False,
        encoding="utf-8",
    )

    print("EXPLORATORY DIAGNOSTICS")
    print(
        exploratory[
            ["factor", "chi_square", "df", "cramers_v",
             "cells_expected_below_5", "min_expected"]
        ].to_string(index=False)
    )
    print()

    # --------------------------------------------------------
    # Confirmatory: model
    # --------------------------------------------------------

    tests = []

    model_table, model_pairs = paired_model_table(frame)

    model_result = stuart_maxwell(model_table)

    tests.append({
        "analysis": "Model: gpt_oss vs qwen",
        "method": "Stuart-Maxwell marginal homogeneity (paired, 3-class)",
        "statistic": model_result["statistic"],
        "df": model_result["df"],
        "p_value": model_result["p_value"],
        "n_units": model_pairs,
        "unit": "prompt set x language",
    })

    model_table.to_csv(
        output_dir / "paired_model_table.csv",
        encoding="utf-8",
    )

    print("PAIRED MODEL TABLE")
    print(model_table.to_string())
    print(
        f"  Stuart-Maxwell: stat={model_result['statistic']} "
        f"df={model_result['df']} p={model_result['p_value']:.3g} "
        f"(n={model_pairs} pairs)"
    )
    print()

    # --------------------------------------------------------
    # Confirmatory: language, pairwise
    # --------------------------------------------------------

    language_tables = {}
    language_frames = []

    for index, first in enumerate(LANGUAGES):
        for second in LANGUAGES[index + 1:]:

            table, pairs = paired_language_table(frame, first, second)

            result = stuart_maxwell(table)

            name = f"{first} vs {second}"

            language_tables[name] = table

            block = table.copy()
            block.insert(0, "comparison", name)
            language_frames.append(block.reset_index())

            tests.append({
                "analysis": f"Language: {name}",
                "method": (
                    "Stuart-Maxwell marginal homogeneity "
                    "(paired, 3-class)"
                ),
                "statistic": result["statistic"],
                "df": result["df"],
                "p_value": result["p_value"],
                "n_units": pairs,
                "unit": "prompt set x model",
            })

            print(f"PAIRED LANGUAGE TABLE — {name}")
            print(table.to_string())
            print(
                f"  Stuart-Maxwell: stat={result['statistic']} "
                f"df={result['df']} p={result['p_value']:.3g} "
                f"(n={pairs} pairs)"
            )
            print()

    pd.concat(language_frames, ignore_index=True).to_csv(
        output_dir / "paired_language_tables.csv",
        index=False,
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Confirmatory: between-prompt factors via GEE
    # --------------------------------------------------------

    for factor in ["attack_category", "variation_id"]:

        result = gee_refusal(frame, factor)

        if result is None:
            continue

        tests.append({
            "analysis": f"{factor} (binary REFUSAL contrast)",
            "method": (
                "GEE logistic, exchangeable, prompt-set clusters; "
                "Wald omnibus"
            ),
            "statistic": result["statistic"],
            "df": result["df"],
            "p_value": result["p_value"],
            "n_units": result["n_clusters"],
            "unit": "prompt-set cluster",
        })

        print(
            f"GEE {factor}: Wald={result['statistic']} "
            f"df={result['df']} p={result['p_value']:.3g} "
            f"(clusters={result['n_clusters']})"
        )

    print()

    # --------------------------------------------------------
    # Benjamini-Hochberg across the confirmatory family
    # --------------------------------------------------------

    confirmatory = pd.DataFrame(tests)

    rejected, corrected, _, _ = multipletests(
        confirmatory["p_value"].to_numpy(),
        alpha=0.05,
        method="fdr_bh",
    )

    confirmatory["p_value_bh"] = corrected
    confirmatory["significant_bh_05"] = rejected

    confirmatory.to_csv(
        output_dir / "confirmatory_tests.csv",
        index=False,
        encoding="utf-8",
    )

    print("CONFIRMATORY TESTS (Benjamini-Hochberg corrected)")
    print(
        confirmatory[
            ["analysis", "statistic", "df", "p_value",
             "p_value_bh", "significant_bh_05"]
        ].to_string(index=False)
    )
    print()

    # --------------------------------------------------------
    # Effect sizes
    # --------------------------------------------------------

    effect_frames = [
        cluster_bootstrap_difference(frame, "model", *MODELS)
    ]

    for index, first in enumerate(LANGUAGES):
        for second in LANGUAGES[index + 1:]:
            effect_frames.append(
                cluster_bootstrap_difference(
                    frame, "language", first, second
                )
            )

    effects = pd.concat(effect_frames, ignore_index=True)

    effects.to_csv(
        output_dir / "effect_sizes_bootstrap.csv",
        index=False,
        encoding="utf-8",
    )

    print("EFFECT SIZES (percentage points, 95% cluster bootstrap)")
    print(
        effects[
            ["comparison", "label", "difference_pp",
             "ci95_low_pp", "ci95_high_pp", "excludes_zero"]
        ].to_string(index=False)
    )
    print()

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report = build_report(
        arguments.label,
        input_path,
        counts,
        exploratory,
        confirmatory,
        model_table,
        language_tables,
        effects,
    )

    (output_dir / "confirmatory_summary.md").write_text(
        report,
        encoding="utf-8",
    )

    print(f"Written to: {output_dir}")
    print()
    print("Existing exploratory outputs were not modified.")
    print("=" * 68)


if __name__ == "__main__":
    main()
