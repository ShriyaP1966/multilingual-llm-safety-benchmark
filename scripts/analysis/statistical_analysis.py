"""
Statistical Analysis for the Multilingual LLM Safety Evaluation Framework.

Input:
    outputs/evaluated/combined_evaluation.csv

Output:
    outputs/analysis/statistics/

The script performs:
    1. Overall model comparison
    2. Model × language analysis
    3. Model × attack-category analysis
    4. Model × variation analysis
    5. Model × language × safety-label analysis
    6. Chi-square tests
    7. Cramer's V effect sizes
    8. Actionable-content comparison
    9. Defensive-context comparison

This script does not modify the evaluator or evaluated dataset.
"""

from pathlib import Path

import argparse
import pandas as pd
from scipy.stats import chi2_contingency


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

# Input and output are overridable so the exploratory suite can be run
# against the frozen final labels without overwriting the original
# outputs/analysis/statistics/ tables, which are preserved.
#
#   python scripts/analysis/statistical_analysis.py #       --input outputs/evaluated_final/combined_final_512.csv #       --outdir outputs/analysis/statistics_final

_parser = argparse.ArgumentParser(
    description="Exploratory statistical analysis (chi-square)."
)

_parser.add_argument(
    "--input",
    default=str(
        BASE_DIR / "outputs" / "evaluated" / "combined_evaluation.csv"
    ),
)

_parser.add_argument(
    "--outdir",
    default=str(BASE_DIR / "outputs" / "analysis" / "statistics"),
)

_arguments, _unknown = _parser.parse_known_args()

INPUT_FILE = Path(_arguments.input)
OUTPUT_DIR = Path(_arguments.outdir)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONSTANTS
# ============================================================

SAFETY_LABELS = [
    "COMPLIANCE",
    "NON_COMPLIANCE",
    "REFUSAL",
]

LANGUAGE_ORDER = [
    "en",
    "hi",
    "mr",
]

MODEL_ORDER = [
    "gpt_oss",
    "qwen",
]


# ============================================================
# HELPERS
# ============================================================

def cramers_v(contingency_table):
    """
    Calculate Cramer's V for a contingency table.
    """

    chi2, _, _, _ = chi2_contingency(contingency_table)

    n = contingency_table.to_numpy().sum()

    if n == 0:
        return 0.0

    rows, columns = contingency_table.shape

    denominator = min(rows - 1, columns - 1)

    if denominator == 0:
        return 0.0

    return (chi2 / (n * denominator)) ** 0.5


def run_chi_square(contingency_table, name):
    """
    Run chi-square test and return a result dictionary.
    """

    chi2, p_value, degrees_of_freedom, expected = chi2_contingency(
        contingency_table
    )

    effect_size = cramers_v(contingency_table)

    return {
        "analysis": name,
        "chi_square": round(chi2, 4),
        "degrees_of_freedom": degrees_of_freedom,
        "p_value": round(p_value, 6),
        "cramers_v": round(effect_size, 4),
    }


def save_csv(dataframe, filename):
    """
    Save dataframe into the statistics output directory.
    """

    path = OUTPUT_DIR / filename
    dataframe.to_csv(path, index=False)
    print(f"Saved: {path}")


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("STATISTICAL ANALYSIS")
print("=" * 60)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print()
print(f"Input file: {INPUT_FILE}")
print(f"Total responses: {len(df)}")

required_columns = [
    "model",
    "language",
    "attack_id",
    "attack_category",
    "variation_id",
    "safety_label",
    "refusal_detected",
    "actionable_content",
    "defensive_context",
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# 1. MODEL × SAFETY LABEL
# ============================================================

print()
print("=" * 60)
print("1. MODEL × SAFETY LABEL")
print("=" * 60)

model_safety = pd.crosstab(
    df["model"],
    df["safety_label"]
).reindex(
    index=MODEL_ORDER,
    columns=SAFETY_LABELS,
    fill_value=0,
)

print(model_safety)

save_csv(
    model_safety.reset_index(),
    "model_safety_counts.csv"
)


# ============================================================
# 2. MODEL COMPARISON — CHI-SQUARE
# ============================================================

print()
print("=" * 60)
print("2. MODEL EFFECT")
print("=" * 60)

model_test = run_chi_square(
    model_safety,
    "Model × Safety Label"
)

model_test_df = pd.DataFrame([model_test])

print(model_test_df.to_string(index=False))

save_csv(
    model_test_df,
    "model_safety_chi_square.csv"
)


# ============================================================
# 3. LANGUAGE × SAFETY LABEL
# ============================================================

print()
print("=" * 60)
print("3. LANGUAGE × SAFETY LABEL")
print("=" * 60)

language_safety = pd.crosstab(
    df["language"],
    df["safety_label"]
).reindex(
    index=LANGUAGE_ORDER,
    columns=SAFETY_LABELS,
    fill_value=0,
)

print(language_safety)

save_csv(
    language_safety.reset_index(),
    "language_safety_counts.csv"
)


# ============================================================
# 4. LANGUAGE EFFECT — CHI-SQUARE
# ============================================================

print()
print("=" * 60)
print("4. LANGUAGE EFFECT")
print("=" * 60)

language_test = run_chi_square(
    language_safety,
    "Language × Safety Label"
)

language_test_df = pd.DataFrame([language_test])

print(language_test_df.to_string(index=False))

save_csv(
    language_test_df,
    "language_safety_chi_square.csv"
)


# ============================================================
# 5. MODEL × LANGUAGE × SAFETY
# ============================================================

print()
print("=" * 60)
print("5. MODEL × LANGUAGE")
print("=" * 60)

model_language_rows = []

for model in MODEL_ORDER:

    subset = df[df["model"] == model]

    table = pd.crosstab(
        subset["language"],
        subset["safety_label"]
    ).reindex(
        index=LANGUAGE_ORDER,
        columns=SAFETY_LABELS,
        fill_value=0,
    )

    test = run_chi_square(
        table,
        f"{model} — Language × Safety Label"
    )

    model_language_rows.append(test)

model_language_tests = pd.DataFrame(model_language_rows)

print(model_language_tests.to_string(index=False))

save_csv(
    model_language_tests,
    "model_language_chi_square.csv"
)


# ============================================================
# 6. ATTACK CATEGORY EFFECT
# ============================================================

print()
print("=" * 60)
print("6. ATTACK CATEGORY EFFECT")
print("=" * 60)

attack_safety = pd.crosstab(
    df["attack_category"],
    df["safety_label"]
).reindex(
    columns=SAFETY_LABELS,
    fill_value=0,
)

print(attack_safety)

save_csv(
    attack_safety.reset_index(),
    "attack_category_safety_counts.csv"
)

attack_test = run_chi_square(
    attack_safety,
    "Attack Category × Safety Label"
)

attack_test_df = pd.DataFrame([attack_test])

print()
print(attack_test_df.to_string(index=False))

save_csv(
    attack_test_df,
    "attack_category_chi_square.csv"
)


# ============================================================
# 7. MODEL × ATTACK CATEGORY
# ============================================================

print()
print("=" * 60)
print("7. MODEL × ATTACK CATEGORY")
print("=" * 60)

model_attack_tests = []

for model in MODEL_ORDER:

    subset = df[df["model"] == model]

    table = pd.crosstab(
        subset["attack_category"],
        subset["safety_label"]
    ).reindex(
        columns=SAFETY_LABELS,
        fill_value=0,
    )

    test = run_chi_square(
        table,
        f"{model} — Attack Category × Safety Label"
    )

    model_attack_tests.append(test)

model_attack_tests_df = pd.DataFrame(model_attack_tests)

print(model_attack_tests_df.to_string(index=False))

save_csv(
    model_attack_tests_df,
    "model_attack_chi_square.csv"
)


# ============================================================
# 8. VARIATION EFFECT
# ============================================================

print()
print("=" * 60)
print("8. VARIATION EFFECT")
print("=" * 60)

variation_safety = pd.crosstab(
    df["variation_id"],
    df["safety_label"]
).reindex(
    columns=SAFETY_LABELS,
    fill_value=0,
)

print(variation_safety)

save_csv(
    variation_safety.reset_index(),
    "variation_safety_counts.csv"
)

variation_test = run_chi_square(
    variation_safety,
    "Variation × Safety Label"
)

variation_test_df = pd.DataFrame([variation_test])

print()
print(variation_test_df.to_string(index=False))

save_csv(
    variation_test_df,
    "variation_safety_chi_square.csv"
)


# ============================================================
# 9. MODEL × VARIATION
# ============================================================

print()
print("=" * 60)
print("9. MODEL × VARIATION")
print("=" * 60)

model_variation_tests = []

for model in MODEL_ORDER:

    subset = df[df["model"] == model]

    table = pd.crosstab(
        subset["variation_id"],
        subset["safety_label"]
    ).reindex(
        columns=SAFETY_LABELS,
        fill_value=0,
    )

    test = run_chi_square(
        table,
        f"{model} — Variation × Safety Label"
    )

    model_variation_tests.append(test)

model_variation_tests_df = pd.DataFrame(
    model_variation_tests
)

print(model_variation_tests_df.to_string(index=False))

save_csv(
    model_variation_tests_df,
    "model_variation_chi_square.csv"
)


# ============================================================
# 10. ACTIONABLE CONTENT
# ============================================================

print()
print("=" * 60)
print("10. ACTIONABLE CONTENT")
print("=" * 60)

actionable_table = pd.crosstab(
    df["model"],
    df["actionable_content"]
).reindex(
    index=MODEL_ORDER,
    fill_value=0,
)

print(actionable_table)

save_csv(
    actionable_table.reset_index(),
    "actionable_content_counts.csv"
)

if actionable_table.shape[1] >= 2:

    actionable_test = run_chi_square(
        actionable_table,
        "Model × Actionable Content"
    )

    actionable_test_df = pd.DataFrame(
        [actionable_test]
    )

    print()
    print(actionable_test_df.to_string(index=False))

    save_csv(
        actionable_test_df,
        "actionable_content_chi_square.csv"
    )


# ============================================================
# 11. DEFENSIVE CONTEXT
# ============================================================

print()
print("=" * 60)
print("11. DEFENSIVE CONTEXT")
print("=" * 60)

defensive_table = pd.crosstab(
    df["model"],
    df["defensive_context"]
).reindex(
    index=MODEL_ORDER,
    fill_value=0,
)

print(defensive_table)

save_csv(
    defensive_table.reset_index(),
    "defensive_context_counts.csv"
)

if defensive_table.shape[1] >= 2:

    defensive_test = run_chi_square(
        defensive_table,
        "Model × Defensive Context"
    )

    defensive_test_df = pd.DataFrame(
        [defensive_test]
    )

    print()
    print(defensive_test_df.to_string(index=False))

    save_csv(
        defensive_test_df,
        "defensive_context_chi_square.csv"
    )


# ============================================================
# 12. SUMMARY
# ============================================================

print()
print("=" * 60)
print("STATISTICAL ANALYSIS COMPLETE")
print("=" * 60)

print()
print(f"Results saved to:")
print(OUTPUT_DIR)