import argparse

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# CONFIGURATION
#
# Input and output are overridable so figures can be regenerated
# from the frozen final labels without overwriting the original
# outputs/analysis/figures/, which are preserved.
#
#   python scripts/analysis/visualize_results.py #       --input outputs/evaluated_final/combined_final_512.csv #       --outdir outputs/analysis/figures_final
# ============================================================

_parser = argparse.ArgumentParser(
    description="Generate benchmark analysis figures."
)

_parser.add_argument(
    "--input",
    default="outputs/evaluated/combined_evaluation.csv",
)

_parser.add_argument(
    "--outdir",
    default="outputs/analysis/figures",
)

_arguments, _unknown = _parser.parse_known_args()

INPUT_FILE = Path(_arguments.input)

OUTPUT_DIR = Path(_arguments.outdir)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Evaluation file not found: {INPUT_FILE}"
    )

df = pd.read_csv(
    INPUT_FILE,
    keep_default_na=False
)


# ============================================================
# BASIC SETTINGS
# ============================================================

models = df["model"].unique()
languages = df["language"].unique()
variations = df["variation_id"].unique()


# ============================================================
# 1. OVERALL REFUSAL RATE BY MODEL
# ============================================================

overall_refusal = (
    df.groupby("model")["refusal_detected"]
    .mean()
    * 100
)

plt.figure(figsize=(8, 5))

plt.bar(
    overall_refusal.index,
    overall_refusal.values
)

plt.ylabel("Refusal Rate (%)")
plt.xlabel("Model")
plt.title("Overall Refusal Rate by Model")
plt.ylim(0, 100)

for i, value in enumerate(overall_refusal.values):
    plt.text(
        i,
        value + 2,
        f"{value:.2f}%",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "01_overall_refusal_rate.png",
    dpi=300
)

plt.close()


# ============================================================
# 2. REFUSAL RATE BY MODEL × LANGUAGE
# ============================================================

language_refusal = (
    df.groupby(
        ["model", "language"]
    )["refusal_detected"]
    .mean()
    * 100
)

language_table = (
    language_refusal
    .unstack()
)

ax = language_table.plot(
    kind="bar",
    figsize=(9, 6)
)

ax.set_ylabel("Refusal Rate (%)")
ax.set_xlabel("Model")
ax.set_title(
    "Refusal Rate Across Languages"
)

ax.set_ylim(0, 100)

plt.xticks(rotation=0)

plt.legend(
    title="Language"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "02_refusal_by_language.png",
    dpi=300
)

plt.close()


# ============================================================
# 3. REFUSAL RATE BY MODEL × VARIATION
# ============================================================

variation_refusal = (
    df.groupby(
        ["model", "variation_id"]
    )["refusal_detected"]
    .mean()
    * 100
)

variation_table = (
    variation_refusal
    .unstack()
)

ax = variation_table.plot(
    kind="bar",
    figsize=(14, 7)
)

ax.set_ylabel("Refusal Rate (%)")
ax.set_xlabel("Model")
ax.set_title(
    "Refusal Rate Across Adversarial Variations"
)

ax.set_ylim(0, 100)

plt.xticks(rotation=0)

plt.legend(
    title="Variation",
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "03_refusal_by_variation.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 4. MODEL × LANGUAGE HEATMAP
# ============================================================

heatmap_data = (
    df.groupby(
        ["model", "language"]
    )["refusal_detected"]
    .mean()
    * 100
)

heatmap_data = (
    heatmap_data
    .unstack()
)

plt.figure(figsize=(8, 5))

plt.imshow(
    heatmap_data.values,
    aspect="auto"
)

plt.colorbar(
    label="Refusal Rate (%)"
)

plt.xticks(
    range(len(heatmap_data.columns)),
    heatmap_data.columns
)

plt.yticks(
    range(len(heatmap_data.index)),
    heatmap_data.index
)

plt.xlabel("Language")
plt.ylabel("Model")

plt.title(
    "Cross-Lingual Safety Robustness"
)

for i in range(
    len(heatmap_data.index)
):
    for j in range(
        len(heatmap_data.columns)
    ):
        plt.text(
            j,
            i,
            f"{heatmap_data.iloc[i, j]:.2f}%",
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "04_cross_lingual_heatmap.png",
    dpi=300
)

plt.close()


# ============================================================
# 5. BASELINE VS VARIATION
# ============================================================

baseline = (
    df[
        df["variation_id"] == "V1- Baseline"
    ]
    .groupby(
        ["model", "language"]
    )["refusal_detected"]
    .mean()
    * 100
)

non_baseline = df[
    df["variation_id"] != "V1- Baseline"
]

variation_rates = (
    non_baseline.groupby(
        ["model", "language", "variation_id"]
    )["refusal_detected"]
    .mean()
    * 100
)

degradation = []

for (
    model,
    language,
    variation
), rate in variation_rates.items():

    base_rate = baseline.loc[
        (model, language)
    ]

    degradation.append({
        "model": model,
        "language": language,
        "variation": variation,
        "baseline": base_rate,
        "variation_rate": rate,
        "change_pp": base_rate - rate
    })

degradation_df = pd.DataFrame(
    degradation
)

largest_decreases = (
    degradation_df
    .sort_values(
        "change_pp",
        ascending=False
    )
    .head(10)
)

labels = (
    largest_decreases["model"]
    + " / "
    + largest_decreases["language"]
    + "\n"
    + largest_decreases["variation"]
)

plt.figure(figsize=(12, 7))

plt.barh(
    labels,
    largest_decreases["change_pp"]
)

plt.xlabel(
    "Decrease in Refusal Rate (percentage points)"
)

plt.ylabel(
    "Model / Language / Variation"
)

plt.title(
    "Largest Decreases in Refusal Rate"
)

plt.axvline(
    0
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "05_largest_refusal_decreases.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 6. ACTIONABLE CONTENT BY MODEL
# ============================================================

actionable_rate = (
    df.groupby("model")["actionable_content"]
    .mean()
    * 100
)

plt.figure(figsize=(8, 5))

plt.bar(
    actionable_rate.index,
    actionable_rate.values
)

plt.ylabel(
    "Actionable Content Rate (%)"
)

plt.xlabel("Model")

plt.title(
    "Actionable Content Rate by Model"
)

for i, value in enumerate(
    actionable_rate.values
):
    plt.text(
        i,
        value + 0.05,
        f"{value:.2f}%",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "06_actionable_content_rate.png",
    dpi=300
)

plt.close()


# ============================================================
# 7. ATTACK CATEGORY × SAFETY BEHAVIOR
# ============================================================

attack_safety = pd.crosstab(
    df["attack_category"],
    df["safety_label"],
    normalize="index"
) * 100

# Ensure consistent label order
safety_order = [
    "COMPLIANCE",
    "NON_COMPLIANCE",
    "REFUSAL"
]

for label in safety_order:
    if label not in attack_safety.columns:
        attack_safety[label] = 0

attack_safety = attack_safety[
    safety_order
]

ax = attack_safety.plot(
    kind="bar",
    stacked=True,
    figsize=(15, 8)
)

ax.set_ylabel(
    "Percentage of Responses (%)"
)

ax.set_xlabel(
    "Attack Category"
)

ax.set_title(
    "Safety Behavior Across Attack Categories"
)

ax.set_ylim(0, 100)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.legend(
    title="Safety Behavior",
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "07_attack_category_safety_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 8. MODEL × LANGUAGE × SAFETY BEHAVIOR
# ============================================================

model_language_safety = pd.crosstab(
    [
        df["model"],
        df["language"]
    ],
    df["safety_label"],
    normalize="index"
) * 100

for label in safety_order:
    if label not in model_language_safety.columns:
        model_language_safety[label] = 0

model_language_safety = (
    model_language_safety[
        safety_order
    ]
)

ax = model_language_safety.plot(
    kind="bar",
    stacked=True,
    figsize=(12, 7)
)

ax.set_ylabel(
    "Percentage of Responses (%)"
)

ax.set_xlabel(
    "Model / Language"
)

ax.set_title(
    "Safety Behavior Across Models and Languages"
)

ax.set_ylim(0, 100)

plt.xticks(
    rotation=0
)

plt.legend(
    title="Safety Behavior",
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "08_model_language_safety_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SAVE DEGRADATION DATA
# ============================================================

degradation_df.to_csv(
    OUTPUT_DIR.parent / "visualization_degradation.csv",
    index=False
)


# ============================================================
# COMPLETION MESSAGE
# ============================================================

print("=" * 60)
print("VISUALIZATION COMPLETE")
print("=" * 60)

print(
    f"\nFigures saved to:\n"
    f"{OUTPUT_DIR.resolve()}"
)

print("\nGenerated figures:")

for file in sorted(
    OUTPUT_DIR.glob("*.png")
):
    print(
        f" - {file.name}"
    )