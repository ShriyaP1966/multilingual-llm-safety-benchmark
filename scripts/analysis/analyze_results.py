import pandas as pd
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path("outputs/evaluated/combined_evaluation.csv")
OUTPUT_DIR = Path("outputs/analysis")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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
# BASIC INFORMATION
# ============================================================

print("=" * 60)
print("MULTILINGUAL SAFETY BENCHMARK ANALYSIS")
print("=" * 60)

print("\nTotal responses:", len(df))
print("Models:", df["model"].nunique())
print("Languages:", df["language"].nunique())
print("Attacks:", df["attack_id"].nunique())
print("Variations:", df["variation_id"].nunique())


# ============================================================
# 1. OVERALL MODEL PERFORMANCE
# ============================================================

overall = pd.crosstab(
    df["model"],
    df["safety_label"]
)

overall_percent = (
    overall.div(overall.sum(axis=1), axis=0) * 100
).round(2)

print("\n" + "=" * 60)
print("1. OVERALL MODEL PERFORMANCE")
print("=" * 60)

print("\nCOUNTS:")
print(overall.to_string())

print("\nPERCENTAGES:")
print(overall_percent.to_string())


overall.to_csv(
    OUTPUT_DIR / "overall_counts.csv"
)

overall_percent.to_csv(
    OUTPUT_DIR / "overall_percentages.csv"
)


# ============================================================
# 2. MODEL × LANGUAGE
# ============================================================

model_language = pd.crosstab(
    [df["model"], df["language"]],
    df["safety_label"]
)

model_language_percent = (
    model_language.div(
        model_language.sum(axis=1),
        axis=0
    ) * 100
).round(2)

print("\n" + "=" * 60)
print("2. MODEL × LANGUAGE")
print("=" * 60)

print("\nCOUNTS:")
print(model_language.to_string())

print("\nPERCENTAGES:")
print(model_language_percent.to_string())


model_language.to_csv(
    OUTPUT_DIR / "model_language_counts.csv"
)

model_language_percent.to_csv(
    OUTPUT_DIR / "model_language_percentages.csv"
)


# ============================================================
# 3. MODEL × ATTACK
# ============================================================

model_attack = pd.crosstab(
    [df["model"], df["attack_id"]],
    df["safety_label"]
)

model_attack_percent = (
    model_attack.div(
        model_attack.sum(axis=1),
        axis=0
    ) * 100
).round(2)

print("\n" + "=" * 60)
print("3. MODEL × ATTACK")
print("=" * 60)

print("\nPERCENTAGES:")
print(model_attack_percent.to_string())

model_attack.to_csv(
    OUTPUT_DIR / "model_attack_counts.csv"
)

model_attack_percent.to_csv(
    OUTPUT_DIR / "model_attack_percentages.csv"
)


# ============================================================
# 4. MODEL × VARIATION
# ============================================================

model_variation = pd.crosstab(
    [df["model"], df["variation_id"]],
    df["safety_label"]
)

model_variation_percent = (
    model_variation.div(
        model_variation.sum(axis=1),
        axis=0
    ) * 100
).round(2)

print("\n" + "=" * 60)
print("4. MODEL × VARIATION")
print("=" * 60)

print("\nPERCENTAGES:")
print(model_variation_percent.to_string())

model_variation.to_csv(
    OUTPUT_DIR / "model_variation_counts.csv"
)

model_variation_percent.to_csv(
    OUTPUT_DIR / "model_variation_percentages.csv"
)


# ============================================================
# 5. REFUSAL RATE
# ============================================================

refusal_rate = (
    df.groupby("model")["refusal_detected"]
    .mean() * 100
).round(2)

print("\n" + "=" * 60)
print("5. OVERALL REFUSAL RATE")
print("=" * 60)

print(refusal_rate.to_string())

refusal_rate.to_csv(
    OUTPUT_DIR / "overall_refusal_rate.csv"
)


# ============================================================
# 6. REFUSAL RATE BY MODEL × LANGUAGE
# ============================================================

language_refusal = (
    df.groupby(
        ["model", "language"]
    )["refusal_detected"]
    .mean() * 100
).round(2)

print("\n" + "=" * 60)
print("6. REFUSAL RATE BY MODEL × LANGUAGE")
print("=" * 60)

print(language_refusal.to_string())

language_refusal.to_csv(
    OUTPUT_DIR / "language_refusal_rates.csv"
)


# ============================================================
# 7. BASELINE VS TRANSFORMED VARIATIONS
# ============================================================

variation_refusal = (
    df.groupby(
        ["model", "language", "variation_id"]
    )["refusal_detected"]
    .mean() * 100
).round(2)

baseline_variation = (
    variation_refusal
    .xs("V1- Baseline", level="variation_id")
)

degradation_rows = []

for (model, language, variation), rate in variation_refusal.items():

    baseline = baseline_variation.loc[
        (model, language)
    ]

    degradation_rows.append({
        "model": model,
        "language": language,
        "variation_id": variation,
        "baseline_refusal_rate": baseline,
        "variation_refusal_rate": rate,
        "degradation_pp": round(
            baseline - rate,
            2
        )
    })


degradation = pd.DataFrame(
    degradation_rows
)

degradation = degradation.sort_values(
    "degradation_pp",
    ascending=False
)

print("\n" + "=" * 60)
print("7. BASELINE VS VARIATION DEGRADATION")
print("=" * 60)

print(degradation.to_string(index=False))

degradation.to_csv(
    OUTPUT_DIR / "variation_degradation.csv",
    index=False
)


# ============================================================
# 8. WORST-CASE VARIATION
# ============================================================

non_baseline = degradation[
    degradation["variation_id"] != "V1- Baseline"
]

if not non_baseline.empty:

    worst_variation = non_baseline.iloc[0]

    print("\n" + "=" * 60)
    print("8. WORST-CASE VARIATION")
    print("=" * 60)

    print(
        f"Model: {worst_variation['model']}"
    )

    print(
        f"Language: {worst_variation['language']}"
    )

    print(
        f"Variation: {worst_variation['variation_id']}"
    )

    print(
        f"Baseline refusal: "
        f"{worst_variation['baseline_refusal_rate']:.2f}%"
    )

    print(
        f"Variation refusal: "
        f"{worst_variation['variation_refusal_rate']:.2f}%"
    )

    print(
        f"Degradation: "
        f"{worst_variation['degradation_pp']:.2f} percentage points"
    )


# ============================================================
# 9. ACTIONABLE CONTENT
# ============================================================

actionable = (
    df.groupby("model")["actionable_content"]
    .sum()
)

actionable_rate = (
    df.groupby("model")["actionable_content"]
    .mean() * 100
).round(2)

print("\n" + "=" * 60)
print("9. ACTIONABLE CONTENT")
print("=" * 60)

print("\nCOUNT:")
print(actionable.to_string())

print("\nRATE:")
print(actionable_rate.to_string())

actionable_rate.to_csv(
    OUTPUT_DIR / "actionable_content_rate.csv"
)


# ============================================================
# 10. SAVE A CLEAN SUMMARY
# ============================================================

summary_rows = []

for model in df["model"].unique():

    subset = df[df["model"] == model]

    summary_rows.append({
        "model": model,
        "total_responses": len(subset),
        "refusal_count": int(
            subset["refusal_detected"].sum()
        ),
        "refusal_rate_percent": round(
            subset["refusal_detected"].mean() * 100,
            2
        ),
        "actionable_count": int(
            subset["actionable_content"].sum()
        ),
        "actionable_rate_percent": round(
            subset["actionable_content"].mean() * 100,
            2
        ),
        "compliance_count": int(
            (subset["safety_label"] == "COMPLIANCE").sum()
        ),
        "partial_compliance_count": int(
            (subset["safety_label"] == "PARTIAL_COMPLIANCE").sum()
        )
    })


summary = pd.DataFrame(summary_rows)

print("\n" + "=" * 60)
print("10. FINAL MODEL SUMMARY")
print("=" * 60)

print(summary.to_string(index=False))

summary.to_csv(
    OUTPUT_DIR / "model_summary.csv",
    index=False
)


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)

print(
    f"\nAnalysis files saved to: {OUTPUT_DIR.resolve()}"
)