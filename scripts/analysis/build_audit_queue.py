import pandas as pd
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "outputs/evaluated/combined_evaluation.csv"
)

OUTPUT_DIR = Path(
    "outputs/analysis/audit"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = (
    OUTPUT_DIR / "targeted_audit_queue.csv"
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
# BUILD AUDIT SCORE
# ============================================================

df["audit_score"] = 0


# ------------------------------------------------------------
# 1. COMPLIANCE / NON-COMPLIANCE AMBIGUITY
# ------------------------------------------------------------

df.loc[
    df["safety_label"].isin(
        ["COMPLIANCE", "NON_COMPLIANCE"]
    ),
    "audit_score"
] += 2


# ------------------------------------------------------------
# 2. ACTIONABLE CONTENT
# ------------------------------------------------------------

df.loc[
    df["actionable_content"] == 1,
    "audit_score"
] += 3


# ------------------------------------------------------------
# 3. REFUSAL + OTHER SAFETY SIGNALS
# ------------------------------------------------------------

df.loc[
    (
        (df["refusal_detected"] == 1)
        &
        (df["actionable_content"] == 1)
    ),
    "audit_score"
] += 4


# ------------------------------------------------------------
# 4. DEFENSIVE CONTEXT
# ------------------------------------------------------------

df.loc[
    df["defensive_context"] == 1,
    "audit_score"
] += 2


# ------------------------------------------------------------
# 5. MULTILINGUAL PRIORITY
# ------------------------------------------------------------

df.loc[
    df["language"].isin(["hi", "mr"]),
    "audit_score"
] += 1


# ------------------------------------------------------------
# 6. VERY SHORT RESPONSES
# ------------------------------------------------------------

response_length = (
    df["response"]
    .astype(str)
    .str.len()
)

df.loc[
    response_length < 100,
    "audit_score"
] += 1


# ------------------------------------------------------------
# 7. LONG RESPONSES
# ------------------------------------------------------------

df.loc[
    response_length > 1500,
    "audit_score"
] += 1


# ============================================================
# PRIORITIZE CASES
# ============================================================

audit_df = (
    df
    .sort_values(
        [
            "audit_score",
            "actionable_content",
            "defensive_context"
        ],
        ascending=False
    )
    .copy()
)


# ============================================================
# SELECT TARGETED REVIEW SET
# ============================================================

# Keep the audit manageable while ensuring
# multilingual/model coverage.

selected_parts = []

for model in df["model"].unique():

    for language in df["language"].unique():

        subset = audit_df[
            (audit_df["model"] == model)
            &
            (audit_df["language"] == language)
        ]

        selected_parts.append(
            subset.head(15)
        )


audit_queue = pd.concat(
    selected_parts,
    ignore_index=True
)

audit_queue = (
    audit_queue
    .drop_duplicates(
        subset=["response"]
    )
    .sort_values(
        "audit_score",
        ascending=False
    )
)


# ============================================================
# SAVE AUDIT QUEUE
# ============================================================

audit_queue.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# SUMMARY
# ============================================================

print("=" * 60)
print("TARGETED AUDIT QUEUE CREATED")
print("=" * 60)

print(
    f"\nOutput:\n{OUTPUT_FILE.resolve()}"
)

print(
    f"\nTotal benchmark responses: {len(df)}"
)

print(
    f"Targeted audit responses: {len(audit_queue)}"
)

print(
    "\nAudit queue by model:"
)

print(
    audit_queue["model"]
    .value_counts()
)

print(
    "\nAudit queue by language:"
)

print(
    audit_queue["language"]
    .value_counts()
)

print(
    "\nAudit queue by safety label:"
)

print(
    audit_queue["safety_label"]
    .value_counts()
)

print(
    "\nAudit score distribution:"
)

print(
    audit_queue["audit_score"]
    .value_counts()
    .sort_index()
)