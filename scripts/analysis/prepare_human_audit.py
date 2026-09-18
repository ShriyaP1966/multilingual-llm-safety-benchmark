import pandas as pd
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

SOURCE_FILE = Path(
    "outputs/evaluated/combined_evaluation.csv"
)

QUEUE_FILE = Path(
    "outputs/analysis/audit/targeted_audit_queue.csv"
)

OUTPUT_FILE = Path(
    "outputs/analysis/audit/human_review_queue.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

if not SOURCE_FILE.exists():
    raise FileNotFoundError(
        f"Source file not found: {SOURCE_FILE}"
    )

if not QUEUE_FILE.exists():
    raise FileNotFoundError(
        f"Audit queue not found: {QUEUE_FILE}"
    )

df = pd.read_csv(
    SOURCE_FILE,
    keep_default_na=False
)

queue = pd.read_csv(
    QUEUE_FILE,
    keep_default_na=False
)


# ============================================================
# IDENTIFY AUDIT ROWS
# ============================================================

key_columns = [
    "attack_id",
    "variation_id",
    "language",
    "model"
]

queue_columns = key_columns + [
    "audit_score"
]

review = df.merge(
    queue[queue_columns],
    on=key_columns,
    how="inner"
)


# ============================================================
# REMOVE ACCIDENTAL DUPLICATES
# ============================================================

review = review.drop_duplicates(
    subset=[
        "attack_id",
        "variation_id",
        "language",
        "model",
        "response"
    ]
)


# ============================================================
# ADD HUMAN REVIEW FIELDS
# ============================================================

review["human_label"] = ""

review["human_agreement"] = ""

review["review_notes"] = ""


# ============================================================
# SELECT USEFUL COLUMNS
# ============================================================

columns = [
    "attack_id",
    "variation_id",
    "attack_category",
    "language",
    "model",
    "prompt",
    "response",
    "safety_label",
    "refusal_detected",
    "actionable_content",
    "defensive_context",
    "audit_score",
    "human_label",
    "human_agreement",
    "review_notes"
]

review = review[columns]


# ============================================================
# SORT FOR REVIEW
# ============================================================

review = review.sort_values(
    [
        "audit_score",
        "model",
        "language",
        "attack_id",
        "variation_id"
    ],
    ascending=[
        False,
        True,
        True,
        True,
        True
    ]
)


# ============================================================
# SAVE
# ============================================================

review.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# SUMMARY
# ============================================================

print("=" * 60)
print("HUMAN REVIEW QUEUE CREATED")
print("=" * 60)

print(
    f"\nOutput:\n{OUTPUT_FILE.resolve()}"
)

print(
    f"\nRows prepared for review: {len(review)}"
)

print("\nCurrent evaluator labels:")

print(
    review["safety_label"]
    .value_counts()
)

print("\nCoverage by model:")

print(
    review["model"]
    .value_counts()
)

print("\nCoverage by language:")

print(
    review["language"]
    .value_counts()
)

print("\nReview columns:")

print(
    ", ".join(review.columns)
)