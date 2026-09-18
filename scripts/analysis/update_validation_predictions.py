import pandas as pd
from pathlib import Path

from evaluate import (
    classify_response,
    detect_refusal,
    detect_actionable_content,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "validation_sample.csv"
OUTPUT_FILE = PROJECT_ROOT / "validation_sample.csv"


df = pd.read_csv(
    INPUT_FILE,
    keep_default_na=False
)


# --------------------------------------------------------
# Re-run the current evaluator on all 63 validation
# responses using the updated classification logic.
# --------------------------------------------------------

df["safety_label"] = df.apply(
    lambda row: classify_response(
        row["response"],
        language=row["language"]
    ),
    axis=1
)


# --------------------------------------------------------
# Refresh the supporting automated detection columns.
# These are predictions from the evaluator, NOT manual
# ground-truth annotations.
# --------------------------------------------------------

df["refusal_detected"] = df["response"].apply(
    detect_refusal
)

df["actionable_content"] = df.apply(
    lambda row: detect_actionable_content(
        row["response"],
        language=row["language"]
    ),
    axis=1
)


# --------------------------------------------------------
# Save the updated predictions.
#
# manual_label and reason are intentionally NOT modified.
# --------------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("VALIDATION PREDICTIONS UPDATED")
print()
print("Automated safety labels:")
print(df["safety_label"].value_counts())
print()
print("Manual ground-truth labels:")
print(df["manual_label"].value_counts())
print()
print(
    "Remaining blank manual labels:",
    (df["manual_label"] == "").sum()
)