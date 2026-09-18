"""
validate_dataset.py

Validates the multilingual benchmark dataset before any API calls.

Author: Shriya Patil
"""

from collections import Counter

from utils import (
    load_config,
    load_dataset,
    setup_logger,
)


# ---------------------------------------------------------------------
# Expected Dataset Structure
# ---------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "attack_id",
    "variation_id",
    "attack_category",
    "prompt_en",
    "prompt_hi",
    "prompt_mr",
]

EXPECTED_CATEGORIES = 13
EXPECTED_VARIATIONS = 8


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def validate_dataset() -> bool:

    config = load_config()

    logger = setup_logger(logger_name="dataset_validation")

    dataset_path = config["dataset"]["path"]

    df = load_dataset(dataset_path)

    passed = True

    print("=" * 60)
    print("DATASET VALIDATION REPORT")
    print("=" * 60)

    print(f"Dataset : {dataset_path}")
    print(f"Rows    : {len(df)}")
    print()

    # -------------------------------------------------
    # Required Columns
    # -------------------------------------------------

    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        print("❌ Missing Columns")
        print(missing_columns)
        logger.error(f"Missing columns: {missing_columns}")
        passed = False
    else:
        print("✓ Required Columns................PASS")

    # -------------------------------------------------
    # Missing Values
    # -------------------------------------------------

    missing = df[REQUIRED_COLUMNS].isnull().sum().sum()

    if missing == 0:
        print("✓ Missing Values.................PASS")
    else:
        print(f"❌ Missing Values.................FAIL ({missing})")
        logger.error(f"Missing values: {missing}")
        passed = False

    # -------------------------------------------------
    # Duplicate English prompts
    # -------------------------------------------------

    duplicates = df["prompt_en"].duplicated().sum()

    if duplicates == 0:
        print("✓ Duplicate English Prompts......PASS")
    else:
        print(f"❌ Duplicate English Prompts......FAIL ({duplicates})")
        logger.error(f"Duplicate prompts: {duplicates}")
        passed = False

    # -------------------------------------------------
    # Categories
    # -------------------------------------------------

    categories = df["attack_category"].nunique()

    if categories == EXPECTED_CATEGORIES:
        print("✓ Attack Categories..............PASS")
    else:
        print(
            f"❌ Attack Categories..............FAIL ({categories})"
        )
        logger.error(f"Category count: {categories}")
        passed = False

    # -------------------------------------------------
    # Variations
    # -------------------------------------------------

    variation_counts = Counter(df["variation_id"])

    if len(variation_counts) == EXPECTED_VARIATIONS:
        print("✓ Variations.....................PASS")
    else:
        print("❌ Variations.....................FAIL")
        logger.error("Variation IDs incorrect")
        passed = False

    # -------------------------------------------------
    # UTF-8 Check
    # -------------------------------------------------

    try:

        df["prompt_hi"].str.encode("utf-8")

        df["prompt_mr"].str.encode("utf-8")

        print("✓ UTF-8 Encoding.................PASS")

    except Exception as e:

        print("❌ UTF-8 Encoding.................FAIL")

        logger.error(str(e))

        passed = False

    print("=" * 60)

    if passed:

        print("STATUS : DATASET READY FOR EXPERIMENT")

    else:

        print("STATUS : VALIDATION FAILED")

    print("=" * 60)

    logger.info(f"Validation Result : {passed}")

    return passed


# ---------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------

if __name__ == "__main__":

    validate_dataset()