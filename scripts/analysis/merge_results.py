"""
merge_results.py

Pivot raw benchmark responses from long format into the per-model wide
format the evaluator consumes.

Input:
    outputs/raw/<provider>_<model>_results.csv

    One row per (attack_id, variation_id, language), carrying the
    prompt, the response and the token counts returned by the provider.

Output:
    outputs/merged/working_dataset_<model>.csv

    One row per (attack_id, variation_id), with prompt and response
    pivoted into per-language columns.

Why this script exists:
    The wide datasets in data/ were originally produced by hand. That
    lost the token-count columns, which is why evaluate.py could not
    populate its `truncated` field and detect_truncation() sat unused
    while 24% of GPT-OSS and 64% of Qwen responses were in fact cut off
    at the 512-token ceiling.

    This script reproduces the pivot reproducibly and carries
    output_tokens through as output_tokens_<language>.

Safety:
    Writes to outputs/merged/ and never to data/. Pass --verify to
    confirm the regenerated prompt/response columns match the existing
    data/working_dataset_<model>.csv exactly, so the original evaluator
    inputs are provably unchanged.
"""

from pathlib import Path

import argparse
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "outputs" / "raw"
MERGED_DIR = PROJECT_ROOT / "outputs" / "merged"
DATA_DIR = PROJECT_ROOT / "data"


# Raw result file -> short model name used throughout the analysis.
RAW_FILES = {
    "gpt_oss": "openai_gpt_oss_20b_results.csv",
    "qwen": "qwen_qwen3.8_27b_results.csv",
}

LANGUAGES = ["en", "hi", "mr"]

KEY_COLUMNS = ["attack_id", "variation_id", "attack_category"]


# ============================================================
# MERGE ONE MODEL
# ============================================================

def merge_model(model_name, raw_path):
    """
    Pivot one model's long-format responses into wide format.
    """

    print()
    print("=" * 60)
    print(f"MERGING: {model_name}")
    print("=" * 60)

    raw = pd.read_csv(
        raw_path,
        keep_default_na=False
    )

    print(f"Raw rows: {len(raw)}")

    required = KEY_COLUMNS + [
        "language",
        "prompt",
        "response",
    ]

    missing = [
        column for column in required
        if column not in raw.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns in {raw_path.name}: {missing}"
        )

    unexpected = set(raw["language"]) - set(LANGUAGES)

    if unexpected:
        raise ValueError(
            f"Unexpected languages in {raw_path.name}: "
            f"{sorted(unexpected)}"
        )

    duplicates = raw.duplicated(
        subset=["attack_id", "variation_id", "language"]
    )

    if duplicates.any():
        raise ValueError(
            f"Duplicate (attack_id, variation_id, language) keys: "
            f"{duplicates.sum()} rows."
        )

    # --------------------------------------------------------
    # Establish the row index from the key columns.
    # --------------------------------------------------------

    base = (
        raw[KEY_COLUMNS]
        .drop_duplicates()
        .sort_values(["attack_id", "variation_id"])
        .reset_index(drop=True)
    )

    print(f"Prompt sets: {len(base)}")

    merged = base.copy()

    # --------------------------------------------------------
    # Pivot prompt, response and output_tokens per language.
    #
    # Column order matches the original hand-built datasets:
    # all prompts, then all responses, then the token counts.
    # --------------------------------------------------------

    has_tokens = "output_tokens" in raw.columns

    if not has_tokens:
        print(
            "WARNING: no output_tokens column; truncation "
            "cannot be detected downstream."
        )

    for field, prefix in [("prompt", "prompt"), ("response", "response")]:

        for language in LANGUAGES:

            subset = raw[raw["language"] == language]

            merged = merged.merge(
                subset[
                    ["attack_id", "variation_id", field]
                ].rename(
                    columns={field: f"{prefix}_{language}"}
                ),
                on=["attack_id", "variation_id"],
                how="left",
            )

    if has_tokens:

        for language in LANGUAGES:

            subset = raw[raw["language"] == language]

            merged = merged.merge(
                subset[
                    ["attack_id", "variation_id", "output_tokens"]
                ].rename(
                    columns={
                        "output_tokens": f"output_tokens_{language}"
                    }
                ),
                on=["attack_id", "variation_id"],
                how="left",
            )

    # --------------------------------------------------------
    # Every prompt set must have all three languages.
    # --------------------------------------------------------

    for language in LANGUAGES:

        for prefix in ["prompt", "response"]:

            column = f"{prefix}_{language}"

            if merged[column].isna().any():
                raise ValueError(
                    f"Missing {column} for "
                    f"{merged[column].isna().sum()} prompt sets."
                )

    MERGED_DIR.mkdir(parents=True, exist_ok=True)

    output_path = (
        MERGED_DIR / f"working_dataset_{model_name}.csv"
    )

    merged.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"Output rows: {len(merged)}")
    print(f"Saved: {output_path}")

    # --------------------------------------------------------
    # Truncation summary — the reason this script carries tokens.
    # --------------------------------------------------------

    if has_tokens:

        print()
        print("Responses at or above the 512-token ceiling:")

        for language in LANGUAGES:

            tokens = pd.to_numeric(
                merged[f"output_tokens_{language}"],
                errors="coerce"
            )

            at_ceiling = (tokens >= 512).sum()

            print(
                f"  {language}: {at_ceiling}/{len(merged)} "
                f"({100 * at_ceiling / len(merged):.1f}%)"
            )

    return merged


# ============================================================
# VERIFY AGAINST THE ORIGINAL HAND-BUILT DATASETS
# ============================================================

def verify_against_original(model_name, merged):
    """
    Confirm the regenerated prompt/response columns match the
    original data/working_dataset_<model>.csv exactly.

    The token columns are new and are excluded from the comparison.
    """

    original_path = (
        DATA_DIR / f"working_dataset_{model_name}.csv"
    )

    if not original_path.exists():
        print(
            f"  SKIP {model_name}: no original at {original_path}"
        )
        return None

    original = pd.read_csv(
        original_path,
        keep_default_na=False
    )

    compare_columns = KEY_COLUMNS + [
        f"{prefix}_{language}"
        for prefix in ["prompt", "response"]
        for language in LANGUAGES
    ]

    missing = [
        column for column in compare_columns
        if column not in original.columns
    ]

    if missing:
        print(
            f"  {model_name}: original lacks {missing}"
        )
        return False

    left = (
        original[compare_columns]
        .sort_values(["attack_id", "variation_id"])
        .reset_index(drop=True)
        .astype(str)
    )

    right = (
        merged[compare_columns]
        .sort_values(["attack_id", "variation_id"])
        .reset_index(drop=True)
        .astype(str)
    )

    if left.equals(right):
        print(
            f"  {model_name}: MATCH — {len(left)} rows, "
            f"{len(compare_columns)} columns identical"
        )
        return True

    print(f"  {model_name}: MISMATCH")

    for column in compare_columns:

        differing = (left[column] != right[column]).sum()

        if differing:
            print(
                f"    {column}: {differing} differing values"
            )

    return False


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Pivot raw benchmark responses into the per-model wide "
            "format, carrying output_tokens through."
        )
    )

    parser.add_argument(
        "--raw-dir",
        default=None,
        help=(
            "Directory holding the raw long-format results. Defaults "
            "to outputs/raw (the 512-token experiment). Use "
            "outputs/raw_2048 for the follow-up experiment."
        ),
    )

    parser.add_argument(
        "--out-dir",
        default=None,
        help=(
            "Directory to write the wide datasets into. Defaults to "
            "outputs/merged."
        ),
    )

    parser.add_argument(
        "--models",
        default=None,
        help=(
            "Comma-separated subset of models to merge, e.g. "
            "'gpt_oss'. Defaults to all models present."
        ),
    )

    parser.add_argument(
        "--verify",
        action="store_true",
        help=(
            "Compare the regenerated prompt/response columns against "
            "the original data/working_dataset_<model>.csv files."
        ),
    )

    arguments = parser.parse_args()

    global RAW_DIR, MERGED_DIR

    if arguments.raw_dir:
        RAW_DIR = Path(arguments.raw_dir)

    if arguments.out_dir:
        MERGED_DIR = Path(arguments.out_dir)

    selected = (
        [name.strip() for name in arguments.models.split(",")]
        if arguments.models else list(RAW_FILES)
    )

    unknown = [name for name in selected if name not in RAW_FILES]

    if unknown:
        raise ValueError(f"Unknown model names: {unknown}")

    print("=" * 60)
    print("MERGE RAW RESULTS INTO WIDE FORMAT")
    print("=" * 60)
    print(f"Raw directory   : {RAW_DIR}")
    print(f"Output directory: {MERGED_DIR}")
    print(f"Models          : {selected}")

    results = {}

    for model_name in selected:

        filename = RAW_FILES[model_name]

        raw_path = RAW_DIR / filename

        if not raw_path.exists():
            print(
                f"WARNING: Missing raw file: {raw_path}"
            )
            continue

        results[model_name] = merge_model(
            model_name,
            raw_path
        )

    if not results:
        print("No raw result files found.")
        return

    if arguments.verify:

        print()
        print("=" * 60)
        print("VERIFY AGAINST ORIGINAL DATASETS")
        print("=" * 60)

        outcomes = {
            model_name: verify_against_original(model_name, merged)
            for model_name, merged in results.items()
        }

        checked = [
            outcome for outcome in outcomes.values()
            if outcome is not None
        ]

        print()

        if checked and all(checked):
            print(
                "All regenerated datasets reproduce the originals."
            )
        elif checked:
            print(
                "At least one dataset does not reproduce its "
                "original. Do not use the merged output until the "
                "difference is explained."
            )

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
