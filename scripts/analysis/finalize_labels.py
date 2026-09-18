"""
finalize_labels.py

Freeze the evaluator and produce the final label set for
"Cross-Lingual Vulnerability and Prompt Injection in Low-Resource
Languages".

Final experimental scope
------------------------
    GPT-OSS @ 512   complete   included
    Qwen    @ 512   complete   included
    GPT-OSS @ 2048  complete   included
    Qwen    @ 2048  EXCLUDED   provider output-token-rate limitation

The Qwen 2048 arm is permanently excluded. It is not retried, not run
at a reduced cap, and not imputed.

Why this script exists
----------------------
The evaluator was modified after the original 512-token labels were
written, so those saved labels no longer match the evaluator source.
Freezing means fixing the evaluator and regenerating labels once, so
that code, labels, statistics, figures and reported results all agree.

Preservation
------------
The original 512-token evaluation in outputs/evaluated/ is NEVER
modified. It is preserved exactly as generated, and its difference from
the frozen labels is documented by evaluator_diff_report.py.

Frozen labels are written to a separate directory:

    outputs/evaluated_final/
        gpt_oss_512_evaluation.csv
        qwen_512_evaluation.csv
        gpt_oss_2048_evaluation.csv
        combined_final_512.csv        the 624-response primary analysis set
        combined_final_all.csv        512 arms plus the GPT-OSS 2048 arm
        FREEZE_MANIFEST.json          evaluator hash, scope, counts

Which file to analyse
---------------------
combined_final_512.csv is the primary analysis set: it is the only
balanced two-model design available, so all model and language
comparisons use it.

combined_final_all.csv adds the GPT-OSS 2048 arm and exists for the
generation-length comparison only. It is deliberately unbalanced and
must not be used for cross-model claims.

Usage
-----
    python scripts/analysis/finalize_labels.py
"""

from pathlib import Path

import argparse
import hashlib
import importlib.util
import json
from datetime import datetime

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EVALUATOR_PATH = PROJECT_ROOT / "scripts" / "analysis" / "evaluate.py"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluated_final"

# Final scope. Each entry is (run label, model, wide dataset path).
#
# The 512 inputs come from outputs/merged/, which merge_results.py has
# verified to reproduce the original data/working_dataset_*.csv exactly
# while additionally carrying output_tokens so truncation can be
# recorded.
SCOPE = [
    (
        "512",
        "gpt_oss",
        PROJECT_ROOT / "outputs" / "merged"
        / "working_dataset_gpt_oss.csv",
    ),
    (
        "512",
        "qwen",
        PROJECT_ROOT / "outputs" / "merged"
        / "working_dataset_qwen.csv",
    ),
    (
        "2048",
        "gpt_oss",
        PROJECT_ROOT / "outputs" / "merged_2048"
        / "working_dataset_gpt_oss.csv",
    ),
]

EXCLUDED = [
    {
        "run": "2048",
        "model": "qwen",
        "reason": (
            "Provider output-tokens-per-minute limit (1000) is below "
            "the configured 2048 cap on the on_demand tier, so "
            "requests are rejected before generation. Permanently "
            "excluded by decision; not retried, not run at a reduced "
            "cap, not imputed."
        ),
    }
]

LANGUAGE_COLUMNS = {
    "en": "response_en",
    "hi": "response_hi",
    "mr": "response_mr",
}

PROMPT_COLUMNS = {
    "en": "prompt_en",
    "hi": "prompt_hi",
    "mr": "prompt_mr",
}

LABELS = ["COMPLIANCE", "NON_COMPLIANCE", "REFUSAL"]


def file_sha256(path):
    """
    SHA256 of a file, used to pin the frozen evaluator.
    """

    digest = hashlib.sha256()

    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)

    return digest.hexdigest()


def load_evaluator():
    """
    Import the evaluator without executing main().
    """

    spec = importlib.util.spec_from_file_location(
        "evaluate_frozen", EVALUATOR_PATH
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def evaluate_dataset(evaluator, run, model_name, path):
    """
    Apply the frozen evaluator to one wide dataset.
    """

    frame = pd.read_csv(path, keep_default_na=False)

    rows = []

    for _, record in frame.iterrows():

        for language, response_column in LANGUAGE_COLUMNS.items():

            response = record[response_column]

            refusal = evaluator.detect_refusal(response)

            actionable = evaluator.detect_actionable_content(
                response, language=language
            )

            defensive = evaluator.detect_defensive_context(response)

            override_key = (
                model_name,
                record["attack_id"],
                record["variation_id"],
                language,
            )

            if override_key in evaluator.MANUAL_OVERRIDES:
                label = evaluator.MANUAL_OVERRIDES[override_key]
            else:
                label = evaluator.classify_response(
                    response, language=language
                )

            token_column = f"output_tokens_{language}"

            if token_column in record.index:
                truncated = evaluator.detect_truncation(
                    response, record[token_column]
                )
                tokens = record[token_column]
            else:
                truncated = ""
                tokens = ""

            rows.append({
                "run": run,
                "attack_id": record["attack_id"],
                "variation_id": record["variation_id"],
                "attack_category": record["attack_category"],
                "language": language,
                "model": model_name,
                "prompt": record[PROMPT_COLUMNS[language]],
                "response": response,
                "safety_label": label,
                "refusal_detected": refusal,
                "actionable_content": actionable,
                "defensive_context": defensive,
                "output_tokens": tokens,
                "truncated": truncated,
            })

    return pd.DataFrame(rows)


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Freeze the evaluator and regenerate the final label set "
            "for the finalized experimental scope."
        )
    )

    parser.add_argument("--outdir", default=str(OUTPUT_DIR))

    arguments = parser.parse_args()

    output_dir = Path(arguments.outdir)

    print("=" * 70)
    print("FREEZE EVALUATOR AND FINALIZE LABELS")
    print("=" * 70)

    evaluator_hash = file_sha256(EVALUATOR_PATH)

    print(f"Evaluator : {EVALUATOR_PATH.name}")
    print(f"SHA256    : {evaluator_hash}")

    evaluator = load_evaluator()

    print(f"Overrides : {len(evaluator.MANUAL_OVERRIDES)}")
    print()

    missing = [
        str(path) for _, _, path in SCOPE if not Path(path).exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Required inputs missing. Run merge_results.py first:\n  "
            + "\n  ".join(missing)
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    frames = []

    for run, model_name, path in SCOPE:

        print(f"Evaluating {model_name} @ {run} ... ", end="")

        frame = evaluate_dataset(evaluator, run, model_name, path)

        frames.append(frame)

        target = output_dir / f"{model_name}_{run}_evaluation.csv"

        frame.to_csv(
            target, index=False, encoding="utf-8-sig"
        )

        print(f"{len(frame)} rows -> {target.name}")

    combined_all = pd.concat(frames, ignore_index=True)

    combined_512 = combined_all[
        combined_all["run"] == "512"
    ].copy()

    combined_512.to_csv(
        output_dir / "combined_final_512.csv",
        index=False,
        encoding="utf-8-sig",
    )

    combined_all.to_csv(
        output_dir / "combined_final_all.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Summaries
    # --------------------------------------------------------

    print()
    print("PRIMARY ANALYSIS SET — combined_final_512.csv")
    print(f"  rows: {len(combined_512)}")

    model_table = pd.crosstab(
        combined_512["model"], combined_512["safety_label"]
    ).reindex(columns=LABELS, fill_value=0)

    model_table["TOTAL"] = model_table.sum(axis=1)

    print(model_table.to_string())

    print()
    print("GENERATION-LENGTH SET — combined_final_all.csv")
    print(f"  rows: {len(combined_all)}")

    run_table = pd.crosstab(
        [combined_all["run"], combined_all["model"]],
        combined_all["safety_label"],
    ).reindex(columns=LABELS, fill_value=0)

    run_table["TOTAL"] = run_table.sum(axis=1)

    print(run_table.to_string())

    # --------------------------------------------------------
    # Freeze manifest
    # --------------------------------------------------------

    manifest = {
        "project": (
            "Cross-Lingual Vulnerability and Prompt Injection in "
            "Low-Resource Languages"
        ),
        "software": "Multilingual LLM Safety Evaluation Framework",
        "frozen_at": datetime.now().isoformat(timespec="seconds"),
        "evaluator": {
            "path": "scripts/analysis/evaluate.py",
            "sha256": evaluator_hash,
            "manual_overrides": len(evaluator.MANUAL_OVERRIDES),
            "taxonomy": LABELS,
        },
        "scope_included": [
            {
                "run": run,
                "model": model_name,
                "input": str(Path(path).relative_to(PROJECT_ROOT)),
                "rows": int(
                    len(frames[index])
                ),
            }
            for index, (run, model_name, path) in enumerate(SCOPE)
        ],
        "scope_excluded": EXCLUDED,
        "outputs": {
            "primary_analysis_set": "combined_final_512.csv",
            "primary_analysis_rows": int(len(combined_512)),
            "generation_length_set": "combined_final_all.csv",
            "generation_length_rows": int(len(combined_all)),
        },
        "label_distribution_primary": {
            label: int((combined_512["safety_label"] == label).sum())
            for label in LABELS
        },
        "preservation": [
            "outputs/evaluated/ (original 512 labels) is unmodified.",
            "Its difference from these frozen labels is documented by "
            "scripts/analysis/evaluator_diff_report.py.",
            "The 63 human-adjudicated labels are unmodified.",
        ],
        "analysis_rules": [
            "combined_final_512.csv is the primary analysis set. It is "
            "the only balanced two-model design and carries all model "
            "and language comparisons.",
            "combined_final_all.csv is unbalanced (no Qwen 2048 arm) "
            "and is for the generation-length comparison only. It must "
            "not be used for cross-model claims.",
            "Labels are evaluator-produced silver labels, not ground "
            "truth.",
        ],
    }

    (output_dir / "FREEZE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=4), encoding="utf-8"
    )

    print()
    print(f"Written to: {output_dir}")
    print()
    print("EVALUATOR FROZEN.")
    print(f"  hash {evaluator_hash[:16]}...")
    print("  outputs/evaluated/ was NOT modified.")
    print("=" * 70)


if __name__ == "__main__":
    main()
