# Reproducibility — Final pipeline (Evaluator v2)

**Project:** Cross-Lingual Vulnerability and Prompt Injection in Low-Resource Languages
**Framework:** Multilingual LLM Safety Evaluation Framework
**Generated:** 2026-09-17T20:04:14Z · repo commit `cd4bb45` (no commit made in this task)

This document records the FINAL software/experimental pipeline. It supplements
`REPRODUCIBILITY.md` (the v1 record), which is preserved unchanged.

## 1. Architecture
```
adversarial prompt -> LLM -> response
                              |
                              +-- SAFETY EVALUATOR (prompt + response)  -> C / NC / R   [research labels]
                              |
                              +-- RESPONSE CLASSIFIER (response only, XLM-R / MuRIL) -> C / NC / R  [demo model]
```
The safety evaluator uses the prompt as CONTEXT; the response is the target.
The classifier is intentionally response-only and never receives the prompt.

## 2. Dataset
- 104 prompt sets = 13 attack categories × 8 prompt variations, in en / hi / mr.
- 312 tasks per model × 2 models (GPT-OSS `openai/gpt-oss-20b`, Qwen `qwen/qwen3.8-27b`, both via Groq during generation only).
- 624 responses (512-token primary set). Unit of independence = prompt set (measured 6×).
- Prompt source: `data/working_dataset.csv`. Merged inputs (prompt+response+tokens): `outputs/merged/working_dataset_{gpt_oss,qwen}.csv`.

## 3. Evaluators
| | v1 (legacy / baseline) | v2 (FINAL) |
|---|---|---|
| Input | response only | prompt + response |
| Source | `scripts/analysis/evaluate.py` | `scripts/analysis/evaluate_v2.py` |
| v2 SHA-256 | — | `4d3ed9c7b1f9bd518a144e242df423e98c42e8efd49b5fe95a1ddfd47dcddfe0` |
| Status | preserved, unchanged | FINAL, frozen |
| Labels | `outputs/evaluated_final/combined_final_512.csv` | `outputs/evaluated_final_v2/combined_final_v2_512.csv` |

Taxonomy: COMPLIANCE / NON_COMPLIANCE / REFUSAL (EMPTY reserved).
v2 thresholds: on_topic overlap ≥ 0.10 AND ≥ 2 shared content terms; substantive ≥ 220 chars OR ≥ 40 words.
Threshold stability: across 36 nearby combinations, max 3.2% label change; REFUSAL invariant. Conclusions are not threshold-sensitive.
v2 vs the 63-case targeted human audit (NOT representative): accuracy 0.635, Cohen's κ +0.311, COMPLIANCE recall 0.667 (v1: 0.286 / −0.320 / 0.212). This is an audit subset, not a population accuracy estimate.

## 4. Final label counts (v2, 624 responses)
REFUSAL 357 · COMPLIANCE 152 · NON_COMPLIANCE 115 (sum = 624; integrity verified: 624 rows, 0 duplicate keys, 13 attack_ids, 8 variations, 312/312 per model, 208×3 per language).

## 5. Labels: silver vs gold
- SILVER = automatic (evaluator v2). 561 rows in the classifier dataset.
- GOLD = 63 human-adjudicated targeted-audit cases (`outputs/analysis/audit/human_review_queue.csv`), unchanged, kept separate. Provenance files (`validation_sample.csv`, `human_review_queue.csv`) are not merged.

## 6. Generation-length / truncation sensitivity (NOT "tokenizer inequity")
- GPT-OSS: ceiling hits 24.4% (512) → 2.6% (2048); mean length 471 → 1127 chars; under v2 only 22/312 (7.1%) labels change, REFUSAL identical (220). Safety labels are largely insensitive to the budget for GPT-OSS.
- Qwen @ 2048: EXCLUDED — provider output-token-per-minute limit (OTPM 1000 < request). Not run, not simulated. Qwen generation-length sensitivity is unmeasured and is a stated limitation.
- Output: `outputs/evaluated_v2/generation_length/`.

## 7. Statistics (on v2 final labels)
- Exploratory (chi-square + Cramér's V): `outputs/analysis/statistics_final_v2/`.
- Confirmatory (repeated-measures Stuart–Maxwell / GEE, Benjamini–Hochberg): `outputs/analysis/statistics_confirmatory_final_v2/`.
- Under v2: Model effect significant (χ²=72.4, BH p=1.3e-15); attack-category and variation REFUSAL-contrasts significant; pairwise LANGUAGE comparisons do NOT survive BH correction (en–hi 0.054, en–mr 0.29, hi–mr 0.19). Associations only; no causal claims.

## 8. Figures
`outputs/analysis/figures_final_v2/` (8 figures). v1 figures preserved in `outputs/analysis/figures_final/`.

## 9. Classifier (response-only)
- Dataset: `outputs/analysis/classifier_final_v2/classifier_dataset.csv` (silver v2 + 63 gold).
- Split: grouped by prompt set, exact-text dedup, gold isolated. `prepare_final_split.py --input … --outdir outputs/analysis/classifier_final_v2`.
- Train COMPLIANCE share rises to 22.4% (v1: 3.5%); train→gold shift reduced.
- Models: XLM-R (`xlm-roberta-base`), MuRIL (`google/muril-base-cased`); label map C=0, NC=1, R=2; seed 42; 3 epochs; lr 2e-5; batch 8; max_len 512; fp16 when CUDA. Weighted + unweighted variants.
- Input is response text only. Gold labels never used for training.
- **Status relative to the paper:** `classifier_final_v2` is a later, supplementary/exploratory
  pipeline (prompt-set grouping, weighted/unweighted variants) and was not used for the
  paper's primary classifier result. The paper-reported response-only classifier results
  remain those from `outputs/analysis/classifier/` (XLM-R 28.57%; MuRIL 23.81%; majority
  baseline 52.38%; COMPLIANCE F1 0.00 for both). Trained checkpoints for both pipelines are
  excluded from the public repository (see Section 12).

## 10. Environment
Python 3.13.15, Windows 11, RTX 4050, CUDA 12.8. Package versions in `requirements.txt`.

## 11. Reproduce (commands)
```bash
# evaluator v2 shadow + freeze
python scripts/analysis/evaluate_v2.py
python scripts/analysis/evaluate_v2_sensitivity.py
# final v2 labels are written to outputs/evaluated_final_v2/
# statistics
python scripts/analysis/statistical_analysis.py  --input outputs/evaluated_final_v2/combined_final_v2_512.csv --outdir outputs/analysis/statistics_final_v2
python scripts/analysis/confirmatory_analysis.py --input outputs/evaluated_final_v2/combined_final_v2_512.csv --outdir outputs/analysis/statistics_confirmatory_final_v2 --label final-v2-512
# figures
python scripts/analysis/visualize_results.py --input outputs/evaluated_final_v2/combined_final_v2_512.csv --outdir outputs/analysis/figures_final_v2
# classifier
python scripts/ml/prepare_final_split.py --input outputs/analysis/classifier_final_v2/classifier_dataset.csv --outdir outputs/analysis/classifier_final_v2
python scripts/ml/train_final_classifier.py --model xlmr  --split-dir outputs/analysis/classifier_final_v2
python scripts/ml/train_final_classifier.py --model muril --split-dir outputs/analysis/classifier_final_v2
# demo
uvicorn app.backend.main:app --port 8077     # local inference, no external key
cd app/frontend && npm install && npm run dev
```

## 12. Preservation guarantees
Unchanged: v1 evaluator, `outputs/evaluated/`, `outputs/evaluated_final/` (fingerprint 7dc8759fb8671e03), v1 figures/statistics, the 63 human gold labels, the 624 raw responses. All v2 work is written to new, clearly versioned locations.
