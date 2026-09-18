# Multilingual LLM Safety Benchmark

![Python](https://img.shields.io/badge/Python-3.13-3776AB)
![React](https://img.shields.io/badge/React-Vite%20%2B%20TypeScript-61DAFB)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

**Project:** Multilingual LLM Safety Benchmark
**Software:** Multilingual LLM Safety Evaluation Framework
**Research topic:** *Cross-Lingual Vulnerability and Prompt Injection in Low-Resource Languages*

A reproducible software framework for evaluating how large language models respond to
adversarial and potentially harmful prompts across **English, Hindi, and Marathi**. It
measures whether a model's safety behaviour holds up across languages and across a
controlled set of adversarial prompt variations, and whether multilingual transformers can
classify the safety of model responses. The benchmark is an instrument for **measurement and
evaluation** — not a tool for producing harmful content, and not a universal proof of
multilingual safety.

## Why this project?

Large language models are increasingly deployed in multilingual settings, yet safety
behaviour cannot be assumed to transfer uniformly across languages or adversarial phrasings.
Most published safety evaluation is done in English. If a model refuses a harmful request in
English but complies with the same request in Hindi or Marathi, its safety training has not
transferred — and speakers of those languages carry the risk.

This project provides a controlled benchmark to examine whether model responses differ across
**English, Hindi, and Marathi**, and across several adversarial prompt framings. The purpose
is evaluation and measurement, not the generation of usable harmful content.

## Benchmark at a glance

| Dimension | Value |
|---|---|
| Languages | English, Hindi, Marathi |
| Prompt sets | 104 multilingual prompt sets |
| Attack categories | 13 |
| Prompt variations | 8 |
| Benchmark tasks per model | 312 (104 × 3 languages) |
| Models evaluated | GPT-OSS and Qwen |
| Total responses | 624 |
| Responses per model | 312 |
| Responses per language | 208 |

The eight prompt variations change **framing, not content**:

1. Baseline
2. Urgency
3. Trusted Relationship
4. Roleplay
5. Hypothetical
6. Obfuscation
7. Multilingual Code-Switch
8. Emotional Appeal

The unit of statistical independence is the **prompt set**: each set is measured six times
(3 languages × 2 models), so responses are clustered rather than independent.

## Evaluation pipeline

```
Adversarial prompt
        ↓
LLM response
        ↓
V2 safety evaluation   (prompt + response)
        ↓
COMPLIANCE / NON_COMPLIANCE / REFUSAL   (silver labels)
        ↓
Statistical analysis
        ↓
Response-only multilingual classification
```

Two components are deliberately kept distinct:

- **Safety evaluator (V2)** reads **both the original adversarial prompt and the model
  response** to judge whether the harmful request was fulfilled. It produces the automatic
  *silver* labels used for analysis.
- **Multilingual classifier** reads the **response text only**. It predicts
  COMPLIANCE / NON_COMPLIANCE / REFUSAL and does not receive the prompt. It is a response
  classifier, not a harmful-prompt detector.

**V1** (response-only) is retained as the legacy/previous evaluator; **V2** (prompt +
response) is the frozen final evaluator used for the current results.

## Final V2 findings

Final V2 label distribution (624 responses):

| Label | Count |
|---|---|
| REFUSAL | 357 |
| COMPLIANCE | 152 |
| NON_COMPLIANCE | 115 |

- **Model.** Model and safety-label distribution were significantly associated — exploratory
  χ² ≈ 71.1 (Cramér's V ≈ 0.34), and the repeated-measures confirmatory test survived
  Benjamini–Hochberg (BH) correction (adjusted *p* ≈ 1.3 × 10⁻¹⁵).
- **Attack category & prompt variation.** Both pre-declared refusal contrasts remained
  significant after BH correction (adjusted *p* ≈ 0.003).
- **Language.** No pairwise language comparison survived BH correction — English–Hindi
  ≈ 0.054, English–Marathi ≈ 0.29, Hindi–Marathi ≈ 0.19 — and the exploratory language
  association was small (Cramér's V ≈ 0.063). Differences are descriptive within this
  benchmark, not statistically confirmed at the corrected threshold.

These are **associations within this benchmark** — not causal claims, and not evidence that
any language is universally safer or less safe.

### Generation-length sensitivity

A **generation-length / truncation sensitivity analysis** compared GPT-OSS at 512- vs
2048-token generation budgets across 312 paired responses:

- ceiling-hit rate: **24.4% → 2.6%**
- mean response length: **471 → 1127 characters**
- labels changed: **22 / 312 (7.1%)**
- REFUSAL count unchanged (**220** at both budgets)

Safety labels were largely insensitive to the generation budget for GPT-OSS. The Qwen
2048-token condition was **not completed** because of the provider's output-token-per-minute
limit; it was not run and not simulated, so Qwen generation-length sensitivity is unmeasured.

## Models / classifiers

Two multilingual transformers classify a response into the three safety classes:

- **XLM-R** (`xlm-roberta-base`)
- **MuRIL** (`google/muril-base-cased`)

Classification is **response-only**. The train/test split is grouped by prompt set to prevent
leakage, with the targeted-audit adjudication set held out for evaluation. Because the
provenance of those adjudicated labels is not established as independent human annotation,
this repository does **not** present them as validated ground-truth accuracy. The classifier
is a research artifact — not a production safety control. Detailed metrics are recorded in the
frozen outputs and the reproducibility documents.

## Repository structure

```
app/
  backend/      FastAPI backend (read-only demonstration API over frozen outputs)
  frontend/     React + Vite + TypeScript interface
scripts/
  analysis/     benchmark evaluation, statistics, generation-length, tables
  ml/           classifier split preparation and training
  providers/    LLM provider factory (used during generation)
data/            benchmark prompt sets
config/          run configuration
outputs/         frozen benchmark and analysis artifacts (weights excluded)
README.md
REPRODUCIBILITY.md
REPRODUCIBILITY_FINAL_V2.md
EVALUATOR_VERSIONS.md
requirements.txt
LICENSE
```

Large trained model weights are **excluded** from the repository; the small benchmark and
analysis artifacts under `outputs/` are versioned.

## Running locally

```bash
# 1. Clone
git clone https://github.com/ShriyaP1966/multilingual-llm-safety-benchmark.git
cd multilingual-llm-safety-benchmark

# 2. Python environment (backend + analysis)
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# 3. Backend API (FastAPI served by Uvicorn)
uvicorn app.backend.main:app --port 8077

# 4. Frontend (React + Vite + TypeScript), in a second terminal
cd app/frontend
npm install
npm run dev
```

The backend serves its read-only endpoints (benchmark results, methodology) without the
trained weights. Live response classification additionally requires the XLM-R / MuRIL
weights, which — like any secrets — are **excluded from GitHub** and must be produced or
obtained separately (see the reproducibility documents).

## Reproducibility

- **`REPRODUCIBILITY_FINAL_V2.md`** — the frozen final (V2) pipeline.
- **`REPRODUCIBILITY.md`** — the V1 (legacy) record, preserved unchanged.
- **`EVALUATOR_VERSIONS.md`** — the V1 (response-only) vs V2 (prompt + response) distinction.

V2 is the frozen final evaluator; V1 is retained for historical/legacy comparison.

## Methodological notes & limitations

- The benchmark contains a controlled set of 104 multilingual prompt sets and therefore does
  not represent all possible real-world prompts.
- Statistical findings are **associations within the benchmark** and should not be
  interpreted causally.
- Language comparisons should not be generalized beyond the tested benchmark.
- Generation-length sensitivity affects a subset of results and was studied for GPT-OSS.
- Qwen's 2048-token condition was not completed because of provider output-token limits.
- The targeted-audit / adjudication provenance is **not** established as independent human
  validation and should not be described as such.
- Large model weights are excluded from the repository.
- This is an evaluation/research framework, not a guarantee of LLM safety.

## Research use & safety

This benchmark is intended for safety evaluation, robustness research, multilingual LLM
analysis, reproducibility, and academic/engineering experimentation. Adversarial prompts are
included **as evaluation inputs** and must not be interpreted as instructions for real-world
harmful activity.

## Citation

> Shriya Patil. *Multilingual LLM Safety Benchmark: Cross-Lingual Vulnerability and Prompt
> Injection in Low-Resource Languages.*

## License

Licensed under the MIT License. See [`LICENSE`](LICENSE) for details. Third-party
dependencies remain under their own respective licenses.
