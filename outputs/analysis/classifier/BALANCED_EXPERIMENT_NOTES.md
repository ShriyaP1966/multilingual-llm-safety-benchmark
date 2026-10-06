# Balanced Classifier Experiments (Non-Frozen)

**Status: This is an additional non-frozen classifier experiment and
does not replace the original V2 research results.**

It does not modify, overwrite, or invalidate any frozen output under
`outputs/analysis/classifier/xlmr/`, `outputs/analysis/classifier/muril/`,
`outputs/analysis/classifier/xlmr_gold/`,
`outputs/analysis/classifier/muril_gold/`, or any existing comparison
CSV. All original frozen classifier scripts
(`train_xlmr.py`, `train_muril.py`, `evaluate_xlmr_gold.py`,
`evaluate_muril_gold.py`) were read for reference only and were not
edited.

## 1. Motivation

The original response-only classifiers (XLM-R and MuRIL) performed
poorly on the 63-case audit/adjudication subset, in particular on the
COMPLIANCE class, which they never predicted at all.

The training data
(`outputs/analysis/classifier/train_silver.csv`, 490 rows) is severely
imbalanced:

| Class | Train count | Train prior |
|---|---|---|
| COMPLIANCE | 17 | 3.47% |
| NON_COMPLIANCE | 141 | 28.78% |
| REFUSAL | 332 | 67.76% |

The 63-case audit/adjudication subset
(`outputs/analysis/classifier/gold_test.csv`) has a very different
distribution:

| Class | Gold count | Gold prior |
|---|---|---|
| COMPLIANCE | 33 | 52.38% |
| NON_COMPLIANCE | 28 | 44.44% |
| REFUSAL | 2 | 3.17% |

This creates a substantial train/test class-distribution shift,
already documented as a limitation of the response-only classifier
elsewhere in this project. The purpose of this experiment is to test
whether inference-time prior calibration and/or class-weighted
training loss can mitigate that shift and improve COMPLIANCE
recall/F1, without touching the frozen baseline in any way.

The audit/adjudication labels used as the evaluation target here are
**not** described as human-validated ground truth in this document;
they are simply the same 63-row evaluation set the frozen baseline was
scored against.

## 2. Frozen baseline (unchanged, for reference only)

From `outputs/analysis/classifier/xlmr_gold/xlmr_gold_metrics.txt` and
`outputs/analysis/classifier/muril_gold/muril_gold_metrics.txt`:

| Model | Accuracy | COMPLIANCE Precision | COMPLIANCE Recall | COMPLIANCE F1 |
|---|---|---|---|---|
| XLM-R | 28.57% | 0.00 | 0.00 | 0.00 |
| MuRIL | 23.81% | 0.00 | 0.00 | 0.00 |
| Majority-class baseline (always predict COMPLIANCE) | 52.38% | 0.5238 | 1.00 | 0.6875 |

Both frozen models predict **zero** COMPLIANCE responses on the 63
GOLD cases; every actual COMPLIANCE case is misclassified as
NON_COMPLIANCE.

## 3. Calibration experiment

**Script:** `scripts/ml/calibrate_and_evaluate_gold.py`

**Method.** No retraining. The original, frozen XLM-R and MuRIL
checkpoints are loaded unmodified. Training-set class priors
`pi_train(y)` are computed only from `train_silver.csv`. Each model's
raw logits on the GOLD set are adjusted before argmax using
training-prior logit adjustment (Saerens, Latinne & Decaestecker,
2002; Menon et al., ICLR 2021):

```
adjusted_logit(y | x) = raw_logit(y | x) - log( pi_train(y) )
```

Subtracting `log(pi_train(y))` removes the bias the model learned
toward the frequent training classes; it is algebraically equivalent
to assuming a **uniform** prior at inference time. The GOLD label
distribution is never used to fit or choose this formula — only
`train_silver.csv` priors are used, and `gold_test.csv` is touched
only afterward, purely to measure the result.

Training priors used: COMPLIANCE = 0.034694, NON_COMPLIANCE =
0.287755, REFUSAL = 0.677551.

**Results** (`outputs/analysis/classifier/xlmr_gold_calibrated/metrics.txt`,
`outputs/analysis/classifier/muril_gold_calibrated/metrics.txt`):

| Model | Accuracy | COMPLIANCE Precision | COMPLIANCE Recall | COMPLIANCE F1 |
|---|---|---|---|---|
| XLM-R (calibrated) | 38.10% | 0.00 | 0.00 | 0.00 |
| MuRIL (calibrated) | 52.38% | 0.5238 | 1.00 | 0.6875 |

XLM-R: accuracy improved (28.57% -> 38.10%) by shifting some
NON_COMPLIANCE/REFUSAL confusion, but COMPLIANCE recall stayed exactly
0 — the logit adjustment was not large enough to flip any COMPLIANCE
case away from NON_COMPLIANCE for this model.

MuRIL: calibration over-corrected and collapsed the model into
predicting **every** response as COMPLIANCE. Its accuracy (52.38%)
and COMPLIANCE F1 (0.6875) are numerically identical to the
majority-class baseline because the calibrated model *is* now a
majority-class predictor — it is no longer discriminating between
classes at all (NON_COMPLIANCE and REFUSAL both dropped to 0 recall).
This is not evidence of a genuinely better classifier; it is a
degenerate collapse produced by an inference-time correction that was
too strong for MuRIL's logit scale.

## 4. Class-weighted XLM-R

**Scripts:** `scripts/ml/train_xlmr_balanced.py` (training, checkpoint
saved to `outputs/analysis/classifier/xlmr_balanced/`, original
`outputs/analysis/classifier/xlmr/` untouched) and
`scripts/ml/evaluate_xlmr_gold_balanced.py` (evaluation, results in
`outputs/analysis/classifier/xlmr_gold_balanced/`).

All settings identical to the original `train_xlmr.py` (seed 42, 3
epochs, lr 2e-5, batch size 8, max length 512, same tokenizer/base
model xlm-roberta-base, same train/dev split, same preprocessing).
The only change is a class-weighted `CrossEntropyLoss`, implemented by
subclassing `Trainer` and overriding `compute_loss`.

Class weights (inverse-frequency, `sklearn.utils.class_weight.
compute_class_weight('balanced', ...)`, computed only from
`train_silver.csv`):

| Class | Weight |
|---|---|
| COMPLIANCE | 9.607843 |
| NON_COMPLIANCE | 1.158392 |
| REFUSAL | 0.491968 |

Development-set result (SILVER dev split, for reference only):
accuracy 63.38%, macro F1 0.4328 (vs. the original frozen XLM-R dev
macro F1 of 0.465325 — see
`outputs/analysis/classifier/xlmr/dev_metrics.txt`).

**GOLD result**
(`outputs/analysis/classifier/xlmr_gold_balanced/metrics.txt`):

| Metric | Value |
|---|---|
| Accuracy | 34.92% |
| COMPLIANCE Precision | 0.00 |
| COMPLIANCE Recall | 0.00 |
| COMPLIANCE F1 | 0.00 |
| NON_COMPLIANCE F1 | 0.5176 |
| REFUSAL F1 | 0.00 |

Class-weighted training raised accuracy over the frozen baseline
(28.57% -> 34.92%) but did **not** produce any COMPLIANCE predictions;
every actual COMPLIANCE case is still misclassified as
NON_COMPLIANCE. With only 17 COMPLIANCE training examples,
up-weighting the loss on those examples ~9.6x was not enough to teach
the model a usable COMPLIANCE decision boundary.

## 5. Class-weighted MuRIL

**Scripts:** `scripts/ml/train_muril_balanced.py` (training, checkpoint
saved to `outputs/analysis/classifier/muril_balanced/`, original
`outputs/analysis/classifier/muril/` untouched) and
`scripts/ml/evaluate_muril_gold_balanced.py` (evaluation, results in
`outputs/analysis/classifier/muril_gold_balanced/`).

Same class weights as above (identical formula, same
`train_silver.csv` source): COMPLIANCE 9.607843, NON_COMPLIANCE
1.158392, REFUSAL 0.491968. All other settings identical to the
original `train_muril.py` (seed 42, 3 epochs, lr 2e-5, batch size 8,
max length 512, google/muril-base-cased, same split/preprocessing).

Development-set result: accuracy 64.79%, macro F1 0.4439 (vs. the
original frozen MuRIL dev macro F1 of 0.425721 — see
`outputs/analysis/classifier/muril/dev_metrics.txt`).

**GOLD result**
(`outputs/analysis/classifier/muril_gold_balanced/metrics.txt`):

| Metric | Value |
|---|---|
| Accuracy | 36.51% |
| COMPLIANCE Precision | 0.00 |
| COMPLIANCE Recall | 0.00 |
| COMPLIANCE F1 | 0.00 |
| NON_COMPLIANCE F1 | 0.5349 |
| REFUSAL F1 | 0.00 |

Same pattern as class-weighted XLM-R: accuracy improved over the
frozen baseline (23.81% -> 36.51%), but COMPLIANCE recall/F1 remained
exactly 0.

## 6. Comparison

Full machine-readable table:
`outputs/analysis/classifier/comparison/balanced_vs_baseline_comparison.csv`
(built by `scripts/ml/compare_balanced_vs_baseline.py`, which only
reads existing prediction files and does not retrain or touch
`gold_test.csv`).

| Model | Experiment | Accuracy | COMPLIANCE P | COMPLIANCE R | COMPLIANCE F1 |
|---|---|---|---|---|---|
| XLM-R | Original Frozen Baseline | 28.57% | 0.00 | 0.00 | 0.00 |
| MuRIL | Original Frozen Baseline | 23.81% | 0.00 | 0.00 | 0.00 |
| XLM-R | Inference-Time Prior Calibration | 38.10% | 0.00 | 0.00 | 0.00 |
| MuRIL | Inference-Time Prior Calibration | 52.38% | 0.5238 | 1.00 | 0.6875 |
| XLM-R | Class-Weighted Retraining | 34.92% | 0.00 | 0.00 | 0.00 |
| MuRIL | Class-Weighted Retraining | 36.51% | 0.00 | 0.00 | 0.00 |
| N/A | Reference: Majority-Class Baseline | 52.38% | 0.5238 | 1.00 | 0.6875 |

## 7. Interpretation

- **COMPLIANCE recall/F1:** In 5 of 6 new experiment cells,
  COMPLIANCE recall/F1 remained exactly 0.00, identical to the frozen
  baseline. The single exception (MuRIL, prior calibration) reached
  the same numbers as the majority-class baseline, but only because
  calibration pushed that model into predicting COMPLIANCE for
  *every* input — a degenerate, non-discriminative collapse, not a
  genuine classifier improvement. It should not be read as evidence
  that calibration "solved" the COMPLIANCE problem.
- **Overall accuracy:** Accuracy improved over the frozen baseline in
  every new experiment cell except it never exceeded the majority
  baseline (52.38%) except in the one degenerate MuRIL-calibration
  case that is equivalent to the majority baseline itself. No
  experiment here produced a classifier that beats the majority
  baseline while still discriminating between classes.
- **Was the problem partially related to class imbalance?** Partially,
  but imbalance correction alone was not sufficient. Class weighting
  and prior calibration both measurably shifted the
  NON_COMPLIANCE/REFUSAL decision boundary (macro metrics moved), but
  neither method gave XLM-R or MuRIL enough signal to recognize
  COMPLIANCE responses as such under normal (non-collapsed)
  operation. With only 17 COMPLIANCE examples in training, the
  models likely never learned distinguishing features of COMPLIANCE
  responses at all — a boundary-correction technique (weighting or
  calibration) cannot recover discriminative signal that was never
  learned in the first place. This points to a data problem (too few
  COMPLIANCE training examples for this response-only formulation)
  rather than a purely fixable prior-mismatch problem.

## 8. Reproducibility

- All original frozen outputs (`outputs/analysis/classifier/xlmr/`,
  `outputs/analysis/classifier/muril/`, `outputs/analysis/classifier/
  xlmr_gold/`, `outputs/analysis/classifier/muril_gold/`, and existing
  comparison CSVs) remain untouched by this experiment.
- `gold_test.csv` was used only for final held-out evaluation of the
  calibrated logits and the balanced checkpoints — never for training,
  class-weight computation, hyperparameter/threshold tuning,
  calibration fitting, or model selection.
- Class weights and training priors were computed only from
  `outputs/analysis/classifier/train_silver.csv`.
- New checkpoints (`xlmr_balanced/`, `muril_balanced/`) and new
  evaluation outputs (`xlmr_gold_calibrated/`, `muril_gold_calibrated/`,
  `xlmr_gold_balanced/`, `muril_gold_balanced/`) live in directories
  separate from the frozen originals.

## 9. Status

**This is an additional non-frozen classifier experiment and does not
replace the original V2 research results.**
