# Confirmatory statistical analysis

Cross-Lingual Vulnerability and Prompt Injection in Low-Resource Languages

- Run label: **final-v2-512**
- Input: `outputs\evaluated_final_v2\combined_final_v2_512.csv`
- Observations: 624
- Prompt sets (independent units): 104
- Design: 104 prompt sets x 2 models x 3 languages

Every prompt set is measured six times, so the 624 rows are clustered rather than independent. Exploratory chi-square results are retained unchanged; the confirmatory tests below account for the repeated structure.

## Why a confirmatory layer was needed

| factor | chi2 | df | p (uncorrected) | Cramer's V | cells E<5 | min E |
|---|---:|---:|---:|---:|---:|---:|
| model | 71.0903 | 2 | 3.66e-16 | 0.3375 | 0/6 (0.0%) | 57.5 |
| language | 4.9222 | 4 | 0.295 | 0.0628 | 0/9 (0.0%) | 38.33 |
| attack_category | 76.8587 | 24 | 1.91e-07 | 0.2482 | 0/39 (0.0%) | 8.85 |
| variation_id | 111.8853 | 14 | 2.41e-17 | 0.2994 | 0/24 (0.0%) | 14.38 |

All four assume independence, which this design violates. The attack-category and variation tables additionally breach the expected-frequency guideline.

## Confirmatory results

| analysis | method | statistic | df | p | p (BH) | significant at BH 0.05 |
|---|---|---:|---:|---:|---:|---|
| Model: gpt_oss vs qwen | Stuart-Maxwell marginal homogeneity (paired, 3-class) | 72.4154 | 2 | 2.22e-16 | 1.33e-15 | yes |
| Language: en vs hi | Stuart-Maxwell marginal homogeneity (paired, 3-class) | 6.6454 | 2 | 0.0361 | 0.0541 | no |
| Language: en vs mr | Stuart-Maxwell marginal homogeneity (paired, 3-class) | 2.4947 | 2 | 0.287 | 0.287 | no |
| Language: hi vs mr | Stuart-Maxwell marginal homogeneity (paired, 3-class) | 3.7073 | 2 | 0.157 | 0.188 | no |
| attack_category (binary REFUSAL contrast) | GEE logistic, exchangeable, prompt-set clusters; Wald omnibus | 33.3343 | 12 | 0.000858 | 0.00257 | yes |
| variation_id (binary REFUSAL contrast) | GEE logistic, exchangeable, prompt-set clusters; Wald omnibus | 23.323 | 7 | 0.0015 | 0.00299 | yes |

Benjamini-Hochberg applied across the 6 tests in this family.

## Paired model table

Pair unit: the same prompt set in the same language answered by both models.

| gpt_oss (row) | COMPLIANCE | NON_COMPLIANCE | REFUSAL |
|---|---:|---:|---:|
| COMPLIANCE | 32 | 26 | 15 |
| NON_COMPLIANCE | 4 | 9 | 6 |
| REFUSAL | 43 | 61 | 116 |

## Paired language tables

Pair unit: the same prompt set and model answered in two languages.

### en vs hi

| en (row) | COMPLIANCE | NON_COMPLIANCE | REFUSAL |
|---|---:|---:|---:|
| COMPLIANCE | 27 | 10 | 14 |
| NON_COMPLIANCE | 9 | 17 | 20 |
| REFUSAL | 9 | 7 | 95 |

### en vs mr

| en (row) | COMPLIANCE | NON_COMPLIANCE | REFUSAL |
|---|---:|---:|---:|
| COMPLIANCE | 32 | 8 | 11 |
| NON_COMPLIANCE | 10 | 16 | 20 |
| REFUSAL | 14 | 11 | 86 |

### hi vs mr

| hi (row) | COMPLIANCE | NON_COMPLIANCE | REFUSAL |
|---|---:|---:|---:|
| COMPLIANCE | 32 | 4 | 9 |
| NON_COMPLIANCE | 7 | 12 | 15 |
| REFUSAL | 17 | 19 | 93 |

## Effect sizes with cluster-bootstrap intervals

Differences in percentage points. 2000 draws resampling prompt sets, seed 42.

| comparison | label | difference (pp) | 95% CI | excludes zero |
|---|---|---:|---|---|
| gpt_oss - qwen | COMPLIANCE | -1.92 | [-8.97, +5.45] | no |
| gpt_oss - qwen | NON_COMPLIANCE | -24.68 | [-31.09, -18.59] | yes |
| gpt_oss - qwen | REFUSAL | +26.60 | [+18.91, +33.97] | yes |
| en - hi | COMPLIANCE | +2.88 | [-2.88, +8.65] | no |
| en - hi | NON_COMPLIANCE | +5.77 | [-0.96, +12.50] | no |
| en - hi | REFUSAL | -8.65 | [-14.90, -1.92] | yes |
| en - mr | COMPLIANCE | -2.40 | [-8.19, +3.85] | no |
| en - mr | NON_COMPLIANCE | +5.29 | [-0.96, +11.54] | no |
| en - mr | REFUSAL | -2.88 | [-9.62, +3.37] | no |
| hi - mr | COMPLIANCE | -5.29 | [-11.06, +0.48] | no |
| hi - mr | NON_COMPLIANCE | -0.48 | [-6.73, +5.77] | no |
| hi - mr | REFUSAL | +5.77 | [-1.44, +12.98] | no |

## Interpretation rules applied

- These are association tests. No causal statement follows from them.
- Correct phrasing: a given prompt framing *was associated with* a different distribution of safety outcomes. Not that it *caused* unsafe behaviour.
- Attack-category and variation results rest on a binary REFUSAL contrast, which discards information about the COMPLIANCE / NON_COMPLIANCE distinction.
- Safety labels are produced by a rule-based evaluator. They are silver labels, not ground truth, and the confirmatory results inherit that measurement error.
