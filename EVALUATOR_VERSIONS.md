# Evaluator Versions

This project uses a rule-based safety evaluator that assigns each model
response one of three labels — **COMPLIANCE**, **NON_COMPLIANCE**, or
**REFUSAL** (with `EMPTY` reserved for empty responses).

Two evaluator versions exist. Both are kept in the repository: the V1
outputs are preserved unchanged, and **V2 is the frozen final evaluator**
used for the current labels, statistics and classifier training.

## V1 — response-only (legacy / baseline)

- **Input:** the model response only.
- **Source:** `scripts/analysis/evaluate.py`
- **SHA-256:** `76b1b83bf4ee55d339163d1a66530dc3cc8880c17456d9bffbb9d579afef25fa`
- **Status:** preserved, unchanged. Its outputs remain in
  `outputs/evaluated_final/` as the legacy baseline.

## V2 — prompt + response (FINAL, frozen)

- **Input:** the original adversarial **prompt and the model response**.
  The prompt is context; the response is the classification target. The
  evaluator judges whether the response fulfils the original request.
- **Source:** `scripts/analysis/evaluate_v2.py`
- **SHA-256:** `4d3ed9c7b1f9bd518a144e242df423e98c42e8efd49b5fe95a1ddfd47dcddfe0`
- **Status:** FINAL, frozen. Its labels are in
  `outputs/evaluated_final_v2/combined_final_v2_512.csv` (624 responses:
  REFUSAL 357, COMPLIANCE 152, NON_COMPLIANCE 115).

## Note on the response classifier

The XLM-R / MuRIL response classifier is a **separate** component. It
reads the **response text only** and never receives the prompt, so it is
not the safety evaluator described above.

See `REPRODUCIBILITY_FINAL_V2.md` for the full final pipeline and
`REPRODUCIBILITY.md` for the V1 record.
