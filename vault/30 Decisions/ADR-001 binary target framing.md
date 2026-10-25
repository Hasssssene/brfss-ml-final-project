# ADR-001 — Binary target framing: Fair/Poor vs rest

- **date:** 2026-10-25
- **status:** accepted
- **tags:** #adr #target
- **scope:** target definition, dataset row filtering

## Context
`GENHLTH` is natively 5-class (1 Excellent 17.7%, 2 Very good 34.0%, 3 Good 31.4%, 4 Fair 12.5%, 5 Poor 4.1%; codes 7 Don't know, 9 Refused, blank ≈ 1,161 rows). The course guidelines list "frame a multi-class problem as a more tractable binary classification task" as a learning objective and grade Basic/Intermediate levels on ROC-AUC with a DummyClassifier baseline — a binary pipeline. Multiclass appears only as an optional Advanced extra.

## Decision
**Main task: y = 1 iff GENHLTH ∈ {4,5} (Fair or Poor), y = 0 iff GENHLTH ∈ {1,2,3}.** Rows with GENHLTH ∈ {7, 9, blank} are dropped, not imputed. Implemented in `src/config.py:make_target()`.

## Alternatives considered
- **Poor (5) vs rest** — 4.1% positives: harsher imbalance, noisier fold scores, thin error-analysis pool. Rejected.
- **Keep 5-class as main task** — contradicts the guidelines' explicit binary objective and makes the required ROC-AUC/Dummy baseline awkward. Rejected (multiclass dropped entirely per ADR-002).
- **Good-or-better vs rest** — identical split to the chosen framing with flipped labels; no distinct experiment.

## Consequences
- ~72,741 positives (16.6%) — realistic imbalance, stable 10-fold AUC, rich error-analysis pool.
- Framing matches CDC's own `_RFHLTH` definition → citable justification in the report.
- **`_RFHLTH` (and anything derived from GENHLTH) becomes target leakage and is blacklisted** — see [[Leakage watchlist]].
- Imbalance-handling criterion (SMOTE / class_weight) is genuinely exercised at 16.6%.
