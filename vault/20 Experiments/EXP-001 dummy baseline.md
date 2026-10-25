# EXP-001 — DummyClassifier sanity baseline

- **exp_id:** EXP-001
- **date:** 2026-10-25
- **phase:** P3
- **status:** done
- **related ADRs:** [[ADR-001 binary target framing]], [[ADR-005 whitelist and matrix v1]]
- **log row:** `results/experiments_log.csv` (EXP-001, 2026-10-25T18:54:31)

## Question
Does the harness produce exactly ROC-AUC = 0.5 for a constant-prior predictor on the canonical folds — i.e. are the scorer, the fold builder and the target wiring correct before any real model runs?

## Setup
- Model / method: `DummyClassifier(strategy='prior')` — predicts the training-fold class prior (constant score)
- Feature set: all-76 (`clean_v1.parquet`, 437,532 × 76 after valid-target mask; matrix ignored by the dummy but full pipeline exercised)
- Imbalance strategy: none
- Hyperparameters: `strategy=prior`
- Validation: `StratifiedKFold(10, shuffle=True, random_state=42)` via `src/config.py:make_folds()` — **folds_sha256 = `3b3a17457039`**, asserted by the harness
- Preprocessing inside the CV pipeline (fit on training folds only): most_frequent impute + one-hot for NOM, median impute for ORD/NUM/ENG — see [[ADR-005 whitelist and matrix v1]]

## Results
| fold | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | mean |
|------|---|---|---|---|---|---|---|---|---|----|------|
| ROC-AUC | 0.500000 | 0.500000 | 0.500000 | 0.500000 | 0.500000 | 0.500000 | 0.500000 | 0.500000 | 0.500000 | 0.500000 | **0.500000** |

- Fold std **0.000000**; range 0.500000–0.500000; runtime_s = 8.3 (log row notes).
- Exactly 0.5 on every fold, as theory predicts for a constant-score predictor (ROC-AUC = P(score_pos > score_neg) + ½·P(tie) = ½ under ties-only).

Figures: none (numeric sanity check; no plot warranted).

## Verdict
**Pass — sanity floor established.** The scorer, folds and target pipeline are wired correctly; every future AUC is comparable against a *measured* floor of 0.500000 rather than an assumed one. This is the rubric's "baseline ROC-AUC from DummyClassifier" item (evidence tag: EXP-001). **Next:** EXP-002 LogisticRegression (linear family) and EXP-003 DecisionTree (non-linear family) on the same folds.
