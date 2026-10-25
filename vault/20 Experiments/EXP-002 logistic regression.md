# EXP-002 — LogisticRegression baseline (manual C grid)

- **exp_id:** EXP-002a…d (4 grid points, one log row each)
- **date:** 2026-10-25
- **phase:** P3
- **status:** done
- **related ADRs:** [[ADR-005 whitelist and matrix v1]]
- **log row:** `results/experiments_log.csv` (EXP-002a–d, 2026-10-25T18:54:45…18:55:23)

## Question
How far does a regularized linear model get on the 76-feature matrix v1 under proper 10-fold CV, and does the regularization strength C matter?

## Setup
- Model / method: `LogisticRegression` + `StandardScaler` (pipeline; scaler fit on training folds only)
- Feature set: all-76 (`clean_v1.parquet`, 437,532 × 76 valid rows; ~267 columns after one-hot, float32)
- Imbalance strategy: none
- Hyperparameters: manual grid **C ∈ {0.01, 0.1, 1, 10}**, `max_iter=2000`, `solver=lbfgs`
- Validation: `StratifiedKFold(10, shuffle=True, random_state=42)` via `src/config.py:make_folds()` — **folds_sha256 = `3b3a17457039`**, asserted in all 4 rows
- Preprocessing inside the CV pipeline: most_frequent impute + one-hot for NOM, median impute for ORD/NUM/ENG (per [[ADR-005 whitelist and matrix v1]])

## Results

**Grid (mean CV ROC-AUC ± fold std):**

| exp_id | C | mean AUC | fold std | runtime_s |
|---|---|---|---|---|
| **EXP-002a** | **0.01** | **0.892588** | 0.001606 | 13.8 |
| EXP-002b | 0.1 | 0.892580 | 0.001607 | 12.7 |
| EXP-002c | 1 | 0.892580 | 0.001607 | 12.8 |
| EXP-002d | 10 | 0.892580 | 0.001607 | 12.3 |

Grid total ≈ 52 s of CV wall time (log runtimes). Fold-1 in-process refit of the winner: lbfgs **n_iter = 18 / 2000**, zero ConvergenceWarnings → the flat grid is *not* an under-convergence artifact.

**Winner EXP-002a fold scores:**
| fold | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | mean |
|------|---|---|---|---|---|---|---|---|---|----|------|
| ROC-AUC | 0.892170 | 0.893388 | 0.893450 | 0.892402 | 0.893962 | 0.889719 | 0.891078 | 0.890693 | 0.895150 | 0.893870 | **0.892588** |

- Fold range 0.889719–0.895150 (spread 0.0054); no anomalous fold.

Figures: none for this EXP (tables above; model-comparison figures come in P4+).

## Verdict
**Winner: C = 0.01 → mean CV ROC-AUC = 0.892588** (EXP-002a), the best P3 baseline. The C-grid is **essentially flat**: 0.1/1/10 all land at 0.892580, only 8e-6 below the winner — far below fold noise (std ≈ 0.0016), so regularization strength is immaterial in this range for one-hot BRFSS data; C=0.01 is kept as the family representative (mild L2 wins by a whisker and shrinks the 267-column design). LogReg beats the best tree ([[EXP-003 decision tree]], 0.880206) by **+0.0124** AUC. **Next:** P4 model zoo (ensembles should exploit the multi-block signal EDA found — see [[EDA v1 summary]]); revisit C only if feature selection changes the design materially.
