# EXP-004 — RandomForest (bagging entry of the P4 zoo)

- **exp_id:** EXP-004
- **date:** 2026-11-02
- **phase:** P4
- **status:** done
- **related ADRs:** [[ADR-005 whitelist and matrix v1]]
- **log row:** `results/experiments_log.csv` (EXP-004, 2026-11-02T15:57:03)

## Question
How far does bagging get on matrix v1 with one sensible config — can RandomForest beat the linear baseline (0.892588, [[EXP-002 logistic regression]]) and hold up against the boosters?

## Setup
- Model / method: `RandomForestClassifier(n_estimators=200, min_samples_leaf=50, max_features='sqrt', random_state=42, n_jobs=1)` — leaf=50 borrowed from the [[EXP-003 decision tree]] lesson (leaf regularization is what makes trees usable)
- Feature set: all-76 (`clean_v1.parquet`, 437,532 × 76 valid rows)
- Imbalance strategy: none
- Preprocessing: P3 shared **impute** preprocessor (RF cannot split NaN — no native missing routing)
- Validation: `StratifiedKFold(10, shuffle=True, random_state=42)` via `src/config.py:make_folds()` — **folds_sha256 = `3b3a17457039`** (identical to EXP-001..003)
- P4 protocol: ONE config per family, tuning deferred to P6

## Results
| fold | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | mean |
|------|---|---|---|---|---|---|---|---|---|----|------|
| ROC-AUC | 0.893310 | 0.893302 | 0.894739 | 0.891585 | 0.894437 | 0.891015 | 0.891906 | 0.892166 | 0.895740 | 0.894946 | **0.893315** |

- fold std **0.001530**; runtime **351.5 s** — the slowest run of all of P4 (2.5× XGBoost, 8.5× LightGBM)

Figures: none for this EXP (comparison figures land with the report stage).

## Verdict
**Reject as final family (rank 4/4 in the zoo).** 0.893315 beats LogReg by only **+0.00073** — under half a fold std, i.e. a statistical tie with the linear baseline — while trailing the boosters by **−0.0100** (vs XGBoost 0.903357, [[EXP-007 xgboost]]). Lesson recorded: on wide one-hot survey data, *ensembles per se* were not the win — **boosting** was; decorrelated `sqrt`-feature trees are individually too weak here, and the accuracy/cost trade is the worst in the zoo. Tuning RF in P6 is not planned unless the picture changes.
