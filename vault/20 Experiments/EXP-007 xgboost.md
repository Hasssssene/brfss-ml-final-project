# EXP-007 — XGBoost (non-sklearn booster #2, P4 zoo winner)

- **exp_id:** EXP-007
- **date:** 2026-11-02
- **phase:** P4
- **status:** done
- **related ADRs:** [[ADR-005 whitelist and matrix v1]]
- **log row:** `results/experiments_log.csv` (EXP-007, 2026-11-02T16:03:51)

## Question
Does XGBoost top the model zoo on identical folds, and is it distinguishable from LightGBM/HistGB?

## Setup
- Model / method: `xgboost.XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, tree_method='hist', random_state=42, n_jobs=1, eval_metric='logloss', verbosity=0)` (verbosity is output hygiene only)
- Feature set: all-76 (`clean_v1.parquet`)
- Imbalance strategy: none
- Preprocessing: **NaN-passthrough** (no imputer; OneHot NaN category + native XGB NaN routing — see [[EXP-005 histgradientboosting]] for the verification)
- Validation: `StratifiedKFold(10, shuffle=True, random_state=42)` via `src/config.py:make_folds()` — **folds_sha256 = `3b3a17457039`**

## Results
| fold | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | mean |
|------|---|---|---|---|---|---|---|---|---|----|------|
| ROC-AUC | 0.903119 | 0.905317 | 0.904325 | 0.902357 | 0.905009 | 0.900190 | 0.902027 | 0.901471 | 0.905539 | 0.904212 | **0.903357** |

- fold std **0.001715**; runtime **138.3 s**

Figures: none for this EXP (comparison figures land with the report stage).

## Verdict
**Best run of P4: 0.903357** — +**0.01077** over LogReg's 0.892588 (≈ **6.3× fold std**, a genuine family-level improvement; 9 of 10 P4 runs beat LogReg, only LogReg+SMOTE lost). vs LightGBM: ahead by **0.000002** — a statistical dead heat (the zoo's deterministic tie-break `max(mean, exp_id)` broke it in XGB's favour); vs HistGB: +0.00014, same story. **XGBoost = family A** for the imbalance phase (EXP-008a/009a/010a all use this exact config plus a strategy). Note for P6: the booster trio being indistinguishable at default-ish configs means family choice can be driven by cost (LGBM 3.3× faster) or post-tuning margins, not by these 5th-decimal gaps.
