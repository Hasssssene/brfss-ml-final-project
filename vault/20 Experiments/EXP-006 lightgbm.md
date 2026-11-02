# EXP-006 — LightGBM (non-sklearn booster #1)

- **exp_id:** EXP-006
- **date:** 2026-11-02
- **phase:** P4
- **status:** done
- **related ADRs:** [[ADR-005 whitelist and matrix v1]]
- **log row:** `results/experiments_log.csv` (EXP-006, 2026-11-02T16:01:27)

## Question
Does Microsoft's LightGBM — histogram boosting outside scikit-learn — beat every sklearn model on identical folds, and at what compute cost?

## Setup
- Model / method: `lightgbm.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=63, random_state=42, n_jobs=1, verbosity=-1)` (verbosity is output hygiene only)
- Feature set: all-76 (`clean_v1.parquet`)
- Imbalance strategy: none
- Preprocessing: **NaN-passthrough** (no imputer; OneHot NaN category + native LGBM NaN routing — see [[EXP-005 histgradientboosting]] for the verification)
- Validation: `StratifiedKFold(10, shuffle=True, random_state=42)` via `src/config.py:make_folds()` — **folds_sha256 = `3b3a17457039`**

## Results
| fold | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | mean |
|------|---|---|---|---|---|---|---|---|---|----|------|
| ROC-AUC | 0.902947 | 0.904705 | 0.904383 | 0.902790 | 0.905059 | 0.900003 | 0.902187 | 0.901398 | 0.905380 | 0.904695 | **0.903355** |

- fold std **0.001691**; runtime **41.4 s** — the **fastest** model of the entire project so far

Figures: none for this EXP (comparison figures land with the report stage).

## Verdict
**Adopted as a top family.** 0.903355 = +**0.01077** over LogReg (6.4× fold std), ahead of every sklearn model including HistGB (0.903220) and RF (0.893315) → rubric "High-performance gradient boosting outside scikit-learn" evidenced together with [[EXP-007 xgboost]]. Dead heat with XGBoost: gap **0.000002** (≈ 0.001× fold std) — statistically indistinguishable; the zoo's deterministic tie-break (max mean, then max exp_id) gave XGBoost the family-A slot for the imbalance phase. At 41 s/run (8.5× faster than RF, 3.3× faster than XGB), LightGBM has the **best accuracy/cost ratio** in the zoo — a strong candidate for the compute-heavy P6 search if tuning budget matters. Rubric evidence: "≥5 classifiers compared" now spans 7 families (Dummy/LogReg/DT/RF/HistGB/LGBM/XGB, EXP-001..007).
