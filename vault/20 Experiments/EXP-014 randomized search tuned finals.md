# EXP-014 — RandomizedSearchCV tuned finals (LightGBM + XGBoost)

- **exp_id:** EXP-014a (LightGBM), EXP-014b (XGBoost)
- **date:** 2026-11-02
- **phase:** P6
- **status:** done
- **related ADRs:** [[ADR-005 whitelist and matrix v1]] (feature matrix, folds and preprocessing unchanged — tuning only, no methodology change)
- **log rows:** `results/experiments_log.csv` (EXP-014a 2026-11-02T17:45:55, EXP-014b 2026-11-02T17:48:26)

## Question
Does automated hyperparameter optimization (`RandomizedSearchCV`) improve the two top boosters beyond their P4 defaults on the canonical folds — and which tuned family becomes the final model?

## Setup
- **Search:** `RandomizedSearchCV(n_iter=15, scoring='roc_auc', random_state=42, n_jobs=-1)` fit on **full X,y** (437,532 rows) — legitimate for *selection*; the selection bias this induces is exactly what [[EXP-015 nested CV]] quantifies.
- **Inner CV (selection only, never comparison):** `StratifiedKFold(5, shuffle=True, random_state=42)`.
- **Spaces** (scipy.stats distributions; verbatim in `results/p6/best_params.json` → `search_config.space`): n_estimators ~ randint[300,600]; learning_rate ~ logU[0.02,0.15]; subsample ~ U[0.6,1.0]; colsample_bytree ~ U[0.5,1.0]; reg_lambda ~ logU[0.1,10]; LGBM adds num_leaves ~ randint[15,127], min_child_samples ~ randint[5,100], subsample_freq=1 fixed; XGB adds max_depth ~ randint[3,10], min_child_weight ~ randint[1,20].
- Fixed booster flags exactly as P4 (LGBM `verbosity=-1`, `n_jobs=1`; XGB `tree_method='hist'`, `eval_metric='logloss'`, `verbosity=0`, `n_jobs=1`); NaN-passthrough preprocessor inside every pipeline; `RANDOM_STATE=42` everywhere.
- Feature set: all-76 · imbalance: none.
- **Tuned finals scored on the canonical folds:** `StratifiedKFold(10, shuffle=True, random_state=42)` via `src/config.py:make_folds()` — folds_sha256 **`3b3a17457039`** asserted in both rows; untuned baselines [[EXP-006 lightgbm]] / [[EXP-007 xgboost]] never re-run.
- **Compute-budget levers (documented):** the timing probe (isolated untuned fits at the inner-5-fold row count: LGBM ≈ 18.8 s stdout, XGB 60.4 s — the latter also in `src/p6_tune.py` comments; aggregate probe wall 79.7 s in `results/p6/timing.json`) plus P4-calibrated parallel-efficiency projections put the spec'd n_iter=20 / n_estimators≤800 worst case at **~110 min > the 90-min budget** → sanctioned levers: **n_iter 20→15** and **n_estimators range [300,800]→[300,600]** (shared by search + nested; untuned 500-tree configs stay bracketed). Actual total P6 wall: **53.3 min**.

## Results

**Search winners** (inner 5-fold AUC; full 15-candidate tables in `results/p6/search_cv_lgbm.csv`, `search_cv_xgb.csv`):

| family | best inner AUC ± std | winning params (`results/p6/best_params.json`) |
|---|---|---|
| LightGBM | **0.904222 ± 0.001051** | n_estimators=512, learning_rate=0.0219, num_leaves=94, min_child_samples=44, subsample=0.8159, colsample_bytree=0.5980, reg_lambda=3.123 |
| XGBoost | **0.903679 ± 0.001059** | n_estimators=451, learning_rate=0.0505, max_depth=7, min_child_weight=4, subsample=0.6226, colsample_bytree=0.5500, reg_lambda=2.003 |

**Tuned finals on the canonical folds:**

| run | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | mean |
|------|---|---|---|---|---|---|---|---|---|----|------|
| EXP-014a LGBM | 0.904011 | 0.905465 | 0.905302 | 0.903872 | 0.905757 | 0.901129 | 0.903399 | 0.902115 | 0.906891 | 0.905271 | **0.904321** |
| EXP-014b XGB | 0.903211 | 0.905068 | 0.904935 | 0.903355 | 0.905751 | 0.900815 | 0.902775 | 0.901682 | 0.906074 | 0.904763 | **0.903843** |

| family | untuned (P4) | tuned (P6) | Δ AUC | paired folds better |
|---|---|---|---|---|
| LGBM | EXP-006 0.903355 ± 0.001691 | **EXP-014a 0.904321 ± 0.001675** | **+0.000966** | **10/10** |
| XGB | EXP-007 0.903357 ± 0.001715 | **EXP-014b 0.903843 ± 0.001667** | **+0.000486** | 9/10 (only fold 2 favours untuned) |

Figures: `figures/p6_tuning_results.png` (dpi 200, 2700×1160 px; ⚠ generated headlessly — **human visual pass required** before report embedding). Artifacts: `results/p6/` (search CSVs, `best_params.json`, `timing.json`); summary `results/p6_summary.md`.

## Verdict
**LightGBM wins; EXP-014a (0.904321) is the project's final best model.** Tuning is real but small: +0.0010 (LGBM, all 10 folds) / +0.0005 (XGB, 9/10) — an order of magnitude below the booster-vs-linear family gain (+0.012). All paired contrasts are statistically significant at the n=10 floor — [[STATISTICAL COMPARISONS]] — and the honest nested tuned estimate is [[EXP-015 nested CV]] **0.904303**. Rubric evidence: **Advanced "Automated hyperparameter optimization (RandomizedSearchCV or better)"**.
