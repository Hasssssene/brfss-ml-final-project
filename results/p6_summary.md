# P6 — Hyperparameter optimization, nested CV, Wilcoxon (EXP-014a/b, EXP-015)

Generated 2026-11-02T18:15:32 by `src/p6_tune.py` on the P3/P4 harness. Methodology unchanged (ADR-locked): y = GENHLTH ∈ {4,5}; feature set `all-76`; ROC-AUC; no resampling (`imbalance_strategy=none`); NaN-passthrough preprocessor inside every pipeline (fit on training data only); RANDOM_STATE=42 everywhere (searches, inner CVs, models).

## Setup

- **Comparison folds**: the canonical `StratifiedKFold(10, shuffle=True, random_state=42)` via `src.config.make_folds()` — fold-index sha256 `3b3a17457039` asserted in every P6 comparison run (EXP-014a/b scored via `run_experiment`, EXP-015 outer loop = the same fold list) and cross-checked against all previously logged rows.
- **Inner CVs (selection only, never comparison)**: Part 1 `StratifiedKFold(5, shuffle=True, random_state=42)`; Part 3 `StratifiedKFold(3, shuffle=True, random_state=42)`.
- **Search**: `RandomizedSearchCV(n_iter=15, scoring='roc_auc', random_state=42, n_jobs=-1, pre_dispatch='n_jobs')` fit on FULL X,y — legitimate for selection; the selection bias it induces is exactly what EXP-015 quantifies.
- Spaces (scipy.stats distributions): LightGBM — n_estimators~randint[300,600]; learning_rate~logU[0.02,0.15]; subsample~U[0.6,1.0]; colsample_bytree~U[0.5,1.0]; reg_lambda~logU[0.1,10]; num_leaves~randint[15,127]; min_child_samples~randint[5,100]; subsample_freq=1 fixed. XGBoost — n_estimators~randint[300,600]; learning_rate~logU[0.02,0.15]; subsample~U[0.6,1.0]; colsample_bytree~U[0.5,1.0]; reg_lambda~logU[0.1,10]; max_depth~randint[3,10]; min_child_weight~randint[1,20].
- Fixed booster flags as in P4: LGBM `verbosity=-1`, `subsample_freq=1`, `n_jobs=1`, `random_state=42`; XGB `tree_method='hist'`, `eval_metric='logloss'`, `verbosity=0`, `n_jobs=1`, `random_state=42`.

## Part 1 — search winners (top-5 per family, inner 5-fold AUC)

**LGBMClassifier** (best inner AUC 0.904222 ± 0.001051; search wall 5.6 min):

| rank | mean inner AUC ± std | params |
|---|---|---|
| 1 | 0.904222 ± 0.001051 | `{"model__colsample_bytree": 0.5979914312095727, "model__learning_rate": 0.021908197340907744, "model__min_child_samples": 44, "model__n_estimators": 512, "model__num_leaves": 94, "model__reg_lambda": 3.1234907221207773, "model__subsample": 0.815876852955632}` |
| 2 | 0.904211 ± 0.001030 | `{"model__colsample_bytree": 0.5079831261101071, "model__learning_rate": 0.03184756542192824, "model__min_child_samples": 96, "model__n_estimators": 563, "model__num_leaves": 61, "model__reg_lambda": 0.22219245475399158, "model__subsample": 0.7564242430292963}` |
| 3 | 0.904198 ± 0.001001 | `{"model__colsample_bytree": 0.5477050582452057, "model__learning_rate": 0.04222004422801932, "model__min_child_samples": 66, "model__n_estimators": 515, "model__num_leaves": 51, "model__reg_lambda": 2.878805718308924, "model__subsample": 0.8550229885420852}` |
| 4 | 0.904196 ± 0.001045 | `{"model__colsample_bytree": 0.7159725093210578, "model__learning_rate": 0.03596444270828742, "model__min_child_samples": 63, "model__n_estimators": 469, "model__num_leaves": 106, "model__reg_lambda": 8.861577452533075, "model__subsample": 0.6931085361721216}` |
| 5 | 0.903835 ± 0.001149 | `{"model__colsample_bytree": 0.9692763545078751, "model__learning_rate": 0.020031407387720055, "model__min_child_samples": 64, "model__n_estimators": 576, "model__num_leaves": 47, "model__reg_lambda": 0.4059611610484306, "model__subsample": 0.8099025726528951}` |

**XGBClassifier** (best inner AUC 0.903679 ± 0.001059; search wall 16.2 min):

| rank | mean inner AUC ± std | params |
|---|---|---|
| 1 | 0.903679 ± 0.001059 | `{"model__colsample_bytree": 0.5499874579090014, "model__learning_rate": 0.05045461670694084, "model__max_depth": 7, "model__min_child_weight": 4, "model__n_estimators": 451, "model__reg_lambda": 2.0034427927560734, "model__subsample": 0.6225646316108401}` |
| 2 | 0.903425 ± 0.001173 | `{"model__colsample_bytree": 0.9143687545759647, "model__learning_rate": 0.041040345896217585, "model__max_depth": 6, "model__min_child_weight": 13, "model__n_estimators": 340, "model__reg_lambda": 0.39133348581043736, "model__subsample": 0.6661067756252009}` |
| 3 | 0.903354 ± 0.000989 | `{"model__colsample_bytree": 0.831261142176991, "model__learning_rate": 0.03747971533796248, "model__max_depth": 8, "model__min_child_weight": 10, "model__n_estimators": 559, "model__reg_lambda": 0.23426581058204046, "model__subsample": 0.9878338511058234}` |
| 4 | 0.902925 ± 0.001200 | `{"model__colsample_bytree": 0.7623873301291946, "model__learning_rate": 0.044764407725988005, "model__max_depth": 10, "model__min_child_weight": 16, "model__n_estimators": 570, "model__reg_lambda": 0.8168455894760165, "model__subsample": 0.9140703845572055}` |
| 5 | 0.902907 ± 0.001247 | `{"model__colsample_bytree": 0.8187787356776066, "model__learning_rate": 0.11950759292024817, "model__max_depth": 3, "model__min_child_weight": 3, "model__n_estimators": 400, "model__reg_lambda": 8.778578373294495, "model__subsample": 0.9395655297064336}` |

Best params per family (verbatim from `results/p6/best_params.json`):

- LightGBM: `{"colsample_bytree": 0.5979914312095727, "learning_rate": 0.021908197340907744, "min_child_samples": 44, "n_estimators": 512, "num_leaves": 94, "reg_lambda": 3.1234907221207773, "subsample": 0.815876852955632}`
- XGBoost: `{"colsample_bytree": 0.5499874579090014, "learning_rate": 0.05045461670694084, "max_depth": 7, "min_child_weight": 4, "n_estimators": 451, "reg_lambda": 2.0034427927560734, "subsample": 0.6225646316108401}`

## Part 2 — tuned finals vs untuned baselines (canonical folds)

| family | untuned (P4) | mean AUC | tuned (P6) | mean AUC | Δ AUC | paired folds better |
|---|---|---|---|---|---|---|
| LGBMClassifier | EXP-006 | 0.903355 ± 0.001691 | EXP-014a | **0.904321 ± 0.001675** | +0.000966 | 10/10 |
| XGBClassifier | EXP-007 | 0.903357 ± 0.001715 | EXP-014b | **0.903843 ± 0.001667** | +0.000486 | 9/10 |

## Part 3 — nested CV: the honest estimate (EXP-015)

- Family nested: **LGBMClassifier** (Part-2 winner: EXP-014a 0.904321 vs EXP-014b 0.903843).
- Protocol: outer = the 10 canonical folds; inner = `RandomizedSearchCV(n_iter=12, StratifiedKFold(3, shuffle=True, random_state=42), roc_auc)` per outer fold, best config refit on the outer training fold, scored on the untouched outer test fold.
- **Unbiased tuned estimate: 0.904303 ± 0.001700** (outer fold AUCs 0.9040 0.9055 0.9053 0.9039 0.9058 0.9011 0.9032 0.9021 0.9069 0.9054).
- **Optimism of the non-nested tuned estimate: +0.000018 AUC** (EXP-014a 0.904321 − EXP-015 0.904303) — the Part-2 number overstates generalization by this much because the same data informed both the search and the score.
- Context: nested estimate vs untuned EXP-006 (0.903355): +0.000949.
- Per-outer-fold inner winners differ across folds — full table in `results/p6/nested_cv_details.csv` (search selection is fold-dependent; the nested protocol prices that in).

## Part 4 — Wilcoxon signed-rank tests (paired canonical folds)

| test | comparison | mean AUC a vs b | median paired Δ | W | p-value | method | verdict |
|---|---|---|---|---|---|---|---|
| T1 | tuned LightGBM vs tuned XGBoost (both P6 winners) | 0.904321 vs 0.903843 | +0.000470 | 0.0 | 0.00195312 | exact | Significant at alpha=0.05: EXP-014a > EXP-014b (median paired delta +0.000471 AUC per fold, 10/10 folds favour EXP-014a). |
| T2 | tuned LGBMClassifier vs its untuned self (EXP-006) | 0.904321 vs 0.903355 | +0.000991 | 0.0 | 0.00195312 | exact | Significant at alpha=0.05: EXP-014a > EXP-006 (median paired delta +0.000991 AUC per fold, 10/10 folds favour EXP-014a). |
| T3 | tuned best booster vs LogisticRegression EXP-002a (effect-size contrast) | 0.904321 vs 0.892588 | +0.011768 | 0.0 | 0.00195312 | exact | Significant at alpha=0.05: EXP-014a > EXP-002a (median paired delta +0.011768 AUC per fold, 10/10 folds favour EXP-014a). |

Plain-language reading:

- **T1** (Significant at alpha=0.05: EXP-014a > EXP-014b (median paired delta +0.000471 AUC per fold, 10/10 folds favour EXP-014a).)
- **T2** (Significant at alpha=0.05: EXP-014a > EXP-006 (median paired delta +0.000991 AUC per fold, 10/10 folds favour EXP-014a).)
- **T3** (Significant at alpha=0.05: EXP-014a > EXP-002a (median paired delta +0.011768 AUC per fold, 10/10 folds favour EXP-014a).)

All three tests are two-sided Wilcoxon signed-rank tests (`scipy.stats.wilcoxon`, `zero_method='wilcox'`) on the 10 paired canonical-fold AUCs; the folds are identical by construction (hash-asserted), so pairing is exact.

## Budget, warnings & artifacts

- Compute budget target ≤ 90 min. Adjustments applied (documented): SEARCH_N_ITER 20 -> 15 (Part 1): probe + P4-calibrated projections put the n_iter=20 worst case over the 90-min budget; n_estimators range [300,800] -> [300,600] (Parts 1+3, same space shared): same reason; untuned configs (500 trees) stay inside the range
- Parent-process warnings during searches: LightGBM — none; XGBoost — none. (Warnings raised inside loky workers cannot be transported to the parent — same caveat as P3; boosters emit none at these verbosity settings.)
- Command walls (results/p6/timing.json): probe 1.3 min; search 21.8 min; finals 3.7 min; nested 26.4 min; wilcoxon 0.0 min; figure 0.0 min.
- Artifacts: `results/p6/search_cv_lgbm.csv`, `search_cv_xgb.csv`, `best_params.json`, `nested_cv_details.csv`, `wilcoxon_tests.csv`, `timing.json`; figure `figures/p6_tuning_results.png` (dpi 200).
- Environment: sklearn RandomizedSearchCV; scipy 1.16.0; xgboost 3.4.1; lightgbm 4.7.0; 12 logical cores.
