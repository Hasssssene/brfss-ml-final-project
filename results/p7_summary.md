# P7 — Error analysis + SHAP explainability (final model = tuned LightGBM, EXP-014a)

Generated 2026-11-02T19:03:25 by `src/p7_error_xai.py`. **No `experiments_log.csv` rows were appended**: every number below derives from the OOF predictions of the already-logged configurations EXP-014a / EXP-014b / EXP-002a (deterministic consequence — see `results/p7/oof_provenance.json`) and from a single full-data refit of the EXP-014a model used as the final-model proxy for SHAP. Its graded CV estimate remains EXP-014a (mean CV AUC 0.904321; nested EXP-015 0.904303).

Folds: the canonical `StratifiedKFold(10, shuffle=True, random_state=42)` via `src.config.make_folds()` (folds_sha256 `3b3a17457039`); OOF per-fold AUCs re-computed from the saved predictions match the logged fold scores to ≤ 4.8e-07 (max abs deviation, all three models).

## Error taxonomy (tuned LGBM, OOF, threshold 0.5)

| quantity | value |
|---|---|
| n (rows) | 437,532 |
| TP / FP / TN / FN | 38,957 / 14,340 / 350,451 / 33,784 |
| FP rate (FPR = FP/N₀) | 0.0393 |
| FN rate (FNR = FN/N₁) | 0.4644 |
| TPR / TNR | 0.5356 / 0.9607 |
| error rate | 0.1100 |

### Errors by raw GENHLTH class — the 3/4 boundary story

| GENHLTH class | n | error type | count | rate |
|---|---|---|---|---|
| 1 (negative: excellent/good) | 77,741 | FP | 709 | 0.0091 |
| 2 (negative: excellent/good) | 149,112 | FP | 2,430 | 0.0163 |
| 3 (negative: excellent/good) | 137,938 | FP | 11,201 | 0.0812 |
| 4 (positive: fair/poor) | 54,736 | FN | 31,248 | 0.5709 |
| 5 (positive: fair/poor) | 18,005 | FN | 2,536 | 0.1408 |

- FP rate climbs monotonically with worse self-rated health within the negatives: class 1: 0.0091 → class 2: 0.0163 → class 3: 0.0812 (of true negatives, class 3 is misflagged most often).
- FN rate falls as health worsens: class 4: 0.5709 → class 5: 0.1408 — class 4 ("fair") is the hardest positive to catch.
- Combined: 42,449 of 48,124 errors (88.2%) sit exactly at the 3/4 boundary — the model's confusion is boundary ambiguity, not random noise.

## Cross-model wrongness (LGBM ∪ XGB ∪ LR, OOF @ 0.5)

| wrong in k/3 models | n | share |
|---|---|---|
| 0 | 380,269 | 0.8691 |
| 1 | 9,075 | 0.0207 |
| 2 | 7,097 | 0.0162 |
| 3 | 41,091 | 0.0939 |

- **41,091 instances are wrong in ALL 3 models** — the systematically hard residue (see `error_candidates.csv`, top 50 by |oof_lgbm − 0.5|, most confident shared errors first).
- Shared-error label mix: genhlth 4: 47, genhlth 5: 3 (of the top 50 candidates).

## Selected cases (full vectors + z-scores in selected_cases.json)

- **Case A** (row 307041, fold 4): most confident FALSE POSITIVE: y=0, highest oof_lgbm among the 14,340 FPs. y=0 (GENHLTH=3); OOF P(y=1): LGBM 0.9595, XGB 0.9815, LR 0.9930; wrong in 3/3 models.
- **Case B** (row 392245, fold 9): most confident FALSE NEGATIVE: y=1, lowest oof_lgbm among the 33,784 FNs. y=1 (GENHLTH=4); OOF P(y=1): LGBM 0.0025, XGB 0.0027, LR 0.0156; wrong in 3/3 models.
- **Case C** (row 208256, fold 4): shared-error instance closest to the 3/4 boundary: wrong in all 3 models, genhlth_raw in {3,4}, |oof_lgbm - 0.5| minimal. y=0 (GENHLTH=3); OOF P(y=1): LGBM 0.5000, XGB 0.5981, LR 0.5632; wrong in 3/3 models.

Selection rationale: A/B isolate the model's most confident failures in each direction (top of the FP/FN confidence ordering); C isolates genuine 3/4-boundary ambiguity (wrong everywhere, probability at the fence).

## SHAP (TreeExplainer, path-dependent) on the final-model proxy

- Explainer: `shap.TreeExplainer`, `feature_perturbation=tree_path_dependent` (default; chosen for report speed — no background pass needed; known quirk: path-dependent can spread credit across correlated one-hot dummies, which is exactly why all contributions are summed back to the 76 ORIGINAL columns before plotting).
- Output space: raw margin (log-odds); base value -2.6095. Additivity asserted (max |base + ΣSHAP − raw_score| = 8.8e-14 on the 20k sample and re-asserted per case).
- Global sample: 20,000 rows, stratified on y, random_state=42.
- Top-5 by mean|SHAP|: PHYSHLTH (0.6813), DIFFWALK (0.2164), DIABETE4 (0.1981), _BMI5 (0.1855), EDUCA (0.1786).
- **SHAP-vs-gain rank agreement: Spearman ρ = 0.911 (p = 3.6e-30, n = 76)** — the tuned model's SHAP ordering broadly agrees with the untuned EXP-006 split-gain ordering (`figures/p7_shap_vs_gain.png`); divergences (rank gaps in the scatter) are typically collinear/split-credit features whose gain concentrates in one tree path but whose SHAP credit is shared.

## Artifacts

- `data/interim/oof_predictions.parquet` (oof_lgbm, oof_xgb, oof_lr, y_true, fold) + `results/p7/oof_provenance.json`
- `results/p7/error_taxonomy.csv`, `error_candidates.csv`, `selected_cases.json`, `shap_global_importance.csv`, `shap_cases.parquet`, `shap_meta.json`
- `figures/p7_shap_summary.png` (beeswarm, top 20 original features), `figures/p7_shap_vs_gain.png`, `figures/p7_case_{A,B,C}_waterfall.png` (all dpi 150)
- Runtimes (results/p7/timing.json): oof 3.7 min; errors 0.0 min; shap 7.0 min; summary 0.0 min
