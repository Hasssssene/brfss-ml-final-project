# P5 — Feature selection, hypothesis H1, PCA (EXP-011..013)

Generated 2026-11-02T17:05:33 by `src/p5_select.py` on the P3/P4 harness. Methodology unchanged (ADR-locked): y = GENHLTH ∈ {4,5}; ROC-AUC; identical canonical folds for EVERY run — `StratifiedKFold(10, shuffle=True, random_state=42)` via `src.config.make_folds()`, fold-index sha256 `3b3a17457039` asserted per run and cross-checked against all logged rows; no resampling (`imbalance_strategy=none`); RANDOM_STATE=42 is the only seed anywhere (MI, PCA, models).

## Hypothesis H1 (verbatim, coordinator-approved)

> The model's predictive power concentrates in five blocks: B1 functional limitation/disability (DIFFWALK, DIFFALON, DIFFDRES, DECIDE, DEAF, BLIND); B2 health days (PHYSHLTH, MENTHLTH, POORHLTH); B3 joint pain/arthritis (JOINPAI2, LMTJOIN3, HAVARTH5, ARTHDIS2, ARTHEXER); B4 chronic conditions (DIABETE4, BPHIGH6, TOLDHI3, CHOLMED3, CVDINFR4, CVDCRHD4, CVDSTRK3, ASTHMA3, ASTHNOW, CHCSCNCR, CHCOCNCR, CHCCOPD3, ADDEPEV3, CHCKDNY2); B5 socio-demographics (_AGE80, INCOME3, EDUCA, EMPLOY1, MARITAL, _RACE, SEXVAR, _BMI5, _SMOKER3). A compact subset of these ~35 columns should match all-76 within fold noise; the complement (diet, screening, state, remaining module vars) contributes little.

## Subset resolution vs the 76-column whitelist

- 37 columns named in H1; **36 present** in the ADR-005 whitelist; **absent: ASTHNOW** (not in the 2021 matrix kept by ADR-005; ASTHMA3 is the only asthma item). No other mismatches: DEAF, BLIND, DECIDE, ARTHEXER, HAVARTH5 etc. are all present.
- H1-subset = 36 columns; complement = 40 columns (diet/fruit-veg ×6, screening/immunization, alcohol ×4, sexual-orientation module, caregiver module, admin/geo ×3, access/utilization, ARTHEDU, CHILDREN, VETERAN3, RENTHOM1, tobacco beyond _SMOKER3).
- Exact lists: `results/p5/subsets.json`.

## All P5 runs (one row per log entry)

| exp_id | model | feature_set | config | mean CV ROC-AUC ± std | wall |
|---|---|---|---|---|---|
| EXP-011a-50 | LGBMClassifier | kbest-50 | SelectKBest(mutual_info_classif | **0.88918 ± 0.00172** | 844s |
| EXP-011a-100 | LGBMClassifier | kbest-100 | SelectKBest(mutual_info_classif | **0.89956 ± 0.00176** | 844s |
| EXP-011a-200 | LGBMClassifier | kbest-200 | SelectKBest(mutual_info_classif | **0.89967 ± 0.00191** | 844s |
| EXP-011b | LGBMClassifier | all-76 | n_estimators=500 | **0.90335 ± 0.00169** | 45s | (importance run)
| EXP-012a | LGBMClassifier | H1-subset | n_estimators=500 | **0.90005 ± 0.00176** | 25s |
| EXP-012b | XGBClassifier | H1-subset | n_estimators=500 | **0.90032 ± 0.00182** | 68s |
| EXP-012c | LGBMClassifier | complement | n_estimators=500 | **0.81184 ± 0.00235** | 26s |
| EXP-012d | XGBClassifier | complement | n_estimators=500 | **0.81188 ± 0.00209** | 86s |
| EXP-013-50 | LGBMClassifier | pca-50 | PCA(n_components=50 | **0.88113 ± 0.00154** | 390s |
| EXP-013-100 | LGBMClassifier | pca-100 | PCA(n_components=100 | **0.88620 ± 0.00133** | 390s |
| EXP-013-200 | LGBMClassifier | pca-200 | PCA(n_components=200 | **0.88880 ± 0.00176** | 390s |

Baselines on the same folds (logged in P4, never re-run): EXP-006 LGBM all-76 = **0.90335 ± 0.00169**; EXP-007 XGB all-76 = **0.90336 ± 0.00172**.

## Verdict 1 — does the H1-subset match all-76 within fold noise?

- LGBM: H1-subset 0.90005 vs all-76 0.90335 → Δ = -0.00330 (paired per-fold Δ mean -0.00330 ± 0.00050, range -0.00398…-0.00246, H1 ≥ full in 0/10 folds).
- XGB: H1-subset 0.90032 vs all-76 0.90336 → Δ = -0.00304 (paired Δ mean -0.00304 ± 0.00051, H1 ≥ full in 0/10 folds).
- Fold-noise reference: baseline fold std = 0.001691 (LGBM) / 0.001715 (XGB).
- **Verdict: NOT within fold noise** — the 36-column H1-subset tracks all-76 to within 0.00330 AUC (criterion |Δ| ≤ 0.001715, one baseline fold-std, on both families). The deficit is small but CONSISTENT (0/10 folds favour the subset on both model families): the subset retains 99.2% of the all-76 lift over 0.5, and the 40 complement columns add a real but tiny ~0.003 AUC when combined with H1 (all-76 = H1 ∪ complement on the same folds).

## Verdict 2 — how weak is the complement?

- LGBM complement 0.81184 vs all-76 0.90335 → Δ = -0.09151 (22.7% of the all-76 lift over 0.5 lost).
- XGB complement 0.81188 vs all-76 0.90336 → Δ = -0.09148.
- Paired per-fold Δ (LGBM): mean -0.09151 ± 0.00256; complement ≥ full in 0/10 folds.
- **Verdict: the complement is dramatically weaker but not uninformative** — standalone it retains 77.3% of the all-76 lift (0.812 vs 0.903), yet its INCREMENTAL value on top of the H1-subset is only ~+0.003 AUC (Verdict 1). Its standalone signal is largely redundant with H1 (e.g. EXERANY2, screening items, alcohol correlate with functional/chronic status) plus module-assignment missingness patterns.

## Verdict 3 — SelectKBest (MI) vs LightGBM gain: overlap

- Original-column top-k agreement between `SelectKBest(mutual_info)` (mean over folds, max over one-hot children) and LightGBM gain share:

| top-k | MI ∩ gain | MI ∩ H1-36 | gain ∩ H1-36 |
|---|---|---|---|
| 10 | 3 / 10 | 7 / 10 | 9 / 10 |
| 20 | 8 / 20 | 15 / 20 | 18 / 20 |
| 30 | 13 / 30 | 20 / 30 | 20 / 30 |
| 36 | 15 / 36 | 22 / 36 | 21 / 36 |

- Best k for SelectKBest: **k=200** → 0.89967 vs all-76 0.90335 (Δ -0.00368); k=50 → 0.88918, k=100 → 0.89956, k=200 → 0.89967.
- Selection stability (k=50): 48 of 76 columns selected in all 10 folds; artifact `results/p5/kbest_selection.csv`.

## Verdict 4 — PCA vs raw features

| n_components | cum. explained var (mean over folds) | AUC | Δ vs EXP-006 |
|---|---|---|---|
| 50 | 43.7% | 0.88113 | -0.02223 |
| 100 | 63.7% | 0.88620 | -0.01715 |
| 200 | 97.8% | 0.88880 | -0.01456 |

- **Verdict: PCA costs 0.01456–0.02223 AUC vs the 76 raw columns** (best at n=200). Unsupervised linear compression smears the one-hot structure (e.g. the 53-level _STATE block) into shared components and forfeits native NaN routing — the impute prep is required before scaling/PCA. Even so, 50 components (43.7% variance) retain most of the signal.
- Solver note: `svd_solver='auto'` (default) resolved to **covariance_eigh** in every fold — sklearn 1.9.1 tall-and-skinny policy (n_features=266/267 ≤ 1000 and n_train=393,779 ≥ 10× n_features): exact eigendecomposition of the covariance matrix, so the n=50/100 variants are provably the first-n components of the n=200 fit (logged notes record the pre-fix attribute probe as 'n/a'; the policy was verified against the installed sklearn source).

## LightGBM gain-importance top-10 (from results/p5/lgbm_importance.csv)

| rank | column | block | gain share |
|---|---|---|---|
| 1 | PHYSHLTH | B2 | 51.69% |
| 2 | DIFFWALK | B1 | 7.99% |
| 3 | DIABETE4 | B4 | 3.31% |
| 4 | INCOME3 | B5 | 2.55% |
| 5 | EDUCA | B5 | 2.53% |
| 6 | _BMI5 | B5 | 2.44% |
| 7 | POORHLTH | B2 | 1.80% |
| 8 | CHCCOPD3 | B4 | 1.71% |
| 9 | EXERANY2 | other | 1.56% |
| 10 | MENTHLTH | B2 | 1.53% |

Block share of total gain: B1 10.1%; B2 55.0%; B3 1.2%; B4 10.6%; B5 11.7%; other 11.4%.

## Reproducibility notes

- MI, gain importances and PCA are computed INSIDE the 10 training folds only (never on full data / test rows); fold loops use joblib with one worker per fold, 1 BLAS thread per worker.
- EXP-011b's fold AUCs replicate the logged EXP-006 row to the log's 6-decimal precision (same pipeline, same folds, n_jobs=1 determinism; recorded max |diff| in its notes field).
- Exp-id scheme: the task's 'one row per k/n' requires unique ids, so rows are EXP-011a-{50,100,200} and EXP-013-{50,100,200}; EXP-011b and EXP-012a..d keep their bare ids.
- Artifacts: `results/p5/subsets.json`, `lgbm_importance.csv`, `kbest_selection.csv`, `pca_variance.csv`; figures `figures/p5_importance_top30.png`, `figures/p5_pca_cumvar.png` (dpi 150).
- Total P5 CV wall time (sum of logged runtime_s): 65.8 min over 11 runs (shared-loop walls counted per row; the kbest/imp/pca loops each ran once).
