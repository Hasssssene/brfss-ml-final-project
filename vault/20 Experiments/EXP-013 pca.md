# EXP-013 — PCA dimensionality reduction vs raw 76 columns

- **exp_id:** EXP-013 (log rows `EXP-013-50`, `EXP-013-100`, `EXP-013-200`)
- **date:** 2026-11-02
- **phase:** P5
- **status:** done
- **related ADRs:** [[ADR-005 whitelist and matrix v1]] (PCA operates on the encoded 76-column matrix inside folds; no methodology change)
- **log row:** `results/experiments_log.csv` (2026-11-02T17:03:26)

## Question
Does unsupervised linear compression (PCA) preserve the signal of the raw one-hot matrix — the rubric's dimensionality-reduction angle and the tractability alternative to feature selection?

## Setup
- Per-fold pipeline (fit on training folds only, never test rows): P3 impute preprocessor → `StandardScaler` → `PCA(n_components ∈ {50,100,200}, whiten=False, random_state=42)` → EXP-006 LightGBM config. Native NaN routing is forfeited (imputation is required before scaling/PCA).
- Model: `LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=63, random_state=42, n_jobs=1, verbosity=-1)`
- Feature set: pca-50 | pca-100 | pca-200
- Imbalance strategy: none
- Validation: `StratifiedKFold(10, shuffle=True, random_state=42)` via `src/config.py:make_folds()` — folds_sha256 **`3b3a17457039`** asserted in all 3 rows; baseline EXP-006 all-76 = 0.903355 (never re-run)
- **Solver note:** `svd_solver='auto'` resolved to **`covariance_eigh`** in every fold — sklearn 1.9.1 tall-and-skinny policy (n_features = 266/267 ≤ 1000 and n_train = 393,779 ≥ 10× n_features) → exact eigendecomposition of the covariance matrix, so the n=50/100 variants are provably the first-n components of the n=200 eigensystem (monotone in n). The log rows record the pre-fix attribute probe as `svd_solver=n/a`; the policy was verified afterwards against the installed sklearn source.

## Results
| run | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | mean |
|------|---|---|---|---|---|---|---|---|---|----|------|
| pca-50 | 0.881290 | 0.879326 | 0.883444 | 0.880420 | 0.882646 | 0.878975 | 0.881303 | 0.879444 | 0.883324 | 0.881093 | **0.881126** |
| pca-100 | 0.886740 | 0.885525 | 0.888571 | 0.885195 | 0.886708 | 0.884568 | 0.886059 | 0.884038 | 0.887645 | 0.886979 | **0.886203** |
| pca-200 | 0.890481 | 0.888683 | 0.889766 | 0.887797 | 0.888586 | 0.886427 | 0.887986 | 0.886107 | 0.892167 | 0.889996 | **0.888800** |

| n_components | cum. explained var (fold mean) | mean AUC | Δ vs EXP-006 |
|---|---|---|---|
| 50 | 43.7% | 0.881126 | **−0.02223** |
| 100 | 63.7% | 0.886203 | **−0.01715** |
| 200 | 97.8% | 0.888800 | **−0.01456** |

Figures: `figures/p5_pca_cumvar.png` (dpi 150). Artifacts: `results/p5/pca_variance.csv`.

## Verdict
**PCA is not competitive for this matrix — honest negative, reported.** Linear compression costs **0.0146–0.0222 AUC** vs the 76 raw columns (best at n=200): unsupervised compression smears the one-hot structure (e.g. the 53-level `_STATE` block) into shared components and forfeits the boosters' native NaN routing (impute prep becomes mandatory). Interesting side-finding: 50 components carry only 43.7% of variance yet already reach 0.8811 — the signal is low-dimensional even though the optimal representation is not. Rubric evidence: **Intermediate "dimensionality reduction (e.g., PCA) used and reported"**. Keep raw columns for the final model.
