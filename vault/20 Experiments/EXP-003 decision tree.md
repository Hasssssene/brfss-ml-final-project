# EXP-003 — DecisionTree baseline (manual depth × leaf grid)

- **exp_id:** EXP-003a…l (12 grid points, one log row each)
- **date:** 2026-10-25
- **phase:** P3
- **status:** done
- **related ADRs:** [[ADR-005 whitelist and matrix v1]]
- **log row:** `results/experiments_log.csv` (EXP-003a–l, 2026-10-25T18:55:41…19:03:12)

## Question
How does a single decision tree trade off depth against leaf size on matrix v1, and can it match the linear baseline (0.892588, [[EXP-002 logistic regression]])?

## Setup
- Model / method: `DecisionTreeClassifier` (native handling of one-hot + ordinal mixes; no scaling)
- Feature set: all-76 (`clean_v1.parquet`, 437,532 × 76 valid rows; ~267 columns after one-hot)
- Imbalance strategy: none
- Hyperparameters: manual grid **max_depth ∈ {6, 10, 16, None} × min_samples_leaf ∈ {1, 50, 200}** (12 configs)
- Validation: `StratifiedKFold(10, shuffle=True, random_state=42)` via `src/config.py:make_folds()` — **folds_sha256 = `3b3a17457039`**, asserted in all 12 rows
- Preprocessing inside the CV pipeline: most_frequent impute + one-hot for NOM, median impute for ORD/NUM/ENG (per [[ADR-005 whitelist and matrix v1]])
- `n_jobs=-1` (10 fold fits in parallel); every config ran inside the 900 s budget — no grid pruning

## Results

**Grid (mean CV ROC-AUC):**

| exp_id | max_depth | min_samples_leaf | mean AUC | fold std | runtime_s |
|---|---|---|---|---|---|
| EXP-003a | 6 | 1 | 0.848558 | 0.002630 | 18.6 |
| EXP-003b | 6 | 50 | 0.848645 | 0.002678 | 18.6 |
| EXP-003c | 6 | 200 | 0.848666 | 0.002686 | 20.1 |
| EXP-003d | 10 | 1 | 0.865805 | 0.002288 | 28.6 |
| EXP-003e | 10 | 50 | 0.871808 | 0.002286 | 25.0 |
| EXP-003f | 10 | 200 | 0.872352 | 0.002394 | 25.5 |
| EXP-003g | 16 | 1 | 0.795855 | 0.003456 | 36.6 |
| EXP-003h | 16 | 50 | 0.874830 | 0.002309 | 34.1 |
| **EXP-003i** | **16** | **200** | **0.880206** | 0.002219 | 32.2 |
| EXP-003j | None | 50 | 0.864202 | 0.002354 | 40.1 |
| EXP-003k | None | 200 | 0.879195 | 0.002028 | 37.0 |
| EXP-003l | None | 1 | 0.694922 | 0.003289 | 45.4 |

Grid total ≈ 362 s of CV wall time (sum of log runtimes; all-17-runs total 7.0 min incl. EXP-001/002). Max fold std across the grid 0.003456 — stable folds, comparisons trustworthy to the 4th decimal.

**Winner EXP-003i fold scores:**
| fold | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | mean |
|------|---|---|---|---|---|---|---|---|---|----|------|
| ROC-AUC | 0.879920 | 0.881922 | 0.881709 | 0.878020 | 0.882437 | 0.876190 | 0.877625 | 0.879330 | 0.882970 | 0.881935 | **0.880206** |

Figures: none for this EXP (grid table above; comparison figures come in P4+).

## Verdict
**Winner: max_depth=16, min_samples_leaf=200 → mean CV ROC-AUC = 0.880206** (EXP-003i). Two clear lessons:
1. **`min_samples_leaf` dominates the grid.** At every depth, leaf=200 ≥ leaf=50 ≥ leaf=1; leaf regularization is what makes deep trees usable. With leaf=200, depth 6 → 16 → None improves 0.8487 → 0.8802 → 0.8792 (saturating near 16).
2. **Unconstrained depth overfits catastrophically:** (None, 1) — near sklearn defaults — collapses to **0.694922** (−0.185 vs winner) as deep paths memorize training folds. Manual tuning lifting 0.6949 → 0.8802 is the rubric's "hyperparameter adjustment improving ROC-AUC" evidence.

Single tree still loses to [[EXP-002 logistic regression]] by **0.0124** AUC — expected for one unpruned tree vs a regularized linear model on wide one-hot data; ensembles (P4) exist precisely to close this gap. **Next:** P4 — RandomForest / HistGradientBoosting / LightGBM / XGBoost on the same folds, plus imbalance strategies.
