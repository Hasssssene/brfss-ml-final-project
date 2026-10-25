# P3 - Baseline experiments (EXP-001..003)

Generated 2026-10-25T19:03:15 by `src/experiments.py` (harness backbone for P4-P6).

Fixed methodology: y = GENHLTH ∈ {4,5}; feature set `all-76` (clean_v1.parquet, 437,532 × 76 after valid-target mask; 72,741 positives = 16.63%); metric ROC-AUC; identical folds for EVERY run — `StratifiedKFold(10, shuffle=True, random_state=42)` via `src.config.make_folds()`, fold-index sha256 `3b3a17457039b258…` asserted in every `run_experiment` call; no resampling (`imbalance_strategy=none`). Preprocessing (most_frequent impute + one-hot for NOM, median impute for ORD/NUM/ENG) lives inside the CV pipeline — fit on training folds only.

## All runs

| exp_id | model | params | mean CV ROC-AUC ± std | wall time |
|---|---|---|---|---|
| EXP-001 | DummyClassifier | strategy=prior | 0.50000 ± 0.00000 | 8s |
| EXP-002a | LogisticRegression | C=0.01;max_iter=2000;solver=lbfgs;scaler=StandardScaler | 0.89259 ± 0.00161 | 14s |
| EXP-002b | LogisticRegression | C=0.1;max_iter=2000;solver=lbfgs;scaler=StandardScaler | 0.89258 ± 0.00161 | 13s |
| EXP-002c | LogisticRegression | C=1;max_iter=2000;solver=lbfgs;scaler=StandardScaler | 0.89258 ± 0.00161 | 13s |
| EXP-002d | LogisticRegression | C=10;max_iter=2000;solver=lbfgs;scaler=StandardScaler | 0.89258 ± 0.00161 | 12s |
| EXP-003a | DecisionTreeClassifier | max_depth=6;min_samples_leaf=1 | 0.84856 ± 0.00263 | 19s |
| EXP-003b | DecisionTreeClassifier | max_depth=6;min_samples_leaf=50 | 0.84864 ± 0.00268 | 19s |
| EXP-003c | DecisionTreeClassifier | max_depth=6;min_samples_leaf=200 | 0.84867 ± 0.00269 | 20s |
| EXP-003d | DecisionTreeClassifier | max_depth=10;min_samples_leaf=1 | 0.86580 ± 0.00229 | 29s |
| EXP-003e | DecisionTreeClassifier | max_depth=10;min_samples_leaf=50 | 0.87181 ± 0.00229 | 25s |
| EXP-003f | DecisionTreeClassifier | max_depth=10;min_samples_leaf=200 | 0.87235 ± 0.00239 | 26s |
| EXP-003g | DecisionTreeClassifier | max_depth=16;min_samples_leaf=1 | 0.79586 ± 0.00346 | 37s |
| EXP-003h | DecisionTreeClassifier | max_depth=16;min_samples_leaf=50 | 0.87483 ± 0.00231 | 34s |
| EXP-003i | DecisionTreeClassifier | max_depth=16;min_samples_leaf=200 | 0.88021 ± 0.00222 | 32s |
| EXP-003j | DecisionTreeClassifier | max_depth=None;min_samples_leaf=50 | 0.86420 ± 0.00235 | 40s |
| EXP-003k | DecisionTreeClassifier | max_depth=None;min_samples_leaf=200 | 0.87919 ± 0.00203 | 37s |
| EXP-003l | DecisionTreeClassifier | max_depth=None;min_samples_leaf=1 | 0.69492 ± 0.00329 | 45s |

## Winners per family

- **Sanity floor (EXP-001)** DummyClassifier(prior): **mean CV AUC = 0.500000** (expected exactly 0.5 for a constant-prior predictor — confirms the scorer and folds).
- **Linear family (EXP-002a–d)**: best **C = 0.01** → **0.89259 ± 0.00161** (EXP-002a). Grid: C=0.01: 0.89259, C=0.1: 0.89258, C=1: 0.89258, C=10: 0.89258.
- **Tree family (EXP-003a–l)**: best **max_depth=16;min_samples_leaf=200** → **0.88021 ± 0.00222** (EXP-003i).

## Fold-score stability

- 10 folds × ~43,753 test rows each (stratified); identical assignment across all 17 runs (hash-asserted per run).
- Dummy (EXP-001): fold AUCs 0.5000–0.5000 (spread 0.0000), std 0.000000.
- LogReg winner (EXP-002a): fold AUCs 0.8897–0.8952 (spread 0.0054), std 0.001606.
- Tree winner (EXP-003i): fold AUCs 0.8762–0.8830 (spread 0.0068), std 0.002219.
- All tree configs: max fold std across the grid = 0.003456 — scores are stable fold-to-fold; no fold is anomalous, so mean CV AUC comparisons between configurations are trustworthy at the 4th decimal.
- Winner margin: best LogReg beats best tree by +0.01238 AUC (0.89259 vs 0.88021); both far above the 0.5 prior floor.

## Diagnostics & runtime

- Per-config CV wall time sums to 7.0 min over all 17 runs; the final `main()` invocation (data load + missing configs + diagnostics) took 3.2 min (the grid was resumed across two processes after a killed session; per-config times come from the log rows).
- Convergence (fold-1 in-process refit of the winning LR, same pipeline): lbfgs n_iter = 18 / 2000, ConvergenceWarnings = 0. cross_val_score worker-side warnings cannot be captured in-process; this refit on 393,779 training rows documents convergence behaviour.
- No grid pruning: all 12 tree configurations ran within the 900s budget.
- Environment: sklearn DecisionTree/LogisticRegression, n_jobs=-1 (10 fold fits in parallel, pre_dispatch='n_jobs'), float32 design matrix (~267 columns after one-hot).
