# P4 - Model zoo + non-sklearn boosting + imbalance comparison (EXP-004..010)

Generated 2026-11-02T16:19:22 by `src/p4_models.py` on top of the P3 harness (`src/experiments.py`).

Methodology unchanged (ADR-locked): y = GENHLTH ∈ {4,5}; feature set `all-76` (437,532 × 76; 72,741 positives = 16.63%); ROC-AUC; identical folds for EVERY run — `StratifiedKFold(10, shuffle=True, random_state=42)` via `src.config.make_folds()`, fold-index sha256 `3b3a17457039b258…` asserted in every `run_experiment` call (same folds as EXP-001..003). Resampling happens INSIDE fit on training folds via imblearn Pipelines; SMOTE/RUS use random_state=42.

## All P4 runs

| exp_id | model | params | prep | mean CV ROC-AUC ± std | wall |
|---|---|---|---|---|---|
| EXP-004 | RandomForestClassifier | n_estimators=200;min_samples_leaf=50;max_features=sqrt;random_state=42;n_jobs=1 | impute | 0.89331 ± 0.00153 | 352s |
| EXP-005 | HistGradientBoostingClassifier | max_iter=300;learning_rate=0.06;random_state=42 | nanpass | 0.90322 ± 0.00175 | 200s |
| EXP-006 | LGBMClassifier | n_estimators=500;learning_rate=0.05;num_leaves=63;random_state=42;n_jobs=1;verbosity=-1 | nanpass | 0.90335 ± 0.00169 | 41s |
| EXP-007 | XGBClassifier | n_estimators=500;learning_rate=0.05;max_depth=6;tree_method=hist;random_state=42;n_jobs=1;eval_metric=logloss | nanpass | 0.90336 ± 0.00172 | 138s |
| EXP-008a | XGBClassifier | n_estimators=500;learning_rate=0.05;max_depth=6;tree_method=hist;random_state=42;n_jobs=1;eval_metric=logloss;scale_pos_weight=5.0149 | nanpass | 0.90326 ± 0.00173 | 136s |
| EXP-009a | XGBClassifier | SMOTE(strategy=0.5;k_neighbors=5;rs=42);n_estimators=500;learning_rate=0.05;max_depth=6;tree_method=hist;random_state=42;n_jobs=1;eval_metric=logloss | impute | 0.89980 ± 0.00183 | 374s |
| EXP-010a | XGBClassifier | RandomUnderSampler(strategy=0.5;rs=42);n_estimators=500;learning_rate=0.05;max_depth=6;tree_method=hist;random_state=42;n_jobs=1;eval_metric=logloss | nanpass | 0.90310 ± 0.00167 | 73s |
| EXP-008b | LogisticRegression | C=0.01;max_iter=2000;solver=lbfgs;class_weight=balanced;scaler=StandardScaler | impute | 0.89304 ± 0.00158 | 13s |
| EXP-009b | LogisticRegression | SMOTE(strategy=0.5;k_neighbors=5;rs=42);C=0.01;max_iter=2000;solver=lbfgs;scaler=StandardScaler | impute | 0.88842 ± 0.00155 | 208s |
| EXP-010b | LogisticRegression | RandomUnderSampler(strategy=0.5;rs=42);C=0.01;max_iter=2000;solver=lbfgs;scaler=StandardScaler | impute | 0.89276 ± 0.00161 | 12s |

Preprocessing variants: `impute` = P3 shared preprocessor (most_frequent + one-hot for NOM, median impute for ORD/NUM/ENG); `nanpass` = NaN-passthrough for boosters — no imputer anywhere: OneHotEncoder(handle_unknown='ignore') encodes NaN as its own category (verified on sklearn 1.9.1, e.g. SOMALE → categories [1,2,3,4,nan]) and ORD/NUM/ENG pass through with NaN intact for native routing by HistGB/LGBM/XGB.

## Part 1 — model zoo verdict (one config each; tuning is P6)

- EXP-007 XGBClassifier: **0.90336 ± 0.00172** (138s)
- EXP-006 LGBMClassifier: **0.90335 ± 0.00169** (41s)
- EXP-005 HistGradientBoostingClassifier: **0.90322 ± 0.00175** (200s)
- EXP-004 RandomForestClassifier: **0.89331 ± 0.00153** (352s)
- **Best family: XGBClassifier (EXP-007) at 0.90336** — used as family A for Part 2.
- P3 references on the same folds: LogReg C=0.01 = 0.89259 (EXP-002a), best single tree = 0.88021 (EXP-003i), Dummy = 0.5 (EXP-001).
- **9 P4 run(s) beat LogReg's 0.892588**: EXP-007 (+0.01077), EXP-006 (+0.01077), EXP-008a (+0.01067), EXP-005 (+0.01063), EXP-010a (+0.01051), EXP-009a (+0.00722), EXP-004 (+0.00073), EXP-008b (+0.00045), EXP-010b (+0.00018).

## Part 2 — imbalance verdict (ΔAUC vs same model with default weights, same folds)

| strategy | family A run | ΔAUC vs Part-1 default | family B (LogReg C=0.01) run | ΔAUC vs EXP-002a |
|---|---|---|---|---|
| class weighting | EXP-008a (XGBClassifier) | -0.00010 | EXP-008b | +0.00045 |
| SMOTE(0.5) | EXP-009a (XGBClassifier) | -0.00355 | EXP-009b | -0.00417 |
| RandomUnderSampler(0.5) | EXP-010a (XGBClassifier) | -0.00026 | EXP-010b | +0.00018 |

- Family A (XGBClassifier): every imbalance strategy HURT AUC vs default weights — consistent with ROC-AUC being threshold-free and rank-based.
- Family B (LogisticRegression): class weighting (B EXP-008b: +0.00045); RandomUnderSampler(0.5) (B EXP-010b: +0.00018) improve AUC.

## Deviations, forced choices & warnings

- **NaN-passthrough verified** (sklearn 1.9.1): OneHotEncoder(handle_unknown='ignore') → NaN is its own category; boosters keep NaN end-to-end. RF/LogReg keep the impute preprocessor.
- **EXP-009a deviation**: SMOTE rejects NaN (imblearn 0.14.2 raises 'Input X contains NaN'), so family A uses the P3 impute preprocessor in the SMOTE run (its Part-1 baseline used NaN-passthrough). The EXP-008a/EXP-010a family-A runs keep their Part-1 preprocessing.
- **scale_pos_weight source**: neg/pos from the full-data prior (364,791/72,741 ≈ 5.0149); StratifiedKFold preserves the ratio in every training fold, so this equals the per-fold value to rounding.
- **SMOTE budget**: per training fold 65,467 pos / 328,312 neg → 1:1 would synthesize ~262,845 rows (656,624 train rows); 0.5 synthesizes ~98,689 (492,468). RUS(0.5) keeps 130,934 majority rows (196,401 train rows).
- HistGB uses sklearn defaults beyond the two tuned-ish params — early_stopping='auto' is ACTIVE (n>10k), so 300 max_iter may stop earlier; this is the family's stock behaviour, not a tuning choice.
- XGB verbosity=0 and LGBM verbosity=-1 are output hygiene only.
- No convergence warnings expected for boosters; the LogReg contrast reuses the P3 pipeline (lbfgs converged in 18 iterations at C=0.01).

## Runtimes

- Sum of per-run CV wall times: 25.8 min over 10 runs; final session wall 0.0 min (runners: cross_val_score n_jobs=-1, model n_jobs=1 inside each fold worker).
