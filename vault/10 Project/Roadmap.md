# Roadmap

Deadline: **2026-11-08 23:59 (C2)**. Statuses: ☐ todo · ◐ in progress · ✅ done

## P0 — Setup ✅
- ✅ Repo, .gitignore, structure, private GitHub remote
- ✅ Agent team (`.opencode/agent/`), AGENTS.md rules
- ✅ Vault bootstrapped: ADR-001..003, leakage watchlist, rubric checklist
- ✅ Python env: pandas, sklearn, seaborn, lightgbm, xgboost, shap, imblearn

## P1 — Profiling & column triage ✅
- ✅ EXP-000: load, shape, dtypes, missingness census, sentinel counts, target distribution ([[EXP-000 data profiling]]; 438,693×303; 16.63% positives; `_RFHLTH` 100% target match)
- ✅ Column triage note: keep/drop per column with reasons (mode-only vars, module vars, leakage) — EXP-000 evidence + open-question resolutions recorded; per-column verdict table finalized during P2 whitelist build
- ✅ Correlation screen: features vs target, duplicate-pair confirmation (e.g. _BMI5 vs HEIGHT3/WEIGHT2) (top assoc DIFFWALK ρ=−0.413; dups confirmed: EXERANY2≡_TOTINDA, _DRDXAR3≡HAVARTH5)

## P2 — Cleaning & feature matrix v1 ✅
- ✅ Recode sentinels → NaN (per-codebook dictionary); ordinal vs nominal encoding plan *(done via per-column **value contracts** in `src/preprocessing.py` — codebook-exact sentinel sets, never pattern matching; ORD ×4 / NUM ×3 / ENG ×15 / NOM ×54 per [[ADR-005 whitelist and matrix v1]])*
- ✅ BMI from _BMI5; day-count vars (88→0); imputation strategy (fold-safe) *(BMI ÷100 + all ENG recodes done; fold-safe imputation design moves to P3 CV pipelines — smoke test used simple median/mode on a 50k subsample)*
- ✅ Checkpoint cleaned matrix → `data/interim/` (parquet) *(`clean_v1.parquet` 438,693×76 + `target_v1.parquet` + `column_meta.json`; 20.54% overall missingness; smoke train AUC 0.8966 — not an experiment)*

## P3 — EDA + baselines ✅ (Basic level complete)
- ✅ EDA figures: target imbalance, missingness map, top associations, correlation heatmap *(5 figures committed: `eda_target_imbalance`, `eda_missingness_matrix`, `eda_association_top25`, `eda_correlation_heatmap`, `eda_key_features_by_class`; [[EDA v1 summary]]; ⚠ figures verified programmatically — human visual pass required before report embedding)*
- ✅ EXP-001 DummyClassifier ROC-AUC baseline (expect 0.5) *(0.500000 exactly on all 10 folds; [[EXP-001 dummy baseline]])*
- ✅ EXP-002 LogisticRegression, EXP-003 DecisionTree (manual tuning, documented) *(LogReg C=0.01 → **0.892588**; Tree (16, 200) → **0.880206**; identical folds sha `3b3a17457039` across all 17 runs; harness `src/experiments.py` now resume-safe — skips logged exp_ids, re-asserts fold hash; [[EXP-002 logistic regression]], [[EXP-003 decision tree]])*

## P4 — Model zoo + imbalance (Intermediate) ◐
- ☐ EXP-004 RandomForest, EXP-005 HistGradientBoosting, EXP-006 LightGBM, EXP-007 XGBoost
- ☐ Imbalance: class_weight='balanced' vs SMOTE(fold-internal) vs none — EXP-008..010

## P5 — Feature selection & hypothesis
- ☐ Hypothesis note: expected driver blocks (disability, health days, chronic conditions, age)
- ☐ SelectKBest + RFECV/permutation importance; selected-subset vs all-features — EXP-011..
- ☐ PCA variant — EXP-0xx

## P6 — Tuning & statistics
- ☐ RandomizedSearchCV on top-2 models (fixed folds)
- ☐ Nested CV unbiased estimate
- ☐ Wilcoxon signed-rank on 10 fold scores, top-2 models

## P7 — Error analysis & XAI
- ☐ ≥3 consistently misclassified respondents, profiled, with figures
- ☐ SHAP global summary + local explanations on best model

## P8 — Deliverables
- ☐ Assemble `deliverables/code.py` (end-to-end, clean process)
- ☐ `report.pdf` full draft through QA audit
- ☐ Screencast script + recording (≤10 min)
- ☐ Submission form + self-assessment
