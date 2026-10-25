# Roadmap

Deadline: **2026-11-08 23:59 (C2)**. Statuses: ☐ todo · ◐ in progress · ✅ done

## P0 — Setup ✅
- ✅ Repo, .gitignore, structure, private GitHub remote
- ✅ Agent team (`.opencode/agent/`), AGENTS.md rules
- ✅ Vault bootstrapped: ADR-001..003, leakage watchlist, rubric checklist
- ✅ Python env: pandas, sklearn, seaborn, lightgbm, xgboost, shap, imblearn

## P1 — Profiling & column triage ◐
- ☐ EXP-000: load, shape, dtypes, missingness census, sentinel counts, target distribution
- ☐ Column triage note: keep/drop per column with reasons (mode-only vars, module vars, leakage)
- ☐ Correlation screen: features vs target, duplicate-pair confirmation (e.g. _BMI5 vs HEIGHT3/WEIGHT2)

## P2 — Cleaning & feature matrix v1
- ☐ Recode sentinels → NaN (per-codebook dictionary); ordinal vs nominal encoding plan
- ☐ BMI from _BMI5; day-count vars (88→0); imputation strategy (fold-safe)
- ☐ Checkpoint cleaned matrix → `data/interim/` (parquet)

## P3 — EDA + baselines (Basic level complete)
- ☐ EDA figures: target imbalance, missingness map, top associations, correlation heatmap
- ☐ EXP-001 DummyClassifier ROC-AUC baseline (expect 0.5)
- ☐ EXP-002 LogisticRegression, EXP-003 DecisionTree (manual tuning, documented)

## P4 — Model zoo + imbalance (Intermediate)
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
