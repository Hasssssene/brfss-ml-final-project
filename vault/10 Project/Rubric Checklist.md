# Rubric Checklist

Every graded item from `docs/guidelines.md` §5. Ticked only when the evidence exists (EXP id / report section). Maintained by the vault-scribe, audited by qa-reviewer.

## Basic (up to 5 pts) — Foundational
- ☑ Data cleaning & preprocessing: missing values, dtypes, one-hot (nominal) + label encoding (ordinal) — *report §Preprocessing* — **evidence: [[ADR-005 whitelist and matrix v1]] (P2): per-column value contracts + ENG recodes in `src/preprocessing.py`, `clean_v1.parquet` gates passed; imputation fold-safe inside CV pipelines (EXP-001..003)**
- ☑ EDA: target variable class imbalance analysis — *figure + EXP id* — **evidence: `figures/eda_target_imbalance.png` (16.63% positive, 1 : 5.0) + [[EDA v1 summary]] §1; figures pending human visual pass before report embedding**
- ☑ EDA: baseline ROC-AUC from DummyClassifier — *EXP-001* — **evidence: [[EXP-001 dummy baseline]] = 0.500000 exactly on all 10 folds**
- ☐ Feature selection: basic algorithm applied (dimensionality reduction for tractability)
- ☑ Modeling: ≥2 scikit-learn classifiers applied and analyzed — **evidence: [[EXP-002 logistic regression]] (LogReg+StandardScaler, 0.892588) & [[EXP-003 decision tree]] (DecisionTree, 0.880206), analyzed incl. winner margins**
- ☑ Parameter tuning: manual hyperparameter adjustment improving ROC-AUC — **evidence: tree grid lifts 0.694922 → 0.880206 (EXP-003l → EXP-003i); LogReg C-grid flat by design, documented in [[EXP-002 logistic regression]]**
- ☑ Validation: stratified 10-fold CV used for ALL experiments — **evidence: `folds_sha256=3b3a17457039` asserted in all 17 P3 log rows via `src/config.py:make_folds()`**
- ☐ Report: preprocessing, EDA, models, results explained

## Intermediate (up to 3 pts) — Deeper experimentation
- ☑ ≥5 classifiers total compared, incl. ensembles (RandomForest / GradientBoosting) — **evidence: 7 families on identical folds (sha `3b3a17457039`): Dummy 0.500000 ([[EXP-001 dummy baseline]]), LogReg 0.892588 ([[EXP-002 logistic regression]]), DecisionTree 0.880206 ([[EXP-003 decision tree]]), RandomForest 0.893315 ([[EXP-004 random forest]]), HistGB 0.903220 ([[EXP-005 histgradientboosting]]), LightGBM 0.903355 ([[EXP-006 lightgbm]]), XGBoost 0.903357 ([[EXP-007 xgboost]])**
- ☑ Imbalance handling: ≥1 explicit technique compared (class_weight / SMOTE / threshold) — **evidence: 3 techniques × 2 families (XGBoost family A + LogReg family B), fold-internal via imblearn Pipelines — class weight Δ −0.00010/+0.00045 ([[EXP-008 class weighting]]), SMOTE(0.5) Δ −0.00355/−0.00417 ([[EXP-009 smote]]), RUS(0.5) Δ −0.00026/+0.00018 ([[EXP-010 random undersampler]]); verdict `none` for AUC (rank-based metric)**
- ☐ Dimensionality reduction (e.g., PCA) used and reported
- ☐ Advanced feature selection + stated hypothesis for the selected variable group
- ☐ Selected-features model vs all-features model compared
- ☐ Statistical comparison: Wilcoxon signed-rank on 10 fold scores, identical partitions (fixed random_state) — top-2 models
- ☐ Error analysis: ≥3 consistently misclassified instances, features analyzed, reasoned hypothesis, visualizations
- ☐ XAI: LIME or SHAP feature importance (global + individual examples)
- ☐ Report: summary tables and plots comparing all models and techniques

## Advanced (up to 2 pts) — State of the art
- ☐ Automated hyperparameter optimization (RandomizedSearchCV or better)
- ☐ Nested cross-validation for unbiased performance estimate
- ☑ High-performance gradient boosting outside scikit-learn (LightGBM/XGBoost) vs sklearn models — **evidence: LightGBM 0.903355 ([[EXP-006 lightgbm]]) & XGBoost 0.903357 ([[EXP-007 xgboost]]) top the 7-model table — both above every sklearn model (HistGB 0.903220, RF 0.893315, LogReg 0.892588, DT 0.880206); margin over linear/bagging families +0.0108 ≈ 6× fold std (booster trio itself statistically tied, spread ≤ 0.00014)**
- ☐ Outstanding report: professional structure + comprehensive final model justification

## Deliverables & admin
- ☐ `report.pdf` with final section **Ethics and Use of Generative AI** (from [[AI Assistance Log]])
- ☐ `code.py` pure Python file, runs end-to-end (no .ipynb)
- ☐ Screencast ≤10 min (URL in submission form)
- ☐ Submission form + self-assessment
- ☐ AI assistance log complete and honest
