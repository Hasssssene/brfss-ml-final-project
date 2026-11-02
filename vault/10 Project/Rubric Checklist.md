# Rubric Checklist

Every graded item from `docs/guidelines.md` §5. Ticked only when the evidence exists (EXP id / report section). Maintained by the vault-scribe, audited by qa-reviewer.

## Basic (up to 5 pts) — Foundational
- ☑ Data cleaning & preprocessing: missing values, dtypes, one-hot (nominal) + label encoding (ordinal) — *report §Preprocessing* — **evidence: [[ADR-005 whitelist and matrix v1]] (P2): per-column value contracts + ENG recodes in `src/preprocessing.py`, `clean_v1.parquet` gates passed; imputation fold-safe inside CV pipelines (EXP-001..003)**
- ☑ EDA: target variable class imbalance analysis — *figure + EXP id* — **evidence: `figures/eda_target_imbalance.png` (16.63% positive, 1 : 5.0) + [[EDA v1 summary]] §1; figures pending human visual pass before report embedding**
- ☑ EDA: baseline ROC-AUC from DummyClassifier — *EXP-001* — **evidence: [[EXP-001 dummy baseline]] = 0.500000 exactly on all 10 folds**
- ☑ Feature selection: basic algorithm applied (dimensionality reduction for tractability) — **evidence: [[EXP-011 selectkbest vs gain importance]] — `SelectKBest(mutual_info_classif, rs=42)` fit inside each training fold, k=50/100/200 → 0.889178 / 0.899556 / 0.899670 (EXP-011a-50..200); best k=200 still −0.0037 below all-76 0.903355 — honest negative, reported with overlap diagnostics (MI∩gain 3/10 @top-10)**
- ☑ Modeling: ≥2 scikit-learn classifiers applied and analyzed — **evidence: [[EXP-002 logistic regression]] (LogReg+StandardScaler, 0.892588) & [[EXP-003 decision tree]] (DecisionTree, 0.880206), analyzed incl. winner margins**
- ☑ Parameter tuning: manual hyperparameter adjustment improving ROC-AUC — **evidence: tree grid lifts 0.694922 → 0.880206 (EXP-003l → EXP-003i); LogReg C-grid flat by design, documented in [[EXP-002 logistic regression]]**
- ☑ Validation: stratified 10-fold CV used for ALL experiments — **evidence: `folds_sha256=3b3a17457039` asserted in all 17 P3 log rows via `src/config.py:make_folds()`**
- ☐ Report: preprocessing, EDA, models, results explained

## Intermediate (up to 3 pts) — Deeper experimentation
- ☑ ≥5 classifiers total compared, incl. ensembles (RandomForest / GradientBoosting) — **evidence: 7 families on identical folds (sha `3b3a17457039`): Dummy 0.500000 ([[EXP-001 dummy baseline]]), LogReg 0.892588 ([[EXP-002 logistic regression]]), DecisionTree 0.880206 ([[EXP-003 decision tree]]), RandomForest 0.893315 ([[EXP-004 random forest]]), HistGB 0.903220 ([[EXP-005 histgradientboosting]]), LightGBM 0.903355 ([[EXP-006 lightgbm]]), XGBoost 0.903357 ([[EXP-007 xgboost]])**
- ☑ Imbalance handling: ≥1 explicit technique compared (class_weight / SMOTE / threshold) — **evidence: 3 techniques × 2 families (XGBoost family A + LogReg family B), fold-internal via imblearn Pipelines — class weight Δ −0.00010/+0.00045 ([[EXP-008 class weighting]]), SMOTE(0.5) Δ −0.00355/−0.00417 ([[EXP-009 smote]]), RUS(0.5) Δ −0.00026/+0.00018 ([[EXP-010 random undersampler]]); verdict `none` for AUC (rank-based metric)**
- ☑ Dimensionality reduction (e.g., PCA) used and reported — **evidence: [[EXP-013 pca]] — PCA(50/100/200) fit inside folds (impute → StandardScaler → PCA), cum var 43.7%/63.7%/97.8% → 0.881126 / 0.886203 / 0.888800 vs all-76 0.903355 (Δ −0.0146…−0.0222); figure `figures/p5_pca_cumvar.png`; verdict: not competitive (smears one-hot structure, forfeits NaN routing) — honest negative**
- ☑ Advanced feature selection + stated hypothesis for the selected variable group — **evidence: [[EXP-011 selectkbest vs gain importance]] (LightGBM gain importances fit inside the 10 training folds, one-hot children summed to original columns: PHYSHLTH 51.7% of gain, DIFFWALK 8.0%; B2 block 55.0%; figure `figures/p5_importance_top30.png`) + hypothesis **H1 stated verbatim** in [[EXP-012 hypothesis H1 subsets]] (five driver blocks B1–B5)
- ☑ Selected-features model vs all-features model compared — **evidence: [[EXP-012 hypothesis H1 subsets]] — H1-subset (36 cols) LGBM 0.900051 / XGB 0.900316 vs all-76 0.903355 / 0.903357 (paired per-fold Δ −0.0033 ± 0.0005; subset ≥ full in 0/10 folds, both families; 99.2% of lift retained); complement (40 cols) standalone 0.811844 / 0.811881 (77.3% of lift) but adds only ~+0.003 on top of H1**
- ☑ Statistical comparison: Wilcoxon signed-rank on 10 fold scores, identical partitions (fixed random_state) — top-2 models — **evidence: [[STATISTICAL COMPARISONS]] + `results/p6/wilcoxon_tests.csv` — T1 tuned LGBM 0.904321 vs tuned XGB 0.903843 (EXP-014a vs EXP-014b): W=0, p=0.001953 exact (n=10 floor), median paired Δ +0.000471, 10/10 folds; folds identical by sha `3b3a17457039`; T2 (vs untuned EXP-006, Δ +0.000991) and T3 (vs LogReg EXP-002a, Δ +0.011768) also significant**
- ☐ Error analysis: ≥3 consistently misclassified instances, features analyzed, reasoned hypothesis, visualizations
- ☐ XAI: LIME or SHAP feature importance (global + individual examples)
- ☐ Report: summary tables and plots comparing all models and techniques

## Advanced (up to 2 pts) — State of the art
- ☑ Automated hyperparameter optimization (RandomizedSearchCV or better) — **evidence: [[EXP-014 randomized search tuned finals]] — RandomizedSearchCV(n_iter=15, inner StratifiedKFold(5, rs=42) selection-only, roc_auc, rs=42) per family on full X,y; tuned LGBM 0.904321 (+0.000966 vs EXP-006, 10/10 folds), tuned XGB 0.903843 (+0.000486 vs EXP-007, 9/10); full candidate tables `results/p6/search_cv_{lgbm,xgb}.csv` + `best_params.json`; budget levers (n_iter 15, n_estimators [300,600]) documented**
- ☑ Nested cross-validation for unbiased performance estimate — **evidence: [[EXP-015 nested CV]] — outer = canonical 10 folds (sha-asserted), inner = 12-candidate RandomizedSearchCV on StratifiedKFold(3, rs=42) refit per fold; unbiased tuned LGBM 0.904303 ± 0.001700; optimism of the non-nested estimate +0.000018 (negligible); 7/10 outer folds re-selected the Part-1 winner; `results/p6/nested_cv_details.csv`**
- ☑ High-performance gradient boosting outside scikit-learn (LightGBM/XGBoost) vs sklearn models — **evidence: LightGBM 0.903355 ([[EXP-006 lightgbm]]) & XGBoost 0.903357 ([[EXP-007 xgboost]]) top the 7-model table — both above every sklearn model (HistGB 0.903220, RF 0.893315, LogReg 0.892588, DT 0.880206); margin over linear/bagging families +0.0108 ≈ 6× fold std (booster trio itself statistically tied, spread ≤ 0.00014)**
- ☐ Outstanding report: professional structure + comprehensive final model justification

## Deliverables & admin
- ☐ `report.pdf` with final section **Ethics and Use of Generative AI** (from [[AI Assistance Log]])
- ☐ `code.py` pure Python file, runs end-to-end (no .ipynb)
- ☐ Screencast ≤10 min (URL in submission form)
- ☐ Submission form + self-assessment
- ☐ AI assistance log complete and honest
