# Rubric Checklist

Every graded item from `docs/guidelines.md` §5. Ticked only when the evidence exists (EXP id / report section). Maintained by the vault-scribe, audited by qa-reviewer.

## Basic (up to 5 pts) — Foundational
- ☐ Data cleaning & preprocessing: missing values, dtypes, one-hot (nominal) + label encoding (ordinal) — *report §Preprocessing*
- ☐ EDA: target variable class imbalance analysis — *figure + EXP id*
- ☐ EDA: baseline ROC-AUC from DummyClassifier — *EXP-001*
- ☐ Feature selection: basic algorithm applied (dimensionality reduction for tractability)
- ☐ Modeling: ≥2 scikit-learn classifiers applied and analyzed
- ☐ Parameter tuning: manual hyperparameter adjustment improving ROC-AUC
- ☐ Validation: stratified 10-fold CV used for ALL experiments
- ☐ Report: preprocessing, EDA, models, results explained

## Intermediate (up to 3 pts) — Deeper experimentation
- ☐ ≥5 classifiers total compared, incl. ensembles (RandomForest / GradientBoosting)
- ☐ Imbalance handling: ≥1 explicit technique compared (class_weight / SMOTE / threshold)
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
- ☐ High-performance gradient boosting outside scikit-learn (LightGBM/XGBoost) vs sklearn models
- ☐ Outstanding report: professional structure + comprehensive final model justification

## Deliverables & admin
- ☐ `report.pdf` with final section **Ethics and Use of Generative AI** (from [[AI Assistance Log]])
- ☐ `code.py` pure Python file, runs end-to-end (no .ipynb)
- ☐ Screencast ≤10 min (URL in submission form)
- ☐ Submission form + self-assessment
- ☐ AI assistance log complete and honest
