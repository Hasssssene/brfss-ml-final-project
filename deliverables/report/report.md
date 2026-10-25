<!--
Master document for the technical report. Compiled to report.pdf in P8.
Traceability rule (AGENTS.md): every number in this report must cite an EXP id that exists
in results/experiments_log.csv. If a number is not logged, it is not written.
This skeleton therefore contains NO results numbers — placeholders only.
Suggested section statuses: [EMPTY] -> [DRAFT] -> [QA-AUDITED] -> [FINAL].
-->

# Predicting Fair/Poor Self-Rated Health from BRFSS 2021: A Binary Classification Study

- **Working title:** Predicting Fair/Poor Self-Rated Health from BRFSS 2021: A Binary Classification Study
- **Author:** [Student name — to fill before submission]
- **Course:** Machine Learning final project — University of Alicante (C2 call, due 2026-11-08)
- **Date:** 2026-10-25
- **Report status: skeleton** *(to update: skeleton → draft → QA-audited → final)*

---

## 1. Introduction & dataset

- BRFSS survey context and why general-health prediction from behavioral/demographic factors is a meaningful task. *(source: docs/guidelines.md §1 Context)*
- Dataset description: source, size, structure, codebook reference. *(feeds: EXP-000 profiling run — all counts filled from the log, none written now; source: vault/40 Data Notes/Codebook highlights.md)*
- Known dataset challenges (high dimensionality, sentinel-coded missingness, target imbalance, variables directly related to the target). *(source: docs/guidelines.md §2; details from EXP-000)*
- How our work addresses the course learning objectives. *(source: docs/guidelines.md §1)*

## 2. Task framing — binary classification per ADR-001

- **Target definition:** y = 1 iff GENHLTH ∈ {4, 5} (Fair/Poor); y = 0 iff GENHLTH ∈ {1, 2, 3}. Rows with GENHLTH ∈ {7, 9, blank} are dropped, not imputed. *(source: ADR-001; implemented in `src/config.py:make_target()`)*
- **CDC `_RFHLTH` framing justification:** the CDC itself derives `_RFHLTH` from GENHLTH as the official "adults with fair or poor health" indicator; our binary target mirrors this established operationalization rather than being an arbitrary cut. Consequence: `_RFHLTH` (and anything derived from GENHLTH) is target leakage and blacklisted — its association with the target is verified empirically before modeling. *(source: ADR-001; probe feeds: EXP-000; source: vault/40 Data Notes/Leakage watchlist.md Tier 1)*
- Alternatives considered and rejected (e.g., Poor-vs-rest; keeping the native multiclass problem as the main task). *(source: ADR-001 "Alternatives considered"; scope decision in ADR-002)*
- Consequences of the chosen framing for imbalance handling and error analysis. *(source: ADR-001 "Consequences"; quantified later from EXP-000 and the imbalance grid EXP-008..010)*

## 3. Data cleaning & preprocessing decisions

- Sentinel recoding per the BRFSS codebook (don't-know / refused / blank / "none" codes) before any modeling; raw data immutable, cleaned matrix checkpointed. *(source: AGENTS.md data integrity rules; vault/40 Data Notes/Codebook highlights.md; sentinel census feeds: EXP-000)*
- **Whitelist principle:** the feature matrix is built from an explicit keep-list in the column-triage note, never by blacklisting drops — a typo in a drop-list must not leak the target. *(source: vault/40 Data Notes/Leakage watchlist.md rule 1; vault/40 Data Notes/Column triage.md)*
- Leakage controls applied: Tier 1 blacklist (`_RFHLTH` and GENHLTH-derived composites — including the `_RFHLTH`≈target probe), Tier 2 duplicate-information pairs resolved (derived bins vs raw variables), Tier 3 construct-suspicion columns decided with data evidence (e.g., POORHLTH). *(source: Leakage watchlist; verification feeds: EXP-000)*
- Encoding and repair plan per column: ordinal vs one-hot, day-count "none" recoding, imputation strategy (fold-safe), engineered features (e.g., continuous BMI). *(source: Column triage verdict table, filled after EXP-000)*

## 4. Exploratory data analysis

- Target imbalance: class-frequency analysis and figure motivating the imbalance-handling experiments. *(feeds: EXP-000; planned figure: `figures/target_imbalance.png`)*
- Missingness census: per-column missing/sentinel rates, missingness map, implications for feature retention. *(feeds: EXP-000; planned figure: `figures/missingness_map.png`)*
- Associations: feature-vs-target screen and duplicate-pair confirmation (e.g., continuous BMI vs height/weight, derived vs raw representations). *(feeds: EXP-000)*
- Multivariate structure: correlation heatmap and top-association ranking. *(feeds: EXP-000 / P3 EDA figures)*

## 5. Baselines

- DummyClassifier ROC-AUC reference under the identical fold scheme. *(feeds: EXP-001)*
- Logistic Regression baseline (encoding-sensitive reference model). *(feeds: EXP-002)*
- Decision Tree baseline (nonlinear reference, interpretable). *(feeds: EXP-003)*
- Manual hyperparameter adjustments for the basic classifiers and their documented effect on ROC-AUC. *(feeds: EXP-002 / EXP-003 rows in results/experiments_log.csv)*

## 6. Model zoo & imbalance handling

- Scikit-learn ensembles: RandomForest, HistGradientBoosting. *(feeds: EXP-004, EXP-005)*
- High-performance boosting outside scikit-learn: LightGBM, XGBoost (Advanced-level criterion). *(feeds: EXP-006, EXP-007; scope: ADR-002)*
- Explicit imbalance handling grid: none vs `class_weight='balanced'` vs fold-internal SMOTE. *(feeds: EXP-008, EXP-009, EXP-010)*
- Summary table and plots comparing all models × imbalance strategies (mean fold AUC with fold spread). *(compiled from all rows in results/experiments_log.csv)*

## 7. Feature selection & hypothesis

- **Hypothesis:** statement of the expected driver blocks (disability, health days, chronic conditions, age) and why this group was selected. *(source: hypothesis note to be written in P5; grounded in EXP-000 associations)*
- Basic selection for tractability + advanced selection (SelectKBest, RFECV / permutation importance). *(feeds: EXP-011 and subsequent P5 rows — ids assigned when logged)*
- Dimensionality reduction variant (PCA). *(feeds: P5 PCA experiment row — id assigned when logged)*
- Selected-subset model vs all-features model comparison against the hypothesis. *(feeds: the P5 comparison rows in results/experiments_log.csv)*

## 8. Hyperparameter tuning & statistical comparison

- Automated search (RandomizedSearchCV) on the top models, executed on the identical fixed folds. *(feeds: P6 experiment rows — ids TBD when logged; scope: ADR-002)*
- Nested cross-validation for the unbiased final performance estimate. *(feeds: P6 experiment rows — ids TBD when logged)*
- Statistical comparison of the top models: Wilcoxon signed-rank test over the fold scores, valid because partitions are identical everywhere (fixed random_state). *(feeds: P6 statistical-comparison note + experiment rows)*
- Fold-identity enforcement across every model, selection, and resampling step. *(source: AGENTS.md reproducibility rules; `src/config.py:make_folds()`)*

## 9. Error analysis & XAI

- At least three consistently misclassified respondents: feature profiles, reasoned failure hypotheses, supporting visualizations. *(feeds: P7 error-analysis note + experiment rows — ids TBD when logged)*
- Global feature importance for the best model via SHAP. *(feeds: P7 XAI rows — ids TBD when logged)*
- Local (per-instance) explanations for the identified error cases, connecting model behavior to the failure hypotheses. *(feeds: P7 XAI rows — ids TBD when logged)*

## 10. Conclusions & recommended model

- Recommended model justified against the full evidence chain (zoo → imbalance grid → selection → tuning → nested CV → significance test), not by a single metric. *(source: synthesis of all logged EXP rows)*
- Honest negative results: what was tried and did not help. *(feeds: rejected/failed rows in results/experiments_log.csv)*
- Limitations: self-reported survey data, no causal claims, generalization caveats. *(source: EDA + error-analysis findings)*
- Comprehensive summary table closing the argument for the final model. *(compiled from results/experiments_log.csv)*

## 11. Ethics and Use of Generative AI — **[DRAFT]**

> Drafted from `vault/10 Project/AI Assistance Log.md` (synced 2026-10-25; the log currently lists **3 entries** — this section must be updated in sync if the log grows). Follows the disclosure structure required by the course guidelines: tool used, purpose, how the generated output was used.

- **Statement of authorship:** the student reviewed, validated, and approved all AI-generated contributions and assumes responsibility for and authorship of the submitted work. *(source: course example statement, docs/guidelines.md §4)*
- **Assistance from other persons:** [to confirm at submission — no assistance beyond course instruction is recorded in the vault].
- **Generative AI use — full disclosure:**

| # | Date | Tool | Purpose | How the output was used |
|---|------|------|---------|-------------------------|
| 1 | 2026-10-25 | OpenCode agent (GLM, Z.ai) | Project planning: read guidelines + codebook, framed the binary target, designed pipeline, agent team, repo and vault structure | Plan reviewed and approved by the student; structure adopted for the project |
| 2 | 2026-10-25 | OpenCode agent (GLM, Z.ai) | Repository scaffolding: `.gitignore`, README, AGENTS.md rules, vault seed content, `src/config.py` | Files reviewed and committed as the initial project setup |
| 3 | 2026-10-25 | OpenCode agent (GLM, Z.ai) | Interpreting the BRFSS codebook structure (missing-value codes, section layout, GENHLTH distribution) | Used to draft the preprocessing rules and leakage watchlist; to be verified against the data in EXP-000 |

- **Responsibility note:** all AI suggestions were personally reviewed and validated before being adapted and incorporated; the student remains the author of the work. *(source: docs/guidelines.md §4; AI Assistance Log)*

---

*End of skeleton. Every [DRAFT]/placeholder above must be replaced by content traceable to `results/experiments_log.csv` or an approved vault note before the report is marked final.*
