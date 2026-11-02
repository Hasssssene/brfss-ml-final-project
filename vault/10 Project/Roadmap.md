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

## P4 — Model zoo + imbalance (Intermediate) ✅ (2026-11-02)
- ✅ EXP-004 RandomForest, EXP-005 HistGradientBoosting, EXP-006 LightGBM, EXP-007 XGBoost *(boosters ≈ **0.9034**: XGB 0.903357 ≥ LGBM 0.903355 ≥ HistGB 0.903220 — statistically tied (spread ≤ 0.00014 ≈ 0.08× fold std); RF 0.893315 ≈ LogReg 0.892588; booster gain over linear **+0.0108 = 6× fold std**; boosters ran imputation-free via verified NaN-passthrough; total 25.8 min CV wall; [[EXP-004 random forest]]–[[EXP-007 xgboost]], `results/p4_summary.md`)*
- ✅ Imbalance: class_weight='balanced' vs SMOTE(fold-internal) vs none — EXP-008..010 *(verdict: **none for AUC** — XGB Δ: class weight −0.00010 / SMOTE −0.00355 / RUS −0.00026; LogReg Δ: +0.00045 / −0.00417 / +0.00018; rationale: ROC-AUC is threshold-free & rank-based, prior-reshaping doesn't improve ranking; [[EXP-008 class weighting]], [[EXP-009 smote]], [[EXP-010 random undersampler]])*

## P5 — Feature selection & hypothesis ✅ (2026-11-02, Intermediate evidenced)
- ✅ Hypothesis note: expected driver blocks (disability, health days, chronic conditions, age) *(H1 — five blocks B1–B5 — stated **verbatim** in [[EXP-012 hypothesis H1 subsets]]; verdict **nearly confirmed**: 36/37 cols retain **99.2% of lift** (−0.0033 AUC) but 0/10 folds favour the subset → not strictly within fold noise (~1.9σ))*
- ✅ SelectKBest + RFECV/permutation importance; selected-subset vs all-features — EXP-011.. *([[EXP-011 selectkbest vs gain importance]]: kbest below all-76 at every k — best k=200 → 0.899670 vs 0.903355, MI∩gain overlap only 3/10 @top-10; adopted fold-internal **gain importances** instead: PHYSHLTH 51.7% of gain, B2 block 55.0%; [[EXP-012 hypothesis H1 subsets]]: H1-subset 0.900051/0.900316 vs baselines, complement standalone 0.811844 but adds only ~+0.003 incrementally; keep all-76)*
- ✅ PCA variant — EXP-0xx *([[EXP-013 pca]]: n=50/100/200 → 0.881126/0.886203/0.888800, **all ≥0.0146 below all-76**; honest negative — compression smears one-hot structure, forfeits NaN routing)*

## P6 — Tuning & statistics ✅ (2026-11-02, Intermediate stats + Advanced tuning evidenced)
- ✅ RandomizedSearchCV on top-2 models (fixed folds) *(budget levers applied & documented after the timing probe projected a ~110-min XGB worst case > the 90-min budget: n_iter 20→15, n_estimators [300,800]→[300,600]; actual wall **53.3 min**; search winners → **tuned LGBM 0.904321 = final best** (+0.000966 vs EXP-006, 10/10 folds), tuned XGB 0.903843 (+0.000486, 9/10); [[EXP-014 randomized search tuned finals]])*
- ✅ Nested CV unbiased estimate *(EXP-015: **0.904303 ± 0.001700** — optimism of the non-nested estimate just **+0.000018**, search width does not overfit; 7/10 outer folds re-selected the Part-1 winner; [[EXP-015 nested CV]])*
- ✅ Wilcoxon signed-rank on 10 fold scores, top-2 models *(**3/3 significant at the n=10 floor p=0.001953**: tuned LGBM > tuned XGB (median Δ +0.000471), > untuned self (+0.000991), > LogReg (+0.011768) — headline: family gain +0.012 ≫ tuning gain +0.001, both real; [[STATISTICAL COMPARISONS]])*

## P7 — Error analysis & XAI ✅ (2026-11-02, Intermediate complete)
- ✅ ≥3 consistently misclassified respondents, profiled, with figures *(3 archetype cases, each wrong in **3/3** model families; **88.2% of all errors (42,449/48,124) sit at the GENHLTH 3/4 boundary** — boundary ambiguity, not noise; 9.4% of rows defeat all three learners. Cases: A label-noise FP (p 0.96/0.98/0.99, PHYSHLTH 30d SHAP +1.94, multimorbid, BMI 51.5), B error-floor FN (p 0.0025, uniformly negative SHAP — unmeasured rating anchors), C fence case (p 0.5000022, comorbid block vs clean current status). `results/analysis/error_analysis.md` + `figures/p7_case_profiles.png` + 3 waterfalls; [[ERROR ANALYSIS AND XAI]])*
- ✅ SHAP global summary + local explanations on best model *(TreeExplainer on the EXP-014a final-model proxy, additivity 8.8e-14, dummy-group aggregation back to 76 columns; top-5 mean|SHAP|: PHYSHLTH 0.681, DIFFWALK 0.216, DIABETE4 0.198, _BMI5 0.185, EDUCA 0.179; **SHAP-vs-gain ρ=0.911 (p=3.6e-30) — confirms the P5/H1 importances**; global beeswarm + 3 local waterfalls; bonus threshold menu τ=0.2886→TPR 0.70/FPR 0.093. Zero log rows by design — OOF is a deterministic consequence of logged EXP-014a/014b/EXP-002a, per-fold AUCs re-verified ≤4.8e-7; [[ERROR ANALYSIS AND XAI]])*

## P8 — Deliverables ◐ *(all agent work done 2026-11-02; remaining items are human-only)*
- ✅ Assemble `deliverables/code.py` (end-to-end, clean process) *(903-line self-contained single file: `--smoke` 20.3 s / `--full` 76.1 s; per-fold deltas vs logged EXP-014a ≤ 4.84e-07, folds-sha prefix `3b3a17457039` asserted, deterministic ×2 runs; one porting bug caught by `--smoke` — pandas label-vs-position — fixed via `.to_numpy(dtype=np.int8)` matching `src/experiments.py:load_xy`; verification artifacts `results/final_run_summary.txt` + `final_run_fold_scores.csv`; final model frozen by [[ADR-006 final model and deliverable freeze]])*
- ✅ `report.pdf` full draft through QA audit *(v1.0: 28 pp A4, 16 numbered tables, 14 figures, 11 sections incl. §11 Ethics matching the 12-entry AI log; built via `build_report.py` → pandoc → HTML → Edge-headless PDF; QA remediation applied — fold-SHA coverage wording corrected in 2 places (41/41 model rows; EXP-000 profiling has no folds), text-layer verified; **QA audit 2026-11-02: OVERALL PASS** — independent sandbox reproduction of `--full` fold AUCs ≤4.84e-7 + the confusion matrix; screencast script `deliverables/screencast_script.md` (6 beats, ≤10:00) drafted alongside)*
- ◐ Screencast script + recording (≤10 min) *(script ✅ — recording is a human action; ⚠ beat timestamps span 0:00–10:00 while the header targets 9:50 — trim ~10 s from Beat 3 if a hard buffer is wanted)*
- ☐ Submission form + self-assessment *(human-only: [Student name] at `report.md:12,414,416`, form fields, self-assessment)*
