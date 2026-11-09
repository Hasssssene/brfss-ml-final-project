# Gamma deck prompt (v2) — paste into Gamma with the report attached

Create a clean, minimal, dark-text-on-light academic presentation deck (8 slides) for a university machine-learning course screencast. Plain professional style, no stock photos, generous whitespace, one idea per slide. Title: "Predicting Fair/Poor General Health — CDC BRFSS 2021". Subtitle: "Hassene Bouamama · Machine Learning Final Project". Use EXACTLY these numbers, no others invented:

**Slide 1 — Title.** "Predicting Fair/Poor General Health — CDC BRFSS 2021". 438,693 respondents × 303 columns. Binary target: Fair/Poor = 72,741 (16.63%, ~1 in 6). Metric: ROC-AUC · stratified 10-fold CV · seed 42. [image slot: bar chart of 5 GENHLTH classes, Fair+Poor highlighted]

**Slide 2 — Leakage first.** `_RFHLTH` agrees with the target on 100.00% of 437,532 rows → dropped, not trained on. Triage funnel, one verdict per column: 4 leaks · 35 admin · 11 constant · 85 sparse · 27 low-coverage · 65 duplicates → **76 kept**. Whitelist, never a drop-list.

**Slide 3 — Codebook traps.** 88 = "zero days" (287,796 answers on PHYSHLTH) — real data, not missing. 7 can mean *Retired*. Sentinel recoding per-column from the codebook. NaN kept; imputation inside CV only. 54 nominal columns one-hot → 322 design columns.

**Slide 4 — Validation discipline.** Same `StratifiedKFold(10, shuffle=True, random_state=42)` for every model. Every run logged: `experiments.csv`, ten fold AUCs per row. Baselines re-run 3× → bit-identical scores.

**Slide 5 — Results.** Dummy 0.500 · LogReg 0.8935 (C-grid flat) · single tree 0.698 → 0.881 pruned · RF 0.895 · HistGB / LightGBM / XGBoost 0.9033–0.9034 (+0.010 over linear; ≤0.0002 apart). Imbalance: weights +0.0005/−0.0002 (wash) · SMOTE −0.0040 (hurts) → none used; AUC is ranking. [image slot: model zoo bar chart]

**Slide 6 — Selection + tuning + statistics.** Driver-block hypothesis: 39 of 76 columns keep 99.5% of AUC (0.8988); every selection route loses → keep all 76. Tuned LightGBM **0.903672** vs tuned XGBoost 0.903640 — Wilcoxon p = 1.000: **an honest tie** → pick LightGBM for speed. Tuning gain +0.000356 (p 0.0059); family gain +0.0103 — **29× more**. Nested CV: same config re-selected in all 10 outer folds → optimism 0.000000.

**Slide 7 — Errors + explainability.** 48,218 OOF errors (11.0%); **88.1% on the Good/Fair boundary** (subjective one-step self-report). Case A: p=0.96, says Good — 30 sick days, BMI 51 (label noise). Case B: p=0.002, says Fair — profile cleaner than medians (feature-set error floor). Case C: p exactly 0.5000 (evidence cancels). SHAP: PHYSHLTH 0.736 ≈ 4× runner-up (diabetes 0.195, BMI 0.187). [image slot: SHAP global importance]

**Slide 8 — Limits + close.** Self-reported target sets the ceiling; scores ranked, not calibrated; SEX/RACE associations need fairness review before any deployment; one documented wart (FLSHTMY3 sentinel). Full AI-assistance disclosure in report §12. "Every number cites a row in experiments.csv."

## Then drop in real figures
- Slide 1: `figures/fig01_target_imbalance.png`
- Slide 5: `figures/fig04_model_zoo.png`
- Slide 7: `figures/fig07_shap_global.png` (and/or `fig06_error_taxonomy.png`)
