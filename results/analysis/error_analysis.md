# P7b — Error analysis & failure-mode interpretation (final model = tuned LightGBM, EXP-014a)

**Author:** data-analyst · **Date:** 2026-11-02 · **Companion to:** `results/p7_summary.md`
**Purpose:** the interpretive error-analysis note for the report. Every number below is either taken verbatim from a P7 artifact (cited inline) or computed from P7 artifacts by `src/analysis/p7b_cases.py` (also cited inline; script prints every number it produces). No new model was fit, no randomness was used, and no `experiments_log.csv` row is needed — all quantities are deterministic consequences of the already-logged runs **EXP-014a** (tuned LGBM, mean CV AUC 0.904321), **EXP-014b** (tuned XGB, 0.9038429) and **EXP-002a** (LogReg, 0.8925882), per `results/p7/oof_provenance.json`.

**Artifacts used:** `results/p7/error_taxonomy.csv`, `results/p7/selected_cases.json`, `results/p7/shap_cases.parquet`, `results/p7/shap_global_importance.csv`, `results/p7/shap_meta.json`, `results/p7/error_candidates.csv`, `data/interim/oof_predictions.parquet`; context from `results/analysis/eda_notes.md` and `results/p4_summary.md`.

**New figure (this note):** `figures/p7_case_profiles.png` (dpi 150) — three panels (Cases A/B/C), horizontal bars = population z-scores of the case's raw values on the top-12 mean|SHAP| features, bar color = sign of that feature's SHAP value for that case (red = pushes toward y=1 Fair/Poor, blue = toward y=0 Excellent/Good), each bar annotated with the raw value and its SHAP contribution, each panel header carrying the case's GENHLTH and the three models' OOF probabilities.

---

## 1. Error structure — boundary ambiguity, not noise

**FACT** (source: `results/p7/error_taxonomy.csv`, tuned LGBM OOF at threshold 0.5, n = 437,532):

| quantity | value |
|---|---|
| TP / FP / TN / FN | 38,957 / 14,340 / 350,451 / 33,784 |
| FPR (FP / 364,791 negatives) | **0.0393** (3.93 %) |
| FNR (FN / 72,741 positives) | **0.4644** (46.44 %) |
| TPR / TNR | 0.5356 / 0.9607 |
| error rate | 0.1100 (48,124 / 437,532) |
| prevalence | 0.1663 (16.63 %) |

**FACT** (same file, errors decomposed by *raw* GENHLTH class):

| GENHLTH class | n | error type | count | rate within class |
|---|---|---|---|---|
| 1 (Excellent) | 77,741 | FP | 709 | 0.0091 |
| 2 (Very good) | 149,112 | FP | 2,430 | 0.0163 |
| 3 (Good) | 137,938 | FP | 11,201 | **0.0812** |
| 4 (Fair) | 54,736 | FN | 31,248 | **0.5709** |
| 5 (Poor) | 18,005 | FN | 2,536 | 0.1408 |

- The FP rate climbs monotonically as self-rated health worsens within the negative class: 0.91 % → 1.63 % → **8.12 %** for classes 1/2/3; the FN rate falls as health worsens: **57.09 %** (class 4) → 14.08 % (class 5). Class-4 TPR is only 0.429 (23,488 of 54,736 caught); class-5 TPR is 0.859.
- 42,449 of the 48,124 errors (88.2 %) sit exactly at the 3/4 boundary (class-3 FPs 11,201 + class-4 FNs 31,248).
- 41,091 instances (9.39 % of all rows; source: `results/p7/error_taxonomy.csv`, `wrong_in_k_models=3`) are misclassified by **all three model families** (tuned LGBM EXP-014a, tuned XGB EXP-014b, LogReg EXP-002a). 86.91 % of rows are wrong in none of the three.
- Of the 50 most confident shared errors (`results/p7/error_candidates.csv`), 47 have GENHLTH = 4 and 3 have GENHLTH = 5.

**INTERPRETATION (hypothesis):** the error mass is *structured*, not stochastic. If errors were noise, they would spread evenly across classes and models; instead they concentrate (i) at one specific ordinal boundary and (ii) in instances that defeat three architecturally different learners simultaneously — the signature of ambiguity in the label itself rather than model capacity. Class 4 ("Fair") is plausibly the hardest positive because it is a heterogeneous catch-all adjacent to "Good": it mixes people pulled down from 3 by early chronic disease with people pulled up from 5 by adaptation, and self-report of "fair" depends on the respondent's internal reference frame (expectations, coping, response shift), which the feature matrix cannot observe. The symmetric pattern — class-3 FP rate 8.12 % climbing toward class 4, class-4 FN rate 57.09 % falling toward class 5 — is exactly what an ordinal variable's fuzzy middle boundary looks like when binarized at 3|4: the 3s nearest the boundary are the easiest negatives to misflag, and the 4s nearest the boundary are the hardest positives to catch. The three-family agreement (9.4 % wrong everywhere) further suggests a shared-information limit: all three models see the same 76 columns, so they inherit the same blind spots.

---

## 2. Case A (row 307041, fold 4) — the multimorbid respondent who rates themselves "Good"

**FACT** (source: `results/p7/selected_cases.json`; SHAP values from `results/p7/shap_cases.parquet`; figure: `figures/p7_case_profiles.png` left panel; waterfall: `figures/p7_case_A_waterfall.png`):

- Selection rule: most confident FALSE POSITIVE — the highest `oof_lgbm` among the 14,340 FPs. y = 0, GENHLTH = 3 ("Good"). Wrong in 3/3 models.
- OOF P(y=1): LGBM **0.9595**, XGB **0.9815**, LR **0.9930**.
- Symptom burden: **PHYSHLTH = 30 days** (z +3.14; SHAP **+1.94**, by far the largest single contribution in any of the three cases), **POORHLTH = 30 days** (z +2.75; SHAP +0.38), MENTHLTH = 5 days (z +0.11; SHAP +0.03).
- Multimorbidity cluster, all "yes": DIFFWALK = 1 (difficulty walking; z −2.28; SHAP +0.47), DIABETE4 = 1 (z −2.49; SHAP +0.33), CVDINFR4 = 1 heart attack (z −4.26; SHAP +0.17), CVDCRHD4 = 1 coronary disease (z −4.24; SHAP +0.37), CVDSTRK3 = 1 stroke (z −4.95; SHAP +0.17), CHCKDNY2 = 1 kidney disease (z −4.96; SHAP +0.36), HAVARTH5 = 1 arthritis (z −1.43; SHAP +0.003). (For these 1/2-coded items the *negative* z means the condition is present — see figure caption.)
- Risk context: `_BMI5` = 51.45 kg/m² (z +3.50; SHAP +0.34), INCOME3 = 1/11, the lowest bracket (z −2.35; SHAP +0.13), retired (EMPLOY1 = 7, z +1.15), age 59 (z +0.25), college graduate (EDUCA 6, z +0.98 — SHAP −0.08, one of the only pushes toward 0).
- Net decomposition (computed by `src/analysis/p7b_cases.py` from `shap_cases.parquet`): base −2.6095 + ΣSHAP **+5.64** = log-odds +3.03 → refit-proxy P(y=1) = 0.954.

**INTERPRETATION (hypothesis):** this is a plausible *label-noise* instance rather than a model failure. The respondent reports the maximum symptom load on both day-count items (30 of 30 days), six chronic conditions, class-III obesity and lowest-bracket income — a profile that in this population is strongly associated with Fair/Poor — yet chooses "Good". Candidate explanations, all unobservable in the feature matrix: (i) *response shift / adaptation* — people with long-standing multimorbidity often recalibrate "health" against their own past rather than against a population norm; (ii) *response style* — stoic or optimistic self-presentation; (iii) *survey artifact* — mode/proxy-respondent effects. On any of these accounts the model's output is defensible: it is faithfully reporting that this vector of measured health is overwhelmingly Fair/Poor-like. Case A also illustrates why class-3 FP rate is 8.12 %: the extreme of that distribution looks like this. We cannot adjudicate the label from the data; we can only locate the disagreement (measured-health markers vs the self-rating) precisely.

---

## 3. Case B (row 392245, fold 9) — the idiosyncratic "Fair" rater with a clean profile

**FACT** (source: `results/p7/selected_cases.json`; `results/p7/shap_cases.parquet`; figure: `figures/p7_case_profiles.png` middle panel; waterfall: `figures/p7_case_B_waterfall.png`):

- Selection rule: most confident FALSE NEGATIVE — the lowest `oof_lgbm` among the 33,784 FNs. y = 1, GENHLTH = 4 ("Fair"). Wrong in 3/3 models.
- OOF P(y=1): LGBM **0.0025**, XGB **0.0027**, LR **0.0156**.
- Clean profile on every top feature: PHYSHLTH = **0 days** (z −0.45; SHAP **−0.61**, the largest single push toward 0), MENTHLTH = 0 days (z −0.50; SHAP −0.09), POORHLTH = **missing** (SHAP −0.12 via the NaN category), DIFFWALK = 2 (no difficulty; SHAP −0.17), DIABETE4 = 3 (no diabetes; SHAP −0.13), `_BMI5` = **22.14** kg/m² (z −0.98; SHAP −0.22), INCOME3 = 9/11 ($100–150k bracket; z +0.95; SHAP −0.28), EDUCA = 6 (college graduate; z +0.98; SHAP −0.21), EXERANY2 = 1 (exercises; SHAP −0.11), BPHIGH6 = 3 (told borderline; SHAP −0.08); no cardiovascular, COPD, cancer or kidney conditions (all = 2); employed, married, female, age 59.
- **Every one of the top-12 SHAP features contributes negatively** (middle panel of `figures/p7_case_profiles.png` is uniformly blue). Net decomposition (`src/analysis/p7b_cases.py`): base −2.6095 + ΣSHAP **−3.37** = log-odds −5.98 → refit-proxy P(y=1) = 0.0025.
- This is not an isolated oddball. Among the 50 most confident shared errors (`results/p7/error_candidates.csv`, profiled by `src/analysis/p7b_cases.py`): PHYSHLTH ≤ 1 day in **50/50** (exactly 0 in 47/50), DIFFWALK = 2 in **50/50**, DIABETE4 = 3 in **50/50**, EXERANY2 = 1 in 49/50, BMI mean 25.62 (range 19.20–33.89), INCOME3 ≥ 7 ($50k+) in 46/50 (median bracket 9/11), EDUCA 5–6 in 50/50, MENTHLTH = 0 in 36/50, POORHLTH missing in 33/50; GENHLTH = 4 in 47/50.
- EDA cross-reference (`results/analysis/eda_notes.md` §5): class-0 vs class-1 means are PHYSHLTH 1.9 vs 13.8 days, DIFFWALK prevalence 8.9 % vs 47.8 %, income median bracket 7 vs 5 — Case B sits at or beyond the *y=0* side of every one of these markers.

**INTERPRETATION (hypothesis):** Case B is the archetype of *irreducible* error for this feature set — a self-rating anchored on constructs the survey's health items do not measure: mental wellbeing beyond the MENTHLTH day count (which is 0 here), expectations, personality, recent life events, or social comparison. The uniformly blue panel is the visual statement of the mechanism: nothing in the model's information set pushes this person toward "Fair", so no model trained on these 76 columns could rank them highly without paying for it elsewhere. The 50-row subtype profile suggests this is a systematic respondent type, not one stray record — and its composition (47/50 GENHLTH=4) again implicates the "Fair" label as the noisy edge of the target. For the report: this is the error floor of the feature set, and the honest framing is "unmeasured rating anchors", not "model weakness".

---

## 4. Case C (row 208256, fold 4) — the ambiguity archetype at p = 0.5000

**FACT** (source: `results/p7/selected_cases.json`; `results/p7/shap_cases.parquet`; figure: `figures/p7_case_profiles.png` right panel; waterfall: `figures/p7_case_C_waterfall.png`):

- Selection rule: shared error (wrong in 3/3 models) with GENHLTH ∈ {3,4} closest to the fence — minimal |oof_lgbm − 0.5|. y = 0, GENHLTH = 3 ("Good").
- OOF P(y=1): LGBM **0.5000** (artifact value 0.5000022), XGB 0.598, LR 0.563. Refit-proxy decomposition (`src/analysis/p7b_cases.py`): base −2.6095 + ΣSHAP **+2.53** = log-odds −0.08 → P = 0.481. Both the CV model and the refit proxy straddle 0.5 from opposite sides.
- Chronic-disease burden pushing toward y=1: DIFFWALK = 1 (z −2.28; SHAP **+0.60**), DIABETE4 = 1 (z −2.49; SHAP **+0.56**), CVDINFR4 = 1 heart attack (z −4.26; SHAP +0.36), CVDSTRK3 = 1 stroke (z −4.95; SHAP +0.30), `_BMI5` = 33.91 (z +0.82; SHAP +0.10), INCOME3 = 2/11, near-lowest bracket (z −1.94; SHAP +0.23), EMPLOY1 = 8 "unable to work" (z +1.50; SHAP +0.34), age 64 (z +0.53), rural (`_URBSTAT` = 2, z +2.44).
- Current-status anchors pushing toward y=0 (or neutral): PHYSHLTH = 2 days (z −0.21; SHAP **−0.02**), POORHLTH = 0 days (z −0.58; SHAP −0.03), MENTHLTH = 2 days (SHAP −0.004), EDUCA = 5 (SHAP −0.01).
- The panel is nearly half red (high-|SHAP| comorbidity items) and half pale (small-magnitude negatives): two blocks offsetting.

**INTERPRETATION (hypothesis):** Case C is what the 3/4 boundary *is*. The feature vector contains two coherent narratives — a heavy chronic-disease/SES profile (diabetes, heart attack, stroke, walking difficulty, unable to work, low income) and a currently symptom-free surface (2 poor-health days, 0 activity-limitation days) — and the model splits the difference exactly. The 8.12 % FP rate among class-3 raters and the 57.09 % FN rate among class-4 raters (§1) imply substantial overlapping densities of GENHLTH=3 and GENHLTH=4 respondents in precisely this covariate region, so a GENHLTH=4 rater with a near-identical vector plausibly exists in the data; distinguishing them would require knowing *how each individual weighs disease history against current function*, which is not in the columns. Unlike Case A (label arguably wrong) and Case B (rating anchored off-grid), Case C is best described as *aleatoric*: the population itself mixes 3s and 4s here, and a probability of 0.5000 is the epistemically correct output, not a malfunction. The binary-error framing of §1 counts it as an error only because thresholding at 0.5 must break exact ties somehow (this row's OOF sits above 0.5 by 2.2e-06).

---

## 5. Threshold and decision framing — FNR 46 % vs FPR 4 % is expected geometry, not a defect

**FACT** (sources: `results/p7/error_taxonomy.csv`; `results/p4_summary.md`; sweep computed by `src/analysis/p7b_cases.py` from `data/interim/oof_predictions.parquet`, provenance `results/p7/oof_provenance.json`):

- At threshold 0.5: FPR 0.0393, FNR 0.4644, precision 0.7309, 53,297 rows flagged (12.2 % of 437,532). The model was tuned and selected on **ROC-AUC** (EXP-014a mean CV AUC 0.904321; nested check EXP-015 0.904303) with **no class weighting or resampling**, at prevalence 16.63 % (1 : 5.0).
- P4 imbalance results on the identical folds (`results/p4_summary.md`): vs default-weight XGB, class weighting ΔAUC −0.00010 (EXP-008a), SMOTE(0.5) ΔAUC −0.00355 (EXP-009a), RUS(0.5) ΔAUC −0.00026 (EXP-010a); for LogReg, class weighting +0.00045 (EXP-008b), RUS +0.00018 (EXP-010b).
- OOF threshold sweep (tuned LGBM; smallest threshold attaining each TPR target):

| τ | TPR | FPR | precision | flagged | share of population |
|---|---|---|---|---|---|
| 0.5000 | 0.536 | 0.039 | 0.731 | 53,297 | 12.2 % |
| 0.4172 | 0.600 | 0.055 | 0.683 | 63,870 | 14.6 % |
| 0.2886 | 0.700 | 0.093 | 0.600 | 84,808 | 19.4 % |
| 0.1771 | 0.800 | 0.158 | 0.502 | 115,839 | 26.5 % |
| 0.0878 | 0.900 | 0.283 | 0.388 | 168,803 | 38.6 % |

**INTERPRETATION (hypothesis):** the 46 %-vs-4 % asymmetry is the *expected* operating point of an unweighted, AUC-selected model read at an arbitrary threshold. AUC optimizes the ranking of positives above negatives across all thresholds; it says nothing about where the calibrated probability mass sits relative to 0.5. With a 1 : 5 positive prior and log-loss-driven training, the natural cut lands where it is cheap to be wrong on the abundant class — few FPs, many FNs. This also explains the P4 finding that resampling did not help AUC: ROC-AUC is threshold-free and prior-insensitive *in ranking terms*, so rebalancing the training set mostly reshuffles where probabilities cross any fixed cut rather than improving separation; hence near-zero (or negative) ΔAUC for EXP-008a/009a/010a while the 0.5-threshold error mix would change substantially. Practical implication *if this model were deployed for screening* (it is not — this is a course project): 0.5 is the wrong knob. A recall-oriented triage policy would read the ROC/PR trade-off explicitly — e.g., τ = 0.2886 buys TPR 0.70 at FPR 0.093 (19.4 % of the population flagged, precision 0.60), and TPR 0.80 costs FPR 0.158 (26.5 % flagged, precision 0.50). The right τ depends on the cost ratio of missed Fair/Poor vs unnecessary follow-up, which only a use case can supply; the model delivers a usable ranking (AUC 0.904, EXP-014a) and the sweep above is the menu.

---

## 6. XAI synthesis — two attribution methods agree; the case stories are the model's story

**FACT** (sources: `results/p7/shap_global_importance.csv`; `results/p7/shap_meta.json`; `results/p7_summary.md`):

- Top-5 by mean|SHAP| (log-odds units, 20,000-row stratified sample, TreeExplainer path-dependent, contributions summed back to the 76 original columns): **PHYSHLTH 0.6813, DIFFWALK 0.2164, DIABETE4 0.1981, `_BMI5` 0.1855, EDUCA 0.1786**. PHYSHLTH is 3.1× the second feature (0.6813 / 0.2164) and alone carries 51.69 % of total split gain.
- SHAP-vs-gain rank agreement: Spearman **ρ = 0.911** (p = 3.6e-30, n = 76) between the tuned model's SHAP ordering and the untuned EXP-006 split-gain ordering (`figures/p7_shap_vs_gain.png`). Divergences are typically collinear features whose gain concentrates in one tree path while SHAP shares credit (`results/p7_summary.md`).
- Additivity asserted: max |base + ΣSHAP − raw score| = 8.8e-14 on the 20k sample and re-asserted per case; base value (log-odds) = −2.6095 (`results/p7/shap_meta.json`).
- Case-level totals (`src/analysis/p7b_cases.py` from `shap_cases.parquet`): A ΣSHAP +5.64 (PHYSHLTH +1.94 is the single largest contribution — about 34 % of the net push); B ΣSHAP −3.37 with all top-12 features negative; C ΣSHAP +2.53 split between a comorbidity block (+0.60/+0.56/+0.36/+0.30) and near-zero current-status anchors. Global picture: `figures/p7_shap_summary.png` (beeswarm, top 20).

**INTERPRETATION (hypothesis):** split gain and SHAP are two attribution mechanisms with different failure modes (gain double-counts correlated features across trees; path-dependent SHAP spreads credit along tree paths), so ρ = 0.911 across 76 features is strong evidence that the *ordering* — current physical-health status first (PHYSHLTH), functional limitation second (DIFFWALK), chronic-disease and socio-demographic context after — is a property of the data, not an artifact of one explanation method. It also reproduces the EDA's univariate ranking (`results/analysis/eda_notes.md` §3: PHYSHLTH ρ = +0.450, DIFFWALK ρ = −0.413 as the top two), so model, attributions, and raw associations all tell one story. Reading the waterfalls (`figures/p7_case_{A,B,C}_waterfall.png`) makes the three failure modes concrete: in **A**, PHYSHLTH's +1.94 does most of the work of dragging the log-odds from the −2.61 base to +3.03 — the model is doing exactly what it learned on the population level, and the "error" is the label disagreeing with its own symptom answers; in **B**, *there is nothing to read* — no feature pushes toward 1, which *is* the explanation of the miss (the rater's "Fair" is off the model's grid); in **C** the waterfall is visibly two-tailed, the algebraic image of a respondent sitting on the population's 3/4 boundary. For the report's ethics section it is worth noting the practical reading of the global ranking: the model leans on self-reported current symptoms, so it inherits self-report bias; that is a feature of GENHLTH as a target, not of LightGBM.

---

## 7. Caveats, contradictions, and what this note adds

- The dispatch brief's numbers all check out against the artifacts, with three refinements: (i) Case C's OOF probability is 0.5000 **to four decimals** but not literally 0.5 (0.5000022) — and its refit-proxy probability is 0.481, a different object (full-data refit vs CV model); the figure states this distinction in its caption. (ii) Case B's "income ≥ $75k" is specifically INCOME3 = 9/11 ($100–150k bracket under the 2021 codebook). (iii) Case C's comorbidity is larger than the brief's "DIFFWALK + DIABETE4 + CVDINFR4": stroke (CVDSTRK3 = 1) and EMPLOY1 = 8 (unable to work) are also present.
- Case B's POORHLTH is missing, so its z-bar is undefined (annotated in the figure); the model still assigns SHAP −0.12 through the NaN one-hot category (nanpass preprocessing, P4).
- The OOF probabilities (CV models, `data/interim/oof_predictions.parquet`) and the SHAP values (full-data refit proxy, `results/p7/shap_cases.parquet`) come from different fitted objects by construction; per-case agreement is close (A: 0.9595 OOF vs 0.954 refit; B: 0.0025 vs 0.0025) but they are not the same number and are never mixed in this note.
- New computations introduced by this note (threshold sweep §5; 50-row subtype profile §3; ΣSHAP totals; PHYSHLTH/DIFFWALK importance ratio) are all produced by `src/analysis/p7b_cases.py` from existing P7 artifacts, printed to stdout for verification, and involve no model fitting, no randomness and no new folds — hence no `experiments_log.csv` row (consistent with the P7 provenance stance in `results/p7/oof_provenance.json`).
- Known limitation of the explanations themselves: path-dependent TreeExplainer spreads credit across correlated one-hot dummies (recorded in `results/p7/shap_meta.json`); per-case attributions are therefore indicative of *which blocks* drive a prediction, not precise per-feature credits.

*Figures referenced: `figures/p7_case_profiles.png` (new, this note), `figures/p7_case_A_waterfall.png`, `figures/p7_case_B_waterfall.png`, `figures/p7_case_C_waterfall.png`, `figures/p7_shap_summary.png`, `figures/p7_shap_vs_gain.png` (all dpi 150).*
