# ERROR ANALYSIS AND XAI — failure modes of the final model (analysis note) #exp

> [!warning] This is an **analysis note, not an experiment.** It has **no EXP id and no `experiments_log.csv` row by design** — every quantity is a deterministic consequence of *already-logged* configurations: the OOF predictions come from re-fitting **EXP-014a** (tuned LGBM), **EXP-014b** (tuned XGB) and **EXP-002a** (LogReg) on the canonical folds, and the SHAP values come from a single full-data refit of the EXP-014a config used as the **final-model proxy** (its graded CV estimate remains EXP-014a 0.904321 / nested EXP-015 0.904303). Inventing a log row for a deterministic re-derivation would violate the one-row-per-run log contract. Provenance recorded in `results/p7/oof_provenance.json`.

- **date:** 2026-11-02
- **phase:** P7
- **status:** done
- **workers:** ml-engineer (P7a computation, `src/p7_error_xai.py`) + data-analyst (P7b interpretation, `src/analysis/p7b_cases.py`)
- **feeding rows:** [[EXP-014 randomized search tuned finals]] (EXP-014a, EXP-014b), [[EXP-002 logistic regression]] (EXP-002a); gain-ordering context from [[EXP-011 selectkbest vs gain importance]] (EXP-011b, replicating [[EXP-006 lightgbm]])
- **primary artifacts:** `results/p7_summary.md`, `results/p7/` (error_taxonomy.csv, error_candidates.csv, selected_cases.json, shap_global_importance.csv, shap_cases.parquet, shap_meta.json, oof_provenance.json, timing.json), `data/interim/oof_predictions.parquet` (gitignored checkpoint)
- **interpretation note (authoritative for report prose):** `results/analysis/error_analysis.md` — 7 sections, FACT/INTERPRETATION discipline

## Provenance stance (why zero log rows is correct)

The canonical folds (`StratifiedKFold(10, shuffle=True, random_state=42)` via `src.config.make_folds()`; `folds_sha256=3b3a17457039` asserted) and the logged params fully determine the OOF predictions — re-running is a re-derivation, not a new comparison. Verified, not assumed: **per-fold OOF AUCs recomputed from the saved predictions match the logged fold scores to ≤ 4.8e-07 max abs deviation for all three models** (LGBM 4.84e-07, XGB 4.74e-07, LogReg 4.14e-07; `matches_logged_run: true` × 3). The full-data SHAP refit is a *different stated object* from the CV models and is never mixed with OOF probabilities (per-case agreement is close but stated separately: A 0.9595 OOF vs 0.954 proxy; B 0.0025 vs 0.0025). P7b's new computations (threshold sweep, 50-row subtype profile, ΣSHAP totals) involve no fitting, no randomness, no new folds — same stance.

## Facts (P7a, tuned LGBM OOF @ threshold 0.5, n = 437,532)

**Error taxonomy** (`results/p7/error_taxonomy.csv`): TP/FP/TN/FN = 38,957 / 14,340 / 350,451 / 33,784 → **FPR 3.93%, FNR 46.44%**, TPR 0.5356, TNR 0.9607, error rate 11.00% (48,124 errors), prevalence 16.63%.

**The 3/4 boundary story** (errors decomposed by raw GENHLTH class):

| GENHLTH class | n | error | rate |
|---|---|---|---|
| 1 (Excellent) | 77,741 | 709 FP | 0.0091 |
| 2 (Very good) | 149,112 | 2,430 FP | 0.0163 |
| 3 (Good) | 137,938 | 11,201 FP | **0.0812** |
| 4 (Fair) | 54,736 | 31,248 FN | **0.5709** |
| 5 (Poor) | 18,005 | 2,536 FN | 0.1408 |

- FP rate climbs monotonically toward the boundary (0.9% → 1.6% → 8.1%); FN rate falls as health worsens (57.1% → 14.1%). **42,449 of 48,124 errors (88.2%) sit exactly at the 3/4 boundary** (class-3 FPs + class-4 FNs).
- **Cross-model wrongness:** 41,091 rows (**9.4%** of all) are misclassified by all three model families simultaneously (LGBM ∪ XGB ∪ LR @ 0.5); 86.9% are wrong in none. Of the 50 most confident shared errors (`error_candidates.csv`), 47 are GENHLTH=4, 3 are GENHLTH=5.

**Three archetype cases** (full vectors + z-scores in `selected_cases.json`; all wrong in 3/3 models):

- **A — most confident FP** (row 307041, fold 4): y=0 (GENHLTH=3 "Good") but P(y=1) = 0.9595 / 0.9815 / 0.9930 (LGBM/XGB/LR). PHYSHLTH = 30 days (z +3.14; SHAP **+1.94**, largest single contribution in any case), POORHLTH = 30 days, DIFFWALK = 1, six named chronic conditions (diabetes, heart attack, coronary disease, stroke, kidney disease, arthritis — the raw vector additionally reports hypertension BPHIGH6=1 and high cholesterol TOLDHI3=1, so "six" is the conservative named-disease count), BMI 51.45 (z +3.50), INCOME3 = 1/11 lowest bracket. ΣSHAP +5.64: base −2.61 → log-odds +3.03.
- **B — most confident FN** (row 392245, fold 9): y=1 (GENHLTH=4 "Fair") but P(y=1) = 0.0025 / 0.0027 / 0.0156. PHYSHLTH = 0 days (SHAP −0.61), no chronic conditions, DIFFWALK = 2, BMI 22.14, INCOME3 = 9/11 ($100–150k), college graduate, exercises. **Every top-12 SHAP feature contributes negatively**; ΣSHAP −3.37 → log-odds −5.98. Subtype profile: the 50 most-confident shared errors are 50/50 PHYSHLTH ≤ 1 day, DIFFWALK = 2, DIABETE4 = 3 — a systematic respondent type, not a stray record.
- **C — fence case** (row 208256, fold 4): y=0 (GENHLTH=3), shared error with minimal |p − 0.5|: LGBM **0.5000022** (XGB 0.598, LR 0.563). Comorbidity block pushing toward y=1 (DIFFWALK=1 SHAP +0.60, DIABETE4=1 +0.56, heart attack +0.36, stroke +0.30, EMPLOY1=8 "unable to work" +0.34, INCOME3=2/11 +0.23) vs near-zero current-status anchors toward y=0 (PHYSHLTH=2 days −0.02, POORHLTH=0 −0.03). ΣSHAP +2.53 → refit-proxy P = 0.481 — a *different stated object* from the OOF 0.5000022, both straddling the fence.

**SHAP XAI** (`shap_global_importance.csv`, `shap_meta.json`; TreeExplainer `tree_path_dependent`, raw-margin output, base −2.6095, 20,000-row y-stratified sample rs=42, additivity asserted max dev 8.8e-14): top-5 mean|SHAP| = **PHYSHLTH 0.6813, DIFFWALK 0.2164, DIABETE4 0.1981, _BMI5 0.1855, EDUCA 0.1786** (PHYSHLTH is 3.1× the #2 feature). One-hot dummy contributions summed back to the **76 original columns** (structural map asserted to partition all transformed columns) — this neutralizes path-dependent SHAP's known credit-spreading across correlated dummies. **SHAP-vs-gain rank agreement: Spearman ρ = 0.911 (p = 3.6e-30, n = 76)** against the P5 fold-internal gain ordering — two attribution methods with different failure modes agree on the ordering.

## Interpretations (P7b, `results/analysis/error_analysis.md` — hypotheses, FACT/INTERPRETATION separated there)

1. **Boundary ambiguity, not noise.** Errors concentrate at one ordinal boundary and defeat three architecturally different learners simultaneously — the signature of label ambiguity (the binarized 3|4 edge), not model capacity. Class 4 "Fair" is a heterogeneous catch-all adjacent to "Good"; self-report depends on the respondent's internal reference frame, which the 76 columns cannot observe.
2. **Case A = plausible label noise** (response shift / adaptation, response style, or survey artifact): the model faithfully reports that this measured-health vector is overwhelmingly Fair/Poor-like; the disagreement is between the label and the respondent's own symptom answers.
3. **Case B = the error floor of the feature set**: the "Fair" rating is anchored on constructs the survey does not measure (wellbeing beyond day counts, expectations, personality, social comparison). No model trained on these columns could rank this person highly without paying elsewhere.
4. **Case C = aleatoric**: the population itself mixes 3s and 4s in this covariate region; p ≈ 0.5000 is the epistemically correct output, counted as an error only because thresholding must break exact ties.
5. **FNR 46% vs FPR 4% is expected geometry** of an unweighted, AUC-selected model at an arbitrary 0.5 cut with a 1:5 prior — consistent with P4's finding that resampling did not help AUC. If deployed for screening (it is not — course project), 0.5 is the wrong knob: **threshold menu** τ=0.2886 → TPR 0.70 / FPR 0.093 (precision 0.60, 19.4% flagged); τ=0.1771 → TPR 0.80 / FPR 0.158 (precision 0.50, 26.5% flagged).
6. **XAI synthesis:** gain and SHAP agreeing at ρ=0.911, plus EDA's univariate ranking, means the driver ordering (current physical-health status → functional limitation → chronic-disease/SES context) is a property of the data, not of one explanation method — and it confirms the P5/H1 importances. The model leans on self-reported current symptoms, so it inherits self-report bias: a feature of GENHLTH as a target, not of LightGBM.

## Figures

- `figures/p7_shap_summary.png` — global beeswarm, top 20 original features
- `figures/p7_shap_vs_gain.png` — SHAP vs P5 gain rank scatter (ρ = 0.911)
- `figures/p7_case_{A,B,C}_waterfall.png` — local explanations: A dragged by PHYSHLTH +1.94, B uniformly negative (nothing to read *is* the explanation), C visibly two-tailed
- `figures/p7_case_profiles.png` — 3-panel z-score/SHAP-sign profiles of the top-12 features per case (geometrically verified programmatically; **human visual pass advised** before report embedding)

## Caveats

- OOF probabilities (CV models) and SHAP values (full-data refit proxy) come from different fitted objects by construction; never mixed (A: 0.9595 vs 0.954; C: 0.5000022 vs 0.481).
- Case B's POORHLTH is missing → z-bar undefined (annotated in the figure); the model still assigns SHAP −0.12 via the NaN one-hot category.
- Path-dependent TreeExplainer spreads credit across correlated dummies — per-case attributions indicate *which blocks* drive a prediction, not precise per-feature credits (recorded in `shap_meta.json`).
- Case A's "six chronic conditions" counts the six named diseases; the raw vector also carries BPHIGH6=1 and TOLDHI3=1 (hypertension, high cholesterol) — an under-, not over-statement of multimorbidity.
- Runtime (`timing.json`): OOF 3.7 min + SHAP 7.0 min ≈ **10.7 min** total (oof 223.8 s, errors 1.0 s, shap 419.2 s).

## Verdict

Rubric evidence: **Intermediate "Error analysis: ≥3 consistently misclassified instances, features analyzed, reasoned hypothesis, visualizations"** (3 archetype cases, each wrong in 3/3 families, profiled with z-scores + SHAP, hypotheses in `error_analysis.md`, 4 figures) and **Intermediate "XAI: LIME or SHAP importance (global + individual)"** (global beeswarm + top-20 CSV + 3 local waterfalls + ρ=0.911 cross-method agreement). With this, every Intermediate experimentation item is evidenced; only the report-dependent items remain. Related: [[STATISTICAL COMPARISONS]] (same no-log-row stance), [[EXP-012 hypothesis H1 subsets]] (H1 blocks guide the XAI reading), [[EXP-015 nested CV]] (final-model honesty).
