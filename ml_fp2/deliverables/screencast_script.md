# Screencast Script (v2) — Predicting Fair/Poor General Health (BRFSS 2021)

**Runtime target: 8:30–9:30.** First sentence states the language (course requirement).
**Only numbers from `ml_fp2/experiments.csv` / the v2 report.** The three traps are at the bottom — read them before recording.

---

## Beat 1 — 0:00–0:45 · Problem + data

> "I'm presenting in English. My project predicts self-rated health from the CDC's BRFSS 2021 survey: 438,693 respondents, 303 raw columns. I turned the 5-point GENHLTH item into a binary problem — Fair or Poor, that's 72,741 people, 16.63%, about 1 in 6 — versus everyone else. Because that's imbalanced, I fixed two contracts up front: the metric is ROC-AUC, not accuracy — 'everyone healthy' already scores 83% accuracy — and every model is scored on the same stratified 10-fold split, seed 42."

**Slide:** title + `fig01_target_imbalance.png`.

---

## Beat 2 — 0:45–2:30 · Preprocessing: 303 → 76

> "With 303 columns, the real danger isn't a weak model — it's a column that secretly *is* the answer. `_RFHLTH` is the CDC's computed health flag: it agrees with my target on 100.00% of 437,532 rows. Training on it would print a fake 0.99. So column one of my funnel is a leakage rule — four columns gone, including the target itself and two CDC bins of raw day counts.
> Every one of the 303 columns got exactly one verdict, first match wins: 4 leaks, 35 admin columns, 11 constants, 85 over-90%-blank, 27 low-coverage, 65 duplicates — 76 survive. Duplicates I verified empirically, not by name: crosstabs show perfect diagonals, like `_TOTINDA` versus `EXERANY2`.
> Two codebook traps. In day-count questions, 88 means 'none' — real zero, not missing; 287,796 people answered 88 on PHYSHLTH. And 7 can mean *Retired* in EMPLOY1. So sentinel recoding is per-column from the codebook, never by pattern. Missing values I kept on purpose — imputation lives inside the CV pipeline, fitted on training folds only."

**Slide:** funnel 303→76 + the 100% agreement callout.

---

## Beat 3 — 2:30–3:15 · Validation setup

> "Same folds for everything — `StratifiedKFold(10, shuffle, random_state=42)` — which is what makes model differences attributable to models, not split luck, and what makes paired statistical tests legal later. Every run lands in `experiments.csv`: one row, ten fold AUCs, mean and std. The baseline block was run three times over development — bit-identical fold scores each time. That's the determinism claim, demonstrated rather than assumed."

**Slide:** fold discipline + experiments.csv excerpt.

---

## Beat 4 — 3:15–5:15 · Results: zoo + imbalance verdict

> "Baselines bracket the problem. Dummy: 0.500 exactly — the floor holds. Logistic regression: 0.8935 — and the C-grid is flat to 8 millionths of AUC, insensitivity, not tuning. A single unpruned tree collapses to 0.698; cap the depth and it recovers to 0.881 — capacity was the problem, not trees. Random Forest adds little: 0.895. The three boosters — scikit's HistGB, LightGBM, XGBoost — land at 0.9033, 0.9033, 0.9034: about +0.010 over linear, and less than 0.0002 apart from each other. The *family* matters, the implementation doesn't.
> Imbalance: hypothesis first — rank metrics shouldn't care about class priors. Class weights: +0.0005, −0.0002 — a wash. SMOTE: **−0.0040** — the biggest effect of the three, and negative. No imbalance handling in the final model, and the reason is the metric: rebalancing moves the best threshold, not the ranking."

**Slide:** `fig04_model_zoo.png` + the three imbalance deltas.

---

## Beat 5 — 5:15–7:15 · Selection, tuning, statistics

> "Feature selection, hypothesis first: five driver blocks — health-days, conditions and limitations, age, socioeconomic, behaviors — 39 of 76 columns. Result: 0.8988, 99.5% of the full model's AUC. Close — but the complement alone still reaches 0.78, and every route to fewer features, including mutual-information selection and PCA, *loses* AUC. Verdict: keep all 76; the survey measures its core constructs redundantly.
> Tuning: RandomizedSearchCV inside inner folds, then the winners scored on the untouched canonical folds. Tuned LightGBM: **0.903672**. Tuned XGBoost: 0.903640. And here's the headline honesty of this project: paired Wilcoxon on the ten shared folds says that difference is a **tie** — p equals 1.0, five fold wins each way. I report a tie and pick LightGBM for speed at equal score. Tuning *within* the family is real but tiny: +0.000356, p 0.0059. Family choice versus tuning: +0.0103 versus +0.0004 — **29 times**. The money is in representation and family, not in knobs.
> And nested CV: inside each outer fold a fresh search re-selected hyperparameters — it re-selected the *same configuration in all ten folds*, so the nested score equals the tuned score exactly, 0.903672. Zero optimism, with the mechanism stated, not shrugged at."

**Slide:** Table 8 + Wilcoxon table (the p=1.000 row highlighted).

---

## Beat 6 — 7:15–9:00 · Errors, XAI, limits, close

> "Where does it fail? Out-of-fold, 48,218 errors — and **88.1% of them sit on the Good-versus-Fair boundary**, exactly where a one-step subjective self-report is softest. Three people make it concrete. Case A: probability 0.96 of Fair/Poor, self-rates Good — but 30 sick days, BMI 51, difficulty walking, diabetic, lowest income. The label is the outlier, not the model. Case B: probability 0.002, self-rates Fair — every measurement clean, college-educated, employed. That's the error floor of these 76 columns — nothing in the feature space separates this person. Case C: probability exactly 0.5000 — age 80 and difficulty walking versus clean health days and BMI 22; the credits cancel. Calibrated indifference.
> SHAP agrees with the gain importances: PHYSHLTH first at 0.736, four times the runner-up — diabetes, BMI, difficulty walking, education, income. That's the driver-blocks hypothesis, confirmed from a second direction.
> Limits, honestly — and measured: the target is a subjective rating — that boundary sets the ceiling; the scores are near-calibrated in practice — Brier 0.081, calibration error 0.005 — but a formal calibration layer still comes before any threshold use; subgroup performance is not uniform — AUC 0.912 for White respondents versus roughly 0.85 to 0.88 for several minority groups — a real fairness gap a deployment must address; and the one preprocessing wart — a flu-shot sentinel — I tested the fix: one millionth of AUC. Noise, verified. Full AI-use disclosure is section 12 of the report. Everything you heard cites a row in `experiments.csv`. Thanks — happy to defend any cell."

**Slides:** `fig06_error_taxonomy.png`, case table, `fig07_shap_global.png`, closing.

---

## The three traps (do NOT say v1 numbers)

1. **Final score: 0.903672** (or 0.9037). Never **0.9043** — that's the retired v1.
2. **Wilcoxon: the LGBM-vs-XGB story is a TIE, p = 1.000** (5/10 folds each way). Never "W=0, p=0.001953, significant win" — that's v1.
3. **H1 driver blocks: 39 columns** (99.5% of AUC). Never **36** — that's v1. Complement-alone: 0.781. Nested-CV inner search: n_iter=4×3.

**Quick-fire facts:** Dummy 0.500000 · LogReg 0.8935 · tree 0.6976→0.8813 · RF 0.8952 · boosters 0.9033–0.9034 · weights +0.0005/−0.0002 · SMOTE −0.0040 · KBest 0.8997 · PCA50 0.8824 · OOF 0.903667 · 48,218 errors (11.0%) · 77.9% of FPs at Good · 92.5 of FNs at Fair · SHAP PHYSHLTH 0.7361 vs DIABETE4 0.1947 · tuned params: leaves 31, trees 500, min_child 200, lr 0.03 · three re-runs bit-identical · calibration Brier 0.081 / ECE 0.005 · subgroup AUC white 0.912 vs black 0.871 / hispanic 0.876 / asian 0.848 · wart-fix Δ −0.000001 (EXP-023).
