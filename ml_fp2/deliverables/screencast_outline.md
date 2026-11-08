# Screencast Outline — Predicting Fair/Poor General Health (BRFSS 2021)

**Total planned runtime: 9:00 (aim to land at 8:30 — leave 30 s buffer).**
**Assets to have open before recording:** the notebook (`ml_fp2/notebooks/BRFSS_project.ipynb`, sections already run), `ml_fp2/experiments.csv` (or its printed summary cell), and the figures folder. Fill every `[PENDING]` tonight, then delete this reminder.

---

## Beat 1 — 0:00–0:45 · Problem + Data *(one slide)*

**Slide title:** "Can we predict who reports Fair/Poor health? (CDC BRFSS 2021)"

**Bullets (speakable, ~10 s each):**
- BRFSS 2021: 438,693 respondents × 303 columns — the largest US health survey
- Task: binary classification — Fair/Poor general health (GENHLTH 4–5, **16.6%**) vs Good-or-better (83.4%)
- Why: a screening-style early flag for poor self-rated health from survey answers alone
- Contract fixed up front: **ROC-AUC**, stratified **10-fold CV**, `random_state=42` everywhere

**Speaker notes (the story):** "Self-rated health is one of the strongest single predictors of mortality, and it's just one question. I asked: can the *rest* of the survey predict that one answer? I'm predicting Fair/Poor — about 1 in 6 respondents — so this is an imbalanced problem, which drove two of my design choices: the metric and the validation setup."

**Asset:** Slide only, with `figures/fig01_target_imbalance.png` embedded (the 5-bar GENHLTH chart, red = Fair/Poor).

---

## Beat 2 — 0:45–2:30 · Preprocessing decisions *(slide + LIVE DIP #1)*

**Slide title:** "Every column had to earn its place: 303 → 76"

**Bullets:**
- Leakage hunt first: `_RFHLTH` is the target wearing a mask — **100.00% agreement** with y on 437,532 rows
- 4 leak columns total (GENHLTH itself, `_RFHLTH`, `_PHYS14D`, `_MENT14D` — CDC bins of raw day counts)
- Six documented triage rules, first match wins: leak 4 · admin 35 · constant 11 · sparse 85 · low-coverage 27 · duplicate 65 → **76 keepers**
- Sentinel traps: 88 = "zero days" (real data!), 7 can mean *Retired* — recoded **per column from the codebook, never by pattern**
- Kept NaN on purpose: imputation happens **inside CV**, on training folds only

**LIVE DIP #1 (~30 s):** Alt-Tab to the notebook → Section 2.1 cell showing `Agreement with y: 100.00%` and the diagonal crosstab → scroll one screen to the Section 3.1 funnel output (`303 columns = 227 dropped + 76 kept`). Say: "That crosstab is diagonal — `_RFHLTH` contains zero information beyond the target. Including it would have printed a fake 0.99 AUC." Then back to slides.

**Speaker notes (the story):** "With 303 columns, the real risk isn't too few features — it's one column that secretly *is* the answer. I audited every CDC-computed column empirically, not by name. The funnel is a whitelist, not a drop-list: a typo in a keep-list loses a column, which is harmless; a typo in a drop-list can leak the target silently."

**Asset:** Slide with the funnel (a simple 303→76 wedge), then the two notebook cells above.

---

## Beat 3 — 2:30–3:30 · Validation setup *(one slide)*

**Slide title:** "One set of folds for everybody: why the comparison is fair"

**Bullets:**
- `StratifiedKFold(10, shuffle=True, random_state=42)` — **identical folds for every model**, every selection, every SMOTE run
- Why identical: score differences come from the *model*, not split luck — and it's what makes the paired Wilcoxon test legal later
- Why AUC, not accuracy: "everyone is healthy" already scores 83.4% accuracy — useless; AUC = **probability a sick respondent ranks above a healthy one**
- Verified the metric from first principles: hand-computed AUC on 6 toy people matched sklearn (0.556)

**Speaker notes (the story):** "Fixing the folds and the seed turned the whole project into a controlled experiment — same data slices, same preprocessing, only the model changes. I also hand-derived the AUC formula on a toy example in Section 1.5 before trusting it, because 90% of my conclusions rest on this one number."

**Asset:** Slide (10-fold diagram + the AUC definition in one line). Optional 10-s dip: notebook Section 1.4 fold table showing every fold at 16.63% positive.

---

## Beat 4 — 3:30–5:30 · Results: model zoo + imbalance verdict *(slide + LIVE DIP #2)*

**Slide title:** "What worked: baselines → ensembles → the imbalance question"

**Bullets:**
- Floor and ceiling of sanity: Dummy = **0.500 exactly**; LogReg = **0.8935** — a *linear* model already captures most of the signal
- Trees need limits: default DecisionTree **0.698** (memorizes), tuned to **0.881**; RandomForest **0.895**
- Boosting added a real but modest step: HistGB **0.9033** · LightGBM **0.9033** · XGBoost **0.9034** — about **+0.01** over LogReg
- Imbalance verdict — hypothesis *confirmed*: class weights (**0.894** LogReg / **0.9031** LightGBM) and SMOTE don't move AUC [PENDING: EXP-014 SMOTE number]
- Why: AUC measures *ranking*; rebalancing moves the best threshold, not the ordering — every run lives in `experiments.csv` with all 10 fold scores

**LIVE DIP #2 (~30 s):** Show the `experiments.csv` table (EXP-001…EXP-014, mean ± std per fold set) → then `figures/fig04_model_zoo.png` [PENDING: figure name if different] or `fig03_baselines.png`. Say: "One row per run, ten fold AUCs each — every number in this talk traces back to a row in this log."

**Speaker notes (the story):** "The LogReg result reframed the whole project: the signal is mostly additive and linear. Boosting still won, so the final model is a booster — but I can say that *after* proving it, not assuming it. And I tested the folk wisdom that imbalanced data needs SMOTE: under a rank-based metric, it doesn't."

**Asset:** Slide (zoo bar chart), then experiments table + one figure in the live dip.

---

## Beat 5 — 5:30–7:00 · Insights: what it learned & where it fails *(slide)*

**Slide title:** "Inside the model: drivers, failures, and why"

**Bullets:**
- Top drivers by gain importance / SHAP [PENDING: top-5 list tonight — expected shape: illness burden (PHYSHLTH/MENTHLTH days), age, access-to-care and functional-difficulty items]
- Consistency check: what boosting ranks highly vs. LogReg coefficients — agree or diverge? [PENDING]
- Three misclassified people, read as case files [PENDING: pick 2 false negatives + 1 false positive tonight — show their actual row values]
- Where the model fails and why: the Good↔Fair boundary is *subjective self-report* — humans disagree on it; plus heavy NaN patterns on exactly the riskiest respondents
- The errors are concentrated near the boundary, not random [PENDING: confirm from error analysis]

**Speaker notes (the story):** "Aggregate AUC doesn't tell me what the model *knows*, so I read individual mistakes. The false negatives I'll show aren't data errors — they're people whose survey answers look healthy while their self-rating says otherwise. That's exactly the boundary where the label itself is noisy, and it's the honest limit of this model."

**Asset:** Slide(s): importance/SHAP figure [PENDING filename] + a small 3-row table of the misclassified respondents.

---

## Beat 6 — 7:00–8:00 · Statistical comparison + final model *(one slide)*

**Slide title:** "Is +0.01 real? Wilcoxon says … — and the final pick"

**Bullets:**
- Same folds → fold scores are *paired* → **Wilcoxon signed-rank** on the 10 AUC differences (booster vs LogReg, booster vs RF) [PENDING: p-values tonight]
- Effect size with honesty: ~+0.01 AUC on 437k rows — statistically significant ≠ practically large, and I say which [PENDING: verdict]
- Tuning done without peeking: selection inside CV / nested where it mattered [PENDING: nested-CV number if run]
- **Final model: [PENDING — expected XGBoost or LightGBM]** — chosen on mean AUC + significance + robustness across folds, with LogReg kept as the interpretable runner-up

**Speaker notes (the story):** "Ten folds give me ten paired differences, so I can test whether the booster's edge is real or fold luck. Whatever the p-value says, my recommendation weighs interpretability: if the gap had been tiny and noisy, I'd have shipped the logistic regression and said so."

**Asset:** Slide with a small paired-differences table [PENDING: fold-by-fold deltas + Wilcoxon result].

---

## Beat 7 — 8:00–9:00 · Conclusions + what I'd do differently *(one slide)*

**Slide title:** "What I learned — and what I'd change"

**Bullets:**
- Biggest lever wasn't the model: it was the **leakage audit + triage** — garbage columns are a bigger threat than a weak learner
- Simple was strong: linear ≈ 0.89 of the ≈ 0.90 ceiling; boosting bought the last ~0.01
- Imbalance folklore, tested and rejected for AUC; would matter only at a fixed operating threshold
- Limits I won't hide: self-reported data, one cross-sectional year, prediction not causation
- Next time: probability calibration for deployment, threshold optimization for screening use, survey weights for population-level claims

**Speaker notes (the story):** "If I keep one lesson: I spent days on 303 columns and minutes on the winning model, and that was the right allocation. The 'what I'd do differently' list is real, not decorative — calibration and thresholds are what stand between this notebook and something a health agency could actually operate." End on one line: "Thanks — happy to defend any cell in the notebook."

**Asset:** Slide only (closing). No dip — finish clean.

---

## Recording tips

- **Tool:** OBS Studio (scene per asset: slide, notebook, CSV) or PowerPoint's built-in recorder — either is fine; 1080p, mic checked with a 10-second test clip first.
- **One dry run, timed, out loud.** Fix any beat that overruns *before* the real take; the dips are where time dies.
- **Never read slides verbatim.** Bullets are telegraph poles; the speaker notes are the sentences. Glance → look at camera → talk.
- **Pre-stage the two live dips:** run the notebook top-to-bottom beforehand, keep Section 2.1, Section 3.1 funnel, and the experiments summary already scrolled to, so each dip is one Alt-Tab, not a hunt.
- **[PENDING] discipline:** no number gets spoken unless it's in `experiments.csv` or a saved figure. Fill all slots tonight, then search the script for "PENDING" — zero hits means ready.
- **Pacing:** aim to finish at 8:30. If a dip stumbles, pause, breathe, restart the *sentence*, not the whole take. Record beat-by-beat if your tool allows it.

## Verified numbers you may quote (from the run notebook)

| Fact | Value |
|---|---|
| Raw data | 438,693 × 303 |
| Target rows dropped (7/9/blank) | 1,161 |
| Fair/Poor share | 72,741 = 16.63% |
| `_RFHLTH` agreement with y | 100.00% (437,532 rows) |
| Triage funnel | 303 = 4 leak + 35 admin + 11 constant + 85 sparse + 27 low-coverage + 65 duplicate + **76 keep** |
| After one-hot (54 nominal) | 322 design columns |
| Dummy / LogReg / tuned tree | 0.5000 / 0.8935 / 0.8813 (default tree 0.6976) |
| RF / HistGB / LightGBM / XGBoost | 0.8952 / 0.9033 / 0.9033 / 0.9034 |
| Class-weighted LogReg / LightGBM | 0.8940 / 0.9031 |
