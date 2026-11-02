# Screencast Script — BRFSS 2021 Fair/Poor Health Prediction (≤ 10 minutes)

**Deliverable:** screencast URL for the submission form. **Total runtime target: 9:50.**
**Format suggestion:** screen-share the repo + `deliverables/report/report.pdf` (or the figures named below); slide-free, demo-style. Speak in English or Spanish (course allows both).

> Every claim below cites the experiment id shown in `results/experiments_log.csv` — if you say a number on camera, it is one of these.

---

## Beat 1 — Introduction & task (0:00–1:00) · 1:00

**On screen:** report.pdf title page + Table 1 (`results/profiling/target_distribution.csv`), then `figures/eda_target_imbalance.png`.

**Speaker notes:**
- BRFSS 2021: 438,693 respondents × 303 columns, the CDC's annual telephone health survey — the world's largest continuously-run health survey (guidelines §1).
- Task: binary classification. Target y = 1 iff GENHLTH ∈ {4,5} Fair/Poor → 16.63% positive (72,741 of 437,532 valid; 1,161 unusable rows dropped) — Figure 1.
- Why this framing (ADR-001): the CDC's own `_RFHLTH` indicator uses exactly this cut; we mirror an official operationalization, we didn't invent a split.
- Evaluation contract: stratified 10-fold CV, **identical folds for every model in the project** (fixed `random_state=42`, fold-hash asserted per run), metric ROC-AUC. Mention that a constant-prior Dummy scores exactly 0.500000 on all 10 folds (EXP-001) — the measured floor.

---

## Beat 2 — Data & cleaning (1:00–3:00) · 2:00

**On screen:** `results/profiling/triage_decisions.csv` (scrolling), `figures/eda_missingness_matrix.png`, Table 2 of the report (triage outcome).

**Speaker notes:**
- The leakage story first (this is the dataset's designed trap): `_RFHLTH` agrees with our target on **100.0% of rows — 437,532/437,532, perfectly diagonal** (EXP-000). It *is* the target. So `GENHLTH`, `_RFHLTH`, `_PHYS14D`, `_MENT14D` are blacklisted, and — key design choice — the matrix is built by **whitelist**, never by drop-lists: a typo can't leak the target (ADR-005).
- 303 → 76 columns via seven recorded rules: drop sparse (>90% missing, 86 cols), duplicates (65 — e.g. EXERANY2 proven identical to `_TOTINDA`), modules under 25% coverage, IDs/mode/admin, constants, leaks. Every one of the 303 columns has a verdict + reason in the CSV — no silent drops. Final: 54 nominal + 15 engineered + 4 ordinal + 3 numeric.
- Sentinel traps (codebook-verified): 7 is *not* always "don't know" — EMPLOY1 7 = Retired (131,546 rows), `_STATE` 9 = Connecticut, `_AGE80` 77 = age 77. Per-column value contracts, not pattern matching.
- The best preprocessing anecdote in the project: day-count code 88 = "none" = **zero** days. PHYSHLTH has 88 on 287,796 rows (~66%). On raw codes PHYSHLTH correlates with the target at ρ = **−0.293**; after 88→0 it flips to **+0.450** — the strongest feature in the data. Sentinels-as-numbers would have inverted our #1 predictor (EXP-000 + EDA notes).
- Missingness after recode: 20.54% overall, bimodal — core survey vs optional modules (Figure 2). Imputation is fold-safe inside CV; boosters get something better (next beat).

---

## Beat 3 — Models & tuning (3:00–6:00) · 3:00

**On screen:** report Tables 4–7 (grids + zoo + imbalance), then `figures/p6_tuning_results.png`.

**Speaker notes:**
- Baselines with manual tuning (EXP-002/003): LogReg C-grid is *flat* (0.892588 at C=0.01; top three settings within 8×10⁻⁶) — honest finding: regularization doesn't matter here. The tree grid is the opposite: min_samples_leaf dominates, and the near-default (depth=None, leaf=1) collapses to **0.694922** vs **0.880206** tuned (EXP-003l → EXP-003i) — manual tuning buys +0.185 AUC on camera.
- Model zoo, one config per family, identical folds (EXP-004..007): RF 0.8933 ≈ LogReg (tie), then the three boosters clear everything: HistGB 0.9032, LightGBM 0.9034, XGBoost 0.9034 — **+0.011 over linear/bagging ≈ 6× fold std**. Boosting was the win, not ensembling per se. LightGBM also runs 41 s vs RF's 351 s.
- A trick worth showing: **NaN-passthrough** — one-hot encodes NaN as its own category and the boosters route missing natively; module missingness stays signal instead of being imputed away.
- Imbalance handling (EXP-008/009/010): class weighting ≈ ±0.0004, undersampling ≈ ±0.0003, **SMOTE hurts (−0.004)** — verdict: none, because ROC-AUC is threshold-free and *rank-based*; resampling reshapes the prior, not the ranking. An honest negative, explained rather than hidden.
- Automated tuning (EXP-014): RandomizedSearchCV, 15 candidates, LightGBM winner **0.904321** (10/10 folds better than untuned), XGBoost 0.903843. Full params in the report (Table 11). Nested CV (EXP-015) says the unbiased estimate is **0.904303** — optimism just +0.000018. Budget levers (n_iter 20→15) documented, not silent.

---

## Beat 4 — Feature selection & statistics (6:00–8:00) · 2:00

**On screen:** `figures/p5_importance_top30.png`, then `figures/p5_pca_cumvar.png`, then report Table 13 (Wilcoxon).

**Speaker notes:**
- Hypothesis H1 (stated up front): predictive power concentrates in five blocks — functional limitation, health days, joint pain, chronic conditions, socio-demographics.
- Basic selection honest negative: SelectKBest(MI) below all-76 at every k (best k=200: 0.8997 vs 0.9034) — univariate filters drop correlated-but-useful columns that trees use for free (EXP-011a).
- Advanced selection (EXP-011b): fold-internal gain importances — **PHYSHLTH alone is 51.7% of total gain**, DIFFWALK 8.0%; the health-days block B2 carries 55.0% (Figure 6).
- H1 test (EXP-012): the 36-column H1-subset reaches 0.9001/0.9003 = **99.2% of the full lift** — but *not* strictly within fold noise: it loses 0/10 folds on both families, so we kept all-76 and used H1 for interpretation. Complement standalone 0.8118 (real but redundant). PCA: honest negative, −0.015 to −0.022 AUC (Figure 7).
- Statistics: Wilcoxon signed-rank on the 10 paired folds — all three tests W=0, **p=0.001953 (the smallest p possible at n=10)**. Headline: **model-family choice (+0.012) is worth 10× more than tuning (+0.001)** — both significant, but that's where the effort belongs.

---

## Beat 5 — Error analysis & XAI (8:00–9:30) · 1:30

**On screen:** `figures/p7_case_profiles.png` (main), optionally `p7_case_A_waterfall.png` + `p7_shap_summary.png`.

**Speaker notes:**
- At threshold 0.5: FPR 3.93%, FNR 46.44%. **88.2% of all errors sit at the Good|Fair (3|4) boundary** — and 41,091 rows are misclassified by all three model families simultaneously. Structured ambiguity in the binarized label, not model failure.
- Three archetypes (Figure 9): **Case A** — GENHLTH=3 "Good" but 30 poor-health days, six conditions, BMI 51; all three models say p≈0.96–0.99 → plausible *label noise* (response shift). **Case B** — GENHLTH=4 "Fair" with a perfectly clean profile; every SHAP contribution negative → the *error floor* of the feature set (unmeasured rating anchors). **Case C** — p = 0.5000 with a comorbidity block vs clean current status → *aleatoric* boundary case.
- Global SHAP (Figure 13): PHYSHLTH 0.681 (3.1× the #2), DIFFWALK, DIABETE4, BMI, education — and SHAP vs gain ranks agree at **ρ = 0.911**: two methods with different failure modes, one story: current physical-health status first.
- Note the FNR/FPR asymmetry is expected geometry for an unweighted AUC-selected model at 1:5 prevalence; the report ships a threshold menu if this were ever a screening tool (it isn't).

---

## Beat 6 — Conclusions & ethics (9:30–10:00) · 0:30

**On screen:** report Table 16 (the 42-row experiment summary) scrolling, then §11 heading.

**Speaker notes:**
- Final model: **tuned LightGBM, 0.9043 ± 0.0017, nested-confirmed** — chosen on the full evidence chain (42 logged runs, identical folds), not one metric. All code reproducible in `deliverables/code.py`; every number in the report cites its experiment id.
- Honest disclosure: an OpenCode agent (GLM, Z.ai) assisted throughout — planning, code, analysis, report drafting — under a human-ratified decision process (ADRs), with every result machine-verified against the append-only experiment log. Full 12-entry disclosure is the report's final section, per course policy.

---

### Production checklist
- [ ] Record at 1080p; zoom into tables/figures when named.
- [ ] Have `results/experiments_log.csv` open once (Beat 3 or 6) to show the append-only log.
- [ ] Keep total ≤ 10:00 (beats sum to 9:50); if over, trim Beat 3's imbalance detail first.
- [ ] Language: English or Spanish — state which in the video's first seconds.
- [ ] Upload, paste URL in the submission form's "General observations" field.
