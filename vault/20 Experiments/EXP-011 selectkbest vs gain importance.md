# EXP-011 — SelectKBest (MI) filter vs LightGBM gain importance

- **exp_id:** EXP-011 (log rows `EXP-011a-50`, `EXP-011a-100`, `EXP-011a-200`, `EXP-011b`)
- **date:** 2026-11-02
- **phase:** P5
- **status:** done
- **related ADRs:** [[ADR-005 whitelist and matrix v1]] (selection runs inside folds on the frozen 76-column matrix; no methodology change, so no new ADR)
- **log row:** `results/experiments_log.csv` (2026-11-02T16:51:58 a-rows, 16:52:52 b-row)

## Question
Two rubric angles in one experiment pair: (a) **basic feature selection** — does a generic univariate filter (`SelectKBest` with mutual information) match all-76 features, and at which k? (b) **model-based (advanced) selection** — where does LightGBM actually concentrate its gain, and do the two notions of "important" agree?

## Setup
- **EXP-011a** (k=50/100/200): `SelectKBest(mutual_info_classif, random_state=42)` computed **inside each of the 10 training folds** on the ENCODED train matrix (P3 impute preprocessor); k=50/100/200 are top-k prefixes of one per-fold MI ranking (monotone selection). EXP-006 LightGBM config on the selected columns. Shared fold loop (`runtime_s` = loop wall per row).
- **EXP-011b**: EXP-006 config on all-76 (NaN-passthrough), **gain importances fitted inside the 10 training folds only** (never full data / test rows), one-hot children summed back to original columns, averaged over folds.
- Model: `LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=63, random_state=42, n_jobs=1, verbosity=-1)`
- Feature set: kbest-50 | kbest-100 | kbest-200 | all-76
- Imbalance strategy: none
- Validation: `StratifiedKFold(10, shuffle=True, random_state=42)` via `src/config.py:make_folds()` — folds_sha256 **`3b3a17457039`** asserted in all 4 rows

## Results
| run | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | mean |
|------|---|---|---|---|---|---|---|---|---|----|------|
| kbest-50 | 0.889471 | 0.888550 | 0.891161 | 0.887374 | 0.890451 | 0.886516 | 0.887554 | 0.888470 | 0.892241 | 0.889989 | **0.889178** |
| kbest-100 | 0.898946 | 0.900157 | 0.900754 | 0.899044 | 0.901703 | 0.896030 | 0.898002 | 0.898110 | 0.901717 | 0.901095 | **0.899556** |
| kbest-200 | 0.899158 | 0.900535 | 0.900955 | 0.899104 | 0.901892 | 0.896135 | 0.897346 | 0.898228 | 0.902260 | 0.901089 | **0.899670** |
| all-76 (EXP-011b) | 0.902947 | 0.904705 | 0.904383 | 0.902790 | 0.905059 | 0.900003 | 0.902187 | 0.901398 | 0.905380 | 0.904695 | **0.903355** |

- EXP-011b fold AUCs **replicate the logged EXP-006 row to 6-decimal precision (max |diff| = 3.7e-07)** — true baseline replication on identical folds.
- Gain top-10 (share of total gain, from `results/p5/lgbm_importance.csv`): **PHYSHLTH 51.69%** [B2], DIFFWALK 7.99% [B1], DIABETE4 3.31% [B4], INCOME3 2.55% [B5], EDUCA 2.53% [B5], _BMI5 2.44% [B5], POORHLTH 1.80% [B2], CHCCOPD3 1.71% [B4], EXERANY2 1.56% [other], MENTHLTH 1.53% [B2].
- **Block share of total gain: B2 health-days 55.0%**, B5 11.7%, B4 10.6%, B1 10.1%, other 11.4%, B3 joint-pain 1.2%.
- MI ∩ gain overlap (original columns, top-k): **3/10 @10**, 8/20 @20, **13/30 @30**, 15/36 @36 — the two rankings disagree heavily at the top.
- SelectKBest stability (k=50): 48 of 76 columns selected in all 10 folds.

Figures: `figures/p5_importance_top30.png` (dpi 150). Artifacts: `results/p5/kbest_selection.csv`, `results/p5/lgbm_importance.csv`.

## Verdict
**Filter selection rejected; gain importances adopted as the advanced selection.** SelectKBest is below all-76 at **every** k (−0.0142 @50, −0.0038 @100, −0.0037 @200): MI is univariate, so it drops correlated-but-useful columns that boosted trees exploit for free (redundancy costs a tree nothing). Gain is extremely concentrated — one column (PHYSHLTH) carries half of it — and sits overwhelmingly inside the H1 blocks (9/10 of top-10 gain columns are in H1-36), which fed the hypothesis test in [[EXP-012 hypothesis H1 subsets]]. Rubric evidence: **Basic "feature selection: basic algorithm applied"** (EXP-011a, honest negative, reported) and — together with H1 — **Intermediate "advanced feature selection + stated hypothesis"** (EXP-011b).
