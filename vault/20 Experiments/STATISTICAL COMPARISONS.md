# STATISTICAL COMPARISONS — Wilcoxon signed-rank tests (analysis note) #exp

> [!warning] This is an **analysis note, not an experiment.** It documents statistical tests computed from *already-logged* fold scores, so it has **no EXP id and no `experiments_log.csv` row by design** — inventing an EXP-016 row for a re-analysis of existing runs would violate the one-row-per-run log contract. Every number below traces to the cited log rows and to `results/p6/wilcoxon_tests.csv` (generated 2026-11-02 by `src/p6_tune.py`, Part 4).

- **date:** 2026-11-02
- **phase:** P6
- **status:** done
- **test:** two-sided Wilcoxon signed-rank (`scipy.stats.wilcoxon`, `zero_method='wilcox'`) on the **10 paired canonical-fold AUCs**; the folds are identical by construction (`folds_sha256=3b3a17457039` asserted in every feeding row), so pairing is exact.
- **feeding rows:** [[EXP-014 randomized search tuned finals]] (EXP-014a, EXP-014b), [[EXP-006 lightgbm]], [[EXP-007 xgboost]], [[EXP-002 logistic regression]] (row EXP-002a)
- **artifact:** `results/p6/wilcoxon_tests.csv`

## Results

| test | comparison | mean AUC a vs b | median paired Δ | folds a>b | W | p | method | verdict |
|---|---|---|---|---|---|---|---|---|
| T1 | tuned LGBM vs tuned XGBoost (both P6 winners) | 0.904321 vs 0.903843 | +0.000471 (0.0004705) | 10/10 | 0.0 | **0.001953** | exact | EXP-014a > EXP-014b — significant |
| T2 | tuned LGBM vs its untuned self (EXP-006) | 0.904321 vs 0.903355 | +0.000991 | 10/10 | 0.0 | **0.001953** | exact | EXP-014a > EXP-006 — significant |
| T3 | tuned LGBM vs LogisticRegression (EXP-002a) | 0.904321 vs 0.892588 | +0.011768 | 10/10 | 0.0 | **0.001953** | exact | EXP-014a > EXP-002a — significant |

All 10 paired differences share the same sign in every test → W = 0 and p = 0.001953125 = 2·2⁻¹⁰ — the **smallest two-sided p attainable at n=10** (the "n=10 floor"): no paired comparison of 10 folds could ever be more significant than this.

## Incident (documented)
`scipy.stats.wilcoxon` (scipy 1.16.0) exposes no `.method` attribute on `WilcoxonResult`, so the method could not be read back from the result object. The `method` column in `wilcoxon_tests.csv` is derived from scipy's documented rule (exact when n ≤ 25, no zeros after the `'wilcox'` drop, no ties among |nonzero diffs|; else normal approximation) and the tests re-run — code and comment in `src/p6_tune.py` Part 4. All three tests resolve to **exact**.

## Verdict
**Three for three significant at α = 0.05.** Headline for the report: the **model-family gain (+0.012, T3) is an order of magnitude larger than the tuning gain (+0.001, T2)** — and both are real (all fold differences same sign, floor p-value). Tuned LightGBM ([[EXP-014 randomized search tuned finals|EXP-014a]]) is confirmed as the final model; its honest selection-bias-free performance is [[EXP-015 nested CV]] 0.904303. Rubric evidence: **Intermediate "Statistical comparison: Wilcoxon signed-rank on 10 fold scores, identical partitions (fixed random_state) — top-2 models"**.
