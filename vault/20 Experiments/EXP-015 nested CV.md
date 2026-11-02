# EXP-015 — Nested cross-validation (unbiased tuned estimate)

- **exp_id:** EXP-015
- **date:** 2026-11-02
- **phase:** P6
- **status:** done
- **related ADRs:** [[ADR-005 whitelist and matrix v1]] (matrix and canonical folds unchanged; nested protocol is the P6-spec evaluation addition, not a methodology change)
- **log row:** `results/experiments_log.csv` (2026-11-02T18:15:00)

## Question
How much does the non-nested P6 protocol (search on full X,y, then score the winner on the canonical folds) **overstate** the tuned model's generalization?

## Setup
- Family: **LightGBM** (Part-2 winner: [[EXP-014 randomized search tuned finals]] 0.904321 vs 0.903843).
- **Outer loop = the canonical 10 folds** (`StratifiedKFold(10, shuffle=True, random_state=42)` via `src/config.py:make_folds()`) — folds_sha256 **`3b3a17457039`** asserted, identical to every other logged row.
- **Inner, per outer fold:** `RandomizedSearchCV(n_iter=12, StratifiedKFold(3, shuffle=True, random_state=42), scoring='roc_auc', random_state=42)` fit on the outer *training* fold only; best config refit on that training fold; scored once on the untouched outer test fold. Same search space as EXP-014 (budget-capped n_estimators [300,600]).
- Feature set all-76 · imbalance none · NaN-passthrough preprocessor inside every pipeline.

## Results
| fold | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | mean |
|------|---|---|---|---|---|---|---|---|---|----|------|
| outer test AUC | 0.904011 | 0.905465 | 0.905302 | 0.903872 | 0.905757 | 0.901092 | 0.903175 | 0.902115 | 0.906891 | 0.905353 | **0.904303 ± 0.001700** |

- **Optimism of the non-nested estimate: +0.000018** (EXP-014a 0.904321 − EXP-015 0.904303) — negligible; the search-induced selection bias is below fold noise.
- **7/10 outer folds re-selected the Part-1 winner** config (folds 1–5, 8, 9; folds 6, 7, 10 chose the Part-1 rank-2 config) — full per-fold table in `results/p6/nested_cv_details.csv`. In the 7 re-selection folds the outer AUCs match EXP-014a's fold scores to logged precision, exactly as expected for a deterministic refit.
- Context: nested estimate vs untuned [[EXP-006 lightgbm]] (0.903355) = **+0.000949** — the tuned gain survives with zero selection bias.

Artifacts: `results/p6/nested_cv_details.csv` (per-outer-fold inner winners, inner AUCs, outer AUCs, walls). Nested wall 26.4 min (`results/p6/timing.json`).

## Verdict
**The search width does not overfit: the honest tuned estimate is 0.904303, only +0.000018 below the non-nested number.** With 437,532 rows and a narrow, regularized space, inner-CV selection is nearly deterministic across resamples. Report both figures with their meaning: **0.904321** = canonical-folds tuned result (EXP-014a, comparable to every other log row), **0.904303** = nested, selection-bias-free estimate (EXP-015). Rubric evidence: **Advanced "Nested cross-validation for unbiased performance estimate"**.
