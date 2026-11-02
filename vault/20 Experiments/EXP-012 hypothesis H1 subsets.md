# EXP-012 — Hypothesis H1: driver-block subset vs all-76 vs complement

- **exp_id:** EXP-012 (log rows `EXP-012a`, `EXP-012b`, `EXP-012c`, `EXP-012d`)
- **date:** 2026-11-02
- **phase:** P5
- **status:** done
- **related ADRs:** [[ADR-005 whitelist and matrix v1]] (subsets are column partitions of the frozen 76-column whitelist; folds/metric/target unchanged → no new ADR)
- **log row:** `results/experiments_log.csv` (2026-11-02T16:53:40–16:56:39)

## Question
Does a compact, hypothesis-driven subset of columns match the full 76-column model within fold noise — and is the remaining complement as uninformative as the hypothesis claims?

## Hypothesis H1 (verbatim, coordinator-approved)

> The model's predictive power concentrates in five blocks: B1 functional limitation/disability (DIFFWALK, DIFFALON, DIFFDRES, DECIDE, DEAF, BLIND); B2 health days (PHYSHLTH, MENTHLTH, POORHLTH); B3 joint pain/arthritis (JOINPAI2, LMTJOIN3, HAVARTH5, ARTHDIS2, ARTHEXER); B4 chronic conditions (DIABETE4, BPHIGH6, TOLDHI3, CHOLMED3, CVDINFR4, CVDCRHD4, CVDSTRK3, ASTHMA3, ASTHNOW, CHCSCNCR, CHCOCNCR, CHCCOPD3, ADDEPEV3, CHCKDNY2); B5 socio-demographics (_AGE80, INCOME3, EDUCA, EMPLOY1, MARITAL, _RACE, SEXVAR, _BMI5, _SMOKER3). A compact subset of these ~35 columns should match all-76 within fold noise; the complement (diet, screening, state, remaining module vars) contributes little.

## Setup
- **H1-subset = 36 columns**: H1 names 37; **ASTHNOW is absent** from the ADR-005 2021 matrix (ASTHMA3 is the sole asthma item; no other mismatches — DEAF, BLIND, DECIDE, ARTHEXER, HAVARTH5 etc. all present).
- **Complement = 40 columns** = whitelist − H1 (diet/fruit-veg ×6, screening/immunization, alcohol ×4, sexual-orientation module, caregiver module, admin/geo ×3, access/utilization, ARTHEDU, CHILDREN, VETERAN3, RENTHOM1, tobacco beyond _SMOKER3). Exact lists: `results/p5/subsets.json`.
- Models: EXP-006 LGBM and EXP-007 XGB configs **unchanged** (NaN-passthrough prep, same as baselines; baselines never re-run).
- Imbalance strategy: none
- Validation: `StratifiedKFold(10, shuffle=True, random_state=42)` via `src/config.py:make_folds()` — folds_sha256 **`3b3a17457039`** asserted in all 4 rows

## Results
| run | model | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | mean |
|-----|-------|---|---|---|---|---|---|---|---|---|----|------|
| EXP-012a H1-subset | LGBM | 0.900190 | 0.901389 | 0.901487 | 0.900331 | 0.901080 | 0.896614 | 0.898446 | 0.897731 | 0.902443 | 0.900796 | **0.900051** |
| EXP-012b H1-subset | XGB | 0.900613 | 0.901946 | 0.901628 | 0.900377 | 0.901528 | 0.896542 | 0.898768 | 0.898220 | 0.902788 | 0.900746 | **0.900316** |
| EXP-012c complement | LGBM | 0.809064 | 0.808651 | 0.812441 | 0.808471 | 0.814755 | 0.811149 | 0.811510 | 0.814348 | 0.813526 | 0.814528 | **0.811844** |
| EXP-012d complement | XGB | 0.808438 | 0.809626 | 0.812661 | 0.808975 | 0.814197 | 0.811340 | 0.811985 | 0.814017 | 0.813615 | 0.813953 | **0.811881** |

Baselines (same folds, P4): EXP-006 LGBM all-76 **0.903355** (fold std 0.001691), EXP-007 XGB all-76 **0.903357** (fold std 0.001715).

- **H1-subset vs all-76**: LGBM Δ −0.00330, XGB Δ −0.00304. Paired per-fold Δ: LGBM −0.00330 ± 0.00050 (range −0.00398…−0.00246); XGB −0.00304 ± 0.00051. Subset ≥ full in **0/10 folds on both families** — the deficit is small but consistent (Δ scatter ≈ 6× smaller than |Δ|).
- **Lift retained: (0.900051−0.5)/(0.903355−0.5) = 99.2%** of the all-76 lift over 0.5.
- **Complement standalone**: 0.811844/0.811881 = **77.3% of the all-76 lift** (Δ ≈ −0.0915; complement ≥ full in 0/10 folds) — real signal, but largely **redundant** with H1 (EXERANY2, screening, alcohol correlate with functional/chronic status) plus module-assignment missingness artifacts (e.g. TRNSGNDR, BPMEDS).
- **Incremental value**: complement adds only **~+0.003 AUC on top of H1** (all-76 = H1 ∪ complement on the same folds) — exactly the size of the H1-subset deficit.
- Cross-check with [[EXP-011 selectkbest vs gain importance]]: 9/10 of the top-10 gain columns and 20/30 of the top-30 are inside H1-36.

Figures: none for this EXP (importance figure lives in EXP-011); subset lists in `results/p5/subsets.json`.

## Verdict
**H1 nearly confirmed — but not strictly within fold noise.** The 36-column H1-subset retains **99.2% of the lift** at −0.0033 AUC on both families, yet fails the strict criterion |Δ| ≤ one baseline fold-std (0.001715) at ~1.9σ, with the subset never winning a single fold (0/10, both families). The complement is dramatically weaker standalone (−0.0915) but not uninformative — its ~+0.003 increment is precisely what the subset gives up. Decision: **keep all-76 for the final model**; H1 blocks drive interpretation (report narrative, error analysis and XAI focus). Rubric evidence: **Intermediate "selected-features model vs all-features model compared"** (two families, paired folds) and, with EXP-011b, **"advanced feature selection + stated hypothesis"**.
