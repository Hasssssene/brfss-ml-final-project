# EDA v1 summary — 76-column matrix v1

- **date:** 2026-10-25 · **phase:** P3 · **author:** data-analyst · **status:** findings logged; figures pending human visual pass #data #exp
- **Inputs:** `data/interim/clean_v1.parquet` (438,693 × 76) + `target_v1.parquet`, valid rows only (**437,532**; 1,161 dropped per [[ADR-001 binary target framing]]) — per [[ADR-005 whitelist and matrix v1]]
- **Full analysis:** `results/analysis/eda_notes.md` (facts verified by script + probe) · script `src/analysis/p3_eda.py` · side tables `eda_missingness.csv`, `eda_association_spearman.csv`
- **No experiments_log row** — EDA is analysis, not a model run; lineage cited instead (see [[EXP-000 data profiling]] for the raw-code comparisons)

> [!warning] The five figures were verified **programmatically** (values re-probed), not by a visual model pass. A **human visual check is required before embedding any of them in the report.**

## Figures (all PNG, dpi 150)
1. `figures/eda_target_imbalance.png` — GENHLTH raw distribution + binary y (16.63% positive, 1 : 5.0 imbalance)
2. `figures/eda_missingness_matrix.png` — per-column missingness, bimodal (core survey vs optional modules); median 5.3%, 19 columns > 50%
3. `figures/eda_association_top25.png` — Spearman |ρ| vs y, top 25 of 76 (no blacklisted columns by construction)
4. `figures/eda_correlation_heatmap.png` — top-40 × top-40 Spearman correlation structure
5. `figures/eda_key_features_by_class.png` — class-conditional distributions (_AGE80, PHYSHLTH, MENTHLTH, INCOME3, DIFFWALK, _SMOKER3)

## Top-10 univariate associations with y (Spearman ρ, pairwise-complete, n = 437,532)

| rank | feature | ρ | reading |
|---|---|---|---|
| 1 | PHYSHLTH | **+0.450** | more poor-physical-health days → Fair/Poor |
| 2 | DIFFWALK | **−0.413** | reverse-coded (1 = difficulty walking) → Fair/Poor |
| 3 | JOINPAI2 | +0.343 | more joint pain → Fair/Poor (68.1% missing — module) |
| 4 | DIFFALON | −0.320 | reverse-coded (1 = difficulty alone) |
| 5 | POORHLTH | +0.307 | more days poor health stops activities (48.0% missing) |
| 6 | LMTJOIN3 | −0.290 | reverse-coded (1 = limited by joints) |
| 7 | DIFFDRES | −0.282 | reverse-coded (1 = difficulty dressing) |
| 8 | INCOME3 | −0.278 | lower income bracket → Fair/Poor |
| 9 | EMPLOY1 | +0.258 | driven by code 8 "unable to work": 2.7% of y=0 vs **21.6%** of y=1 |
| 10 | CHCCOPD3 | −0.252 | reverse-coded (1 = COPD) |

Weakest of 76: POTATOE1 −0.0005, CRGVEXPT −0.001, _STATE +0.007, USENOW3/ECIGNOW1 ≈ −0.009.

## Key callouts

**① PHYSHLTH sign flip — the ENG recode is not optional.** On raw codes ρ = **−0.293** ([[EXP-000 data profiling]], the 88 = "none" sentinel dominated); after the 88→0 recode ρ = **+0.450** — the *strongest* whitelist feature and the #1 predictor. Sentinels-as-numbers would have handed the model an inverted signal. Evidence recorded in [[Codebook highlights]]; recode mandated by [[ADR-004 tier-3 leakage resolution]].

**② No leak-like outlier.** Max |ρ| ≈ 0.45 — the target is multi-causal; gains must come from combining feature blocks (argument for ensembles over single shallow models). Also confirms the leakage blacklist worked: nothing approaches `_RFHLTH`'s 100% agreement.

**③ Correlation structure is tame in the top-40 view — the full matrix peaks higher.** *(corrected 2026-11-03 via notebook verification)* Within the **top-40-by-target-association** view — the scope of `figures/eda_correlation_heatmap.png` and of the original claim — max off-diagonal |ρ| = **0.590** (CHOLMED3–TOLDHI3); only 4 pairs > 0.5; **zero pairs > 0.7** — no severe collinearity post-dedup ([[ADR-005 whitelist and matrix v1]] dup-drop vindicated). The **full 76×76 pairwise-complete matrix** (computed live in `notebooks/BRFSS_walkthrough.ipynb` §7) peaks at |ρ| = **0.744** — **AVEDRNK3–MAXDRNKS** (+0.744), then **DRNK3GE5–MAXDRNKS** (+0.719) — the co-asked alcohol block: weakly target-associated (hence invisible to the top-40 view) but strongly mutually correlated among drinkers, where survey skip-logic co-observes all three items; overall 7 pairs > 0.5 and 2 > 0.7 across all 76 columns. Contained: the ADR-005 duplicate-pair resolutions cover the deterministic duplicates, the redundancy is empirically benign — [[EXP-012 hypothesis H1 subsets]]'s complement (which holds the alcohol block) adds only ~+0.003 AUC on top of H1 — and boosted trees simply pick one member of a correlated group per split. Lesson recorded: "max correlation" is only defined relative to a feature set; state the scope. 3 NaN pairs (HAVARTH5 × JOINPAI2/LMTJOIN3/ARTHDIS2) are a skip-logic artifact: the arthritis-management module is asked only of diagnosed cases (HAVARTH5 constant = 1 on the pairwise subset).

**④ Oddities / data-quality flags** (all FACTS):
- `_AGE80` top-coded: **7.85%** of rows exactly 80 (spike, no NaN).
- Day-count zero spikes from 88→0: PHYSHLTH **65.7%**, MENTHLTH **61.8%** at exactly 0 — mixed distribution (true 0 + former "none"), both classes share the spike.
- FV "<1/month" 0.5-codes (raw 300): FRUITJU2 **16,343 rows**, FRENCHF1 12,079, POTATOE1 9,933, FVGREEN1 5,736, FRUIT2 3,552, VEGETAB2 2,891; all six FV columns share decoded max 3,009.6 (codebook-legal edge).
- `_BMI5 > 80.0`: 88 rows among 390,957 non-missing valid rows (negligible for trees; clip if a linear/scaled model ever uses it).
- Strongest predictors sit in the high-missingness regime (JOINPAI2 68.1%, POORHLTH 48.0%) — their imputed values deserve extra distrust.

## Feed into P5 (hypothesis)

**Functional-limitation / joint-pain items dominate: 7 of the top-12** associations (DIFFWALK, DIFFALON, DIFFDRES, DECIDE, JOINPAI2, LMTJOIN3, ARTHDIS2). Block mean |ρ| ranking: health days **0.316** > functional limitation **0.306** > chronic conditions 0.167 > socio-demo 0.144 > access 0.088 > behaviors 0.082 > diet 0.052 > admin/geo 0.035. Current-health-status features carry ~2× the marginal signal of chronic conditions and ~4× behaviors — but with 32–48% less coverage. Expect P5 importances led by PHYSHLTH/DIFFWALK/JOINPAI2-class features; diet and admin/geo (incl. the 53-level `_STATE` one-hot, ρ = +0.007) are empirical-pruning candidates. Limitation items are mutually correlated (≤ 0.43 internal) → tree importances will show within-block sharing, not one dominant member.

## Verdict
EDA closes P3's analysis half; [[Rubric Checklist]] imbalance item evidenced by `eda_target_imbalance.png`. Figures blocked for report embedding until the human visual pass. Next: **P4 model zoo + imbalance** (EXP-004..010) on the identical folds.
