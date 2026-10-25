# P3 EDA — 76-column matrix v1

**Scope:** exploratory analysis of the final feature matrix for the P5 hypothesis note and the report's EDA section.
**Inputs:** `data/interim/clean_v1.parquet` (438,693 × 76, sentinels → NaN, ENG applied), `data/interim/target_v1.parquet` (y, valid, genhlth_raw) — both per ADR-005. Valid mask applied first: **437,532 rows** (1,161 dropped: GENHLTH 7/9/blank).
**Method:** pandas/matplotlib/seaborn only; Spearman ρ is pairwise-complete (NaN-aware). Script: `src/analysis/p3_eda.py` (facts verified by `src/analysis/p3_probe.py`). Side artifacts: `results/analysis/eda_missingness.csv` (76 rows), `results/analysis/eda_association_spearman.csv` (76 rows: ρ, |ρ|, n_pairwise, no NaN ρ).
**Author:** data-analyst, 2026-10-25. No experiments-log entry (EDA, not a model run).

---

## 1. Target definition and imbalance — `figures/eda_target_imbalance.png`

**FACT** (panel a, raw GENHLTH, n = 437,532 valid): Excellent(1) 77,741 · 17.8 % — Very good(2) 149,112 · 34.1 % — Good(3) 137,938 · 31.5 % — Fair(4) 54,736 · 12.5 % — Poor(5) 18,005 · 4.1 %. 1,161 rows excluded (DK/Refused/blank), matching ADR-005.
**FACT** (panel b, binary): y = 1 (Fair/Poor) **72,741 = 16.63 %**; y = 0 **364,791 = 83.37 %**; imbalance ratio 1 : 5.0.

**INTERPRETATION (hypothesis):** moderate imbalance — every fold will have ≈ 7.3k positives (stratified 10-fold), so ROC-AUC estimation is stable. The majority-class baseline (83.4 % accuracy) is trivially high, confirming AUC over accuracy as the project metric. Downstream P4/P5 may test class weights or SMOTE for recall-oriented behavior, but no evidence from this figure demands resampling for AUC itself.

## 2. Missingness — `figures/eda_missingness_matrix.png`

**FACT:** median column missingness 5.3 %; 4 complete columns (SEXVAR, QSTLANG, _STATE, _AGE80); 21 columns > 25 %, 19 > 50 %. Top-10 (crimson): SOMALE 74.8 %, PREDIAB1 74.0 %, HOMBPCHK 71.5 %, HOMRGCHK 71.4 %, SOFEMALE 71.2 %, MARIJAN1 69.0 %, CRGVEXPT 68.6 %, ARTHDIS2 68.3 %, JOINPAI2 68.1 %, ARTHEXER 68.0 %. Overall matrix missingness after recode 20.54 % (ADR-005 figure, full matrix; my table is valid-rows-only). Distribution is bimodal: a core-survey mode (most columns < 15 %) and an optional-module mode (57–75 %).
**FACT:** two of the strongest single predictors sit in the high-missingness regime: JOINPAI2 (ρ = +0.343) 68.1 % missing, POORHLTH (ρ = +0.307) 48.0 % missing.

**INTERPRETATION (hypothesis):** the bimodal split reflects questionnaire design (module routing / skip logic), so missingness is structural rather than respondent-driven — plausibly close to MCAR *within* module-asked subsets, but module assignment correlates with state and language, so "missing" may carry mild signal. Imputation must be fold-safe (already mandated by ADR-005); the 19 columns > 50 % missing are the imputation-risk set, and their post-imputation coefficients should be distrusted more than core-survey features of equal ρ.

## 3. Univariate association with the target — `figures/eda_association_top25.png`

> Caption note: **no blacklisted column is present by construction** — GENHLTH, `_RFHLTH`, `_PHYS14D`, `_MENT14D` are excluded from matrix v1 by ADR-005 rule R1 (hard assert in `src/preprocessing.py`).

**FACT:** top-10 whitelist features by |ρ| (n = 437,532, pairwise complete):

| rank | feature | ρ | reading (higher code → Fair/Poor?) |
|---|---|---|---|
| 1 | PHYSHLTH | **+0.450** | more poor-physical-health days → yes |
| 2 | DIFFWALK | **−0.413** | reverse-coded: 1 = "yes, difficulty walking" → yes |
| 3 | JOINPAI2 | **+0.343** | more joint pain → yes |
| 4 | DIFFALON | −0.320 | reverse-coded (1 = yes difficulty alone) → yes |
| 5 | POORHLTH | +0.307 | more days poor health stops activities → yes |
| 6 | LMTJOIN3 | −0.290 | reverse-coded (1 = yes limited) → yes |
| 7 | DIFFDRES | −0.282 | reverse-coded (1 = yes difficulty dressing) → yes |
| 8 | INCOME3 | −0.278 | lower income bracket → yes |
| 9 | EMPLOY1 | +0.258 | driven by code 8 "unable to work": 2.7 % of y=0 vs 21.6 % of y=1 |
| 10 | CHCCOPD3 | −0.252 | reverse-coded (1 = yes COPD) → yes |

**FACT:** ranks 11–25 (values in figure): DECIDE −0.250, ARTHDIS2 −0.248, EXERANY2 +0.247, HAVARTH5 −0.217, BPHIGH6 −0.198, MENTHLTH +0.192, DIABETE4 −0.191, EDUCA −0.190, CVDCRHD4 −0.189, HOMBPCHK −0.187, PRIMINSR +0.181, ADDEPEV3 −0.181, CVDINFR4 −0.179, BLIND −0.173, CHCKDNY2 −0.168. Weakest of all 76: POTATOE1 −0.0005, CRGVEXPT −0.001, _STATE +0.007, USENOW3 −0.009, ECIGNOW1 −0.009.
**FACT (encoding consequence):** PHYSHLTH flips from ρ = −0.293 on raw codes (EXP-000, where the "88 = none" sentinel dominated) to **+0.450** after the 88→0 ENG recode — the recode was prerequisite to any sensible ranking.

**INTERPRETATION (hypothesis):** the ceiling |ρ| ≈ 0.45 with no leak-like outlier says the target is multi-causal — no single feature separates Fair/Poor, so gains will come from combining feature blocks (supports trees/ensembles over single shallow linear baselines). Functional-status items dominate (7 of the top 12 are limitation/joint-pain items). `_STATE` at |ρ| ≈ 0.007 suggests the 53-level one-hot is mostly noise dimensionality — a P5 pruning candidate (dropping it is a selection decision, allowed within the frozen whitelist).

## 4. Correlation structure — `figures/eda_correlation_heatmap.png`

**FACT** (top-40 by |ρ|, pairwise-complete Spearman; ordering = three named blocks first, then remaining 28 by signed ρ):
- **Health days** (PHYSHLTH·POORHLTH·MENTHLTH): internal ρ +0.234 … +0.409 (strongest PHYSHLTH–POORHLTH +0.409).
- **Disability** (DIFFWALK·DIFFALON·DIFFDRES·DECIDE): internal ρ +0.213 … +0.432 (strongest DIFFALON–DIFFDRES +0.432); DECIDE is the loosest member.
- **Socio-demo** (INCOME3·EMPLOY1·EDUCA·_AGE80): internal ρ −0.363 … +0.562 — heterogeneous: EMPLOY1–_AGE80 +0.562 (retirement), INCOME3–EDUCA +0.433, INCOME3–EMPLOY1 −0.363, EDUCA–_AGE80 −0.009.
- Cross-block and elsewhere: **maximum off-diagonal |ρ| = 0.590 (CHOLMED3–TOLDHI3)**; only 4 pairs exceed 0.5 (also EMPLOY1–_AGE80 0.562, BPHIGH6–HOMBPCHK 0.538, HOMBPCHK–HOMRGCHK 0.509); **zero pairs > 0.7**.
- **Blank cells (ρ undefined):** HAVARTH5 × {JOINPAI2, LMTJOIN3, ARTHDIS2} — on the pairwise-complete subsets (n ≈ 138.8k–140.7k) HAVARTH5 is constant (= 1), because the arthritis-management module is asked only of diagnosed arthritis cases.
- Visible secondary structure in the sorted remainder: the cardiovascular triple CVDCRHD4/CVDINFR4/CVDSTRK3 (ρ = −0.189/−0.179/−0.154) lands adjacent, as do DEAF/BLIND (−0.125/−0.173) and the cholesterol pair TOLDHI3/CHOLMED3.

**INTERPRETATION (hypothesis):** redundancy within blocks is moderate (≤ 0.43–0.59), so one-hot + regularized models or trees should cope without explicit de-duplication beyond ADR-005; linear models may still benefit from block-aware regularization. The HAVARTH5 blank cells are a skip-logic artifact: conditional-module items implicitly encode their gate (HAVARTH5 = 1), so JOINPAI2/LMTJOIN3/ARTHDIS2 partly *are* "has arthritis" indicators — worth remembering when interpreting their coefficients in P5.

## 5. Class-conditional distributions — `figures/eda_key_features_by_class.png`

**FACT** (y = 0 vs y = 1):
- **_AGE80**: mean 53.7 vs 59.9 years; 34,355 rows (7.85 %) exactly 80 (top-coding spike, no NaN).
- **PHYSHLTH**: mean 1.9 vs 13.8 days; 65.7 % of all rows are 0 (88→0 recode); 2.1 % missing.
- **MENTHLTH**: mean 3.2 vs 8.7 days; 61.8 % zeros; 1.8 % missing.
- **INCOME3**: median bracket 7 vs 5 (1 = lowest … 11 = highest); 21.5 % missing.
- **DIFFWALK**: P(difficulty | class) 8.9 % vs 47.8 % — a 5.4× rate ratio (n non-missing 417,855).
- **_SMOKER3**: current-every-day 8.0 % vs 16.3 %, current-some-days 3.2 % vs 5.4 %, former 26.2 % vs 33.1 %, never 62.5 % vs 45.3 % (ρ = −0.142).
- Supporting: EXERANY2 "any exercise" 80.2 % vs 51.7 %.

**INTERPRETATION (hypothesis):** the positive class skews older, lower-income, functionally limited and less physically active — an epidemiologically coherent signature (not surprising, but a useful sanity check that the target recode is correct). Day-count features are zero-inflated with heavy right tails; trees handle this natively, while any linear model would need transforms — an argument for the tree-based end of the P4/P5 model pool. The DIFFWALK rate ratio (5.4×) makes it the cleanest single "explainer" for a report narrative.

## 6. Distribution oddities & data-quality flags (all FACTS)

- **_AGE80 top-coding:** 7.85 % of rows are exactly 80 (rightmost-bin spike visible in panel 1 of fig 5).
- **Day-count zero spikes:** PHYSHLTH 65.7 %, MENTHLTH 61.8 % at exactly 0 — the ENG 88→0 recode creates a mixed distribution (true "0 days" + former "none" answers); both classes contain the spike, so it is not target-informative by itself beyond genuine prevalence.
- **FV "<1/month" 0.5 codes** (raw 300): FRUITJU2 16,343 rows, FRENCHF1 12,079, POTATOE1 9,933, FVGREEN1 5,736, FRUIT2 3,552, VEGETAB2 2,891 — a deliberate half-count band distinct from 0 ("never", raw 555). All six FV columns share the decoded maximum 3,009.6 (= 99/day × 30.4, raw code 199) — extreme but codebook-legal edge values.
- **_BMI5 > 80.0:** 88 rows among 390,957 non-missing valid rows (ADR-005 flagged 89 on the full matrix) — negligible for trees, worth clipping if a linear/scaled model is used.
- **Conditional-module missingness:** POORHLTH 48.0 %, JOINPAI2 68.1 % (skip logic, see §4).
- `_STATE` ρ = +0.007, CRGVEXPT ρ = −0.001, POTATOE1 ρ = −0.0005 — effectively uninformative marginals.

## 7. Which feature blocks look most predictive (feeds P5 hypothesis)

**FACT** (interpretive grouping — my grouping, not an ADR construct; mean |ρ| / mean non-missing coverage across group members):

| block | k | mean \|ρ\| | mean coverage |
|---|---|---|---|
| health days (PHYSHLTH, MENTHLTH, POORHLTH) | 3 | **0.316** | 82.7 % |
| functional limitation / joint pain (DIFFWALK, DIFFALON, DIFFDRES, DECIDE, JOINPAI2, LMTJOIN3, ARTHDIS2) | 7 | **0.306** | 68.2 % |
| chronic conditions (CHCCOPD3, HAVARTH5, BPHIGH6, DIABETE4, … ×12) | 12 | 0.167 | 99.5 % |
| socio-demographics (INCOME3, EDUCA, _AGE80, EMPLOY1, _RACE, SEXVAR, MARITAL) | 7 | 0.144 | 96.1 % |
| access / utilization (PRIMINSR, PERSDOC3, MEDCOST1, CHECKUP1, …) | 7 | 0.088 | 94.3 % |
| behaviors (EXERANY2, _SMOKER3, ALCDAY5, … ×9) | 9 | 0.082 | 78.5 % |
| diet / fruit-veg (×6) | 6 | 0.052 | 89.6 % |
| admin / geo (_STATE, QSTLANG, _URBSTAT) | 3 | 0.035 | 99.5 % |

**INTERPRETATION (hypothesis for P5):** *current-health-status* features (health days + functional limitation) carry roughly twice the marginal signal of chronic conditions and four times that of behaviors, but with 32–48 % less coverage on average. Expect P5 feature-importance rankings to be led by PHYSHLTH/DIFFWALK/JOINPAI2-class features; expect diet and admin/geo blocks to contribute little (candidates for empirical pruning within the whitelist). Because limitation items are also mutually correlated (§4), a handful of them may saturate the block's combined contribution — tree importances will show within-block sharing rather than one dominant member.

---
*Figures (all PNG, dpi 150): `figures/eda_target_imbalance.png`, `figures/eda_missingness_matrix.png`, `figures/eda_association_top25.png`, `figures/eda_correlation_heatmap.png`, `figures/eda_key_features_by_class.png`. All numbers above were computed by `src/analysis/p3_eda.py` / `p3_probe.py` on `clean_v1.parquet` + `target_v1.parquet` (valid rows only) unless explicitly attributed (EXP-000, ADR-005).*
