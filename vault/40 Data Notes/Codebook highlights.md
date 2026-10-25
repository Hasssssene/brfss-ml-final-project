# Codebook Highlights (LLCP2021)

Source: `docs/codebook.pdf` (CDC BRFSS 2021 codebook, llcp-v2-508). Notes below were extracted while planning; **verify against the actual CSV in EXP-000** before relying on them. #data

## Dataset shape
- **438,693 respondent records × 303 columns** (confirmed EXP-000; all numeric dtype in the CSV)
- Two interview modes: landline and cell phone — many columns asked only to one mode's respondents (blank for the rest) → pruning criterion
- Optional modules (state-selected): prediabetes, CFS, hepatitis, HPV, tetanus, shingles, lung-cancer screening, cancer screening, caregiver, ACEs, marijuana — partial coverage → pruning criterion unless module adds value

## Target: GENHLTH (Section: Health Status)
| Code | Meaning | n | % |
|---|---|---|---|
| 1 | Excellent | 77,741 | 17.7% |
| 2 | Very good | 149,112 | 34.0% |
| 3 | Good | 137,938 | 31.4% |
| 4 | Fair | 54,736 | 12.5% |
| 5 | Poor | 18,005 | 4.1% |
| 7 | Don't know | 788 | — |
| 9 | Refused | 369 | — |
| blank | missing | 4 | — |

Binary framing per [[ADR-001]]: {4,5} → 1 (16.6%), {1,2,3} → 0. **Confirmed in EXP-000:** 437,532 valid rows, 72,741 positives = **16.63%**; 1,161 rows dropped (788 DK + 369 refused + 4 blank).

### `_RFHLTH` coding (verified EXP-000)
1 = Good or Better (y = 0), **2 = Fair or Poor (y = 1)**. Agreement with our binary target: **100.0%** (perfectly diagonal crosstab) → Tier-1 blacklist empirically confirmed ([[Leakage watchlist]]).

## Missing-value sentinels (general pattern)
| Pattern | Meaning | Examples |
|---|---|---|
| 7 / 77 / 777 | Don't know | most 1-digit/2-digit/3-digit categoricals — **but see the warning below: 7 is sometimes a VALID code** |
| 9 / 99 / 999 | Refused | idem — **9 is sometimes a VALID code** |
| 777777 / 999999 | DK / Refused (6-digit items) | FLSHTMY3, HIVTSTD3 (the EXP-000 scan missed these → HIVTSTD3 true coverage 20.3%, not 29.2%) |
| blank | Not asked / missing (mode or module skip) | CTELENM1-block, modules |
| 88 | **None** (a valid zero!) | PHYSHLTH, MENTHLTH, POORHLTH day counts; AVEDRNK3, DRNK3GE5, MARIJAN1, CHILDREN |
| 7777 / 9999 | DK / Refused | HEIGHT3, WEIGHT2 |
| 9999900-series | computed-impossible markers | some `_`-calculated cols |

> [!warning] **7 and 9 are NOT always sentinels** (verified against codebook + data in P2, 2026-10-25). Columns where 7 and/or 9 are **valid values** that must NOT be recoded to NaN:
> - **EMPLOY1: 7 = "Retired"** (131,546 rows — the census miscounted them as sentinels; only 9 is a sentinel)
> - **INCOME3**: 7 and 9 are valid income brackets among 1–11 (sentinels are 77/99 only)
> - **PRIMINSR**: 7 (TRICARE/VA) and 9 (state plan) are valid plans; **IMFVPLA2**: 7 (ER) and 9 (other place) valid among 1–12
> - **JOINPAI2**: 7 and 9 are valid 0–10 pain ratings (sentinels 77/99 only)
> - **_RACE**: 7 = Multiracial (valid 1–8) · **_STATE**: 9 = Connecticut (valid FIPS) · **_AGE80**: 77 = age 77 (no sentinels at all)
> - **CHILDREN** has **no 77 code** (88 = none → 0; 99 = refused); **AVEDRNK3/MAXDRNKS**: 7/9 drinks are valid, 88 = none → 0 (AVEDRNK3)
>
> Consequence (binding): sentinel recoding uses **per-column value contracts** (codebook-exact sets, asserted post-recode in `src/preprocessing.py`), never generic 7/9/77/99 pattern matching — see [[ADR-005 whitelist and matrix v1]].

### Column-specific decode notes (P2, verified)
- **ALCDAY5**: valid ranges **101–107** (days/week → ×4.33) and **201–230** (days/month → −200); 888 = no drinks → 0; 777/999 → NaN.
- **Fruit/vegetable items** (FRUIT2, FRUITJU2, FVGREEN1, FRENCHF1, POTATOE1, VEGETAB2): 1xx per-day ×30.4 · 2xx per-week ×4.33 · 3xx per-month −300 · **555 = "never" → 0** · **300 = "<1/month" → 0.5**; 777/999 → NaN.
- **_BMI5**: two implied decimals (÷100); **89 codebook-legal rows > 80.0 exist and are kept** (flagged for P3/P5 attention).
- **FLSHTMY3**: MMYYYY codes kept as nominal; 777777/999999 → NaN; 37 malformed-but-real codes retained as categories.

> [!warning] Never treat sentinels as numbers. 88-days = 0 days, not 88. **EXP-000:** PHYSHLTH alone has **287,796** rows coded 88 (≈66% of all rows) — the 88→0 recode is **mandatory** before modeling ([[ADR-004 tier-3 leakage resolution]]).
>
> **Empirical proof (P3 EDA):** PHYSHLTH's Spearman ρ vs y **flips sign** after the recode — **−0.293 on raw codes** (sentinel dominates, [[EXP-000 data profiling]]) → **+0.450 after 88→0** (strongest whitelist feature, [[EDA v1 summary]]). Sentinel-as-number would have handed the model an *inverted* health signal.

## Key section map (columns to triage)
- **Identifiers/admin (drop):** SEQNO, _PSU, IDATE, IMONTH, IDAY, IYEAR, FMONTH, DISPCODE
- **Weights (drop):** _LLCPWT, _STRWT, _RAWRAKE, _WT2RAKE, _STSTR, _STRATA...
- **Landline/cell flow (drop, mode-only):** CTELENM1..CSTATE1 block (landline), CELPHON1/CCELLFON etc.
- **Health status:** GENHLTH (target), PHYSHLTH, MENTHLTH, POORHLTH
- **Health care access:** PRIMINSR, PERSDOC3, MEDCOST1, CHECKUP1
- **Exercise:** EXERANY2 (+ calculated _PA family → [[Leakage watchlist]])
- **Chronic conditions:** BPHIGH6, BPMEDS, TOLDHI3, CHOLMED3, CVDINFR4, CVDCRHD4, CVDSTRK3, ASTHMA3, ASTHNOW, CHCSCNCR, CHCOCNCR, CHCCOPD3, HAVARTH5, ADDEPEV3, CHCKDNY2, DIABETE4, DIABAGE3
- **Demographics:** MARITAL, EDUCA, RENTHOM1, EMPLOY1, CHILDREN, INCOME3, WEIGHT2, HEIGHT3, DEAF, BLIND, DECIDE, DIFFWALK, DIFFDRES, DIFFALON (disability block), SMOKE100, SMOKDAY2, USENOW3, ECIGNILD1..., ALCDAY5, DRNK3GE5..., FVBRAND..., HIVRISK5...
- **Calculated (`_`-prefixed):** mixed — health-status ones blacklisted, demographic/behavioral ones deduplicated per watchlist
