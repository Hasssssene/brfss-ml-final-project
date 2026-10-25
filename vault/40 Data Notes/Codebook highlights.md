# Codebook Highlights (LLCP2021)

Source: `docs/codebook.pdf` (CDC BRFSS 2021 codebook, llcp-v2-508). Notes below were extracted while planning; **verify against the actual CSV in EXP-000** before relying on them. #data

## Dataset shape
- ~438,693 respondent records, 300+ columns (exact count confirmed in EXP-000)
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

Binary framing per [[ADR-001]]: {4,5} → 1 (16.6%), {1,2,3} → 0.

## Missing-value sentinels (general pattern)
| Pattern | Meaning | Examples |
|---|---|---|
| 7 / 77 / 777 | Don't know | most 1-digit/2-digit/3-digit categoricals |
| 9 / 99 / 999 | Refused | idem |
| blank | Not asked / missing (mode or module skip) | CTELENM1-block, modules |
| 88 | **None** (a valid zero!) | PHYSHLTH, MENTHLTH, POORHLTH day counts |
| 7777 / 9999 | DK / Refused | HEIGHT3, WEIGHT2 |
| 9999900-series | computed-impossible markers | some `_`-calculated cols |

> [!warning] Never treat sentinels as numbers. 88-days = 0 days, not 88.

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
