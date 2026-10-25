# ADR-005 — Feature whitelist & matrix v1: 303 → 76 columns via seven triage rules

- **date:** 2026-10-25
- **status:** accepted
- **tags:** #adr #features #data
- **scope:** feature matrix v1 / column whitelist (all 303 census columns)

## Context
[[EXP-000 data profiling]] produced a full census of the 303 raw columns (coverage, sentinels, distinct values, target association, duplicate-pair probes) and [[ADR-004 tier-3 leakage resolution]] settled the Tier-3 leakage question. Modeling needs a frozen feature matrix — and 303 candidates, of which 92 are >90% blank+sentinel and dozens are calculated duplicates of each other, is not one. The whitelist therefore had to be decided once, explicitly, **per column**: every column gets a verdict + reason (no silent drops), and every judgment call gets recorded.

Evidence base (EXP-000 + codebook cross-check in `src/preprocessing.py`):
- 438,693 rows × 303 cols; 437,532 valid target rows; 72,741 positives (16.63%).
- `_RFHLTH` shows **100.0% agreement** with y — the true-leakage signature; blacklist confirmed.
- Duplicate pairs proven identical: EXERANY2 ≡ _TOTINDA (n = 436,659), _DRDXAR3 ≡ HAVARTH5 (n = 434,721).
- The census's generic 7/9/77/99 sentinel scan **overcounts** sentinels wherever 7/9 are valid codes (EMPLOY1 7 = "Retired" — 131,546 rows; INCOME3 / PRIMINSR / IMFVPLA2 / JOINPAI2 / _RACE code 7; _STATE 9 = Connecticut; _AGE80 77) and **undercounts** 6-digit sentinels (777777 / 999999 in FLSHTMY3, HIVTSTD3) → sentinel handling must be per-column value contracts, not pattern matching.

## Decision
The whitelist is defined by **seven rules**, applied with precedence **R1 > R3 > R2 > R6 > R4 > R5 > keep** (each column takes the first matching verdict):

1. **DROP-LEAK (R1) — leakage blacklist.** `GENHLTH` (the target source), `_RFHLTH`, `_PHYS14D`, `_MENT14D` (calculated from GENHLTH / its family). Enforced by a hard assert in `src/preprocessing.py`: these can never enter the matrix.
2. **DROP-ID / DROP-MODE (R2) — admin, weights, flow.** Identifiers, dates, dispatch codes, weights/strata (`SEQNO`, `_PSU`, `_LLCPWT`, `_STSTR`, …), instrument admin and the landline/cell intro-flow block — including the "etc." telephone-admin columns `NUMHHOL3`, `CPDEMO1B`, `MSCODE`. No respondent health content.
3. **DROP-CONSTANT (R3).** Census-constant columns (≤1 distinct value): the 11 constants + `IYEAR` (2 values, admin).
4. **DROP-SPARSE (R4).** >90% blank + sentinel (micro-coverage modules: diabetes management, hepatitis, HPV, tetanus, shingles, lung-cancer & cancer screening, cancer survivorship, caregiver detail, firearms, marijuana follow-ups, phone count).
5. **Coverage gate (R5).** Survival requires **≥25% effective coverage after sentinel recode**; 10–25% → **DROP-MODULE** (optional-module columns; ACEs get no special treatment). Borderline cases resolve by the rule, not discretion: PREDIAB1 25.98% kept, PDIABTST 24.86% dropped, SOMALE 25.23% kept (the minimum among kept).
6. **DROP-DUP (R6) with keep-raw preference.** One representation per duplicate pair / computed family, preferring the raw or simplest form:
   - `_BMI5` (over HEIGHT3 / WEIGHT2 / HTIN4 / HTM4 / WTKG3 / _BMI5CAT / _RFBMI5)
   - `_SMOKER3` (over SMOKE100 / SMOKDAY2 / _RFSMOK3)
   - `_AGE80` (over _AGEG5YR / _AGE65YR / _AGE_G)
   - `_RACE` (over _PRACE1 / _MRACE1 / _HISPANC / _RACEG21 / _RACEGR3 / _RACEPRV / _IMPRACE)
   - `EDUCA` (over _EDUCAG) · `INCOME3` (over _INCOMG1) · `PRIMINSR` (over _HLTHPLN / _HCVU652)
   - `EXERANY2` (≡ _TOTINDA, proven EXP-000) · `HAVARTH5` (≡ _DRDXAR3, proven EXP-000)
   - alcohol raws `ALCDAY5` / `AVEDRNK3` / `DRNK3GE5` / `MAXDRNKS` (over DRNKANY5 / DROCDY3_ / _RFBING5 / _DRNKWK1 / _RFDRHV7)
   - same principle for e-cig (`ECIGNOW1` over `_CURECI1`), vaccine/HIV raw items over computed flags, fruit-vegetable raw frequency items over computed indices, `SEXVAR` over `_SEX`, BP/chol/asthma/MI raw items over computed flags.
7. **Judgment-call protocol (R7).** Ambiguous columns get an explicit recorded call (verdict + reason row in `results/profiling/triage_decisions.csv`); **no silent decisions**. All R7 calls:

| Call | Decision | Rationale |
|---|---|---|
| QSTLANG vs QSTVER | **keep QSTLANG** (NOM), drop QSTVER | interview language is a respondent attribute; questionnaire version is instrument admin |
| _URBSTAT vs _METSTAT | **keep _URBSTAT**, drop _METSTAT | urbanicity pair — one representation per pair |
| MSCODE | drop | MSA geo-stratum = weighting geography; redundant with kept _STATE + _URBSTAT |
| NUMHHOL3 | drop (despite passing R5 at 26.6% coverage) | residential-telephone count = dual-frame weighting admin (rule 2 "etc.") |
| CPDEMO1B | drop | cell-phone usage share = _DUALUSE weighting input (telephone admin) |
| AVEDRNK3 / MAXDRNKS | **keep** | raw alcohol items not enumerated in rule 6 — "prefer the raw/simplest representation" |
| FLSHTMY3 | **keep as NOMINAL** | raw MMYYYY immunization item surviving rule 5 (44.3% effective coverage); 777777/999999 → NaN; 37 malformed-but-real codes retained as categories |
| CHOLCHK3 | **keep as NOMINAL, not ordinal** | codebook order is non-monotone (1 = "Never" vs 2–8 = decreasing recency); one-hot preserves everything |
| CAREGIV1 value 8 | keep as its own category | 8 = "caregiving recipient died" (328 rows) is a valid codebook response, not a sentinel |
| CHECKUP1 | **keep as ORDINAL**, 8 = "never" as its own high value | 1<2<3<4 time buckets; 7/9 → NaN |
| HTIN4 / HTM4 / WTKG3 | drop | linear metric transforms of dropped HEIGHT3 / WEIGHT2; BMI family resolved to `_BMI5` |
| _CHLDCNT / _HCVU652 | drop | computed binning / flag of kept CHILDREN / PRIMINSR (+ age) |
| _STATE | **keep** (NOM, 53 FIPS codes) | legitimate predictor; FIPS 9 = Connecticut is a valid code; P5 selection can prune empirically if useless |

Fixed encodings: **ORD** = CHECKUP1, EDUCA, INCOME3, _SMOKER3 · **NUM** = JOINPAI2, MAXDRNKS, _AGE80 · **ENG** recodes = day-count 88→0 (PHYSHLTH/MENTHLTH/POORHLTH), ALCDAY5 → drinks/month, FV items → times/month (555→0, 300→0.5), _BMI5 ÷100, CHILDREN 88→0 (no 77 code exists), MARIJAN1 88→0, AVEDRNK3/DRNK3GE5 88→0, FLSHTMY3 MMYYYY · everything else **NOM** (one-hot at train time).

Sentinel handling is defined by **per-column value contracts** (codebook-exact sentinel sets asserted post-recode) — see the codebook-trap warning in [[Codebook highlights]].

## Alternatives considered
- **Keep computed indices instead of raws** (e.g. `_RFBMI5`, `_RFDRHV7`, `_TOTINDA`) — rejected: they carry duplicate information of the kept raws (EXERANY2 ≡ _TOTINDA is literally identical, EXP-000); raws hold strictly more detail.
- **Impute micro-modules under 25% coverage** instead of dropping — rejected: >75% imputed values would be imputation artifacts, not signal (same reasoning as the caregiver block in [[ADR-004 tier-3 leakage resolution]]).
- **Drop `_STATE`** (admin-adjacent) — rejected: state is a legitimate nominal predictor (care access, policy, environment); with one-hot + P5 selection it gets pruned empirically if useless, not by decree.

## Consequences
- **The 76-column whitelist is frozen as matrix v1**: `data/interim/clean_v1.parquet` (438,693 × 76, float64, sentinels → NaN, ENG applied, no target column; gitignored interim checkpoint) + `target_v1.parquet` (row-aligned y + valid mask) + `column_meta.json` (per-column encoding/recode/verdict/coverage).
- **76 kept / 227 dropped**: KEEP-NOM 54, ENG 15, KEEP-ORD 4, KEEP-NUM 3; DROP-SPARSE 86, DROP-DUP 65, DROP-MODULE 26, DROP-ID 18, DROP-MODE 16, DROP-CONSTANT 12, DROP-LEAK 4.
- Per-column fates for **all 303 columns** are authoritative in `results/profiling/triage_decisions.csv` (verdict, rule, reason, census + effective coverage, encoding, recode).
- Sanity gates all passed: blacklist absent; EXERANY2 in / _TOTINDA out; HAVARTH5 in / _DRDXAR3 out; JOINPAI2 & POORHLTH present; 437,532 valid / 72,741 positives / 16.63%; zero residual sentinels (value-contract scan); all kept columns ≥25% effective coverage (min SOMALE 25.23%).
- Missingness after recode is **20.54% overall** (worst: SOMALE 74.8%, PREDIAB1 74.0%) — the P3 imputation strategy must be **fold-safe**.
- **Any change to the whitelist from here on requires a new ADR.** [[Column triage]] status → FINALIZED v1; [[Leakage watchlist]] Tier 2 closed.
- 89 codebook-legal `_BMI5` values > 80.0 are kept but flagged for attention in P3/P5.
