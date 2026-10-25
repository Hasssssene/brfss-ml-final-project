# Column Triage

> [!info] Status: **EXP-000 evidence recorded 2026-10-25 — per-column verdict table to be filled during the P2 whitelist build**. The feature matrix is built from the **keep** list here (whitelist principle, see [[Leakage watchlist]]). #data

## Method
1. EXP-000 profiles every column: dtype, % blank, % sentinel-coded (7/9/77/99/777/999/7777/9999...), n distinct values, top values, association with binary target.
2. Each column gets a verdict + reason in the table below. Verdicts:
   - **KEEP-ORD** keep, ordinal encoding
   - **KEEP-NOM** keep, one-hot encoding
   - **KEEP-NUM** keep, numeric (sentinels → NaN)
   - **ENG** keep, engineered (e.g., BMI, day-count 88→0)
   - **DROP-ID / DROP-MODE / DROP-MODULE / DROP-LEAK / DROP-DUP / DROP-SPARSE** drop, with reason
3. qa-reviewer audits the final matrix against this note.

## EXP-000 evidence (verified 2026-10-25, [[EXP-000 data profiling]])

**Shape & missingness**
- 438,693 rows × **303 columns** (all numeric dtype); 437,532 valid target rows (1,161 dropped per [[ADR-001 binary target framing]]).
- **92/303** columns >90% blank+sentinel → default DROP-SPARSE/DROP-MODULE territory; **155/303** >50%.
- **3 completely empty:** TOLDCFS, HAVECFS, WORKCFS (CFS module never asked) → DROP-SPARSE.
- **247/303** contain ≥1 sentinel code → sentinel→NaN recode is the default path, not the exception.
- **11 constant** (≤1 distinct value post-sentinels): COLGHOUS, STATERE1, CELPHON1, SAFETIME, CTELNUM1, CELLFON5, CADULT1, CCLGHOUS + the 3 empty CFS cols → DROP-MODE/admin; near-constant admin vars (e.g. IYEAR, 2 values) drop alongside → DROP-ID/DROP-MODE.

**Associations vs y (top non-blacklisted, Spearman)** — DIFFWALK −0.413 · JOINPAI2 +0.325 (28.1% cov) · DIFFALON −0.319 · INCOME3 −0.296 (54.1% cov) · PHYSHLTH −0.293 (96.3% cov) · LMTJOIN3 −0.290 (32.2% cov) · EMPLOY1 +0.286 (68.2% cov). GENHLTH itself ρ=+0.672 (target source — excluded, see [[ADR-004 tier-3 leakage resolution]]).

**Micro-coverage traps:** CDASSIST −0.316, CDHOUSE −0.307, CDSOCIAL −0.283 — but only ~1.0% asked (4.2–4.3k rows, caregiver module) → candidate **DROP-SPARSE** per [[ADR-004 tier-3 leakage resolution]].

**Confirmed duplicate pairs (Tier-2 evidence):** EXERANY2 ≡ _TOTINDA (identical n = 436,659, identical ρ = 0.2465); _DRDXAR3 ≡ HAVARTH5 (identical n = 434,721, identical ρ = −0.2172) → keep ONE of each in the whitelist.

**Day-count 88 code:** PHYSHLTH has 287,796 rows coded 88 ("none") → 88→0 recode mandatory before modeling ([[ADR-004 tier-3 leakage resolution]]).

**Hard exclusions for the whitelist build:** GENHLTH, _RFHLTH, _PHYS14D, _MENT14D ([[ADR-004 tier-3 leakage resolution]]; `_RFHLTH` empirically 100.0% = target, EXP-000).

## Verdict table (filled after EXP-000)
| Column | Section | Verdict | Missing/sentinel % | Reason |
|---|---|---|---|---|
| (pending — filled during P2 whitelist build) | | | | |

## Open questions for triage
- ~~POORHLTH: keep or drop?~~ **RESOLVED → KEEP** per [[ADR-004 tier-3 leakage resolution]] (ρ = −0.104, distinct construct, not GENHLTH-derived; 49.2% missing → sentinel→NaN + imputation; final fate decided empirically in P5 selection)
- _AGE80 vs _AGEG5YR: continuous only or both? → settle in P2 matrix build
- Module columns with 10–30% coverage: any worth keeping as "asked=0/1 + answer"? → settle in P2 matrix build
- Duplicate-pair choices (which side of EXERANY2≡_TOTINDA, _DRDXAR3≡HAVARTH5, INCOME3 vs _IMPINC, etc.) → settle in P2 matrix build
