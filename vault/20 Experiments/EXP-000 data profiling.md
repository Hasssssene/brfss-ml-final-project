# EXP-000 — Data profiling (shape, missingness, sentinels, target, associations, leakage probe)

- **exp_id:** EXP-000
- **date:** 2026-10-25
- **phase:** P1
- **status:** done
- **related ADRs:** [[ADR-001 binary target framing]], [[ADR-004 tier-3 leakage resolution]]
- **log row:** `results/experiments_log.csv` (EXP-000, 2026-10-25T17:30:54)

## Question
What does the raw BRFSS 2021 file actually contain before any cleaning — shape and dtypes, per-column missingness and sentinel load, target distribution under the ADR-001 framing, feature–target associations, and the empirical leakage status of the [[Leakage watchlist]] suspects?

## Setup
- Model / method: **none — descriptive profiling** (`src/profile_data.py`, run once on the raw CSV)
- Feature set: all 303 raw columns (described, not modeled)
- Imbalance strategy: n/a
- Hyperparameters: n/a
- Validation: n/a (no model, no folds; the fold scheme is untouched)

Artifacts (all committed):
- `results/profiling/column_census.csv` — 303 rows: dtype, blanks, sentinel counts, n distinct, top-5 values
- `results/profiling/target_distribution.csv` — GENHLTH codes and binary y
- `results/profiling/association_screen.csv` — Spearman ρ vs y for the 300 columns with defined ρ
- `results/profiling/leakage_probe.csv` — `_RFHLTH` vs target crosstab

## Results
*No fold AUCs — this is a profiling run (log row has empty fold columns by design).*

**Shape & target (matches [[ADR-001 binary target framing]] expectations)**
- 438,693 rows × 303 columns, all numeric dtype.
- Valid target rows: **437,532**; dropped per ADR-001: **1,161** (788 Don't know + 369 Refused + 4 blank).
- y = 1 (Fair/Poor): **72,741 → 16.63%** positive rate; y = 0: 364,791.

**Leakage probe (watchlist Tier 1)**
- `_RFHLTH` vs y: **100.0% agreement** (437,532 / 437,532; crosstab perfectly diagonal) — blacklist **empirically confirmed**.
- Coding correction: in `_RFHLTH`, **2 = Fair/Poor (y = 1)** and 1 = Good or better (y = 0); the reversed mapping agrees on 0 rows.
- `GENHLTH` itself: Spearman ρ = **+0.672** vs y — it is the target source; added to the explicit feature-exclusion list (see [[ADR-004 tier-3 leakage resolution]]).

**Top non-blacklisted Spearman ρ vs y** (coverage = n_used / 437,532 valid rows)
| Column | ρ | n_used | Coverage |
|---|---|---|---|
| DIFFWALK | −0.413 | 417,855 | 95.5% |
| JOINPAI2 | +0.325 | 122,864 | 28.1% |
| DIFFALON | −0.319 | 416,268 | 95.2% |
| CDASSIST | −0.316 | 4,292 | ~1.0% |
| CDHOUSE | −0.307 | 4,273 | ~1.0% |
| INCOME3 | −0.296 | 236,712 | 54.1% |
| PHYSHLTH | −0.293 | 421,384 | 96.3% |
| LMTJOIN3 | −0.290 | 140,700 | 32.2% |
| EMPLOY1 | +0.286 | 298,279 | 68.2% |

- **Caregiver micro-module trap:** CDASSIST / CDHOUSE / CDSOCIAL show |ρ| = 0.28–0.32 but were asked of only ~1% of respondents (~4.2–4.3k rows, a cognitively-impaired subpopulation) — the association is a coverage artifact, not general signal. Candidate **DROP-SPARSE** in [[Column triage]] (per [[ADR-004 tier-3 leakage resolution]]).
- **Tier-3 health-days inputs:** PHYSHLTH ρ = −0.293 (96.3% cov), MENTHLTH ρ = −0.099, POORHLTH ρ = −0.104 (only 50.8% cov) → all **kept** per [[ADR-004 tier-3 leakage resolution]].

**Missingness census**
- **92 / 303** columns >90% blank+sentinel; **155 / 303** >50%.
- **3 completely empty:** TOLDCFS, HAVECFS, WORKCFS.
- **247 / 303** contain ≥1 sentinel code (7/77/777, 9/99/999, 7777/9999 family).
- **11 constant columns** (≤1 distinct value after sentinel exclusion → DROP-MODE/admin): COLGHOUS, STATERE1, CELPHON1, SAFETIME, CTELNUM1, CELLFON5, CADULT1, CCLGHOUS + the 3 empty CFS columns; near-constant admin vars (e.g. IYEAR, 2 values) drop alongside.
- Association screen covers 300/303 columns (the 3 empty ones excluded).

**Duplicate evidence (watchlist Tier 2)**
- **EXERANY2 ≡ _TOTINDA**: identical n_used (436,659) and identical ρ = 0.2465 — same information twice.
- **_DRDXAR3 ≡ HAVARTH5**: identical n_used (434,721), identical ρ = −0.2172, identical 0.66% missing rate.

**Day-count sentinels**
- 88 ("none") dominates: PHYSHLTH top value is 88 with **287,796** occurrences (≈66% of all rows) → **88→0 recode mandatory** before modeling ([[ADR-004 tier-3 leakage resolution]], [[Codebook highlights]]).

Figures: `figures/exp000_target_distribution.png`, `figures/exp000_missingness_top40.png`, `figures/exp000_association_top20.png`

## Verdict
Profiling complete; P1 evidence closed. The dataset matches ADR-001 expectations (16.63% positives); the `_RFHLTH` blacklist is empirically airtight (100% target agreement); the whitelist must explicitly exclude **GENHLTH, _RFHLTH, _PHYS14D, _MENT14D**. Feeds: [[Column triage]] EXP-000 evidence section + [[ADR-004 tier-3 leakage resolution]]. **Next: P2 matrix build** — sentinel→NaN recodes, 88→0, ordinal/nominal encoding plan, whitelist → `data/interim/` parquet checkpoint; then EXP-001 Dummy baseline.
