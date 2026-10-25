# Leakage Watchlist

> [!danger] Columns here must NEVER appear in the feature matrix. Audited by qa-reviewer, enforced by `src/preprocessing.py` (final authority: the drop-list in code).

## Tier 1 — Direct target derivation (absolute)
| Column | Why | Status |
|---|---|---|
| `GENHLTH` | **The target source itself** (y is its recode; EXP-000: ρ = +0.672 vs y) | 🔴 blacklist — target, excluded from whitelist per [[ADR-004 tier-3 leakage resolution]] |
| `_RFHLTH` | Calculated directly from GENHLTH: adults with fair/poor health — **it IS the target** | 🔴 blacklist — **empirically confirmed (EXP-000)**: 100.0% agreement with y, crosstab perfectly diagonal. Coding note: **2 = Fair/Poor (y = 1)**, 1 = Good or better |
| `_RFHLTH`-adjacent health-status composites | Derived from the same self-report construct family | 🔴 blacklist |

## Tier 2 — Derived from features we keep (duplicate information → drop one side)
| Column | Duplicate of | Rule | Status |
|---|---|---|---|
| `_PHYS14D`, `_MENT14D` | binned `PHYSHLTH` / `MENTHLTH` | keep raw vars, drop derived | 🟡 confirm in EXP-000 |
| `_BMI5CAT`, `_RFBMI5` | binned `_BMI5` | keep `_BMI5` continuous; drop bins | 🟡 confirm |
| `_SMOKER3`, `_RFSMOK3` | `SMOKE100` + `SMOKDAY2` | keep ONE representation | 🟡 decide in triage |
| `_AGEG5YR`, `_AGE65YR`, `_AGE80` | age bins vs continuous | keep `_AGE80` (+ optionally `_AGEG5YR` as ordinal) | 🟡 decide |
| `_RACE`, `_RACEG21`, `_RACEGR3`, `MRACASC1` | race groupings | keep ONE (`_RACE`) | 🟡 decide |
| `_EDUCAG` | `EDUCA` | keep `EDUCA` ordinal | 🟡 |
| `_INCOMG`, `_IMPINC` | `INCOME3` | keep `INCOME3` ordinal | 🟡 |
| `_HLTHPL1` | `PRIMINSR` grouping | keep ONE | 🟡 |
| `_PA...` (activity indices), `_PACAT_`, `_MINAC...` | `EXERANY2` + `_TOTINDA` family | keep ONE representation | 🟡 **EXERANY2 ≡ _TOTINDA confirmed duplicate (EXP-000: identical n = 436,659, identical ρ = 0.2465)** — choose in P2 |
| `_DRDXAR3` | `HAVARTH5` (derived arthritis dx count vs ever-told) | keep ONE representation | 🟡 **confirmed duplicate (EXP-000: identical n = 434,721, identical ρ = −0.2172, identical 0.66% missing)** — choose in P2 |
| `_DRNKWK`, `_RFDRHV7`, `DRNKANY5` + AUDITC items | alcohol computed vs raw | keep ONE | 🟡 |
| `_SEX` vs `SEXVAR` | identical info | keep `SEXVAR` | 🟡 |
| `_FLSHOT7` etc. computed vaccine flags | raw `FLUSHOT7`-style items | keep raw | 🟡 |

## Tier 3 — Construct-suspicion (same latent construct as target; decide with data evidence)
*Resolved 2026-10-25 by [[ADR-004 tier-3 leakage resolution]] using EXP-000 numbers (contrast case: `_RFHLTH` = 100.0% target agreement is what true derivation looks like; none of these is GENHLTH-derived).*

| Column | Concern | Status |
|---|---|---|
| `POORHLTH` (days poor health kept you from usual activities) | conceptually near-identical to GENHLTH; BUT not derived from it | 🟢 **keep per ADR-004** — ρ = −0.104 (EXP-000); 49.2% missing → sentinel→NaN + imputation; final fate decided empirically by P5 feature selection |
| `PHYSHLTH`, `MENTHLTH` (unhealthy days) | strongly associated but legitimate predictors of *general* health | 🟢 **keep per ADR-004** — distinct constructs (ρ = −0.293 / −0.099, EXP-000); 88→0 recode mandatory; monitor |
| `_RFSEAT2`/seatbelt, `_DRNKDY`_ etc. | behavioral, no leakage risk | 🟢 |

## Rules
1. Feature matrix is built by **whitelist** (from [[Column triage]]), not by blacklisting — a typo in a drop-list must not leak the target.
2. Any Tier-2/🟡 decision gets finalized in the triage note during the P2 matrix build and recorded here. Tier-3 was resolved 2026-10-25 via [[ADR-004 tier-3 leakage resolution]].
3. qa-reviewer cross-checks matrix columns against this list at every audit.
