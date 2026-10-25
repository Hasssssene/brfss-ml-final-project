# Leakage Watchlist

> [!danger] Columns here must NEVER appear in the feature matrix. Audited by qa-reviewer, enforced by `src/preprocessing.py` (final authority: the drop-list in code).

## Tier 1 — Direct target derivation (absolute)
| Column | Why | Status |
|---|---|---|
| `_RFHLTH` | Calculated directly from GENHLTH: adults with fair/poor health — **it IS the target** | 🔴 blacklist |
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
| `_PA...` (activity indices), `_PACAT_`, `_MINAC...` | `EXERANY2` + `_TOTINDA` family | keep ONE representation | 🟡 |
| `_DRNKWK`, `_RFDRHV7`, `DRNKANY5` + AUDITC items | alcohol computed vs raw | keep ONE | 🟡 |
| `_SEX` vs `SEXVAR` | identical info | keep `SEXVAR` | 🟡 |
| `_FLSHOT7` etc. computed vaccine flags | raw `FLUSHOT7`-style items | keep raw | 🟡 |

## Tier 3 — Construct-suspicion (same latent construct as target; decide with data evidence)
| Column | Concern | Status |
|---|---|---|
| `POORHLTH` (days poor health kept you from usual activities) | conceptually near-identical to GENHLTH; BUT not derived from it | 🟠 measure assoc. in EXP-000, then ADR |
| `PHYSHLTH`, `MENTHLTH` (unhealthy days) | strongly associated but legitimate predictors of *general* health | 🟢 keep (distinct construct), monitor |
| `_RFSEAT2`/seatbelt, `_DRNKDY`_ etc. | behavioral, no leakage risk | 🟢 |

## Rules
1. Feature matrix is built by **whitelist** (from [[Column triage]]), not by blacklisting — a typo in a drop-list must not leak the target.
2. Any Tier-2/🟡 decision gets finalized in the triage note after EXP-000 and recorded here.
3. qa-reviewer cross-checks matrix columns against this list at every audit.
