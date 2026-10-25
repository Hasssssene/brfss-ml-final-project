# Leakage Watchlist

> [!danger] Columns here must NEVER appear in the feature matrix. Audited by qa-reviewer, enforced by `src/preprocessing.py` (final authority: the drop-list in code).

## Tier 1 — Direct target derivation (absolute)
| Column | Why | Status |
|---|---|---|
| `GENHLTH` | **The target source itself** (y is its recode; EXP-000: ρ = +0.672 vs y) | 🔴 blacklist — target, excluded from whitelist per [[ADR-004 tier-3 leakage resolution]] |
| `_RFHLTH` | Calculated directly from GENHLTH: adults with fair/poor health — **it IS the target** | 🔴 blacklist — **empirically confirmed (EXP-000)**: 100.0% agreement with y, crosstab perfectly diagonal. Coding note: **2 = Fair/Poor (y = 1)**, 1 = Good or better |
| `_RFHLTH`-adjacent health-status composites | Derived from the same self-report construct family | 🔴 blacklist |

## Tier 2 — Derived from features we keep (duplicate information → drop one side)
*All rows resolved 2026-10-25 by [[ADR-005 whitelist and matrix v1]] (rule 6 keep-raw preference + EXP-000 duplicate proofs); per-column record: `results/profiling/triage_decisions.csv`.*

| Column | Duplicate of | Rule | Status |
|---|---|---|---|
| `_PHYS14D`, `_MENT14D` | binned `PHYSHLTH` / `MENTHLTH` (GENHLTH-family derived) | drop derived | ✅ **resolved (ADR-005)**: DROP-LEAK rule 1 — hard blacklist; raws `PHYSHLTH`/`MENTHLTH` kept |
| `_BMI5CAT`, `_RFBMI5` | binned `_BMI5` | keep `_BMI5` continuous; drop bins | ✅ **resolved**: dropped (DROP-DUP); `_BMI5` kept as ENG ÷100 |
| `_SMOKER3` vs `SMOKE100` + `SMOKDAY2`, `_RFSMOK3` | smoking representations | keep ONE | ✅ **resolved**: kept `_SMOKER3` (ORD 1–4); dropped `SMOKE100`, `SMOKDAY2`, `_RFSMOK3` |
| `_AGEG5YR`, `_AGE65YR`, `_AGE80` | age bins vs continuous | keep `_AGE80` (+ optionally `_AGEG5YR` as ordinal) | ✅ **resolved**: `_AGE80` only (KEEP-NUM, imputed, no sentinels); all bins dropped (incl. `_AGE_G`) |
| `_RACE`, `_RACEG21`, `_RACEGR3`, `MRACASC1` | race groupings | keep ONE (`_RACE`) | ✅ **resolved**: kept `_RACE` (NOM 1–8, code 7 = Multiracial valid); dropped `_PRACE1`, `_MRACE1`, `_HISPANC`, `_RACEG21`, `_RACEGR3`, `_RACEPRV`, `_IMPRACE` (MRACASC1 absent from 2021) |
| `_EDUCAG` | `EDUCA` | keep `EDUCA` ordinal | ✅ **resolved**: dropped; `EDUCA` kept (ORD) |
| `_INCOMG`, `_IMPINC` | `INCOME3` | keep `INCOME3` ordinal | ✅ **resolved**: dropped `_INCOMG1`; `INCOME3` kept (ORD 1–11; codes 7/9 are valid brackets); `_IMPINC` absent from 2021 |
| `_HLTHPL1` | `PRIMINSR` grouping | keep ONE | ✅ **resolved**: absent from 2021; its counterparts `_HLTHPLN` + `_HCVU652` dropped (DROP-DUP); `PRIMINSR` kept (NOM, 11 categories) |
| `_PA...` (activity indices), `_PACAT_`, `_MINAC...` | `EXERANY2` + `_TOTINDA` family | keep ONE representation | ✅ **resolved**: `_TOTINDA` dropped (≡ `EXERANY2` proven, EXP-000: identical n = 436,659, identical ρ = 0.2465); `EXERANY2` kept (NOM); the `_PA...` index family is absent from 2021 |
| `_DRDXAR3` | `HAVARTH5` (derived arthritis dx count vs ever-told) | keep ONE representation | ✅ **resolved**: dropped (EXP-000: identical n = 434,721, identical ρ = −0.2172); `HAVARTH5` kept (NOM) |
| `_DRNKWK`, `_RFDRHV7`, `DRNKANY5` + AUDITC items | alcohol computed vs raw | keep ONE | ✅ **resolved**: kept raws `ALCDAY5`/`AVEDRNK3`/`DRNK3GE5`/`MAXDRNKS`; dropped `DRNKANY5`, `DROCDY3_`, `_RFBING5`, `_DRNKWK1`, `_RFDRHV7` |
| `_SEX` vs `SEXVAR` | identical info | keep `SEXVAR` | ✅ **resolved**: `_SEX` dropped; `SEXVAR` kept |
| `_FLSHOT7` etc. computed vaccine flags | raw `FLUSHOT7`-style items | keep raw | ✅ **resolved**: dropped `_FLSHOT7`, `_PNEUMO3`, `_AIDTST4`; kept raw `FLUSHOT7`, `PNEUVAC4`, `HIVTST7` |

> [!warning] Codebook trap (learned in P2, binding for all future recodes): codes 7 and 9 are **not** always sentinels. Valid 7/9 codes exist in EMPLOY1 (7 = "Retired", 131,546 rows), INCOME3 / PRIMINSR / IMFVPLA2 / JOINPAI2 (7/9 = valid brackets/places/ratings), `_RACE` (7 = Multiracial), `_STATE` (9 = Connecticut), `_AGE80` (77 = age 77) — and 6-digit sentinels 777777/999999 exist (FLSHTMY3, HIVTSTD3). Sentinel recoding must therefore use **per-column value contracts** (codebook-exact sets, as implemented and asserted in `src/preprocessing.py`), never generic 7/9/77/99 pattern matching.

## Tier 3 — Construct-suspicion (same latent construct as target; decide with data evidence)
*Resolved 2026-10-25 by [[ADR-004 tier-3 leakage resolution]] using EXP-000 numbers (contrast case: `_RFHLTH` = 100.0% target agreement is what true derivation looks like; none of these is GENHLTH-derived).*

| Column | Concern | Status |
|---|---|---|
| `POORHLTH` (days poor health kept you from usual activities) | conceptually near-identical to GENHLTH; BUT not derived from it | 🟢 **keep per ADR-004** — ρ = −0.104 (EXP-000); 49.2% missing → sentinel→NaN + imputation; final fate decided empirically by P5 feature selection |
| `PHYSHLTH`, `MENTHLTH` (unhealthy days) | strongly associated but legitimate predictors of *general* health | 🟢 **keep per ADR-004** — distinct constructs (ρ = −0.293 / −0.099, EXP-000); 88→0 recode mandatory; monitor |
| `_RFSEAT2`/seatbelt, `_DRNKDY`_ etc. | behavioral, no leakage risk | 🟢 |

## Rules
1. Feature matrix is built by **whitelist** (from [[Column triage]]), not by blacklisting — a typo in a drop-list must not leak the target.
2. Any Tier-2/🟡 decision gets finalized in the triage note during the P2 matrix build and recorded here. Tier-3 was resolved 2026-10-25 via [[ADR-004 tier-3 leakage resolution]]; Tier-2 was resolved 2026-10-25 via [[ADR-005 whitelist and matrix v1]].
3. qa-reviewer cross-checks matrix columns against this list at every audit.
