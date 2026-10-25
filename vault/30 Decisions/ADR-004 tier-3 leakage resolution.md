# ADR-004 — Tier-3 leakage resolution: keep health-days vars, 88→0 recode, micro-coverage drop flags

- **date:** 2026-10-25
- **status:** accepted
- **tags:** #adr #features #data
- **scope:** feature hygiene / Tier-3 leakage resolution

## Context
The [[Leakage watchlist]] Tier 3 flagged the health-days block for a data-driven decision after EXP-000. The probe ([[EXP-000 data profiling]]) delivered the numbers:
- `PHYSHLTH` ρ = −0.293 (96.3% coverage), `MENTHLTH` ρ = −0.099, `POORHLTH` ρ = −0.104 with only 50.8% coverage (49.2% missing).
- Contrast case — `_RFHLTH`: **100.0% agreement** with the binary target (perfectly diagonal crosstab). That is the signature of actual target derivation. None of the three health-days variables is derived from GENHLTH; they measure distinct constructs (days of poor physical / mental health; days of activity limitation).
- Day-count sentinel 88 = "none" (PHYSHLTH: 287,796 of 438,693 rows coded 88) — a valid zero, not a missing code.
- Caregiver module (CDASSIST, CDHOUSE, CDSOCIAL): |ρ| = 0.28–0.32 but asked of only ~1% of respondents — association is an artifact of a tiny selected subpopulation, not general signal.
- `GENHLTH` itself (ρ = +0.672 vs y) is the target source and must never enter features.

## Decision
1. **Keep `PHYSHLTH`, `MENTHLTH`, `POORHLTH` as features.** None is derived from GENHLTH; each is a distinct construct (contrast: `_RFHLTH`'s 100% agreement is what true leakage looks like).
2. **`POORHLTH`:** sentinel→NaN + fold-safe imputation to handle its 49.2% missingness; its final inclusion is decided **empirically by feature selection in P5**, not by decree.
3. **Day-count 88→0 recode is mandatory** before any modeling (PHYSHLTH / MENTHLTH / POORHLTH; 88 means zero days).
4. **Caregiver micro-coverage block** (CDASSIST, CDHOUSE, CDSOCIAL — ~1% asked) → candidate **DROP-SPARSE** in [[Column triage]].
5. **`GENHLTH` itself joins the explicit feature-exclusion list** (target source). The whitelist build must exclude: **GENHLTH, _RFHLTH, _PHYS14D, _MENT14D**.

## Alternatives considered
- **Drop `POORHLTH` now for sparsity** (49.2% missing) — rejected: let P5 feature selection decide empirically; imputation cost is low and the construct is legitimate.
- **Drop `PHYSHLTH` as "too predictive"** (ρ = −0.293) — rejected: it is a legitimate predictor of a *distinct* construct, not a derivation of GENHLTH; predictive strength is the point of the exercise, and it is nowhere near the 100%-agreement leakage signature.

## Consequences
- Tier-3 watchlist rows close as 🟢 keep (`POORHLTH`: keep + impute; final fate in P5).
- P2 cleaning must implement 88→0 for day counts and sentinel→NaN for `POORHLTH` before imputation.
- Triage gains a DROP-SPARSE candidate class backed by EXP-000 coverage numbers.
- The P2 whitelist hard-excludes GENHLTH, _RFHLTH, _PHYS14D, _MENT14D; qa-reviewer audits the matrix against this list and [[Leakage watchlist]].
