# Column Triage

> [!info] Status: **stub — to be filled after EXP-000** (profiling run). The feature matrix is built from the **keep** list here (whitelist principle, see [[Leakage watchlist]]). #data

## Method
1. EXP-000 profiles every column: dtype, % blank, % sentinel-coded (7/9/77/99/777/999/7777/9999...), n distinct values, top values, association with binary target.
2. Each column gets a verdict + reason in the table below. Verdicts:
   - **KEEP-ORD** keep, ordinal encoding
   - **KEEP-NOM** keep, one-hot encoding
   - **KEEP-NUM** keep, numeric (sentinels → NaN)
   - **ENG** keep, engineered (e.g., BMI, day-count 88→0)
   - **DROP-ID / DROP-MODE / DROP-MODULE / DROP-LEAK / DROP-DUP / DROP-SPARSE** drop, with reason
3. qa-reviewer audits the final matrix against this note.

## Verdict table (filled after EXP-000)
| Column | Section | Verdict | Missing/sentinel % | Reason |
|---|---|---|---|---|
| (pending) | | | | |

## Open questions for triage
- POORHLTH: keep or drop? (construct suspicion, [[Leakage watchlist]] Tier 3) → needs EXP-000 association numbers
- _AGE80 vs _AGEG5YR: continuous only or both?
- Module columns with 10–30% coverage: any worth keeping as "asked=0/1 + answer"?
- INCOME3 vs _IMPINC imputed income coverage comparison
