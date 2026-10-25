# ADR-002 — Scope: Basic + Intermediate + Advanced (minus multiclass)

- **date:** 2026-10-25
- **status:** accepted
- **tags:** #adr #scope
- **scope:** experiment plan, rubric coverage

## Context
Rubric: Basic (5 pts), Intermediate (3 pts), Advanced (2 pts). Deadline 2026-11-08. Grading weights reasoning/report quality over raw metric. The user wants full-rubric ambition but chose a **binary-only** main track (no 3-class or 5-class side experiment).

## Decision
1. Cover all Basic + Intermediate items.
2. Advanced points via: **LightGBM + XGBoost** (non-sklearn boosting), **RandomizedSearchCV** automated tuning, **nested cross-validation** for the unbiased final estimate.
3. **No multiclass experiment** (explicitly declined).
4. Local Python development; final deliverable assembled as a single `deliverables/code.py`.

## Alternatives considered
- Basic-only — safe but leaves 5 pts on the table. Rejected.
- Binary + 3-class bonus — maximal rubric coverage, but doubles compute and report surface in a 2-week window. Declined by the student.

## Consequences
- Advanced "problem framing" bullet is forfeited; the other Advanced bullets must all land to compensate (boosting libraries, automated search, nested CV, outstanding report).
- Roadmap P4–P6 ([[Roadmap]]) sized for one model zoo run + imbalance grid + selection + tuning.
