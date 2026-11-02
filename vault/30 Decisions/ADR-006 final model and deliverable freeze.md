# ADR-006 — Final model and deliverable freeze

- **date:** 2026-11-02
- **status:** accepted
- **tags:** #adr #final #deliverables
- **scope:** final recommended model / graded estimate / deliverable artifacts (P8 close-out)

## Context
P4–P6 compared 7 model families on identical canonical folds (sha `3b3a17457039`), 3 imbalance techniques, 3 selection/compression strategies, automated tuning (EXP-014) and nested validation (EXP-015), with Wilcoxon signed-rank tests on the paired folds ([[STATISTICAL COMPARISONS]]). P7 confirmed the winner via error analysis and SHAP XAI on its OOF predictions. P8 packaged everything into the two graded deliverables and the pre-submission QA audit passed. A freeze decision is needed: the submission deadline (2026-11-08) leaves only human actions, and any further model churn now costs review time and risks introducing an unverified change after the audit.

## Decision
1. **Final recommended model: tuned LightGBM — [[EXP-014 randomized search tuned finals|EXP-014a]]**, parameters verbatim in `deliverables/code.py` (`TUNED_LGBM_PARAMS`: n_estimators=512, learning_rate=0.0219, num_leaves=94, min_child_samples=44, subsample=0.816, colsample_bytree=0.598, reg_lambda=3.12, rs=42) inside the NaN-passthrough pipeline (NOM one-hot `handle_unknown='ignore'`; ORD/NUM/ENG passthrough with native missing routing).
2. **Graded estimate, two stated numbers:** canonical 10-fold **0.904321 ± 0.001675** (EXP-014a; the headline, and the exact number `code.py --full` reproduces to ≤4.84e-07 per fold) and the unbiased nested-CV estimate **0.904303** (EXP-015) — optimism +0.000018, negligible. Report §10 cites both with this framing.
3. **Deliverables frozen:** `deliverables/code.py` (903 lines, self-contained, QA-executed `--smoke` + `--full`, deterministic) and `deliverables/report/report.pdf` v1.0 (28 pp, 16 tables, 14 figures, §11 Ethics matching the 12→16-entry AI log). `report.html` retained as the committed build intermediate.
4. **Any further model, fold, feature-set or preprocessing change requires a new ADR** — there is no budget for unaudited changes before submission. Remaining changes are human-only: §11 personalization (`report.md:12,414,416`), figure visual pass, screencast recording, submission form.

## Alternatives considered
- **Tuned XGBoost (EXP-014b), runner-up 0.903843** — rejected: Wilcoxon T1 (paired canonical folds) favours LGBM in 10/10 folds, W=0, p=0.001953 (n=10 exact floor), median paired Δ +0.000471.
- **Best scikit-learn model (HistGradientBoosting, 0.903220)** — rejected on the same family-level test (booster trio tops every sklearn family by ≥0.011 ≈ 6× fold std over linear/bagging; see [[EXP-005 histgradientboosting]]).
- **Keep iterating (e.g. wider search, stacking)** — rejected: nested CV shows search width does not overfit (optimism +0.000018) and 7/10 outer folds re-selected the same winner; expected gains ≪ fold std, cost = post-audit churn.

## Consequences
- Report, screencast script, `code.py` and all vault claims now cite exactly one final model; no number drift possible between deliverables.
- `results/experiments_log.csv` stays at 42 rows; `code.py --full` is a reproduction of EXP-014a, not a new run (verification artifacts: `results/final_run_summary.txt`, `results/final_run_fold_scores.csv`).
- [[Rubric Checklist]] Advanced "outstanding report" carries the caveat *final human read-through pending*; the QA audit is the agent-side gate, the human read is the submission-side one.
- Deferred nits recorded (T1 median rounding display; `requirements.txt` `>=` bounds vs exact verified versions Python 3.13.7 / sklearn 1.9.1 / lightgbm 4.7.0 / pandas 2.3.0 / numpy 2.2.6) — accepted because `code.py --full` fails loudly on environment drift via the fold-hash assert.
