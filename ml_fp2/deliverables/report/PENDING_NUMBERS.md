# PENDING NUMBERS — report.md fill-in checklist

Status after the 2026-11-08 number-filling pass. Rule unchanged: a placeholder may
ONLY be filled with a number that exists in `experiments.csv` (cite the EXP id) or a
figure/notebook output that actually exists. Never invent or extrapolate.

## Filled this pass (verify against sources before submission)

- [x] §7 Table 7 EXP-014 row (0.889475 ± 0.001468), Figure 4 embed, §7.2 final verdict
      (SMOTE Δ −0.0040 vs EXP-004; no imbalance technique helps AUC).
- [x] §8 rewritten: H1 blocks (39/76 cols) stated-before-testing, Table 8 (EXP-010/015/
      016/017/018/019), Figure 5 embed, verdict "keep all 76", gain-importance reading.
- [x] §9 rewritten: search design (RandomizedSearchCV n_iter=10, inner 3-fold rs=42),
      best params + inner scores, Table 9 (EXP-020/021 vs EXP-010/011), Wilcoxon Table 10
      (three pairs, exact p; median Δ; fold wins), nested CV EXP-022 with zero-optimism
      mechanism explained.
- [x] §10 rewritten: OOF AUC 0.903667 (vs EXP-020), error taxonomy (77.9% / 92.5% /
      88.1%; 48,218 of 437,532 = 11.0%), Cases A (row 307808) and B (row 393280),
      Figure 6 embed, SHAP global top-12 (Figure 7), local SHAP Figures 8A/8B.
- [x] §11 rewritten: 8-point summary, final recommendation (tuned LGBM 0.903672,
      EXP-020), Known limitations (FLSHTMY3 777777 wart; canonical log =
      ml_fp2/experiments.csv; self-report ceiling; no calibration; demographic
      associations; whitelist bound), "what we would do next" (FLSHTMY3 fix,
      calibration, 5-class).
- [x] §6 honesty fix: LogReg C grid presented as FLAT/insensitive (0.893493/0.893491/
      0.893499, spread 0.000008), "kept C=0.01, no tuning win".
- [x] §5.4 determinism note sharpened: three independent re-runs → bit-identical fold
      scores; `experiments.csv` labeled the canonical log.
- [x] Abstract: one-sentence final-result summary added (EXP-020).
- [x] §12.1 table refreshed word-for-word from the current `ml_fp2/AI_ASSISTANCE_LOG.md`
      (12 rows) + two † rows (report drafting; tonight's fill-in pass); §12.2 Purpose
      extended with item (5).

## Remaining placeholders

1. **§10 Case C** — `[PENDING: case C profile]`: Table 11 row C (row index / true class),
   the Case C paragraph in §10.2, and the Figure 8C caption reading. Numbers arrive tonight.
2. **[STUDENT NAME]** — header (line 6) and §12.2 signature (plus the "[Adjust the
   previous sentence…]" note directly under it).
3. **code.py** — export the notebook to `deliverables/code.py` (course requires .py, not
   .ipynb); check the §1 deliverables sentence still matches.
4. **PDF** — compile `report.md` → `report.pdf` (not done in this pass, per instructions).

## Also outstanding (discovered this pass)

- Append the two † rows of §12.1 (report drafting; tonight's number-filling pass)
  verbatim to `ml_fp2/AI_ASSISTANCE_LOG.md` — the log file itself does not contain them
  yet, and course policy requires the log and the report section to match.
