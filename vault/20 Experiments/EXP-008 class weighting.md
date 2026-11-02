# EXP-008 — Class weighting vs default weights (imbalance strategy 1/3)

- **exp_id:** EXP-008a (XGBoost) + EXP-008b (LogReg contrast)
- **date:** 2026-11-02
- **phase:** P4
- **status:** done
- **related ADRs:** [[ADR-005 whitelist and matrix v1]]
- **log row:** `results/experiments_log.csv` (EXP-008a 2026-11-02T16:06:21, EXP-008b 2026-11-02T16:06:39)

## Question
Does explicitly reweighting the minority class improve mean CV ROC-AUC on this 16.63%-positive problem, for a booster and for a linear model?

## Setup
- **EXP-008a — family A (zoo winner, [[EXP-007 xgboost]]):** `scale_pos_weight = neg/pos = 364,791/72,741 ≈ 5.0149` from the full-data class prior; stratified folds preserve the ratio in every training fold by construction, so this equals the per-fold value to rounding. Config otherwise identical to EXP-007; NaN-passthrough preprocessing kept.
- **EXP-008b — family B (LogReg C=0.01, baseline [[EXP-002 logistic regression]]):** sklearn `class_weight='balanced'` (computed inside fit on the train fold); same StandardScaler + impute prep as EXP-002a.
- Validation: `StratifiedKFold(10, shuffle=True, random_state=42)` via `src/config.py:make_folds()` — **folds_sha256 = `3b3a17457039`**; ΔAUC computed against the same-model default-weight run on the same folds.

## Results
| exp_id | model | weighting | mean CV ROC-AUC | fold std | ΔAUC vs default | runtime |
|---|---|---|---|---|---|---|
| EXP-008a | XGBClassifier | scale_pos_weight=5.0149 | 0.903261 | 0.001733 | **−0.000096** | 136.0 s |
| EXP-008b | LogisticRegression | class_weight=balanced | 0.893035 | 0.001577 | **+0.000447** | 12.7 s |

EXP-008a folds: 0.903165, 0.904784, 0.904264, 0.902496, 0.904891, 0.900186, 0.901936, 0.900857, 0.905416, 0.904613
EXP-008b folds: 0.892604, 0.893821, 0.893840, 0.892784, 0.894473, 0.890283, 0.891533, 0.891178, 0.895592, 0.894243

Figures: none for this EXP.

## Verdict
**No effect on AUC in either family** — |ΔAUC| ≤ 0.00045, at most ~0.3× fold std (XGB slightly hurt, LogReg slightly helped; both noise-level). Expected and now measured: **ROC-AUC is threshold-free and rank-based** — class weighting reshapes the class prior and moves the implicit operating point, but does not improve the *ranking* of positives above negatives. Feeds the P4 imbalance verdict (with [[EXP-009 smote]], [[EXP-010 random undersampler]]): **`imbalance_strategy = none` for the AUC objective**. Class weighting stays in the report as the cheapest technique to compare (rubric "imbalance handling" evidenced) and the honest negative result.
