# EXP-009 — SMOTE(0.5) inside folds (imbalance strategy 2/3)

- **exp_id:** EXP-009a (XGBoost) + EXP-009b (LogReg contrast)
- **date:** 2026-11-02
- **phase:** P4
- **status:** done
- **related ADRs:** [[ADR-005 whitelist and matrix v1]]
- **log row:** `results/experiments_log.csv` (EXP-009a 2026-11-02T16:13:01, EXP-009b 2026-11-02T16:16:51)

## Question
Does fold-internal oversampling of the minority class via SMOTE improve mean CV ROC-AUC — or does synthetic interpolation hurt the ranking?

## Setup
- Method: `SMOTE(sampling_strategy=0.5, k_neighbors=5, random_state=42)` **inside the CV folds** (imblearn `Pipeline` → fit on train folds only, test folds untouched)
- Why 0.5 and not 1:1: per training fold 65,466 pos / 328,312 neg — a full 1:1 rebalance would synthesize **262,846 rows** (train 393,778 → 656,624); 0.5 caps synthesis at **98,690** (492,468) — the standard compromise between class signal and synthetic density
- **EXP-009a — family A** ([[EXP-007 xgboost]] config otherwise identical). **Documented deviation:** SMOTE rejects NaN (imblearn 0.14.2 raises `Input X contains NaN`), so this run uses the **P3 impute preprocessor** instead of the Part-1 NaN-passthrough — see caveat below.
- **EXP-009b — family B** (LogReg C=0.01, baseline [[EXP-002 logistic regression]]): pipeline order prep → StandardScaler → **SMOTE → LR**, so SMOTE distances see standardized features (not dominated by large-scale NUM columns).
- Validation: `StratifiedKFold(10, shuffle=True, random_state=42)` via `src/config.py:make_folds()` — **folds_sha256 = `3b3a17457039`**

## Results
| exp_id | model | mean CV ROC-AUC | fold std | ΔAUC vs default | runtime |
|---|---|---|---|---|---|
| EXP-009a | XGBClassifier + SMOTE(0.5) | 0.899803 | 0.001828 | **−0.003554** | 373.5 s |
| EXP-009b | LogisticRegression + SMOTE(0.5) | 0.888419 | 0.001545 | **−0.004169** | 207.9 s |

EXP-009a folds: 0.899122, 0.901391, 0.900687, 0.899033, 0.901392, 0.896815, 0.897564, 0.898173, 0.902599, 0.901257
EXP-009b folds: 0.887851, 0.889954, 0.888067, 0.888401, 0.890230, 0.886173, 0.886483, 0.886709, 0.890063, 0.890255

Figures: none for this EXP.

## Verdict
**Worst imbalance strategy on BOTH families** — the only P4 comparison where a booster drops below its default by ~2× fold std, and **the only P4 run to lose to the LogReg baseline** (EXP-009b −0.0042). Interpretation: SMOTE's synthetic minority points blur the decision boundary — they change the class prior without making the *ranking* of true positives above negatives any better, which is all ROC-AUC measures. It is also the most expensive run of P4 (373.5 s, 2.7× XGB default). **Caveat (honest):** EXP-009a's Δ conflates the strategy with the forced preprocessing switch (nanpass → impute), so part of the drop may be imputation, not SMOTE; EXP-009b (same prep as its baseline, −0.0042) shows the damage is real for SMOTE itself. **Reject for the AUC objective** — feeds the P4 verdict `imbalance_strategy = none` ([[EXP-008 class weighting]], [[EXP-010 random undersampler]]).
