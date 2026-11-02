# EXP-005 — HistGradientBoosting (sklearn booster of the P4 zoo)

- **exp_id:** EXP-005
- **date:** 2026-11-02
- **phase:** P4
- **status:** done
- **related ADRs:** [[ADR-005 whitelist and matrix v1]]
- **log row:** `results/experiments_log.csv` (EXP-005, 2026-11-02T16:00:32)

## Question
What does sklearn's own gradient booster deliver on matrix v1 — and does running it without any imputation (NaN-passthrough) work?

## Setup
- Model / method: `HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, random_state=42)`
  - sklearn default `early_stopping='auto'` is **ACTIVE** at this sample size (n_train > 10k), so 300 `max_iter` may stop earlier — stock family behaviour, documented, not a tuning choice
- Feature set: all-76 (`clean_v1.parquet`)
- Imbalance strategy: none
- Preprocessing: **NaN-passthrough** variant (new in P4, `build_nan_passthrough_preprocessor()`): no imputer anywhere — verified on sklearn 1.9.1 that `OneHotEncoder(handle_unknown='ignore')` encodes NaN as its **own category** (e.g. SOMALE → categories [1.0, 2.0, 3.0, 4.0, nan]; **51/54 NOM columns carry NaN**, verified against `clean_v1.parquet`) and ORD/NUM/ENG pass through with NaN intact, which HistGB routes natively at split time
- Validation: `StratifiedKFold(10, shuffle=True, random_state=42)` via `src/config.py:make_folds()` — **folds_sha256 = `3b3a17457039`**

## Results
| fold | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | mean |
|------|---|---|---|---|---|---|---|---|---|----|------|
| ROC-AUC | 0.902916 | 0.904437 | 0.904351 | 0.902754 | 0.904975 | 0.899863 | 0.901755 | 0.901152 | 0.905430 | 0.904570 | **0.903220** |

- fold std **0.001748**; runtime **199.9 s**

Figures: none for this EXP (comparison figures land with the report stage).

## Verdict
**First family to clearly beat linear:** +**0.01063** over LogReg (0.892588) ≈ **6.1× fold std** — a real family-level improvement, not fold noise. Trails XGBoost by only 0.00014 (≈ 0.08× fold std): the booster trio is **statistically indistinguishable**; HistGB loses the zoo only on the deterministic tie-break and on runtime (200 s vs LightGBM's 41 s). The NaN-passthrough answer is **yes**: HistGB ran with **zero imputation**, missingness stays signal instead of being blurred by most-frequent imputation — a core reason boosters suit module-sparse survey data. See [[EXP-006 lightgbm]], [[EXP-007 xgboost]] for the non-sklearn comparison.
