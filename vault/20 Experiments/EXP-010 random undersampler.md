# EXP-010 — RandomUnderSampler(0.5) inside folds (imbalance strategy 3/3)

- **exp_id:** EXP-010a (XGBoost) + EXP-010b (LogReg contrast)
- **date:** 2026-11-02
- **phase:** P4
- **status:** done
- **related ADRs:** [[ADR-005 whitelist and matrix v1]]
- **log row:** `results/experiments_log.csv` (EXP-010a 2026-11-02T16:18:10, EXP-010b 2026-11-02T16:18:31)

## Question
Does fold-internal majority-class undersampling hold ROC-AUC steady (and save compute), or does throwing away ~60% of majority rows cost ranking quality?

## Setup
- Method: `RandomUnderSampler(sampling_strategy=0.5, random_state=42)` **inside the CV folds** (imblearn `Pipeline`) — covers the majority-class rubric angle
- Budget per training fold: majority drops **328,312 → 130,932** rows (train 393,778 → 196,398); RUS **accepts NaN** (verified on imblearn 0.14.2), so family A **keeps its NaN-passthrough preprocessor** — no forced deviation here (contrast [[EXP-009 smote]])
- **EXP-010a — family A** ([[EXP-007 xgboost]] config otherwise identical); **EXP-010b — family B** (LogReg C=0.01, baseline [[EXP-002 logistic regression]], same impute prep)
- Validation: `StratifiedKFold(10, shuffle=True, random_state=42)` via `src/config.py:make_folds()` — **folds_sha256 = `3b3a17457039`**

## Results
| exp_id | model | mean CV ROC-AUC | fold std | ΔAUC vs default | runtime |
|---|---|---|---|---|---|
| EXP-010a | XGBClassifier + RUS(0.5) | 0.903096 | 0.001674 | **−0.000261** | **73.0 s** |
| EXP-010b | LogisticRegression + RUS(0.5) | 0.892765 | 0.001610 | **+0.000177** | 11.6 s |

EXP-010a folds: 0.902766, 0.904760, 0.904105, 0.902867, 0.904584, 0.899645, 0.901843, 0.901298, 0.905234, 0.903860
EXP-010b folds: 0.892361, 0.893397, 0.893516, 0.892571, 0.894132, 0.889934, 0.891202, 0.890944, 0.895456, 0.894135

Figures: none for this EXP.

## Verdict
**AUC-neutral (|ΔAUC| ≤ 0.0003, ≈0.15× fold std) — but 1.9× faster:** EXP-010a wall time **73.0 s vs 138.3 s** for the identical XGBoost on full folds. Confirms the P4 verdict (with [[EXP-008 class weighting]], [[EXP-009 smote]]): resampling reshapes the class prior, not the ranking → **`imbalance_strategy = none` for the AUC objective**. **Side-finding adopted for P6:** RUS(0.5) is a legitimate *acceleration trick* — random/tuning searches can iterate ~2× cheaper at −0.0003 AUC, with the final selected config always re-validated on full folds.
