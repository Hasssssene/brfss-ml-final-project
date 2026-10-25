# EXP-xxx — <short name>

- **exp_id:** EXP-xxx
- **date:** YYYY-MM-DD
- **phase:** P1|P2|P3|P4|P5|P6|P7
- **status:** planned|running|done|failed
- **related ADRs:** [[ADR-001 ...]]
- **log row:** `results/experiments_log.csv` (exp_id column)

## Question
What is this experiment trying to answer?

## Setup
- Model / method:
- Feature set: all | selected (which) | PCA (components)
- Imbalance strategy: none | class_weight | SMOTE
- Hyperparameters:
- Validation: `StratifiedKFold(10, shuffle=True, random_state=42)` via `src/config.py`

## Results
| fold | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | mean |
|------|---|---|---|---|---|---|---|---|---|----|------|
| ROC-AUC | | | | | | | | | | | |

Figures: `figures/...`

## Verdict
What did we learn? Adopt / reject / iterate. Next step.
