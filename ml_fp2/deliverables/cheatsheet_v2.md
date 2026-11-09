# Cheat sheet v2 (submission version) — presenting the rebuild

**Now the rule is inverted: on camera you say v2 numbers only.** v1 is retired;
its numbers are the traps. The right column is why you can answer follow-ups
confidently — you ran v1 end-to-end before rebuilding it yourself.

| Topic | Say this (v2) | Trap (v1 — never say) |
|---|---|---|
| Target | 438,693 × 303 → 437,532 labeled, 72,741 pos = **16.63%** | same in both — safe |
| Floor | Dummy = **0.500000** all 10 folds | same in both — safe |
| Leakage | `_RFHLTH` **100.00%** agreement → dropped | same in both — safe |
| Funnel | leak 4 · admin 35 · constant 11 · sparse 85 · low-cov 27 · dup 65 → **76** | same in both — safe |
| 88 trap | 88 = zero days; 287,796 answers; ρ −0.29 → **+0.450** after recode | v1: +0.445 (close but not yours) |
| Design matrix | 54 nominal one-hot with `dummy_nan` → **322** columns | v1: 76-col matrix, no dummy_nan |
| LogReg | **0.8935**, C-grid flat (spread 0.000008) — "insensitivity, not optimization" | v1: 0.8926 |
| Tree | default **0.6976** → depth 16 **0.8813** | v1: 0.6949 → 0.8802 |
| Zoo | HistGB/LGBM/XGB **0.9033–0.9034**, ≤0.0002 apart — family, not implementation | v1: 0.9032–0.9034 (same story, slightly different digits) |
| Imbalance | weights +0.0005/−0.0002 · **SMOTE −0.0040** → none | same story both; v1 SMOTE −0.0042/−0.0036 |
| H1 | **39 of 76** cols keep **99.5%** (0.8988); complement alone 0.781; KBest/PCA lose → keep 76 | **v1: 36 cols, 99.2%** |
| Tuned final | **0.903672** — leaves 31, trees 500, min_child 200, lr 0.03 | **v1: 0.904321 — leaves 94, trees 512, lr 0.022** |
| Wilcoxon | LGBM vs XGB: **tie, p = 1.000** (5/10 each) → pick LGBM on speed; report a tie | **v1: W=0, p=0.001953, significant win** |
| Nested CV | **0.903672**, optimism 0.000000 — same config in all 10 outer folds (n_iter=4×3) | v1: 0.904303, +0.000018 |
| OOF | 0.903667 · 48,218 errors (11.0%) | v1: 88.2% boundary share |
| Boundary | **88.1%** of errors at 3/4 boundary; FP 77.9% at Good; FN 92.5% at Fair | v1: 88.2% |
| Case A | row 307808, p **0.9632**, 30 sick days, BMI 51.4 → label noise | v1: row 307041, p 0.96 |
| Case B | row 393280, p **0.0023**, clean profile → feature-set floor | v1: row 392245, p 0.0025 |
| Case C | row 21595, p exactly **0.5000** — credits cancel | v1: row 208256, p 0.5000022 |
| SHAP | PHYSHLTH **0.7361** ≈ 4× #2 (DIABETE4 0.1947) | v1: 0.681, 3.1× |
| Headline stat | family gain (+0.0103) = **29×** tuning gain (+0.000356) | v1: 10× |
| Wart | FLSHTMY3 777777 kept as category — documented, first fix | (v2-only honesty point) |
| Determinism | 3 re-runs → bit-identical folds; log refreshes rows in place | v1: fold-hash assert |

## Recording flow
1. Script: `deliverables/screencast_script.md` (6 beats, 8:30–9:30).
2. Deck: Gamma prompt at `deliverables/gamma_deck_prompt.md`; drop real figures into slides 1/5/7.
3. Dry run once, out loud, timed. Then record (PowerPoint Record or OBS).
4. First sentence states the language.
