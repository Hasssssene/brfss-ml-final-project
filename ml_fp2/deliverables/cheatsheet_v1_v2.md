# Cheat sheet — presenting v1 with knowledge from the v2 weekend

**The one rule: on camera you say v1 numbers only.** The v2 numbers in the right
column are your *memory anchors* — the experience that proves you own the concepts.

| Slide / topic | Say this (v1) | You know it because (v2 weekend) |
|---|---|---|
| Target | 438,693 × 303 → 437,532 labeled, 72,741 pos = **16.63%** | You built y yourself (NB §1.3); your fig01 shows the same 16.63% |
| Floor | Dummy = **0.500000** all 10 folds | You ran it (EXP-001) |
| Leakage | `_RFHLTH` **100.0%** agreement, diagonal crosstab | You ran that exact check yourself (NB §2.1) |
| Whitelist rule | build by keep-list, never drop-list | You answered this comprehension question verbatim |
| Funnel | leak 4 · admin 35 · constant 11 · sparse 85 · low-cov 27 · dup 65 → **76** | Your own funnel printed the same six rules (NB §3.2) |
| 88 trap | 88 = zero days; PHYSHLTH ρ −0.29 → **+0.45** after recode | You recoded 88→0 yourself (NB §4.1) |
| 7-trap | EMPLOY1 7 = *Retired*, 131k rows | You answered this exact comprehension question |
| Folds | same `StratifiedKFold(10, shuffle, rs=42)` everywhere; fold-hash asserted | You printed the identical fold table (NB §1.4) |
| Tree tuning | default tree **0.6949** → tuned **0.8802** (+0.185) | Your own EXP-005/007: 0.698→0.881 — "unlimited tree memorizes" (your words) |
| LogReg | **0.8926**, C-grid flat | Your EXP-002..004: flat, 0.8935 — you called it "insensitivity" |
| Zoo | boosters 0.9032–0.9034, +0.011 ≈ 6× fold std | Your EXP-008..011: same cluster, same gap |
| Imbalance | weights ±0.0004 · RUS ±0.0003 · **SMOTE −0.004** → none | You ran SMOTE yourself and watched it lose |
| Tuned final | **0.904321**, n_est 512, leaves 94, lr 0.022 | Different params than v2 — just read them off the slide |
| Nested CV | **0.904303**, optimism +0.000018 | You lived the identical-scores mystery + its explanation |
| H1 | 36 cols = **99.2%** of lift, loses 10/10 folds → keep all 76 | Your EXP-015: 39 cols, 99.5% — same verdict |
| KBest/PCA | honest negatives, below all-76 | Your EXP-017/018/019 — same result |
| Wilcoxon | all three tests **W=0, p=0.001953** (n=10 floor) | ⚠️ different detail than v2 (v2: tie p=1.0) — say the v1 line |
| Headline stat | family gain (+0.012) = **10×** tuning gain (+0.001) | Your own table said the same thing |
| Errors | **88.2%** at the 3\|4 boundary | Your taxonomy: 88.1% — same story |
| Case A | p≈0.96, "Good", 30 sick days, BMI 51 → label noise | Your case A: p 0.9632, same person profile |
| Case B | p≈0.003, "Fair", clean → error floor | Your case B: p 0.0023, clean profile |
| Case C | p = 0.5000 → boundary | Your case C: p exactly 0.5000 |
| SHAP | PHYSHLTH 0.681, 3.1× #2; agrees with gain ρ=0.911 | Your SHAP: PHYSHLTH 0.736 dominant — same conclusion |
| Ethics | full disclosure, final report section | Same disclosure exists in both versions |

## The three traps (do NOT mix versions)
1. **Final score**: say **0.9043** — never 0.9037 (that's v2).
2. **Wilcoxon**: v1's LGBM beats XGB significantly (W=0). v2's was a tie. Tell the **v1** story.
3. **H1 subset**: v1 = 36 columns. v2 = 39. Say **36**.

## Recording tips
- Open `deliverables/screencast_deck.pptx` → Slide Show → Presenter View (notes visible to you only).
- PowerPoint: Record → From Beginning; or OBS over the slideshow.
- ~1 minute per slide = 9:30. One dry run, then record; edit between beats freely.
- First words: which language you're speaking (course requirement).
