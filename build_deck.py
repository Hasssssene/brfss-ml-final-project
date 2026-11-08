"""Build the v1 screencast deck (16:9) with v1 figures + speaker notes."""
import os
from pptx import Presentation
from pptx.util import Inches, Pt

ROOT = os.getcwd()
FIG = os.path.join(ROOT, 'figures')
OUT = os.path.join('deliverables', 'screencast_deck.pptx')

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


def add_slide(title, bullets, img=None, notes='', img_full=False):
    s = prs.slides.add_slide(BLANK)

    tb = s.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(12.3), Inches(0.9))
    p = tb.text_frame.paragraphs[0]
    r = p.runs[0] if p.runs else p.add_run()
    r.text = title
    r.font.size = Pt(30)
    r.font.bold = True

    if img and img_full:
        s.shapes.add_picture(os.path.join(FIG, img), Inches(2.2), Inches(1.4),
                             height=Inches(5.7))
    elif img:
        tb2 = s.shapes.add_textbox(Inches(0.5), Inches(1.5), Inches(6.6), Inches(5.6))
        tf = tb2.text_frame
        tf.word_wrap = True
        for i, b in enumerate(bullets):
            para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            run = para.add_run()
            run.text = ('•  ' if not b.startswith(('→', '   ')) else '') + b
            run.font.size = Pt(15)
            para.space_after = Pt(8)
        s.shapes.add_picture(os.path.join(FIG, img), Inches(7.3), Inches(1.5),
                             height=Inches(5.4))
    else:
        tb2 = s.shapes.add_textbox(Inches(0.7), Inches(1.7), Inches(12.0), Inches(5.3))
        tf = tb2.text_frame
        tf.word_wrap = True
        for i, b in enumerate(bullets):
            para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            run = para.add_run()
            run.text = ('•  ' if not b.startswith(('→', '   ')) else '') + b
            run.font.size = Pt(17 if len(bullets) < 10 else 15)
            para.space_after = Pt(10)

    if notes:
        s.notes_slide.notes_text_frame.text = notes
    return s


# 1 — title
add_slide('Predicting Fair/Poor General Health — BRFSS 2021', [
    'Hassene Bouamama  ·  ML final project  ·  screencast',
    '',
    'Can the REST of a 438,693-person health survey predict one self-rating question?',
    'Binary target: GENHLTH ∈ {4,5} = Fair/Poor (16.63%) vs {1,2,3} (83.37%)',
    'Metric: ROC-AUC · stratified 10-fold CV · identical folds, seed 42',
    'Final model: tuned LightGBM — CV ROC-AUC 0.9043 ± 0.0017 (nested-confirmed)',
], notes='State language (English). 30 seconds max. One breath: biggest health survey, '
         'one question predicts a lot — we predict that question from everything else.')

# 2 — problem & data
add_slide('The data & the imbalance', [
    '438,693 respondents × 303 columns (CDC telephone survey)',
    '1,161 unusable target rows dropped → 437,532 labeled',
    '72,741 Fair/Poor = 16.63% → class ratio 1 : 5',
    'Why AUC, not accuracy: "everyone healthy" = 83.4% accuracy, useless',
    'Measured floor: DummyClassifier = 0.500000 on all 10 folds (EXP-001)',
], img='eda_target_imbalance.png',
   notes='Figure: the 5-point scale, red = our positive class. 16.6% positive is why '
         'accuracy lies and AUC (ranking) is the honest metric.')

# 3 — leakage
add_slide('The trap: _RFHLTH is the target in disguise', [
    'Calculated columns (_-prefixed) = derived by CDC after the interview',
    '_RFHLTH vs our y: 100.0% agreement — 437,532 / 437,532, diagonal crosstab',
    'It does not PREDICT Fair/Poor — it IS Fair/Poor (answer key with a fake mustache)',
    'Blacklisted: GENHLTH, _RFHLTH, _PHYS14D, _MENT14D',
    'Design rule: build the matrix by WHITELIST, never drop-lists —',
    '   a typo in a keep-list loses a column (harmless); in a drop-list it leaks the target',
], notes='This is the dataset designed trap. Association is fine (that is prediction); '
         'DERIVATION is cheating. 100% agreement = derivation.')

# 4 — triage funnel + sentinels
add_slide('303 → 76 columns + the sentinel traps', [
    'Six-rule triage, every column gets a verdict + reason: leak 4 · admin 35 · constant 11 ·',
    '   sparse 85 · low-coverage 27 · duplicate 65 → KEEP 76',
    'Duplicates proven: EXERANY2 ≡ _TOTINDA (diagonal crosstabs)',
    'Trap 1: day-count 88 = "none" = ZERO days — PHYSHLTH: 88 on 287,796 rows (~66%)',
    '   ρ with target: −0.29 raw → +0.450 after 88→0 — the #1 feature, un-inverted',
    'Trap 2: 7 is not always "don\'t know": EMPLOY1 7 = Retired (131k rows)',
    '   → per-column recode contracts from the codebook, never pattern matching',
], img='eda_missingness_matrix.png',
   notes='The 88-anecdote is the best preprocessing story: sentinels-as-numbers would have '
         'INVERTED our strongest predictor. Missingness is bimodal: core survey vs optional modules.')

# 5 — validation + baselines
add_slide('One set of folds for everybody', [
    'StratifiedKFold(10, shuffle=True, random_state=42) — fold hash asserted per run',
    'Identical folds → score differences come from the MODEL, not split luck',
    '→ also makes the paired Wilcoxon test valid later',
    'Manual tuning: DecisionTree grid 0.6949 → 0.8802 (+0.185 AUC) — limits beat depth',
    'LogReg C-grid is FLAT (0.8926; spread 8e-6) — honest: regularization does not matter here',
    'NaN-passthrough: one-hot NaN = own category; boosters route missing natively',
], notes='Fold-hash story: every logged experiment carries the same fold fingerprint. '
         'The tree grid shows manual tuning mattering where it should (overfitting control).')

# 6 — model zoo + imbalance
add_slide('Model zoo & the imbalance verdict', [
    'Dummy 0.5000 · LogReg 0.8926 · Tree 0.8802 · RF 0.8933',
    'HistGB 0.9032 · LightGBM 0.9034 · XGBoost 0.9034 — boosters win by +0.011 ≈ 6× fold std',
    'Imbalance tested, not assumed:',
    '   class weights ±0.0004 · undersampling ±0.0003 · SMOTE HURTS (−0.004)',
    '   → verdict: none — ROC-AUC is rank-based; resampling reshapes the prior, not the ranking',
    'Automated tuning (RandomizedSearchCV): LightGBM 0.9043 (10/10 folds over untuned)',
    'Nested CV unbiased estimate: 0.9043 (optimism +0.000018 — the search does not overfit)',
], img='p6_tuning_results.png',
   notes='Family gain (+0.012) is 10x the tuning gain (+0.001) — remember this line for the '
         'statistics slide. SMOTE negative = honest finding, explained not hidden.')

# 7 — feature selection & statistics
add_slide('What actually matters (H1) + statistics', [
    'H1 stated BEFORE testing: power concentrates in 5 blocks (health days, limitation,',
    '   chronic conditions, age, socioeconomics)',
    'Gain importances: PHYSHLTH alone = 51.7% of total gain; DIFFWALK 8.0%',
    'H1 subset (36 cols): 99.2% of full lift — but loses 10/10 folds → keep all 76',
    'SelectKBest & PCA: honest negatives (below all-76; variance kept ≠ signal kept)',
    'Wilcoxon signed-rank (paired folds): all three tests p = 0.001953 — the n=10 floor',
    '   → family choice (+0.012) is worth 10× tuning (+0.001): both real, effort goes to family',
], img='p5_importance_top30.png',
   notes='Point at the figure: one feature dwarfs everything. The subset result validates the '
         'hypothesis without changing the model. PCA/KBest: honest negatives.')

# 8 — errors & XAI
add_slide('Where it fails: the 3|4 boundary', [
    '88.2% of all errors sit at the Good|Fair boundary — label ambiguity, not model failure',
    'Case A: truth "Good", p=0.96 — 30 sick days, 6 conditions, BMI 51 → label noise',
    'Case B: truth "Fair", p=0.003 — perfectly clean profile → error floor of the features',
    'Case C: p = 0.5000 exactly — mixed evidence, calibrated indifference',
    'SHAP: PHYSHLTH 0.681 (3.1× the #2) · SHAP vs gain ranks agree ρ = 0.911',
    '   two methods, different failure modes, one story: current physical-health status first',
], img='p7_shap_summary.png',
   notes='The three archetypes map to three failure MODES: label noise, unmeasured factors, '
         'genuine boundary. If asked about FNR/FPR asymmetry: expected geometry at 1:5 prevalence.')

# 9 — conclusions & ethics
add_slide('Conclusions', [
    'Final model: tuned LightGBM — 0.9043 ± 0.0017, nested-confirmed, SHAP-interpretable',
    'Evidence chain: 42 logged runs on identical folds; every report number cites its EXP id',
    'Fully reproducible: deliverables/code.py (single file, end-to-end, seed 42 everywhere)',
    'What I would do differently:',
    '   calibrate probabilities for fixed-threshold screening · per-column sentinel audit earlier',
    '   · the original 5-class problem (macro-F1 / multi-class AUC)',
    'Ethics & GenAI: full disclosure in the report\'s final section — an OpenCode agent (GLM) '
    'assisted throughout; every result human-reviewed and machine-verified',
], notes='Close in under 40 seconds. The disclosure sentence must be said plainly — it is a '
         'course requirement and it is honest.')

prs.save(OUT)
print(f'deck written: {OUT} ({len(prs.slides._sldIdLst)} slides)')
