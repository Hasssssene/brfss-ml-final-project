"""Build deliverables/presentation/BRFSS_screencast.pptx

9-slide academic deck for the 10-minute screencast. Mono + single red accent.
Every number copied verbatim from deliverables/report/report.md (v1.2) -- verified
by grep before this script was written; see the NUMBERS table in the QA notes.

Palette:  BG_LIGHT FAFAFA | BG_DARK 1A1A1A | INK 212121 | MUTED 6E6E6E
          LINE D9D9D9 | CARD FFFFFF | RED D7263D (key numbers + motif only)
Fonts:    Trebuchet MS (headers) / Calibri (body)
Canvas:   13.333 x 7.5 in (16:9)
"""
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "figures"
OUT = ROOT / "deliverables" / "presentation" / "BRFSS_screencast.pptx"

BG_LIGHT = RGBColor(0xFA, 0xFA, 0xFA)
BG_DARK = RGBColor(0x1A, 0x1A, 0x1A)
INK = RGBColor(0x21, 0x21, 0x21)
MUTED = RGBColor(0x6E, 0x6E, 0x6E)
LINE = RGBColor(0xD9, 0xD9, 0xD9)
CARD = RGBColor(0xFF, 0xFF, 0xFF)
RED = RGBColor(0xD7, 0x26, 0x3D)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
DARKMUTED = RGBColor(0xB5, 0xB5, 0xB5)

HEAD = "Trebuchet MS"
BODY = "Calibri"

W, H = 13.333, 7.5


def solid(shape, color):
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()


def txt(slide, x, y, w, h, runs, size=14, font=BODY, color=INK, bold=False,
        align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, leading=None, spc=None,
        space_after=None):
    """runs: str OR list of (text, dict-overrides) tuples; each tuple starts a new
    paragraph if the override contains break=True."""
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    if isinstance(runs, str):
        runs = [(runs, {})]
    p = None
    for text, ov in runs:
        if p is None:  # first paragraph of the box
            p = tf.paragraphs[0]
            p.alignment = ov.get("align", align)
            if leading:
                p.line_spacing = ov.get("leading", leading)
            if space_after is not None:
                p.space_after = Pt(ov.get("space_after", space_after))
        r = p.add_run()
        r.text = text
        f = r.font
        f.name = ov.get("font", font)
        f.size = Pt(ov.get("size", size))
        f.bold = ov.get("bold", bold)
        f.color.rgb = ov.get("color", color)
        sp = ov.get("spc", spc)
        if sp:
            r.font._rPr.set("spc", str(sp))
        if ov.get("break"):  # next tuple starts a NEW paragraph
            p = tf.add_paragraph()
            p.alignment = ov.get("align", align)
            if leading:
                p.line_spacing = ov.get("leading", leading)
            if space_after is not None:
                p.space_after = Pt(ov.get("space_after", space_after))
    return box


def rect(slide, x, y, w, h, color):
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    solid(sh, color)
    return sh


def hline(slide, x, y, w, color=LINE, weight=1.0):
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Pt(weight))
    solid(sh, color)
    return sh


def chip(slide, n, dark=False):
    """Red square + slide number, top-right motif."""
    c = rect(slide, W - 0.85, 0.42, 0.42, 0.42, RED)
    tf = c.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = str(n)
    r.font.name = HEAD
    r.font.size = Pt(15)
    r.font.bold = True
    r.font.color.rgb = WHITE


def kicker(slide, text, dark=False):
    txt(slide, 0.75, 0.52, 9.5, 0.3, text.upper(), size=11, font=HEAD,
        color=(DARKMUTED if dark else MUTED), bold=True, spc=200)


def title(slide, text, dark=False, size=30):
    txt(slide, 0.75, 0.82, 11.0, 0.65, text, size=size, font=HEAD,
        color=(WHITE if dark else INK), bold=True)


def light_slide(prs, kick, heading, n):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = BG_LIGHT
    kicker(s, kick)
    title(s, heading)
    chip(s, n)
    return s


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def figure(slide, name, x, y, w=None, h=None, caption=None):
    path = FIG / name
    pic = slide.shapes.add_picture(str(path), Inches(x), Inches(y),
                                   Inches(w) if w else None, Inches(h) if h else None)
    pic.line.color.rgb = LINE
    pic.line.width = Pt(0.75)
    if caption:
        txt(slide, x, y + pic.height / 914400 + 0.08, (pic.width / 914400), 0.25,
            caption, size=10, color=MUTED, align=PP_ALIGN.CENTER)
    return pic


def card(slide, x, y, w, h):
    c = rect(slide, x, y, w, h, CARD)
    c.shadow.inherit = False
    ln = c.line
    ln.color.rgb = LINE
    ln.width = Pt(1.0)
    return c


prs = Presentation()
prs.slide_width = Inches(W)
prs.slide_height = Inches(H)

# ============================ SLIDE 1 -- TITLE (dark) ============================
s = prs.slides.add_slide(prs.slide_layouts[6])
s.background.fill.solid()
s.background.fill.fore_color.rgb = BG_DARK
kicker(s, "Machine Learning Final Project  ·  BRFSS 2021", dark=True)
chip(s, 1, dark=True)
txt(s, 0.75, 2.15, 11.9, 1.9, [
    ("Predicting ", {}),
    ("Fair/Poor", {"color": RED}),
    (" General Health", {}),
], size=44, font=HEAD, bold=True, color=WHITE)
txt(s, 0.75, 3.35, 11.9, 0.5,
    "Binary classification  ·  438,693 respondents × 303 columns  ·  ROC-AUC, stratified 10-fold CV",
    size=17, color=DARKMUTED)
hline(s, 0.78, 4.15, 2.2, RED, 2.2)
txt(s, 0.75, 4.45, 8.0, 0.4, "Hassene Bouamama", size=16, font=HEAD, bold=True, color=WHITE)
txt(s, 0.75, 4.85, 8.0, 0.35, "Final project screencast  ·  November 2026", size=12, color=DARKMUTED)
notes(s, "BEAT 1 (0:00-1:00). BRFSS 2021: 438,693 respondents x 303 columns, the CDC's annual "
         "telephone health survey. Task: binary classification, y = 1 iff GENHLTH in {4,5} "
         "Fair/Poor. Evaluation contract: stratified 10-fold CV, IDENTICAL folds for every model "
         "(fixed random_state=42, fold-hash asserted per run), metric ROC-AUC. State language "
         "(English). ~55 seconds.")

# ===================== SLIDE 2 -- DATA & IMBALANCE =====================
s = light_slide(prs, "Data & imbalance", "One in six respondents reports Fair or Poor health", 2)
txt(s, 0.75, 1.75, 5.6, 1.1, [
    ("16.63%", {"size": 72, "color": RED, "bold": True, "font": HEAD}),
], leading=0.95)
txt(s, 0.75, 2.9, 5.4, 0.35, "positive class  ·  72,741 of 437,532 valid rows", size=14, color=MUTED)
for i, b in enumerate([
    [("Class ratio ", {}), ("1 : 5.0", {"bold": True}), (" — majority-class = ", {}),
     ("83.37%", {"bold": True}), (" accuracy", {})],
    [("Accuracy is meaningless here → metric is ", {}), ("ROC-AUC", {"bold": True})],
    [("Floor: DummyClassifier = ", {}), ("0.500000", {"bold": True, "color": RED}),
     (" on all 10 folds", {})],
]):
    txt(s, 0.75, 3.6 + i * 0.6, 5.6, 0.5, b, size=15.5, leading=1.05)
figure(s, "eda_target_imbalance.png", 6.7, 1.8, w=6.1,
       caption="GENHLTH raw classes (left) and binary target (right) — EXP-000 / Figure 1")
notes(s, "BEAT 1 cont. Target distribution: 16.63% positive (72,741) vs 83.37% negative "
         "(364,791), ratio 1 : 5.0. A model predicting 'healthy' for everyone scores 83.37% "
         "accuracy — that is why the project evaluates with ROC-AUC. Measured floor: "
         "DummyClassifier = 0.500000 on every fold (EXP-001). ~45 seconds.")

# ===================== SLIDE 3 -- LEAKAGE TRAP =====================
s = light_slide(prs, "The leakage trap", "One column IS the answer", 3)
txt(s, 0.75, 1.8, 5.0, 1.45, [
    ("100.0%", {"size": 96, "color": RED, "bold": True, "font": HEAD}),
], leading=0.95)
txt(s, 0.75, 3.3, 5.2, 0.75, [
    ("_RFHLTH agrees with the target on 100.0% of rows — ", {}),
    ("perfectly diagonal crosstab", {"bold": True}),
], size=16, leading=1.1)
txt(s, 0.75, 4.2, 5.2, 0.8, "It does not predict the answer.\nIt IS the answer.",
    size=19, font=HEAD, bold=True, leading=1.15)
# crosstab mini-card
card(s, 6.35, 1.75, 6.25, 2.1)
txt(s, 6.65, 1.95, 5.7, 0.3, "CROSSTAB — GENHLTH × _RFHLTH  (437,532 / 437,532)",
    size=11, font=HEAD, bold=True, color=MUTED, spc=100)
rows = [
    ("GENHLTH 1–3  (83.37%)", "→  _RFHLTH = 1", "100.0%"),
    ("GENHLTH 4–5  (16.63%)", "→  _RFHLTH = 2", "100.0%"),
]
for i, (a, b, c) in enumerate(rows):
    y = 2.42 + i * 0.52
    txt(s, 6.65, y, 2.9, 0.4, a, size=14, bold=True)
    txt(s, 9.55, y, 1.7, 0.4, b, size=14, color=MUTED)
    txt(s, 11.35, y, 1.0, 0.4, c, size=14, color=RED, bold=True, align=PP_ALIGN.RIGHT)
    hline(s, 6.65, y + 0.44, 5.7)
# blacklist card
card(s, 6.35, 4.15, 6.25, 1.55)
txt(s, 6.65, 4.35, 5.7, 0.3, "BLACKLISTED — NEVER FEATURES", size=11, font=HEAD,
    bold=True, color=MUTED, spc=100)
txt(s, 6.65, 4.75, 5.7, 0.85, [
    ("GENHLTH  ·  _RFHLTH  ·  _PHYS14D  ·  _MENT14D", {"bold": True, "size": 15.5, "break": True}),
    (("Design rule: build by whitelist, never drop-lists"), {"size": 13.5, "color": MUTED}),
], leading=1.2)
txt(s, 0.75, 5.15, 5.2, 0.9, [
    (("Whitelist beats blacklists:"), {"bold": True, "break": True}),
    (("a typo in a drop-list cannot leak the target"), {"color": MUTED}),
], size=14, leading=1.15)
notes(s, "BEAT 2 (1:00-3:00), part 1. The dataset's designed trap: _RFHLTH agrees with our "
         "target on 100.0% of rows (437,532/437,532, perfectly diagonal, EXP-000). So GENHLTH, "
         "_RFHLTH, _PHYS14D, _MENT14D are blacklisted — and the matrix is built by WHITELIST, "
         "never by drop-lists: a typo cannot leak the target (ADR-005). ~40 seconds.")

# ===================== SLIDE 4 -- 303 -> 76 =====================
s = light_slide(prs, "Preprocessing", "303 → 76 columns — every column gets a verdict", 4)
# left: verdict rows
txt(s, 0.75, 1.7, 6.0, 0.3, "SEVEN TRIAGE RULES (ADR-005)", size=11, font=HEAD,
    bold=True, color=MUTED, spc=100)
verds = [
    ("DROP-SPARSE", "86", "over 90% blank + sentinel"),
    ("DROP-DUP", "65", "computed duplicates of kept raws"),
    ("DROP-MODULE", "26", "optional modules under 25% coverage"),
    ("DROP-ID", "18", "identifiers, dates, weights, strata"),
    ("DROP-MODE", "16", "telephone-flow / interview-mode admin"),
    ("DROP-CONSTANT", "12", "constant columns incl. 3 empty"),
    ("DROP-LEAK", "4", "GENHLTH family — the target itself"),
]
for i, (name, n, why) in enumerate(verds):
    y = 2.1 + i * 0.42
    txt(s, 0.75, y, 2.35, 0.35, name, size=13, font=HEAD, bold=True)
    txt(s, 3.1, y, 0.75, 0.35, n, size=14, bold=True, color=RED, align=PP_ALIGN.RIGHT)
    txt(s, 4.0, y, 2.85, 0.35, why, size=12, color=MUTED)
txt(s, 0.75, 5.18, 6.0, 0.45, [
    ("→ keep ", {"size": 16}),
    ("76", {"size": 20, "bold": True, "color": RED, "font": HEAD}),
    ("  (54 nominal · 15 engineered · 4 ordinal · 3 numeric)", {"size": 13, "color": MUTED}),
], leading=1.0)
figure(s, "exp000_missingness_top40.png", 8.15, 1.55, h=3.85,
       caption="40 most-missing columns (EXP-000)")
# sentinel traps bottom strip
card(s, 7.35, 5.75, 5.25, 1.35)
txt(s, 7.6, 5.92, 4.8, 0.28, "SENTINEL TRAPS — CODEBOOK-VERIFIED", size=11, font=HEAD,
    bold=True, color=MUTED, spc=100)
txt(s, 7.6, 6.28, 4.85, 0.75, [
    ("Code 88 = zero days → PHYSHLTH ρ flips ", {"size": 12.5, "break": True}),
    ("        ", {"size": 12.5}),
    ("−0.293 → +0.450", {"size": 12.5, "bold": True, "color": RED}),
    ("   (287,796 rows)", {"size": 11.5, "color": MUTED, "break": True}),
    ("Code 7 is not always “don’t know”: EMPLOY1 7 = Retired (131,546 rows)",
     {"size": 12.5}),
], leading=1.1)
notes(s, "BEAT 2, part 2. 303 -> 76 via seven recorded rules (ADR-005): sparse 86, duplicates 65 "
         "(e.g. EXERANY2 proven identical to _TOTINDA), modules 26, IDs 18, mode 16, constants 12, "
         "leaks 4. Every one of the 303 columns has a verdict + reason in triage_decisions.csv — "
         "no silent drops. Sentinel traps: code 88 = 'none' = zero days (PHYSHLTH, 287,796 rows): "
         "correlation flips -0.293 -> +0.450 after recode — the strongest feature in the data was "
         "INVERTED on raw codes. And 7 is not always don't-know: EMPLOY1 7 = Retired (131,546 "
         "rows). Per-column value contracts, not pattern matching. ~75 seconds.")

# ===================== SLIDE 5 -- VALIDATION & BASELINES =====================
s = light_slide(prs, "Validation & baselines", "One splitter. Three baselines. Honest grids.", 5)
card(s, 0.75, 1.7, 11.85, 1.15)
txt(s, 1.05, 1.92, 11.3, 0.75, [
    ("StratifiedKFold(10, shuffle, seed 42)", {"bold": True, "size": 18, "font": HEAD, "break": True}),
    ("Identical folds for every model in the project — fold-hash asserted in all 41 model rows of the log",
     {"size": 13, "color": MUTED}),
], leading=1.15)
base = [
    ("DummyClassifier", "0.500000", "prior baseline — exactly the theoretical floor"),
    ("LogisticRegression", "0.892588", "C-grid flat: top settings within 8×10⁻⁶"),
    ("DecisionTree", "0.694922 → 0.880206", "depth/leaf limits control overfitting"),
]
for i, (m, v, why) in enumerate(base):
    y = 3.15 + i * 0.95
    card(s, 0.75, y, 11.85, 0.8)
    txt(s, 1.05, y + 0.16, 3.3, 0.5, m, size=16, font=HEAD, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    txt(s, 4.45, y + 0.16, 3.6, 0.5, v, size=19, bold=True, color=RED, anchor=MSO_ANCHOR.MIDDLE)
    txt(s, 8.2, y + 0.16, 4.2, 0.5, why, size=12.5, color=MUTED, anchor=MSO_ANCHOR.MIDDLE)
txt(s, 0.75, 6.15, 11.85, 0.5,
    "Manual tuning on the tree buys +0.185 AUC — regularization, not depth, is what deep trees need",
    size=13.5, color=INK)
notes(s, "BEAT 3 (3:00-6:00), part 1. Baselines with manual tuning: LogReg C-grid is FLAT "
         "(0.892588 at C=0.01; top three settings within 8e-6) — regularization doesn't matter "
         "here. The tree grid is the opposite: min_samples_leaf dominates; near-default "
         "(depth=None, leaf=1) collapses to 0.694922 vs 0.880206 tuned — manual tuning buys "
         "+0.185 AUC (EXP-002/003). ~50 seconds.")

# ===================== SLIDE 6 -- MODEL ZOO & IMBALANCE =====================
s = light_slide(prs, "Model zoo & imbalance", "Boosting wins; resampling does not help AUC", 6)
zoo = [
    ("DummyClassifier", "0.500000", False),
    ("DecisionTree (tuned)", "0.880206", False),
    ("LogisticRegression", "0.892588", False),
    ("RandomForest", "0.893315", False),
    ("HistGradientBoosting", "0.903220", True),
    ("LightGBM", "0.903355", True),
    ("XGBoost", "0.903357", True),
]
txt(s, 0.75, 1.7, 5.6, 0.3, "MODEL ZOO — ONE CONFIG EACH, IDENTICAL FOLDS", size=11,
    font=HEAD, bold=True, color=MUTED, spc=100)
for i, (m, v, hot) in enumerate(zoo):
    y = 2.1 + i * 0.5
    txt(s, 0.75, y, 3.6, 0.4, m, size=14.5, bold=hot)
    txt(s, 4.4, y, 1.6, 0.4, v, size=15.5, bold=True,
        color=(RED if hot else INK), align=PP_ALIGN.RIGHT)
    if hot:
        hline(s, 0.75, y + 0.42, 5.3)
txt(s, 0.75, 5.75, 5.6, 0.9, [
    ("Booster trio: ", {"size": 14}),
    ("+0.011", {"size": 17, "bold": True, "color": RED, "font": HEAD}),
    (" over every sklearn model ≈ ", {"size": 14}),
    ("6× fold std", {"size": 14, "bold": True}),
], leading=1.1)
card(s, 6.9, 1.7, 5.7, 2.35)
txt(s, 7.2, 1.9, 5.1, 0.3, "IMBALANCE TECHNIQUES — ΔAUC", size=11, font=HEAD,
    bold=True, color=MUTED, spc=100)
imb = [
    ("Class weights", "±0.00045"),
    ("SMOTE (0.5, in folds)", "−0.004"),
    ("Undersampling (0.5)", "±0.0003"),
]
for i, (k, v) in enumerate(imb):
    y = 2.3 + i * 0.5
    txt(s, 7.2, y, 3.6, 0.4, k, size=14)
    txt(s, 10.9, y, 1.4, 0.4, v, size=15, bold=True, align=PP_ALIGN.RIGHT)
txt(s, 7.2, 3.72, 5.1, 0.3, "verdict: none — AUC is rank-based", size=13, bold=True)
card(s, 6.9, 4.35, 5.7, 2.3)
txt(s, 7.2, 4.55, 5.1, 1.95, [
    ("Why boosters suit survey data", {"bold": True, "size": 14.5, "font": HEAD, "break": True}),
    ("NaN-passthrough: one-hot encodes missing as its own category", {"size": 13, "break": True}),
    ("Boosters route missing natively — module missingness stays signal, not noise",
     {"size": 13, "color": MUTED}),
], leading=1.15, space_after=6)
notes(s, "BEAT 3, part 2. Model zoo (EXP-004..007): RF 0.8933 ties LogReg, then the boosters "
         "clear everything: HistGB 0.9032, LightGBM 0.9034, XGBoost 0.9034 — +0.011 over "
         "linear/bagging, ~6x fold std. Boosting was the win. NaN-passthrough: one-hot encodes "
         "NaN as its own category; boosters route missing natively. Imbalance (EXP-008/009/010): "
         "class weighting within 0.00045, undersampling 0.0003, SMOTE HURTS (-0.004). Verdict: "
         "none — ROC-AUC is threshold-free and rank-based; resampling reshapes the prior, not "
         "the ranking. Honest negative, explained not hidden. ~70 seconds.")

# ===================== SLIDE 7 -- WHAT MATTERS + STATISTICS =====================
s = light_slide(prs, "Feature selection & statistics", "One feature dominates — and family beats tuning", 7)
figure(s, "p5_importance_top30.png", 9.05, 1.55, h=3.55,
       caption="Gain importances, H1 blocks colored (EXP-011b / Figure 6)")
txt(s, 0.75, 1.7, 8.2, 1.1, [
    ("PHYSHLTH = ", {"size": 24, "font": HEAD, "bold": True}),
    ("51.69%", {"size": 40, "color": RED, "bold": True, "font": HEAD}),
    (" of total gain", {"size": 24, "font": HEAD, "bold": True}),
], leading=1.0)
bul = [
    [("H1 subset — 36 columns: ", {}), ("99.2%", {"bold": True, "color": RED}),
     (" of full lift, loses 10/10 folds → keep all 76", {})],
    [("Honest negatives: ", {}), ("PCA and SelectKBest both cost AUC", {"bold": True})],
    [("Complement (40 cols) standalone: ", {}), ("0.8118", {"bold": True}),
     (" — real but redundant", {})],
]
for i, b in enumerate(bul):
    txt(s, 0.75, 2.95 + i * 0.55, 8.1, 0.5, b, size=15, leading=1.05)
# statistics strip
card(s, 0.75, 4.85, 8.05, 1.95)
txt(s, 1.05, 5.05, 7.5, 0.3, "WILCOXON SIGNED-RANK — 10 IDENTICAL FOLDS", size=11,
    font=HEAD, bold=True, color=MUTED, spc=100)
txt(s, 1.05, 5.45, 7.5, 1.2, [
    ("W = 0  ·  p = 0.001953", {"size": 20, "bold": True, "font": HEAD, "break": True}),
    ("family choice ", {"size": 14}),
    ("+0.012", {"size": 16, "bold": True, "color": RED}),
    ("  =  10 ×  tuning gain ", {"size": 14}),
    ("+0.001", {"size": 16, "bold": True}),
    ("   — both significant", {"size": 14, "color": MUTED}),
], leading=1.2, space_after=4)
txt(s, 9.05, 5.65, 3.6, 0.9, [
    ("Final: tuned LightGBM ", {"size": 13}),
    ("0.904321", {"size": 17, "bold": True, "color": RED, "font": HEAD, "break": True}),
    ("nested CV:  0.904303", {"size": 13, "color": MUTED}),
], leading=1.25)
notes(s, "BEAT 4 (6:00-8:00). H1 stated up front: power concentrates in five blocks. SelectKBest: "
         "honest negative (best k=200: 0.8997 vs 0.9034). Gain importances: PHYSHLTH alone 51.69% "
         "of total gain, DIFFWALK 8.0%; health-days block carries 55.0%. H1 test: the 36-column "
         "subset reaches 99.2% of the full lift, but the deficit is consistent — it loses 10/10 "
         "folds against all-76 — so we kept all-76 and used H1 for interpretation. Complement "
         "standalone 0.8118 — real but redundant. PCA: -0.015 to -0.022. Statistics: Wilcoxon "
         "on 10 paired folds, all three tests W=0, p=0.001953 (smallest possible at n=10). "
         "Headline: family choice (+0.012) worth 10x more than tuning (+0.001). Final: tuned "
         "LightGBM 0.904321, nested CV 0.904303. ~110 seconds.")

# ===================== SLIDE 8 -- ERRORS & XAI =====================
s = light_slide(prs, "Error analysis & XAI", "Errors live at the label boundary", 8)
txt(s, 0.75, 1.6, 4.6, 0.95, [
    ("88.2%", {"size": 60, "color": RED, "bold": True, "font": HEAD}),
], leading=0.95)
txt(s, 0.75, 2.6, 5.2, 0.7, "of all errors sit at the Good | Fair (3|4) boundary —\nlabel ambiguity, not model failure",
    size=14, leading=1.15)
txt(s, 0.75, 3.35, 5.2, 0.4, "41,091 rows misclassified by all three model families", size=13, color=MUTED)
cases = [
    ("A", "p = 0.9595", "label noise — symptoms scream Fair/Poor, rater says Good"),
    ("B", "p = 0.0025", "error floor — clean profile, unmeasured rating anchors"),
    ("C", "p = 0.5000022", "true boundary — comorbid profile vs clean current status"),
]
txt(s, 0.75, 4.0, 5.3, 0.3, "THREE CONSISTENTLY MISCLASSIFIED ARCHETYPES", size=11,
    font=HEAD, bold=True, color=MUTED, spc=100)
for i, (c, p, why) in enumerate(cases):
    y = 4.4 + i * 0.72
    rect(s, 0.75, y, 0.42, 0.42, INK)
    txt(s, 0.75, y + 0.05, 0.42, 0.35, c, size=15, bold=True, color=WHITE,
        align=PP_ALIGN.CENTER, font=HEAD)
    txt(s, 1.32, y + 0.02, 1.75, 0.4, p, size=15, bold=True, color=RED)
    txt(s, 3.1, y + 0.03, 3.6, 0.65, why, size=11.5, color=MUTED, leading=1.0)
figure(s, "p7_case_profiles.png", 6.95, 1.7, w=5.7,
       caption="Cases A / B / C — z-scores vs population (Figure 9)")
card(s, 6.95, 4.85, 5.7, 1.9)
txt(s, 7.25, 5.05, 5.1, 0.3, "GLOBAL SHAP — TWO METHODS, ONE STORY", size=11,
    font=HEAD, bold=True, color=MUTED, spc=100)
txt(s, 7.25, 5.45, 5.2, 1.2, [
    ("PHYSHLTH 0.681", {"size": 16, "bold": True, "font": HEAD, "color": RED}),
    ("  (3.1× the #2)", {"size": 13, "color": MUTED, "break": True}),
    ("SHAP vs gain rank agreement:  ", {"size": 13}),
    ("ρ = 0.911", {"size": 15, "bold": True}),
], leading=1.15, space_after=4)
notes(s, "BEAT 5 (8:00-9:30). At threshold 0.5: FPR 3.93%, FNR 46.44%. 88.2% of all errors sit at "
         "the Good|Fair boundary; 41,091 rows wrong under all three families simultaneously — "
         "structured ambiguity in the binarized label, not model failure. Case A: GENHLTH=3 but "
         "30 poor-health days, six conditions, BMI 51; all models p about 0.96-0.99 — plausible "
         "label noise. Case B: GENHLTH=4 with a clean profile, p 0.0025 — error floor of the "
         "feature set. Case C: p 0.5000022 with a comorbidity block — aleatoric boundary case. "
         "Global SHAP: PHYSHLTH 0.681 (3.1x the #2); SHAP vs gain ranks agree at rho=0.911 — "
         "current physical-health status first. FNR/FPR asymmetry is expected geometry for an "
         "unweighted AUC model at 1:5 prevalence; report ships a threshold menu. ~85 seconds.")

# ===================== SLIDE 9 -- CONCLUSIONS & ETHICS (dark) =====================
s = prs.slides.add_slide(prs.slide_layouts[6])
s.background.fill.solid()
s.background.fill.fore_color.rgb = BG_DARK
kicker(s, "Conclusions & ethics", dark=True)
chip(s, 9, dark=True)
txt(s, 0.75, 1.5, 11.9, 0.35, "Final model", size=16, font=HEAD, bold=True, color=DARKMUTED)
txt(s, 0.75, 2.0, 11.9, 1.1, [
    ("Tuned LightGBM  ", {"size": 30, "font": HEAD, "bold": True}),
    ("0.9043 ± 0.0017", {"size": 40, "color": RED, "bold": True, "font": HEAD}),
], leading=1.0)
txt(s, 0.75, 3.1, 11.5, 0.4, "nested-CV confirmed (0.904303)  ·  SHAP-interpretable  ·  honest negatives reported",
    size=14, color=DARKMUTED)
hline(s, 0.78, 3.85, 2.2, RED, 2.2)
fin = [
    [("42 logged runs", {"bold": True}), (" on identical folds — every number traceable to an experiment id", {})],
    [("Fully reproducible:", {"bold": True}), (" deliverables/code.py re-derives the final folds to ≤ 5×10⁻⁷", {})],
    [("Ethics:", {"bold": True}), (" full AI-assistance disclosure — final section of the report", {})],
]
for i, b in enumerate(fin):
    txt(s, 0.75, 4.2 + i * 0.55, 11.9, 0.5, b, size=15.5, color=WHITE, leading=1.05)
txt(s, 0.75, 6.55, 8.0, 0.4, "Hassene Bouamama  ·  Machine Learning Final Project  ·  November 2026",
    size=12, color=DARKMUTED)
notes(s, "BEAT 6 (9:30-10:00). Final model: tuned LightGBM, 0.9043 +/- 0.0017, nested-confirmed "
         "(0.904303) — chosen on the full evidence chain (42 logged runs, identical folds), not "
         "one metric. Reproducible end-to-end in deliverables/code.py (re-derives the final "
         "folds to within 5e-7). Honest disclosure: an OpenCode agent (GLM, Z.ai) assisted "
         "throughout — planning, code, analysis, report drafting — under a human-ratified "
         "decision process (ADRs), machine-verified against the append-only experiment log. "
         "Full disclosure is the report's final section, per course policy. Thanks. ~30 seconds.")

OUT.parent.mkdir(parents=True, exist_ok=True)
prs.save(str(OUT))
print(f"saved {OUT} ({OUT.stat().st_size} bytes, {len(prs.slides.__iter__.__self__._sldIdLst)} slides)")
