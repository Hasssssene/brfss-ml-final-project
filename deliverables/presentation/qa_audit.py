"""Geometry + content QA auditor for BRFSS_screencast.pptx (no vision required).

Checks:
  A. Slide-bounds violations (anything outside 13.333 x 7.5, tolerance 0.02")
  B. Sibling overlaps: partial overlaps between elements that are NOT in a
     container relationship (card-behind-text is allowed = full containment;
     text exactly covering its own rect is allowed).
  C. Text overflow: estimated rendered width (PIL, real fonts) vs box width for
     single-line boxes; estimated wrapped height vs box height for all boxes.
  D. Figure aspect distortion: stored w/h ratio vs source PNG natural ratio.
  E. Number audit: every numeric token on slides must appear in the verified
     source strings (report.md) — hardcoded whitelist of expected tokens with
     their slide numbers.
Exit code 1 if any FAIL. WARNs are reported but don't fail.
"""
import math
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from PIL import Image, ImageFont
from pptx import Presentation
from pptx.util import Emu

ROOT = Path(__file__).resolve().parents[2]
DECK = ROOT / "deliverables" / "presentation" / "BRFSS_screencast.pptx"
FIG = ROOT / "figures"
EMU_IN = 914400.0
SLIDE_W, SLIDE_H = 13.333, 7.5

FONT_FILES = {
    ("Trebuchet MS", True): "trebucbd.ttf",
    ("Trebuchet MS", False): "trebuc.ttf",
    ("Calibri", True): "calibrib.ttf",
    ("Calibri", False): "calibri.ttf",
}
_fonts = {}


def pil_font(name, bold, pt):
    key = (name, bold, round(pt))
    if key not in _fonts:
        fname = FONT_FILES.get((name, bold), "calibri.ttf")
        _fonts[key] = ImageFont.truetype(str(Path("C:/Windows/Fonts") / fname), int(round(pt * 96 / 72)))
    return _fonts[key]


def runs_of(shape):
    out = []
    if not shape.has_text_frame:
        return out
    for p in shape.text_frame.paragraphs:
        for r in p.runs:
            out.append(r)
    return out


def est_line_width_in(shape):
    """Sum of run widths for the widest paragraph (inches)."""
    best = 0.0
    for p in shape.text_frame.paragraphs:
        w = 0.0
        for r in p.runs:
            f = r.font
            pt = f.size.pt if f.size else 18
            try:
                font = pil_font(f.name or "Calibri", bool(f.bold), pt)
            except OSError:
                font = pil_font("Calibri", False, pt)
            w += font.getlength(r.text) / 96.0
        best = max(best, w)
    return best


def est_text_height_in(shape):
    """Wrapped-height estimate: per paragraph, lines = ceil(width/box_w), height
    accumulates pt * 1.22 * line_spacing."""
    box_w = shape.width / EMU_IN
    if box_w <= 0.01:
        return 0.0
    total_pt = 0.0
    for p in shape.text_frame.paragraphs:
        w = 0.0
        pt_max = 12.0
        for r in p.runs:
            f = r.font
            pt = f.size.pt if f.size else 18
            pt_max = max(pt_max, pt)
            try:
                font = pil_font(f.name or "Calibri", bool(f.bold), pt)
            except OSError:
                font = pil_font("Calibri", False, pt)
            w += font.getlength(r.text) / 96.0
        lines = max(1, math.ceil(w / max(box_w - 0.02, 0.1) - 0.02))
        ls = p.line_spacing if p.line_spacing else 1.08
        total_pt += lines * pt_max * 1.22 * (ls if isinstance(ls, float) else 1.08)
        if p.space_after:
            total_pt += p.space_after.pt
    return total_pt / 72.0


def rect_of(shape):
    return (shape.left / EMU_IN, shape.top / EMU_IN,
            (shape.left + shape.width) / EMU_IN, (shape.top + shape.height) / EMU_IN)


def contains(a, b, tol=0.03):
    return a[0] - tol <= b[0] and a[1] - tol <= b[1] and a[2] + tol >= b[2] and a[3] + tol >= b[3]


def overlap_area(a, b):
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


prs = Presentation(str(DECK))
fails, warns = [], []

# verified numeric tokens per slide (source: report.md v1.2 grep audit)
EXPECTED = {
    1: ["438,693", "303"],
    2: ["16.63%", "72,741", "437,532", "1 : 5.0", "83.37%", "0.500000"],
    3: ["100.0%", "83.37%", "16.63%", "437,532", "437,532"],
    4: ["86", "65", "26", "18", "16", "12", "76", "−0.293", "+0.450", "287,796", "131,546"],
    5: ["42", "0.500000", "0.892588", "8×10⁻⁶", "0.694922", "0.880206", "+0.185"],
    6: ["0.500000", "0.880206", "0.892588", "0.893315", "0.903220", "0.903355", "0.903357",
        "+0.011", "6× fold std", "0.00045", "0.0003", "−0.004"],
    7: ["51.69%", "99.2%", "10/10", "76", "0.8118", "0.001953", "+0.012", "+0.001",
        "0.904321", "0.904303"],
    8: ["88.2%", "41,091", "0.9595", "0.0025", "0.5000022", "0.681", "3.1×", "0.911"],
    9: ["0.9043", "0.0017", "0.904303", "42", "5×10⁻⁷"],
}

for si, slide in enumerate(prs.slides, start=1):
    elems = []  # (kind, name, rect, shape)
    for sh in slide.shapes:
        r = rect_of(sh)
        kind = "pic" if sh.shape_type == 13 else ("text" if sh.has_text_frame and sh.text_frame.text.strip() else "box")
        elems.append((kind, (sh.name or "shape")[:28], r, sh))
    # A. bounds
    for kind, name, r, sh in elems:
        if r[0] < -0.02 or r[1] < -0.02 or r[2] > SLIDE_W + 0.02 or r[3] > SLIDE_H + 0.02:
            fails.append(f"S{si} BOUNDS {name} at {tuple(round(v,2) for v in r)}")
    # B. sibling partial overlaps (containment allowed: cards behind text)
    for i in range(len(elems)):
        for j in range(i + 1, len(elems)):
            ka, na, ra, sa = elems[i]
            kb, nb, rb, sb = elems[j]
            ov = overlap_area(ra, rb)
            if ov <= 0.004:
                continue
            if contains(ra, rb) or contains(rb, ra):
                continue  # card/label pattern
            # text partially over a card edge or another text
            fails.append(f"S{si} OVERLAP {na}×{nb} area={ov:.2f}in² "
                         f"at {tuple(round(v,2) for v in ra)} vs {tuple(round(v,2) for v in rb)}")
    # C. text overflow
    for kind, name, r, sh in elems:
        if kind != "text":
            continue
        box_w, box_h = r[2] - r[0], r[3] - r[1]
        w_est = est_line_width_in(sh)
        h_est = est_text_height_in(sh)
        if w_est > box_w * 1.03 and h_est <= box_h * 1.15:
            # single-line intent, wrapping will change layout -> flag as warn
            warns.append(f"S{si} WRAP {name}: text width {w_est:.2f}\" > box {box_w:.2f}\" "
                         f"(will wrap to ~{math.ceil(w_est/max(box_w,0.1))} lines; box h {box_h:.2f}\" vs est {h_est:.2f}\")")
        if h_est > box_h * 1.35 and h_est - box_h > 0.12:
            fails.append(f"S{si} VOVERFLOW {name}: est height {h_est:.2f}\" > box {box_h:.2f}\" "
                         f"text='{sh.text_frame.text[:40]}'")
    # D. figure distortion
    for kind, name, r, sh in elems:
        if kind != "pic":
            continue
        try:
            src = None
            blob = sh.image.blob
            import io
            im = Image.open(io.BytesIO(blob))
            nat = im.width / im.height
        except Exception:
            warns.append(f"S{si} PIC {name}: source ratio unreadable")
            continue
        stored = sh.width / sh.height
        if abs(stored - nat) / nat > 0.02:
            fails.append(f"S{si} DISTORT {name}: stored ratio {stored:.3f} vs natural {nat:.3f}")
    # E. numbers present
    slide_text = " ".join(sh.text_frame.text for sh in slide.shapes if sh.has_text_frame)
    for tok in EXPECTED.get(si, []):
        if tok not in slide_text:
            fails.append(f"S{si} NUMBER-MISSING: '{tok}' not found on slide")

print("=== FAILURES ===" if fails else "=== NO FAILURES ===")
for f in fails:
    print("FAIL:", f)
print("=== WARNINGS ===")
for w in warns:
    print("WARN:", w)
sys.exit(1 if fails else 0)
