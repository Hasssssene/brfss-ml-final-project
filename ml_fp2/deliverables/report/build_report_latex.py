#!/usr/bin/env python3
"""Build report.pdf from report.md via LaTeX (pandoc -> tectonic/XeTeX).

v2 port of the proven v1.2 builder: default-LaTeX article, 11pt, A4, Latin
Modern, auto-numbered sections, TOC, plain page numbers, all black.
Content frozen: report.md is never modified (in-memory copy only).

Adapted for the ml_fp2 report format:
  - front matter: subtitle / course / Author: / Date: plain lines (not list items)
  - manual "## Contents" block removed (LaTeX generates the TOC)
  - figures: alt text blanked; the explicit *Figure N: ...* caption lines are
    kept (source parentheticals stripped), centered under the image
  - MAX_TABULAR_ROWS = 12: the 14-row assistance-log table must stay longtable
    (taller-than-page tabulars clip silently!)
"""
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent          # ml_fp2/
MD = HERE / 'report.md'
PDF = HERE / 'report.pdf'
TEX = HERE / 'report_build.tex'
TPL = HERE / 'template_build.tex'
FIGABS = (ROOT / 'figures').resolve().as_posix()

TEMPLATE = r'''\documentclass[11pt,a4paper]{article}
\usepackage[a4paper,margin=2.5cm]{geometry}
\usepackage{lmodern}
\usepackage[T1]{fontenc}
\usepackage{graphicx}
\setkeys{Gin}{width=0.88\linewidth,height=0.68\textheight,keepaspectratio}
\usepackage{longtable,booktabs,array,etoolbox,calc}
\usepackage{enumitem}
\setlist{topsep=3pt,itemsep=2pt,parsep=0pt}
\usepackage{xcolor}
\usepackage[hidelinks]{hyperref}
\setlength{\emergencystretch}{3em}
\setlength{\parskip}{5pt plus 2pt}
\widowpenalty=10000
\providecommand{\tightlist}{\setlength{\itemsep}{2pt}\setlength{\parskip}{0pt}}
\providecommand{\pandocbounded}[1]{#1}
\begin{document}

\begin{center}
{\LARGE\bfseries $title$}\par\vspace{0.8em}
{\small\itshape $subtitle$}\par\vspace{1.0em}
{\normalsize $author$}\par\vspace{0.3em}
{\small $course$}\par\vspace{0.8em}
{\small\itshape $date$}\par
\end{center}
\vspace{0.6em}

{\small $abstract$\par}
\vspace{1.2em}
\hrule
\vspace{1.2em}

\tableofcontents
\newpage
$body$
\end{document}
'''

# unicode -> LaTeX (math mode where Latin Modern text lacks the glyph)
UNICODE_MAP = {
    '\u2192': r'$\rightarrow$', '\u00d7': r'$\times$', '\u03c1': r'$\rho$',
    '\u00b1': r'$\pm$', '\u0394': r'$\Delta$', '\u2248': r'$\approx$',
    '\u2208': r'$\in$', '\u2264': r'$\le$', '\u2265': r'$\ge$',
    '\u2212': '-', '\u2013': '--', '\u03c4': r'$\tau$', '\u03a3': r'$\Sigma$',
    '\u2261': r'$\equiv$', '\u00f7': r'$\div$', '\u00bd': r'$\frac{1}{2}$',
    '\u2229': r'$\cap$', '\u222a': r'$\cup$', '\u03c3': r'$\sigma$',
    '\u2286': r'$\subseteq$', '\u2260': r'$\neq$',
    '\u00b7': r'$\cdot$', '\u03b1': r'$\alpha$', '\u2026': r'\ldots{}',
    '\u2014': '---', '\u00a7': r'\S{}',
    '\u207b': r'\textsuperscript{-}', '\u2070': r'\textsuperscript{0}',
    '\u00b9': r'\textsuperscript{1}', '\u00b2': r'\textsuperscript{2}',
    '\u00b3': r'\textsuperscript{3}', '\u2074': r'\textsuperscript{4}',
    '\u2076': r'\textsuperscript{6}', '\u2077': r'\textsuperscript{7}',
}
# parenthetical tails we strip from figure captions (sources, not content)
CAPTION_TAIL = re.compile(
    r'\s*\(([^()]*(?:results/|notebook|src/|vault/|\u00a7|EXP-)[^()]*)\)\s*$', re.I)

MAX_TABULAR_ROWS = 12   # <= this becomes unbreakable tabular; bigger tables must
                        # stay longtable (taller-than-page tabulars clip silently!)


def preprocess(md: str):
    md = md.lstrip('\ufeff')
    # title
    title_m = re.search(r'^# (.+)$', md, flags=re.M)
    title = title_m.group(1).strip()
    md = md.replace(title_m.group(0), '', 1)

    # front matter: subtitle / course / author / date (v2 plain-line format)
    sub_m = re.search(r'^\*\*([^\n*]+)\*\*\s*$', md, flags=re.M)
    subtitle = sub_m.group(1).strip() if sub_m else ''
    if sub_m:
        md = md.replace(sub_m.group(0), '', 1)
    course_m = re.search(r'^(Machine Learning[^\n]*Final Project[^\n]*)$', md, flags=re.M)
    course = course_m.group(1).strip() if course_m else 'Machine Learning --- Final Project'
    if course_m:
        md = md.replace(course_m.group(0), '', 1)
    auth_m = re.search(r'^Author: (.+)$', md, flags=re.M)
    author = auth_m.group(1).strip() if auth_m else 'Hassene Bouamama'
    if auth_m:
        md = md.replace(auth_m.group(0), '', 1)
    date_m = re.search(r'^Date: (.+)$', md, flags=re.M)
    date = date_m.group(1).strip() if date_m else 'November 2026'
    if date_m:
        md = md.replace(date_m.group(0), '', 1)

    # abstract blockquote -> variable (renders above the rule, under title)
    abs_m = re.search(r'^> \*\*Abstract\.\*\*(.+?)(?:\n\n|\n## )', md, flags=re.M | re.S)
    abstract = ''
    if abs_m:
        abstract = ' '.join(abs_m.group(1).split())
        md = md.replace(abs_m.group(0), '', 1)

    # manual contents block -> gone (LaTeX builds the real TOC)
    md = re.sub(r'\n## Contents\n.*?\n---\n', '\n', md, flags=re.S)
    # decorative horizontal rules (sections separate visually in LaTeX)
    md = re.sub(r'^---\s*$\n?', '', md, flags=re.M)

    # figures: blank alt text (explicit caption lines follow in this report)
    md = re.sub(r'^!\[[^\]]*\]\(([^)]+)\)$', r'![](\1)', md, flags=re.M)
    # caption lines: strip source parentheticals (notebook section / EXP ids)
    def cap(m):
        return '*{}*'.format(CAPTION_TAIL.sub('', m.group(1).rstrip()))
    md = re.sub(r'^\*(Figure \d+[A-C]?:[^\n]*)\*$', cap, md, flags=re.M)

    # absolute figure paths
    md = re.sub(r'\((?:\.\./)+figures/([^)]+)\)', rf'({FIGABS}/\1)', md)

    # heading numbers off (LaTeX numbers sections itself)
    md = re.sub(r'^(#{2,4}) \d+(?:\.\d+)*\.? ', r'\1 ', md, flags=re.M)

    return md, title, subtitle, course, author, date, abstract


def fix_tables(tex: str) -> tuple[str, int, int]:
    """longtable -> centered unbreakable tabular for short tables (spliced in
    place; everything around each table is preserved byte-for-byte)."""
    n_tab, n_long = 0, 0
    out, last = [], 0
    for m in re.finditer(r'\\begin\{longtable\}.*?\\end\{longtable\}', tex, re.S):
        inner = m.group(0)
        # strip longtable-only markers (single-header pandoc 3.12 layout)
        stripped = re.sub(r'\\endfirsthead\s*\n?', '', inner)
        stripped = re.sub(r'\\endhead\s*\n?', '', stripped)
        stripped = re.sub(r'\\endfoot\s*\n?', '', stripped)
        stripped = re.sub(r'\\bottomrule\s*\\noalign\{\}\s*\\endlastfoot\s*\n?', '', stripped)
        stripped = re.sub(r'\\endlastfoot\s*\n?', '', stripped)
        n_rows = len(re.findall(r'\\\\\n', stripped))
        if n_rows <= MAX_TABULAR_ROWS:
            conv = stripped.replace(r'\begin{longtable}[]{@{}', r'\begin{tabular}{@{}', 1)
            conv = re.sub(r'\\bottomrule\s*(?:\\noalign\{\}\s*)?\\end\{longtable\}',
                          '\\\\bottomrule\n\\\\end{tabular}', conv)
            conv = conv.replace(r'\end{longtable}', '\\bottomrule\n\\end{tabular}')
            out.append(tex[last:m.start()])
            out.append('\n\\begin{center}\\small\n' + conv + '\n\\end{center}\n')
            last = m.end()
            n_tab += 1
        else:
            n_long += 1
    out.append(tex[last:])
    return ''.join(out), n_tab, n_long


def center_figures(tex: str) -> str:
    lines = tex.splitlines()
    out = []
    for l in lines:
        s = l.strip()
        if re.fullmatch(r'(\\pandocbounded\{)?\\includegraphics\{[^}]+\}\}?', s):
            out.append(r'\begin{center}' + s + r'\end{center}')
        elif re.fullmatch(r'\\emph\{(Figure|Table)[^}]*\}', s):
            out.append(r'{\small\begin{center}' + s + r'\end{center}}')
        else:
            out.append(l)
    return '\n'.join(out)


def main():
    md, title, subtitle, course, author, date, abstract = preprocess(
        MD.read_text(encoding='utf-8'))
    TPL.write_text(TEMPLATE, encoding='utf-8')
    body_md = HERE / '_body_build.md'
    body_md.write_text(md, encoding='utf-8')

    r = subprocess.run(
        ['pandoc', str(body_md), '--from', 'markdown-implicit_figures', '--to', 'latex',
         '--template', str(TPL), '--number-sections', '--shift-heading-level-by', '-1',
         '--wrap=none', '--no-highlight',
         '-V', f'title={title}', '-V', f'subtitle={subtitle}',
         '-V', f'author={author}', '-V', f'course={course}',
         '-V', f'date={date}', '-V', f'abstract={abstract}',
         '-o', str(TEX)], capture_output=True, text=True)
    body_md.unlink(missing_ok=True)
    if r.returncode != 0:
        sys.exit('pandoc failed: ' + r.stderr[:2000])

    tex = TEX.read_text(encoding='utf-8')
    tex, n_tab, n_long = fix_tables(tex)
    tex = center_figures(tex)
    # unicode -> LaTeX math AFTER pandoc (pandoc escapes $ in plain text,
    # so math injection must happen on the final .tex)
    for k, v in UNICODE_MAP.items():
        tex = tex.replace(k, v)
    tex = tex.replace('$\\frac{1}{2}$', '$1/2$')
    TEX.write_text(tex, encoding='utf-8')
    print(f'[1/3] tex ready: {n_tab} tabular (unbreakable), {n_long} longtable (split w/ header)')

    r = subprocess.run(['tectonic', str(TEX)], capture_output=True, text=True, timeout=1200)
    if r.returncode != 0 or not (HERE / 'report_build.pdf').exists():
        tail = (r.stdout + r.stderr)[-2500:]
        sys.exit('tectonic failed:\n' + tail)
    (HERE / 'report_build.pdf').replace(PDF)
    print(f'[2/3] compiled: {PDF} {PDF.stat().st_size:,} bytes')
    print('[3/3] OK')


if __name__ == '__main__':
    main()
