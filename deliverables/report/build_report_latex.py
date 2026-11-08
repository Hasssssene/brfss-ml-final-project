#!/usr/bin/env python3
"""Build report.pdf from report.md via LaTeX (pandoc -> tectonic/XeTeX), v1.2 style.

Matches the 'inspire' template: default-LaTeX article, 11pt, A4, Latin Modern,
auto-numbered sections, TOC, plain page numbers, all black.
Content frozen: report.md is never modified (in-memory copy only).

Fixes vs first attempt:
  - small tables converted longtable -> tabular (never split across pages);
    only the 42-row appendix table stays longtable (with repeated header)
  - figures: centered image + clean caption line (no pandoc alt-text captions,
    no source parentheticals)
  - unicode -> LaTeX math so Latin Modern covers every glyph
"""
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
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
{\LARGE\bfseries $title$}\par\vspace{1.0em}
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
    '\u00b7': r'$\cdot$', '\u03b1': r'$\alpha$', '\u2026': r'\ldots{}',
    '\u2014': '---', '\u2013': '--', '\u00a7': r'\S{}',
    '\u207b': r'\textsuperscript{-}', '\u2070': r'\textsuperscript{0}',
    '\u00b9': r'\textsuperscript{1}', '\u00b2': r'\textsuperscript{2}',
    '\u00b3': r'\textsuperscript{3}', '\u2074': r'\textsuperscript{4}',
    '\u2076': r'\textsuperscript{6}', '\u2077': r'\textsuperscript{7}',
}
# parenthetical tails we strip from figure captions (sources, not content)
CAPTION_TAIL = re.compile(
    r'\s*\(([^()]*(?:results/|notebook|src/|vault/|\u00a7)[^()]*)\)\s*$', re.I)

MAX_TABULAR_ROWS = 18   # <= this becomes unbreakable tabular; bigger tables must
                        # stay longtable (taller-than-page tabulars clip silently!)


def preprocess(md: str) -> str:
    md = md.lstrip('\ufeff')
    # title block
    title_m = re.search(r'^# (.+)$', md, flags=re.M)
    title = title_m.group(1).strip()
    md = md.replace(title_m.group(0), '', 1)
    md = re.sub(r'^- \*\*Author:\*\*.*$\n?', '', md, flags=re.M)
    md = re.sub(r'^- \*\*Date:\*\*.*$\n?', '', md, flags=re.M)
    course = 'Machine Learning final project --- University of Alicante (C2 call)'
    md = re.sub(r'^- \*\*Course:\*\*.*$\n?', '', md, flags=re.M)
    md = md.replace('Report status: final draft v1.0', 'Report status: final draft v1.2')

    # abstract blockquote -> variable (renders above the rule, under title)
    abs_m = re.search(r'^> \*\*Abstract\.\*\*(.+?)(?:\n\n|\n## )', md, flags=re.M | re.S)
    abstract = ''
    if abs_m:
        abstract = ' '.join(abs_m.group(1).split())
        md = md.replace(abs_m.group(0), abs_m.group(0)[-4:] if abs_m.group(0).endswith('## ') else '', 1)

    # figures: clean alt text, emit centered image + italic caption line
    def figure(m):
        alt, path = m.group(1), m.group(2)
        alt = CAPTION_TAIL.sub('', alt.strip())
        cap = f'\n\n*{alt}*' if alt else ''
        return f'![]({path}){cap}'
    md = re.sub(r'^!\[([^\]]*)\]\(([^)]+)\)$', figure, md, flags=re.M)

    # absolute figure paths
    md = re.sub(r'\((?:\.\./)+figures/([^)]+)\)', rf'({FIGABS}/\1)', md)

    # heading numbers off (LaTeX numbers sections itself)
    md = re.sub(r'^(#{2,4}) \d+(?:\.\d+)*\.? ', r'\1 ', md, flags=re.M)

    return md, title, course, abstract


def fix_tables(tex: str) -> tuple[str, int, int]:
    """longtable -> centered unbreakable tabular for short tables (spliced in
    place; everything around each table is preserved byte-for-byte)."""
    n_tab, n_long = 0, 0
    out, last = [], 0
    for m in re.finditer(r'\\begin\{longtable\}.*?\\end\{longtable\}', tex, re.S):
        inner = m.group(0)
        # strip longtable-only markers (single-header pandoc 3.12 layout:
        # header rows ... \endhead ... \bottomrule\noalign{}\endlastfoot)
        stripped = re.sub(r'\\endfirsthead\s*\n?', '', inner)
        stripped = re.sub(r'\\endhead\s*\n?', '', stripped)
        stripped = re.sub(r'\\endfoot\s*\n?', '', stripped)
        stripped = re.sub(r'\\bottomrule\s*\\noalign\{\}\s*\\endlastfoot\s*\n?', '', stripped)
        stripped = re.sub(r'\\endlastfoot\s*\n?', '', stripped)
        n_rows = len(re.findall(r'\\\\\n', stripped))
        if n_rows <= MAX_TABULAR_ROWS:
            conv = stripped.replace(r'\begin{longtable}[]{@{}', r'\begin{tabular}{@{}', 1)
            # one clean bottom rule at the end, then close tabular
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
    md, title, course, abstract = preprocess(MD.read_text(encoding='utf-8'))
    TPL.write_text(TEMPLATE, encoding='utf-8')
    body_md = HERE / '_body_build.md'
    body_md.write_text(md, encoding='utf-8')

    r = subprocess.run(
        ['pandoc', str(body_md), '--from', 'markdown-implicit_figures', '--to', 'latex',
         '--template', str(TPL), '--number-sections', '--shift-heading-level-by', '-1',
         '-V', f'title={title}', '-V', 'author=Hassene Bouamama',
         '-V', f'course={course}',
         '-V', 'date=November 2026 --- final draft v1.2 (compiled 2026-11-08)',
         '-V', f'abstract={abstract}',
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
