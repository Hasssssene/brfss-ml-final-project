#!/usr/bin/env python3
"""Build report.pdf from report.md via real LaTeX (pandoc -> tectonic/XeTeX).

v1.2 typography: Times New Roman (system), justified, longtable+booktabs,
numbered-page footer, running head. CONTENT IS FROZEN: report.md is never
modified; a preprocessed in-memory copy only (a) moves title/author/date into
a LaTeX title block, (b) rewrites figure paths to absolute, (c) bumps the
status label to v1.2. Every number, table and figure is byte-preserved.
"""
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
MD = HERE / 'report.md'
PDF = HERE / 'report.pdf'
FIGABS = (ROOT / 'figures').resolve().as_posix()

TEMPLATE = r'''
\documentclass[11pt,a4paper]{article}
\usepackage[margin=1in]{geometry}
\usepackage{fontspec}
\setmainfont{Times New Roman}
\setmonofont{Consolas}
\usepackage{graphicx}
\setkeys{Gin}{width=0.9\linewidth,height=0.72\textheight,keepaspectratio}
\usepackage{longtable,booktabs,array,etoolbox,calc}
\AtBeginEnvironment{longtable}{\footnotesize}
\usepackage{enumitem}
\setlist{topsep=3pt,itemsep=2pt,parsep=0pt}
\usepackage{fancyhdr}
\pagestyle{fancy}\fancyhf{}
\fancyhead[L]{\small\itshape BRFSS 2021 --- Fair/Poor Health Prediction}
\fancyhead[R]{\small\sffamily v1.2}
\fancyfoot[C]{\thepage}
\renewcommand{\headrulewidth}{0.3pt}
\usepackage{xcolor}
\usepackage[colorlinks=true,linkcolor=blue!45!black,urlcolor=blue!45!black]{hyperref}
\setlength{\emergencystretch}{3em}
\setlength{\parskip}{4pt plus 2pt}
\widowpenalty=10000
\providecommand{\tightlist}{\setlength{\itemsep}{2pt}\setlength{\parskip}{0pt}}
\providecommand{\pandocbounded}[1]{#1}
\begin{document}
\begin{center}
{\LARGE\bfseries $title$}\par\vspace{0.9em}
{\large $author$}\par\vspace{0.35em}
{\small\itshape $date$}\par\vspace{0.25em}
\rule{0.6\linewidth}{0.4pt}
\end{center}
$body$
\end{document}
'''

def main():
    md = MD.read_text(encoding='utf-8')

    # (a) title / author / date -> LaTeX title block
    title_m = re.search(r'^# (.+)$', md, flags=re.M)
    title = title_m.group(1).strip()
    md = md.replace(title_m.group(0), '', 1)
    author = 'Hassene Bouamama'
    md = re.sub(r'^- \*\*Author:\*\*.*$\n?', '', md, flags=re.M)
    date = 'November 2026 --- final draft v1.2 (compiled 2026-11-08)'
    md = re.sub(r'^- \*\*Date:\*\*.*$\n?', '', md, flags=re.M)
    md = md.replace('Report status: final draft v1.0', 'Report status: final draft v1.2')

    # (b) absolute figure paths (forward slashes, XeTeX-safe)
    md = re.sub(r'\((?:\.\./)+figures/([^)]+)\)', rf'({FIGABS}/\1)', md)

    (HERE / 'report_v1_2.tex') if False else None
    tpl = HERE / 'template_v1_2.tex'
    tpl.write_text(TEMPLATE, encoding='utf-8')

    tmp_md = HERE / '_body_v1_2.md'
    tmp_md.write_text(md, encoding='utf-8')

    cmd = [
        'pandoc', str(tmp_md), '--from', 'markdown', '--to', 'latex',
        '--template', str(tpl), '--shift-heading-level-by', '-1',
        '--pdf-engine', 'tectonic',
        '-V', f'title={title}', '-V', f'author={author}', '-V', f'date={date}',
        '-o', str(PDF),
    ]
    print('[1/2] pandoc + tectonic ...')
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    tmp_md.unlink(missing_ok=True)
    if r.returncode != 0 or not PDF.exists():
        print(r.stdout[-3000:]); print(r.stderr[-3000:])
        sys.exit(f'FAILED rc={r.returncode}')
    print(f'[2/2] OK: {PDF} {PDF.stat().st_size:,} bytes')

if __name__ == '__main__':
    main()
