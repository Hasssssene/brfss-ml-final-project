#!/usr/bin/env python3
"""Build report.pdf from report.md.

Pipeline (verified on this machine):
  1. pandoc report.md -> HTML fragment (markdown -> html5)
  2. Wrap in a standalone HTML page with report.css inlined and figure paths
     rewritten to absolute file:/// URIs (so Edge resolves them regardless of
     the HTML file's location).
  3. Microsoft Edge headless --print-to-pdf -> report.pdf.

Usage:  python build_report.py
Exit code 0 on success; non-zero with a message otherwise.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent  # repo root
MD = HERE / "report.md"
CSS = HERE / "report.css"
HTML = HERE / "report.html"  # intermediate, kept for debugging
PDF = HERE / "report.pdf"

EDGE_CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]

IMG_SRC = re.compile(r'src="((?:\.\./)+figures/[^"]+)"')


def find_edge() -> Path:
    for cand in EDGE_CANDIDATES:
        p = Path(cand)
        if p.exists():
            return p
    found = shutil.which("msedge")
    if found:
        return Path(found)
    raise SystemExit("Microsoft Edge not found (tried: " + ", ".join(EDGE_CANDIDATES) + ")")


def run_pandoc() -> str:
    if shutil.which("pandoc") is None:
        raise SystemExit("pandoc not found on PATH")
    proc = subprocess.run(
        ["pandoc", str(MD), "--from", "markdown+implicit_figures", "--to", "html5",
         "--wrap=none"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=True,
    )
    return proc.stdout


def absolutize_images(fragment: str) -> str:
    """Rewrite ../figures/x.png -> absolute file:/// URI so Edge always finds it."""

    def repl(m: re.Match) -> str:
        rel = m.group(1)
        uri = (MD.parent / rel).resolve().as_uri()
        return f'src="{uri}"'

    out = IMG_SRC.sub(repl, fragment)
    # Fail loudly if any relative image survived (a missing figure would print blank).
    leftover = re.findall(r'src="(\.\./[^"]+)"', out)
    if leftover:
        raise SystemExit(f"Unrewritten relative image src(s): {leftover}")
    missing = [u for u in IMG_SRC.findall(fragment) if not (MD.parent / u).resolve().exists()]
    if missing:
        raise SystemExit(f"Figure file(s) not found: {missing}")
    return out


def build_html(body: str) -> None:
    css = CSS.read_text(encoding="utf-8")
    page = (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        "<title>BRFSS 2021 GENHLTH binary classification — technical report</title>\n"
        f"<style>\n{css}\n</style>\n</head>\n<body>\n{body}\n</body>\n</html>\n"
    )
    HTML.write_text(page, encoding="utf-8")


def print_pdf() -> None:
    edge = find_edge()
    cmd = [
        str(edge),
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={PDF}",
        HTML.as_uri(),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=300)
    if proc.returncode != 0 or not PDF.exists():
        raise SystemExit(
            f"Edge print-to-pdf failed (rc={proc.returncode}): "
            f"{proc.stdout}\n{proc.stderr}"
        )


def pdf_stats() -> tuple[int, int | None]:
    """Return (size bytes, page count or None if undetectable)."""
    data = PDF.read_bytes()
    size = len(data)
    pages = None
    # Best-effort page count: uncompressed /Type /Page markers, else /Count N in the tree.
    n_marker = len(re.findall(rb"/Type\s*/Page[^s]", data))
    if n_marker:
        pages = n_marker
    else:
        counts = [int(m) for m in re.findall(rb"/Type\s*/Pages[^>]*?/Count\s+(\d+)", data)]
        counts += [int(m) for m in re.findall(rb"/Count\s+(\d+)[^>]*?/Type\s*/Pages", data)]
        if counts:
            pages = max(counts)
    return size, pages


def main() -> None:
    if not MD.exists():
        raise SystemExit(f"missing {MD}")
    print(f"[1/4] pandoc  : {MD.name} -> html fragment")
    body = absolutize_images(run_pandoc())
    print(f"[2/4] wrap    : standalone html with inlined {CSS.name}, absolute figure URIs -> {HTML.name}")
    build_html(body)
    print(f"[3/4] edge    : headless print-to-pdf -> {PDF.name}")
    print_pdf()
    size, pages = pdf_stats()
    print(f"[4/4] verify  : {PDF} exists, {size:,} bytes, pages={pages}")
    if size < 200_000:
        raise SystemExit(f"PDF suspiciously small ({size:,} bytes < 200,000) — check figures")
    print("OK")


if __name__ == "__main__":
    main()
