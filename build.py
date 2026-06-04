#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["pyyaml"]
# ///
"""Canonical Piatra paper build.

Renders <paper_dir>/PAPER.md -> <paper_dir>/PAPER.pdf via pandoc + xelatex,
using a superset LaTeX preamble (inlined below) whose only per-paper variable
is the running-header text.

Two modes, one file:
  * vendored   — copied to a paper repo root; `uv run build.py` builds that repo
                 (REPO = the directory this file sits in).
  * canonical  — `uv run build/build.py <repo_path>` builds an arbitrary repo
                 (REPO = argv[1]); this is what `papers build <slug>` invokes.

Configuration comes from <REPO>/metadata.yaml when present:
  paper_dir : "paper" (default) | "docs"   — where PAPER.md lives
  header    : running-header text          — falls back to `title`, then folder name
Everything else is convention. pandoc + xelatex are the only external tools;
the inlined preamble means the build has zero repo-relative dependencies.
"""

import os
import sys
import subprocess
import tempfile
import pathlib

import yaml

# Superset preamble: the union of every per-paper preamble that existed before
# the workspace was consolidated. The fullest papers (owned-causality, P-JEPA,
# gluing-leviathan) already carried all of this; humanities papers carried a
# subset. amsmath/amssymb/mathtools/amsthm are inert in prose-only papers, so
# the union is safe — verified by rebuilding every paper and diffing the PDFs.
# @@HEADER@@ is replaced with the running-header text at build time.
PREAMBLE = r"""
% Running headers
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small\textit{@@HEADER@@}}
\fancyhead[R]{\small\thepage}
\renewcommand{\headrulewidth}{0.4pt}

% Title page: no header, just centered page number at bottom
\fancypagestyle{plain}{\fancyhf{}\fancyfoot[C]{\thepage}\renewcommand{\headrulewidth}{0pt}}

% Title page styling
\usepackage{titling}
\pretitle{\vspace*{\fill}\begin{center}\LARGE\bfseries}
\posttitle{\end{center}\vskip 3em}
\preauthor{\begin{center}\large}
\postauthor{\end{center}\vskip 1.5em}
\predate{\begin{center}\large}
\postdate{\end{center}\vspace*{\fill}\thispagestyle{empty}\newpage}

% Math packages
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{mathtools}

% Tables
\usepackage{booktabs}

% Theorem environments
\usepackage{amsthm}
\newtheorem{definition}{Definition}
\newtheorem{proposition}{Proposition}
\newtheorem{lemma}{Lemma}
""".lstrip()


def tex_escape(s: str) -> str:
    """Escape the handful of TeX specials that can appear in a header title."""
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("#", r"\#"), ("_", r"\_"), ("$", r"\$")):
        s = s.replace(a, b)
    return s


def load_meta(repo: pathlib.Path) -> dict:
    p = repo / "metadata.yaml"
    if p.exists():
        return yaml.safe_load(p.read_text()) or {}
    return {}


def main() -> int:
    if len(sys.argv) > 1:
        repo = pathlib.Path(sys.argv[1]).resolve()
    else:
        repo = pathlib.Path(__file__).resolve().parent

    meta = load_meta(repo)
    paper_dir = repo / meta.get("paper_dir", "paper")
    header = meta.get("header") or meta.get("title") or repo.name

    md = paper_dir / "PAPER.md"
    pdf = paper_dir / "PAPER.pdf"
    if not md.exists():
        print(f"error: {md} not found", file=sys.stderr)
        return 1

    preamble = PREAMBLE.replace("@@HEADER@@", tex_escape(str(header)))
    tmp = tempfile.NamedTemporaryFile("w", suffix=".tex", delete=False)
    try:
        tmp.write(preamble)
        tmp.close()
        cmd = [
            "pandoc", str(md),
            "-o", str(pdf),
            "--pdf-engine=xelatex",
            "--resource-path", str(paper_dir),
            "-H", tmp.name,
            "-V", "mainfont=Palatino",
            "-V", "mathfont=Palatino",
            "-V", "monofont=Menlo",
            "-V", "fontsize=11pt",
            "-V", "geometry:margin=1in",
            "-V", "linestretch=1.15",
        ]
        subprocess.run(cmd, check=True)
    finally:
        os.unlink(tmp.name)

    print(pdf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
