#!/usr/bin/env python3
"""Assemble and compile the v7 LaTeX report in report_v7/.

report_v7/main.tex is the single source of the text and refers to figures by
label (\\ref{fig:<key>}). For every figure below this script copies the
panels PDF to report_v7/figures/ and writes figures/<key>.tex, the figure
environment with the caption written by the plotting script, so numbering and
cross-references come from LaTeX. Compiles main.pdf with tectonic; the web
page shows this same PDF.

Usage: python build_report_v7.py
"""
import argparse
import re
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
WD2 = HERE.parent / "wheel_decoding2" / "output"
REPORT = HERE / "report_v7"

FIGURES = {  # key in main.tex: output stem (path without suffix)
    "windows": OUT / "dorsal_windows",
    "counts": OUT / "v2" / "v2_counts",
    "tuning": OUT / "v2" / "v2_tuning",
    "decoding_n20": OUT / "v2" / "v2_decoding_n20",
    "decoding_all": OUT / "v2" / "v2_decoding_all",
    "decoding_top10": OUT / "v2" / "v2_decoding_top10",
    "region_variance": OUT / "v2" / "region_variance",
    "summary": OUT / "v2" / "v2_summary",
    "meso_size_scaling": OUT / "v2" / "meso_size_scaling",
    "depth": OUT / "v2" / "si_depth_profile",
    "depth_coverage": OUT / "v2" / "si_depth_coverage",
    "rasters": WD2 / "fig1_rasters",
    "checks": WD2 / "fig2_checks",
    "population": WD2 / "fig3_population",
}


def caption_of(stem):
    tex = stem.with_suffix(".tex").read_text()
    return re.search(r"\\caption\{(.*)\}\s*\n\s*\\label", tex, re.S).group(1)


def width_of(stem):
    tex = stem.with_suffix(".tex").read_text()
    return float(re.search(r"\\begin\{minipage\}\{([\d.]+)in\}", tex).group(1))


def figure_tex(key, stem):
    w = width_of(stem)
    w = "\\linewidth" if w > 6.9 else f"{w:.2f}in"
    return (f"\\begin{{figure}}[!htbp]\n  \\centering\n"
            f"  \\includegraphics[width={w},height=0.8\\textheight,keepaspectratio]{{figures/{key}.pdf}}\n"
            f"  \\caption{{{caption_of(stem)}}}\n  \\label{{fig:{key}}}\n\\end{{figure}}\n")


def build(decoding_only=False, figures=None):
    fig_dir = REPORT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    keys = ("decoding_n20", "decoding_all", "decoding_top10") if decoding_only else (figures or FIGURES)
    for key in keys:
        stem = FIGURES[key]
        shutil.copy(stem.parent / f"{stem.name}_panels.pdf", fig_dir / f"{key}.pdf")
        (fig_dir / f"{key}.tex").write_text(figure_tex(key, stem))
    subprocess.run(["tectonic", "-X", "compile", "main.tex"], cwd=REPORT, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--decoding-only", action="store_true",
                       help="Replace Figures 1 through 3 and reuse the other report panels.")
    group.add_argument("--figure", action="append", choices=FIGURES,
                       help="Replace a named figure. Repeat to select multiple figures.")
    args = parser.parse_args()
    build(decoding_only=args.decoding_only, figures=args.figure)
